using System;
using System.Runtime.InteropServices;
using NativeRender;
using Nri;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Maximal-fidelity replication of the original RTXPT ToneMappingPasses.cpp auto-exposure + tone-map
    /// path, end to end, reusing the original shaders:
    ///
    ///   1. luminance  — <c>Luminance.rastershader</c> (#includes luminance_ps.hlsl) writes
    ///                   log2(luminance) into mip-0 of a pow2 luminance pyramid (RTV).
    ///   2. mip reduce — <c>LuminanceMip.computeshader</c> (donut MipMapGenPass-style 2x2 box average,
    ///                   native compute via a bindless per-mip UAV array) builds the pyramid; the 1x1
    ///                   top mip is the geometric-mean log-luminance. NOT Unity GenerateMips, so it
    ///                   never leaves a sampler heap bound between native dispatches.
    ///   3. capture    — <c>capture_cs.computeshader</c> (#includes ToneMapping.hlsl's capture_cs
    ///                   VERBATIM) copies the top mip into the 1-float auto-exposure buffer.
    ///   4. apply      — <c>ToneMapping.rastershader</c> (#includes ToneMapping.ps.hlsli) tone-maps.
    ///
    /// This is the faithful "luminance_ps + GenerateMips + capture_cs + tonemap" chain, done natively.
    /// Selected by NativeRtxptSetting.useMipChainAutoExposure; otherwise the single-pass ReduceLuminance
    /// path (NativeRtxptToneMappingPass / -RasterPass) is used.
    /// </summary>
    public class NativeRtxptToneMappingMipChainPass : ScriptableRenderPass, IDisposable
    {
        private const float kExposureKey = 0.042f;
        private const uint  kR32F  = (uint)DXGI_FORMAT.DXGI_FORMAT_R32_FLOAT;
        private const uint  kRGBA16F = (uint)DXGI_FORMAT.DXGI_FORMAT_R16G16B16A16_FLOAT;

        [StructLayout(LayoutKind.Sequential)]
        private struct ToneMappingConstants
        {
            public float   whiteScale, whiteMaxLuminance;
            public uint    toneMapOperator, clamped, autoExposure;
            public float   avgLuminance, autoExposureLumValueMin, autoExposureLumValueMax;
            public Vector4 colorTransform0, colorTransform1, colorTransform2;
            public uint    enabled, _padding0, _padding1, _padding2;
        }

        // MipConstants — root constants for LuminanceMip.computeshader (8 × u32).
        [StructLayout(LayoutKind.Sequential)]
        private struct MipConstants
        {
            public uint srcMip, dstMip, srcW, srcH, dstW, dstH, _p0, _p1;
        }

        // ── Resolution-independent pipelines (built once) ──
        private readonly NativeRasterPipeline       _lumRaster;
        private readonly NativeRasterDescriptorSet  _lumDs;
        private readonly NativeComputePipeline       _mipCs;
        private readonly NativeComputePipeline       _captureCs;
        private readonly NativeComputeDescriptorSet  _captureDs;
        private readonly NativeRasterPipeline        _applyRaster;
        private readonly NativeRasterDescriptorSet   _applyDs;

        private readonly DeviceBuffer           _avgLuminanceBuffer;
        private readonly VolatileConstantBuffer _applyCb;

        private readonly IntPtr[] _applyColorRes = new IntPtr[1];
        private readonly uint[]   _applyColorFmt = { kRGBA16F };
        private readonly IntPtr[] _lumColorRes   = new IntPtr[1];
        private readonly uint[]   _lumColorFmt   = { kR32F };

        // ── Resolution-dependent (rebuilt on resize) ──
        private NriTextureResource           _lumTex;       // pow2, mips, R32F
        private BindlessUAVTexture           _mipUav;       // one slot per mip
        private NativeComputeDescriptorSet[] _mipSets;      // one per destination mip (1..top)
        private int  _lumW, _lumH, _mipCount;

        private NativeRtxptPassContext _ctx;
        private IntPtr                 _sourcePtr;
        private IntPtr                 _outputPtr;

        public NativeRtxptToneMappingMipChainPass(
            NativeRasterShader  luminanceRasterShader,
            NativeComputeShader luminanceMipCs,
            NativeComputeShader captureCs,
            NativeRasterShader  toneMapRasterShader)
        {
            _lumRaster   = new NativeRasterPipeline(luminanceRasterShader,
                               NativeRenderPlugin.RasterPipelineStateDesc.FullscreenOpaque(kR32F));
            _lumDs       = new NativeRasterDescriptorSet(_lumRaster);

            _mipCs       = new NativeComputePipeline(luminanceMipCs);

            _captureCs   = new NativeComputePipeline(captureCs);
            _captureDs   = new NativeComputeDescriptorSet(_captureCs);

            _applyRaster = new NativeRasterPipeline(toneMapRasterShader,
                               NativeRenderPlugin.RasterPipelineStateDesc.FullscreenOpaque(kRGBA16F));
            _applyDs     = new NativeRasterDescriptorSet(_applyRaster);

            _avgLuminanceBuffer = new DeviceBuffer(sizeof(uint));
            _applyCb            = new VolatileConstantBuffer(Marshal.SizeOf<ToneMappingConstants>());
        }

        public void Dispose()
        {
            _lumDs?.Dispose();        _lumRaster?.Dispose();
            DisposeMipSets();
            _mipUav?.Dispose();
            _mipCs?.Dispose();
            _captureDs?.Dispose();    _captureCs?.Dispose();
            _applyDs?.Dispose();      _applyRaster?.Dispose();
            _avgLuminanceBuffer?.Dispose();
            _applyCb?.Dispose();
            _lumTex?.Release();
        }

        private void DisposeMipSets()
        {
            if (_mipSets == null) return;
            foreach (var s in _mipSets) s?.Dispose();
            _mipSets = null;
        }

        public void Setup(NativeRtxptPassContext ctx, NriTextureResource source, NriTextureResource output)
        {
            _ctx       = ctx;
            _sourcePtr = source.NativePtr;
            _outputPtr = output.NativePtr;
            EnsureResources(ctx.DisplayResolution);
        }

        // (Re)create the pow2 luminance pyramid + per-mip bindless UAV + per-mip descriptor sets.
        private void EnsureResources(int2 displayRes)
        {
            int w = 1 << (int)Mathf.Floor(Mathf.Log(Mathf.Max(2, displayRes.x), 2f));
            int h = 1 << (int)Mathf.Floor(Mathf.Log(Mathf.Max(2, displayRes.y), 2f));
            if (_lumTex != null && _lumTex.IsCreated && w == _lumW && h == _lumH)
                return;

            _lumW = w; _lumH = h;

            _lumTex ??= new NriTextureResource("Rtxpt_Luminance", GraphicsFormat.R32_SFloat,
                            new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE_STORAGE,
                                                   layout = Layout.SHADER_RESOURCE_STORAGE, stageBits = 1 << 10 });
            _lumTex.Allocate(new int2(w, h), 1, useMipMap: true);
            _mipCount = _lumTex.rt.mipmapCount;

            // Bindless UAV: slot k → mip k of the luminance texture.
            _mipUav?.Dispose();
            _mipUav = new BindlessUAVTexture(_mipCount);
            for (int k = 0; k < _mipCount; k++)
                _mipUav.SetTexture(k, _lumTex.rt, mipSlice: k, dxgiFormat: kR32F);

            // One descriptor set per destination mip (1..mipCount-1). Separate sets keep each mip's
            // root-constants/binding snapshot in its own ring slot (a single set's 8-deep ring would
            // wrap for >8 mips and clobber in-flight dispatch data).
            DisposeMipSets();
            int dstCount = Mathf.Max(0, _mipCount - 1);
            _mipSets = new NativeComputeDescriptorSet[dstCount];
            for (int i = 0; i < dstCount; i++)
                _mipSets[i] = new NativeComputeDescriptorSet(_mipCs);
        }

        private static ToneMappingConstants BuildConstants(NativeRtxptSetting s)
        {
            bool auto = s.autoExposure;
            float exposureScale       = Mathf.Pow(2f, s.exposureCompensation);
            float manualExposureScale = 1f;
            if (!auto)
            {
                float ev      = s.exposureValue;
                float shutter = Mathf.Clamp(Mathf.Pow(2f, ev) / (s.fNumber * s.fNumber), 0.001f, 10000f);
                manualExposureScale = (s.filmSpeed / 100f) / (shutter * s.fNumber * s.fNumber);
            }
            float k = exposureScale * manualExposureScale;
            float lumMin = Mathf.Pow(2f, auto ? s.exposureValueMin : -16f);
            float lumMax = Mathf.Pow(2f, auto ? s.exposureValueMax :  16f);
            return new ToneMappingConstants
            {
                whiteScale = s.toneMapWhiteScale, whiteMaxLuminance = s.toneMapWhiteMaxLuminance,
                toneMapOperator = (uint)s.toneMapOperator, clamped = s.toneMapClamped ? 1u : 0u,
                autoExposure = auto ? 1u : 0u, avgLuminance = 0f,
                autoExposureLumValueMin = lumMin, autoExposureLumValueMax = lumMax,
                colorTransform0 = new Vector4(k, 0, 0, 0),
                colorTransform1 = new Vector4(0, k, 0, 0),
                colorTransform2 = new Vector4(0, 0, k, 0),
                enabled = s.enableToneMapping ? 1u : 0u,
            };
        }

        private class PassData
        {
            internal NativeRasterPipeline       LumRaster;
            internal NativeRasterDescriptorSet  LumDs;
            internal NativeComputePipeline       MipCs;
            internal NativeComputeDescriptorSet[] MipSets;
            internal BindlessUAVTexture          MipUav;
            internal NativeComputePipeline       CaptureCs;
            internal NativeComputeDescriptorSet  CaptureDs;
            internal NativeRasterPipeline        ApplyRaster;
            internal NativeRasterDescriptorSet   ApplyDs;
            internal DeviceBuffer                AvgLumBuffer;
            internal VolatileConstantBuffer      ApplyCb;
            internal ToneMappingConstants        Cb;
            internal IntPtr SourcePtr, OutputPtr, LumTexPtr;
            internal int2   Resolution;
            internal int    LumW, LumH, MipCount;
            internal bool   AutoExposure;
            internal IntPtr[] ApplyColorRes, LumColorRes;
            internal uint[]   ApplyColorFmt, LumColorFmt;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxpt.ToneMappingMipChain", out var pd);

            pd.LumRaster = _lumRaster; pd.LumDs = _lumDs;
            pd.MipCs = _mipCs; pd.MipSets = _mipSets; pd.MipUav = _mipUav;
            pd.CaptureCs = _captureCs; pd.CaptureDs = _captureDs;
            pd.ApplyRaster = _applyRaster; pd.ApplyDs = _applyDs;
            pd.AvgLumBuffer = _avgLuminanceBuffer; pd.ApplyCb = _applyCb;
            pd.Cb = BuildConstants(_ctx.Setting);
            pd.SourcePtr = _sourcePtr; pd.OutputPtr = _outputPtr;
            pd.LumTexPtr = _lumTex != null ? _lumTex.NativePtr : IntPtr.Zero;
            pd.Resolution = _ctx.DisplayResolution;
            pd.LumW = _lumW; pd.LumH = _lumH; pd.MipCount = _mipCount;
            pd.AutoExposure = _ctx.Setting.autoExposure;
            pd.ApplyColorRes = _applyColorRes; pd.ApplyColorFmt = _applyColorFmt;
            pd.LumColorRes = _lumColorRes; pd.LumColorFmt = _lumColorFmt;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }

        private static unsafe void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            cmd.BeginSample(RenderPassMarkers.ToneMapping);

            if (data.AutoExposure && data.LumTexPtr != IntPtr.Zero && data.MipUav != null && data.MipSets != null)
            {
                // 1. luminance → mip 0 (raster, reuses luminance_ps).
                var lds = data.LumDs;
                lds.SetTexture("gColorTex", data.SourcePtr);
                data.LumColorRes[0] = data.LumTexPtr;
                var lumDraw = new RasterDrawDesc
                {
                    numRenderTargets = 1, colorResources = data.LumColorRes, colorFormats = data.LumColorFmt,
                    depthResource = IntPtr.Zero, viewport = new Rect(0, 0, data.LumW, data.LumH),
                    vertexCount = 3, instanceCount = 1,
                };
                data.LumRaster.Draw(cmd, lds, in lumDraw);

                // 2. mip reduction — one destination mip per dispatch (donut-style box average).
                for (int dst = 1; dst < data.MipCount; dst++)
                {
                    int srcW = Mathf.Max(1, data.LumW >> (dst - 1));
                    int srcH = Mathf.Max(1, data.LumH >> (dst - 1));
                    int dstW = Mathf.Max(1, data.LumW >> dst);
                    int dstH = Mathf.Max(1, data.LumH >> dst);

                    var mds = data.MipSets[dst - 1];
                    mds.SetBindlessRWTexture("u_mips", data.MipUav);
                    var mc = new MipConstants
                    {
                        srcMip = (uint)(dst - 1), dstMip = (uint)dst,
                        srcW = (uint)srcW, srcH = (uint)srcH, dstW = (uint)dstW, dstH = (uint)dstH,
                    };
                    mds.SetRootConstants("MipConstants", &mc, 8);

                    uint gx = ((uint)dstW + 7u) / 8u;
                    uint gy = ((uint)dstH + 7u) / 8u;
                    data.MipCs.Dispatch(cmd, mds, gx, gy, 1);
                }

                // 3. capture top mip → 1-float buffer (reuses ToneMapping.hlsl capture_cs).
                var cds = data.CaptureDs;
                cds.SetTexture("t_CaptureSource", data.LumTexPtr);
                cds.SetRWTypedBuffer("u_CaptureTarget", data.AvgLumBuffer, 1, kR32F);
                data.CaptureCs.Dispatch(cmd, cds, 1, 1, 1);
            }

            // 4. tone-map apply (raster, reuses ToneMapping.ps.hlsli) → ProcessedOutputColor.
            data.ApplyCb.UploadDirect(context.cmd, data.Cb);
            var ads = data.ApplyDs;
            ads.SetConstantBuffer("PerImageCB", data.ApplyCb);
            ads.SetTexture("gColorTex", data.SourcePtr);
            ads.SetTypedBuffer("t_AvgLuminance", data.AvgLumBuffer, 1, kR32F);

            data.ApplyColorRes[0] = data.OutputPtr;
            var applyDraw = new RasterDrawDesc
            {
                numRenderTargets = 1, colorResources = data.ApplyColorRes, colorFormats = data.ApplyColorFmt,
                depthResource = IntPtr.Zero, viewport = new Rect(0, 0, data.Resolution.x, data.Resolution.y),
                vertexCount = 3, instanceCount = 1,
            };
            data.ApplyRaster.Draw(cmd, ads, in applyDraw);

            cmd.EndSample(RenderPassMarkers.ToneMapping);
        }
    }
}
