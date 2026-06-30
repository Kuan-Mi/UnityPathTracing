using System;
using System.Runtime.InteropServices;
using NativeRender;
using PathTracing.NativeInterop.NRI;
using PathTracing.Profiling;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Faithful native replica of the original RTXPT ToneMappingPasses.cpp auto-exposure + tone-map,
    /// built entirely from the original shaders (each Unity asset is a thin verbatim #include wrapper):
    ///
    ///   1. luminance  — Luminance.rastershader (= donut fullscreen_vs + RTXPT luminance_ps.hlsl)
    ///                   writes log2-luminance into mip-0 of a pow2 luminance pyramid.
    ///   2. mip reduce — mipmapgen_cs.computeshader (donut passes/mipmapgen_cs.hlsl MODE_COLOR,
    ///                   verbatim: t_input SRV mip + u_output UAV mip array) builds the pyramid.
    ///   3. capture    — capture_cs.computeshader (= ToneMapping.hlsl capture_cs, verbatim) copies the
    ///                   1x1 top mip (log2 geometric-mean luminance) into a 1-float UAV buffer.
    ///   4. read-back  — the buffer is read to the CPU (UploadBuffer.RequestReadback/TryGetReadback) and
    ///                   fed into gParams.avgLuminance (exp2), exactly as ToneMappingPasses.cpp does with
    ///                   its cReadbackLag ring (steady-state identical; ~1-2 frame latency here).
    ///   5. apply      — ToneMapping.rastershader (= donut fullscreen_vs + RTXPT ToneMapping.hlsl main_ps,
    ///                   verbatim, TONEMAPPING_AUTOEXPOSURE_CPU==1 path) tone-maps the HDR source.
    ///
    /// PIX markers mirror ToneMappingPasses.cpp: "Luminance" { draw, "MipMapGen::Dispatch" { mip
    /// dispatches }, capture, read-back copy } then "ToneMapping" { apply draw }.
    /// </summary>
    public class RtxptToneMappingMipChainPass : ScriptableRenderPass, IDisposable
    {
        private const uint kR16F    = (uint)DXGI_FORMAT.DXGI_FORMAT_R16_FLOAT;
        private const uint kR32F    = (uint)DXGI_FORMAT.DXGI_FORMAT_R32_FLOAT;
        private const uint kRGBA16F = (uint)DXGI_FORMAT.DXGI_FORMAT_R16G16B16A16_FLOAT;

        // Mirrors ToneMapping_cb.h ToneMappingConstants (96 bytes).
        [StructLayout(LayoutKind.Sequential)]
        private struct ToneMappingConstants
        {
            public float   whiteScale,      whiteMaxLuminance;
            public uint    toneMapOperator, clamped,                 autoExposure;
            public float   avgLuminance,    autoExposureLumValueMin, autoExposureLumValueMax;
            public Vector4 colorTransform0, colorTransform1,         colorTransform2;
            public uint    enabled,         _padding0,               _padding1, _padding2;
        }

        // donut MipmmapGenConstants (mipmapgen_cb.h): { uint dispatch; uint numLODs; uint padding[2]; }.
        // Bound as root constants for mipmapgen_cs.computeshader (varies per pass).
        [StructLayout(LayoutKind.Sequential)]
        private struct MipMapGenCB
        {
            public uint dispatch; // pass index
            public uint numLODs; // output mips this pass (1..NUM_LODS)
            public uint padding0, padding1;
        }

        // donut MipMapGenPass constants (GROUP_SIZE / NUM_LODS / MAX_PASSES).
        private const int kGroupSize = 16;
        private const int kNumLods   = 4;
        private const int kMaxPasses = 4;

        // TONEMAPPING_EXPOSURE_KEY (ToneMapping_cb.h:15) — the "key" gray that auto-exposure maps to.
        private const float kExposureKey = 0.042f;

        // ── Pipelines (built once) ──
        private readonly NativeRasterPipeline       _lumRaster;
        private readonly NativeRasterDescriptorSet  _lumDs;
        private readonly NativeComputePipeline      _mipCs;
        private readonly NativeComputeDescriptorSet _mipDs; // donut runs ≤4 passes/frame (< ring depth)
        private readonly NativeComputePipeline      _captureCs;
        private readonly NativeComputeDescriptorSet _captureDs;
        private readonly NativeRasterPipeline       _applyRaster;
        private readonly NativeRasterDescriptorSet  _applyDs;

        private readonly UploadBuffer           _avgLumBuffer; // 1 × float, DEFAULT heap + UAV + readback
        private readonly VolatileConstantBuffer _applyCb;
        private readonly float[]                _avgLumReadback = new float[1];
        private          float                  _avgLumLog2; // last captured log2(avg luminance)

        private readonly IntPtr[] _applyColorRes = new IntPtr[1];
        private readonly uint[]   _applyColorFmt = { kRGBA16F };
        private readonly IntPtr[] _lumColorRes   = new IntPtr[1];
        private readonly uint[]   _lumColorFmt   = { kR16F }; // single-channel log-luminance pyramid

        // ── Resolution-dependent ──
        private NriTextureResource _lumTex;
        private int                _lumW, _lumH, _mipCount;

        private RtxptPassContext _ctx;
        private IntPtr           _sourcePtr;
        private IntPtr           _outputPtr;

        public RtxptToneMappingMipChainPass(
            NativeRasterShader luminanceRasterShader,
            NativeComputeShader mipMapGenCs,
            NativeComputeShader captureCs,
            NativeRasterShader toneMapRasterShader)
        {
            _lumRaster = new NativeRasterPipeline(luminanceRasterShader,
                NativeRenderPlugin.RasterPipelineStateDesc.FullscreenOpaque(kR16F));
            _lumDs     = new NativeRasterDescriptorSet(_lumRaster);
            _mipCs     = new NativeComputePipeline(mipMapGenCs);
            _mipDs     = new NativeComputeDescriptorSet(_mipCs);
            _captureCs = new NativeComputePipeline(captureCs);
            _captureDs = new NativeComputeDescriptorSet(_captureCs);
            _applyRaster = new NativeRasterPipeline(toneMapRasterShader,
                NativeRenderPlugin.RasterPipelineStateDesc.FullscreenOpaque(kRGBA16F));
            _applyDs = new NativeRasterDescriptorSet(_applyRaster);

            // UAV-capable + readback-capable single-float buffer for the capture target.
            // Names mirror the original RTXPT ToneMappingPasses.cpp debugName strings for PIX parity.
            _avgLumBuffer = new UploadBuffer(1, sizeof(float), UploadBuffer.UploadMode.Ranges, allowUAV: true, debugName: "AvgLuminanceBuffer");
            _applyCb      = new VolatileConstantBuffer(Marshal.SizeOf<ToneMappingConstants>(), "ToneMappingConstants");
        }

        public void Dispose()
        {
            _lumDs?.Dispose();
            _lumRaster?.Dispose();
            _mipDs?.Dispose();
            _mipCs?.Dispose();
            _captureDs?.Dispose();
            _captureCs?.Dispose();
            _applyDs?.Dispose();
            _applyRaster?.Dispose();
            _avgLumBuffer?.Dispose();
            _applyCb?.Dispose();
            _lumTex?.Release();
        }

        public void Setup(RtxptPassContext ctx, NriTextureResource source, NriTextureResource output)
        {
            _ctx       = ctx;
            _sourcePtr = source.NativePtr;
            _outputPtr = output.NativePtr;

            // Poll the CPU read-back of the average luminance (filled a frame or two after capture),
            // mirroring ToneMappingPass::Render's map/read of avgLuminanceBufferReadback.
            if (_avgLumBuffer.TryGetReadback(_avgLumReadback, 1))
            {
                _avgLumLog2 = _avgLumReadback[0];
                // Debug.Log($"[ToneMapping] Captured log2(avg luminance) = {_avgLumLog2:F4} (avg luminance = {Mathf.Pow(2f, _avgLumLog2):F4})");
            }
            // else
            // {
            //     Debug.LogWarning($"[ToneMapping] Avg luminance read-back not ready yet (frame lag), using last value log2(avg luminance) = {_avgLumLog2:F4}");
            // }

            // Pyramid is sized from the RENDER resolution, replicating the original's quirk:
            // Sample.cpp:1290 constructs ToneMappingPass with the render-res *m_view (the pyramid is
            // pow2-floored from that view's extent at construction), while Render() runs with the
            // display-res fullscreenView on the display-res HDR. The luminance draw simply samples
            // the display-res source over the render-res-derived pow2 viewport (UV-based, lossless).
            EnsureResources(ctx.RenderResolution);
        }

        private void EnsureResources(int2 renderRes)
        {
            int w = 1 << (int)Mathf.Floor(Mathf.Log(Mathf.Max(2, renderRes.x), 2f));
            int h = 1 << (int)Mathf.Floor(Mathf.Log(Mathf.Max(2, renderRes.y), 2f));
            if (_lumTex != null && _lumTex.IsCreated && w == _lumW && h == _lumH)
                return;

            _lumW = w;
            _lumH = h;

            // Single-channel R16_FLOAT log-luminance pyramid — the format ToneMappingPasses.cpp picks
            // for non-RGBA32F sources. luminance_ps writes float4(logLum,0,0,1) → only .r is stored;
            // mipmapgen_cs's typed views and capture_cs's Texture2D<float> read it back component-wise.
            _lumTex ??= new NriTextureResource("Luminance Texture", GraphicsFormat.R16_SFloat,
                new NriResourceState
                {
                    accessBits = AccessBits.SHADER_RESOURCE_STORAGE,
                    layout     = Layout.SHADER_RESOURCE_STORAGE, stageBits = 1 << 10
                });
            _lumTex.Allocate(new int2(w, h), 1, useMipMap: true);
            _mipCount = _lumTex.rt.mipmapCount;
        }

        // Scalar of m_ColorTransform (ToneMappingPasses.cpp:431-440). White-balance is identity here, so the
        // transform is k·I where k = exposureScale · manualExposureScale.
        private static float ExposureScale(RtxptSetting s)
        {
            float exposureScale       = Mathf.Pow(2f, s.exposureCompensation);
            float manualExposureScale = 1f;
            if (!s.autoExposure)
            {
                float ev      = s.exposureValue;
                float shutter = Mathf.Clamp(Mathf.Pow(2f, ev) / (s.fNumber * s.fNumber), 0.001f, 10000f);
                manualExposureScale = (s.filmSpeed / 100f) / (shutter * s.fNumber * s.fNumber);
            }

            return exposureScale * manualExposureScale;
        }

        /// <summary>
        /// Mirrors Sample.cpp:1508 — <c>constants.preExposedGrayLuminance = EnableToneMapping ?
        /// luminance(GetPreExposedGray(0)) : 1.0</c>. With the diagonal (white-balance-identity) color
        /// transform used here, <c>GetPreExposedGray = inverse(k·I)·0.18 = 0.18/k</c>, divided by
        /// <c>EXPOSURE_KEY / avgLuminance</c> when auto-exposure is on; <c>luminance(x,x,x) == x</c>.
        /// Uses the last CPU-read-back average luminance (1-2 frame lag, exactly like
        /// ToneMappingPass::GetPreExposedGray reading avgLuminanceLastCaptured). Returns 1.0 when tone
        /// mapping is disabled — the neutral value the path tracer's firefly/DLSS clamps assume.
        /// </summary>
        public float GetPreExposedGrayLuminance(RtxptSetting s)
        {
            if (!s.enableToneMapping) return 1.0f;
            float gray = 0.18f / ExposureScale(s);
            if (s.autoExposure)
                gray *= Mathf.Pow(2f, _avgLumLog2) / kExposureKey; // /= (EXPOSURE_KEY / avgLuminance)
            return gray;
        }

        // Mirrors SetParameters / Update* and the CPU avgLuminance feed (exp2 of the read-back log2 value).
        private static ToneMappingConstants BuildConstants(RtxptSetting s, float avgLumLog2)
        {
            bool  auto   = s.autoExposure;
            float k      = ExposureScale(s);
            float lumMin = Mathf.Pow(2f, auto ? s.exposureValueMin : -16f);
            float lumMax = Mathf.Pow(2f, auto ? s.exposureValueMax : 16f);
            return new ToneMappingConstants
            {
                whiteScale              = s.toneMapWhiteScale, whiteMaxLuminance = s.toneMapWhiteMaxLuminance,
                toneMapOperator         = (uint)s.toneMapOperator, clamped       = s.toneMapClamped ? 1u : 0u,
                autoExposure            = auto ? 1u : 0u,
                avgLuminance            = Mathf.Pow(2f, avgLumLog2), // exp2(log2 avg) = linear avg, fed to main_ps CPU path
                autoExposureLumValueMin = lumMin, autoExposureLumValueMax = lumMax,
                colorTransform0         = new Vector4(k, 0, 0, 0),
                colorTransform1         = new Vector4(0, k, 0, 0),
                colorTransform2         = new Vector4(0, 0, k, 0),
                enabled                 = s.enableToneMapping ? 1u : 0u,
            };
        }

        private class PassData
        {
            internal NativeRasterPipeline       LumRaster;
            internal NativeRasterDescriptorSet  LumDs;
            internal NativeComputePipeline      MipCs;
            internal NativeComputeDescriptorSet MipDs;
            internal NativeComputePipeline      CaptureCs;
            internal NativeComputeDescriptorSet CaptureDs;
            internal NativeRasterPipeline       ApplyRaster;
            internal NativeRasterDescriptorSet  ApplyDs;
            internal UploadBuffer               AvgLumBuffer;
            internal VolatileConstantBuffer     ApplyCb;
            internal ToneMappingConstants       Cb;
            internal IntPtr                     SourcePtr, OutputPtr, LumTexPtr;
            internal int2                       Resolution;
            internal int                        LumW, LumH, MipCount;
            internal bool                       AutoExposure;
            internal IntPtr[]                   ApplyColorRes, LumColorRes;
            internal uint[]                     ApplyColorFmt, LumColorFmt;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("ToneMapping", out var pd);

            pd.LumRaster     = _lumRaster;
            pd.LumDs         = _lumDs;
            pd.MipCs         = _mipCs;
            pd.MipDs         = _mipDs;
            pd.CaptureCs     = _captureCs;
            pd.CaptureDs     = _captureDs;
            pd.ApplyRaster   = _applyRaster;
            pd.ApplyDs       = _applyDs;
            pd.AvgLumBuffer  = _avgLumBuffer;
            pd.ApplyCb       = _applyCb;
            pd.Cb            = BuildConstants(_ctx.Setting, _avgLumLog2);
            pd.SourcePtr     = _sourcePtr;
            pd.OutputPtr     = _outputPtr;
            pd.LumTexPtr     = _lumTex != null ? _lumTex.NativePtr : IntPtr.Zero;
            pd.Resolution    = _ctx.DisplayResolution;
            pd.LumW          = _lumW;
            pd.LumH          = _lumH;
            pd.MipCount      = _mipCount;
            pd.AutoExposure  = _ctx.Setting.autoExposure;
            pd.ApplyColorRes = _applyColorRes;
            pd.ApplyColorFmt = _applyColorFmt;
            pd.LumColorRes   = _lumColorRes;
            pd.LumColorFmt   = _lumColorFmt;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }

        private static unsafe void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            if (data.AutoExposure && data.LumTexPtr != IntPtr.Zero && data.MipDs != null)
            {
                // "Luminance" marker (ToneMappingPasses.cpp:220) wraps draw + mip chain + capture + copy.
                cmd.BeginSample(RenderPassMarkers.RtxptLuminance);

                // 1. luminance → mip 0 (raster, verbatim luminance_ps).
                var lds = data.LumDs;
                lds.SetTexture("gColorTex", data.SourcePtr);
                data.LumColorRes[0] = data.LumTexPtr;
                var lumDraw = new RasterDrawDesc
                {
                    numRenderTargets = 1, colorResources     = data.LumColorRes, colorFormats = data.LumColorFmt,
                    depthResource    = IntPtr.Zero, viewport = new Rect(0, 0, data.LumW, data.LumH),
                    vertexCount      = 4, instanceCount      = 1, // donut fullscreen_vs.hlsl = 4-vertex triangle strip
                };
                data.LumRaster.Draw(cmd, lds, in lumDraw);

                // 2. mip reduction — donut MipMapGenPass::Dispatch replicated exactly (same scheme as
                //    RtxptEnvMapBakerPass.GenerateDonutMipChain): pass i reads mip i*NUM_LODS via
                //    the t_input SRV and writes mips i*NUM_LODS+1.. via the u_output UAV array, with the
                //    base-mip group count for EVERY pass (donut over-dispatches; out-of-range UAV writes
                //    are dropped). The native per-subresource barriers make the SRV-read-mip /
                //    UAV-write-mips split on one resource legal under Unity's per-resource state tracker.
                cmd.BeginSample(RenderPassMarkers.RtxptEnvMapMipMapGen); // "MipMapGen::Dispatch"
                int  nmip    = data.MipCount;
                var  mds     = data.MipDs;
                uint groupsX = (uint)((data.LumW + kGroupSize - 1) / kGroupSize);
                uint groupsY = (uint)((data.LumH + kGroupSize - 1) / kGroupSize);
                for (int i = 0; i < kMaxPasses; i++)
                {
                    int inputMip = i * kNumLods;
                    if (inputMip >= nmip) break;
                    int numLODs = Mathf.Min(nmip - inputMip - 1, kNumLods);
                    if (numLODs <= 0) break;

                    var mc = new MipMapGenCB { dispatch = (uint)i, numLODs = (uint)numLODs };
                    mds.SetTexture("t_input", data.LumTexPtr, inputMip, 1); // SRV: MostDetailedMip = inputMip, 1 mip
                    mds.SetRWTextureMipArray("u_output", data.LumTexPtr, inputMip + 1); // UAV array: mips inputMip+1 .. +NUM_LODS
                    mds.SetRootConstants("c_MipMapgen", &mc, 4);
                    data.MipCs.Dispatch(cmd, mds, groupsX, groupsY, 1);
                }

                cmd.EndSample(RenderPassMarkers.RtxptEnvMapMipMapGen);

                // 3. capture top mip → 1-float UAV buffer (verbatim capture_cs).
                var cds = data.CaptureDs;
                cds.SetTexture("t_CaptureSource", data.LumTexPtr);
                cds.SetRWTypedBuffer("u_CaptureTarget", data.AvgLumBuffer, 1, kR32F);
                data.CaptureCs.Dispatch(cmd, cds, 1, 1, 1);

                // 4. queue the GPU→CPU read-back of the captured value (consumed in Setup next frame).
                data.AvgLumBuffer.RequestReadback(cmd, 0, 1);

                cmd.EndSample(RenderPassMarkers.RtxptLuminance);
            }

            // 5. tone-map apply (raster, verbatim main_ps) → ProcessedOutputColor, under the original's
            //    "ToneMapping" marker (ToneMappingPasses.cpp:295). Auto-exposure reads
            //    gParams.avgLuminance (CPU-fed in BuildConstants).
            cmd.BeginSample(RenderPassMarkers.RtxptToneMapping);
            data.ApplyCb.UploadDirect(context.cmd, data.Cb);
            var ads = data.ApplyDs;
            ads.SetConstantBuffer("PerImageCB", data.ApplyCb);
            ads.SetTexture("gColorTex", data.SourcePtr);

            data.ApplyColorRes[0] = data.OutputPtr;
            var applyDraw = new RasterDrawDesc
            {
                numRenderTargets = 1, colorResources     = data.ApplyColorRes, colorFormats = data.ApplyColorFmt,
                depthResource    = IntPtr.Zero, viewport = new Rect(0, 0, data.Resolution.x, data.Resolution.y),
                vertexCount      = 4, instanceCount      = 1, // donut fullscreen_vs.hlsl = 4-vertex triangle strip
            };
            data.ApplyRaster.Draw(cmd, ads, in applyDraw);
            cmd.EndSample(RenderPassMarkers.RtxptToneMapping);
        }
    }
}