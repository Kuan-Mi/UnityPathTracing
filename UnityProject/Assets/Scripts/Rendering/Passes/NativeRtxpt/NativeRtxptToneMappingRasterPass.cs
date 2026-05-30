using System;
using System.Runtime.InteropServices;
using NativeRender;
using Nri;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Raster variant of <see cref="NativeRtxptToneMappingPass"/>: identical pipeline (native
    /// ReduceLuminance compute → tone-map apply), but the apply step runs the original RTXPT
    /// ToneMapper as a <b>fullscreen pixel shader</b> through the new native raster pipeline
    /// (<c>ToneMapping.rastershader</c>, which #includes ToneMapper/ToneMapping.ps.hlsli and reuses
    /// its operators verbatim) instead of <c>ToneMappingApply.computeshader</c>.
    ///
    /// This is the faithful "replicate the C++ ToneMappingPasses.cpp draw" path the raster support
    /// was added for. Auto-exposure still uses the native log-luminance reduction (not luminance_ps
    /// + GenerateMips) to keep the whole chain off Unity sampler-heap ops.
    /// Writes the LDR result into ProcessedOutputColor (a Unity RenderTexture → RTV-capable).
    /// </summary>
    public class NativeRtxptToneMappingRasterPass : ScriptableRenderPass, IDisposable
    {
        private const float kExposureKey = 0.042f; // TONEMAPPING_EXPOSURE_KEY (ToneMapping_cb.h)

        // Mirrors ToneMapping_cb.h ToneMappingConstants (96 bytes) — same layout as the compute pass.
        [StructLayout(LayoutKind.Sequential)]
        private struct ToneMappingConstants
        {
            public float   whiteScale;
            public float   whiteMaxLuminance;
            public uint    toneMapOperator;
            public uint    clamped;
            public uint    autoExposure;
            public float   avgLuminance;             // unused on the GPU path
            public float   autoExposureLumValueMin;
            public float   autoExposureLumValueMax;
            public Vector4 colorTransform0;
            public Vector4 colorTransform1;
            public Vector4 colorTransform2;
            public uint    enabled;
            public uint    _padding0;
            public uint    _padding1;
            public uint    _padding2;
        }

        private readonly NativeComputePipeline      _reduceCs;
        private readonly NativeComputeDescriptorSet _reduceDs;
        private readonly NativeRasterPipeline        _applyRaster;
        private readonly NativeRasterDescriptorSet   _applyDs;

        private readonly DeviceBuffer           _avgLuminanceBuffer; // 1 × float, native heap (UAV + SRV)
        private readonly VolatileConstantBuffer _applyCb;

        // Reused per-frame draw buffers (no per-frame GC).
        private readonly IntPtr[] _colorRes = new IntPtr[1];
        private readonly uint[]   _colorFmt = { (uint)DXGI_FORMAT.DXGI_FORMAT_R16G16B16A16_FLOAT };

        private NativeRtxptPassContext _ctx;
        private IntPtr                 _sourcePtr;
        private IntPtr                 _outputPtr;

        public NativeRtxptToneMappingRasterPass(
            NativeComputeShader reduceLuminanceCs,
            NativeRasterShader  toneMapRasterShader)
        {
            _reduceCs = new NativeComputePipeline(reduceLuminanceCs);
            _reduceDs = new NativeComputeDescriptorSet(_reduceCs);

            // Single opaque RTV, fullscreen-triangle, no depth — matches ProcessedOutputColor (RGBA16F).
            var state = NativeRenderPlugin.RasterPipelineStateDesc.FullscreenOpaque(
                (uint)DXGI_FORMAT.DXGI_FORMAT_R16G16B16A16_FLOAT);
            _applyRaster = new NativeRasterPipeline(toneMapRasterShader, state);
            _applyDs     = new NativeRasterDescriptorSet(_applyRaster);

            _avgLuminanceBuffer = new DeviceBuffer(sizeof(uint));
            _applyCb            = new VolatileConstantBuffer(Marshal.SizeOf<ToneMappingConstants>());
        }

        public void Dispose()
        {
            _reduceDs?.Dispose();
            _reduceCs?.Dispose();
            _applyDs?.Dispose();
            _applyRaster?.Dispose();
            _avgLuminanceBuffer?.Dispose();
            _applyCb?.Dispose();
        }

        public void Setup(NativeRtxptPassContext ctx, NriTextureResource source, NriTextureResource output)
        {
            _ctx       = ctx;
            _sourcePtr = source.NativePtr;
            _outputPtr = output.NativePtr;
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
                whiteScale              = s.toneMapWhiteScale,
                whiteMaxLuminance       = s.toneMapWhiteMaxLuminance,
                toneMapOperator         = (uint)s.toneMapOperator,
                clamped                 = s.toneMapClamped ? 1u : 0u,
                autoExposure            = auto ? 1u : 0u,
                avgLuminance            = 0f,
                autoExposureLumValueMin = lumMin,
                autoExposureLumValueMax = lumMax,
                colorTransform0         = new Vector4(k, 0f, 0f, 0f),
                colorTransform1         = new Vector4(0f, k, 0f, 0f),
                colorTransform2         = new Vector4(0f, 0f, k, 0f),
                enabled                 = s.enableToneMapping ? 1u : 0u,
            };
        }

        private class PassData
        {
            internal NativeComputePipeline      ReduceCs;
            internal NativeComputeDescriptorSet ReduceDs;
            internal NativeRasterPipeline        ApplyRaster;
            internal NativeRasterDescriptorSet   ApplyDs;
            internal IntPtr                     SourcePtr, OutputPtr;
            internal DeviceBuffer               AvgLumBuffer;
            internal VolatileConstantBuffer     ApplyCb;
            internal ToneMappingConstants       Cb;
            internal int2                       Resolution;
            internal bool                       AutoExposure;
            internal IntPtr[]                   ColorRes;
            internal uint[]                     ColorFmt;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxpt.ToneMappingRaster", out var pd);

            pd.ReduceCs     = _reduceCs;
            pd.ReduceDs     = _reduceDs;
            pd.ApplyRaster  = _applyRaster;
            pd.ApplyDs      = _applyDs;
            pd.SourcePtr    = _sourcePtr;
            pd.OutputPtr    = _outputPtr;
            pd.AvgLumBuffer = _avgLuminanceBuffer;
            pd.ApplyCb      = _applyCb;
            pd.Cb           = BuildConstants(_ctx.Setting);
            pd.Resolution   = _ctx.DisplayResolution;
            pd.AutoExposure = _ctx.Setting.autoExposure;
            pd.ColorRes     = _colorRes;
            pd.ColorFmt     = _colorFmt;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            cmd.BeginSample(RenderPassMarkers.ToneMapping);

            if (data.AutoExposure)
            {
                var rds = data.ReduceDs;
                rds.SetTexture("gColorTex", data.SourcePtr);
                rds.SetRWTypedBuffer("u_AvgLuminance", data.AvgLumBuffer, 1, (uint)DXGI_FORMAT.DXGI_FORMAT_R32_FLOAT);
                data.ReduceCs.Dispatch(cmd, rds, 1, 1, 1);
            }

            data.ApplyCb.UploadDirect(context.cmd, data.Cb);

            var ads = data.ApplyDs;
            ads.SetConstantBuffer("PerImageCB", data.ApplyCb);
            ads.SetTexture("gColorTex", data.SourcePtr);
            ads.SetTypedBuffer("t_AvgLuminance", data.AvgLumBuffer, 1, (uint)DXGI_FORMAT.DXGI_FORMAT_R32_FLOAT);

            data.ColorRes[0] = data.OutputPtr;
            var draw = new RasterDrawDesc
            {
                numRenderTargets = 1,
                colorResources   = data.ColorRes,
                colorFormats     = data.ColorFmt,
                depthResource    = IntPtr.Zero,
                viewport         = new Rect(0, 0, data.Resolution.x, data.Resolution.y),
                vertexCount      = 3,   // fullscreen triangle
                instanceCount    = 1,
            };
            data.ApplyRaster.Draw(cmd, ads, in draw);

            cmd.EndSample(RenderPassMarkers.ToneMapping);
        }
    }
}
