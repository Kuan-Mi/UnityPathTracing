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
    /// Port of the RTXPT tone-mapper (ToneMapper/ToneMappingPasses.cpp + ToneMapping.hlsl),
    /// adapted to the native compute pipeline. Runs after DLSS-RR on the display-resolution
    /// HDR image:
    ///
    ///   1. ReduceLuminance — one native thread group strides the whole image and reduces to the
    ///      average log2 luminance (geometric mean) in a one-float native buffer.
    ///   2. ToneMappingApply — auto-exposure + color transform + operator (ACES, …) → LDR.
    ///
    /// The original luminance_ps + GenerateMips + capture_cs chain is intentionally NOT used:
    /// Unity's GenerateMips is a sampler-heap op, and interleaving it with native plugin dispatches
    /// left a SAMPLER descriptor heap bound, tripping D3D12 SetComputeRootDescriptorTable validation
    /// (and crashing). The reduction keeps the whole path native (no Unity ops, no samplers, no mips),
    /// and the average luminance stays on the GPU — no CPU read-back latency.
    /// </summary>
    public class NativeRtxptToneMappingPass : ScriptableRenderPass, IDisposable
    {
        private const float kExposureKey = 0.042f; // TONEMAPPING_EXPOSURE_KEY (ToneMapping_cb.h)

        // ── Constant buffer (mirrors ToneMapping_cb.h ToneMappingConstants, 96 bytes) ──
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
            public Vector4 colorTransform0;          // float3x4 row 0 (row_major)
            public Vector4 colorTransform1;          // row 1
            public Vector4 colorTransform2;          // row 2
            public uint    enabled;
            public uint    _padding0;
            public uint    _padding1;
            public uint    _padding2;
        }

        private readonly NativeComputePipeline      _reduceCs;
        private readonly NativeComputeDescriptorSet _reduceDs;
        private readonly NativeComputePipeline      _applyCs;
        private readonly NativeComputeDescriptorSet _applyDs;

        private readonly DeviceBuffer           _avgLuminanceBuffer; // 1 × float, native heap (UAV + SRV)
        private readonly VolatileConstantBuffer _applyCb;

        private NativeRtxptPassContext _ctx;
        private IntPtr                 _sourcePtr;
        private IntPtr                 _outputPtr;

        public NativeRtxptToneMappingPass(
            NativeComputeShader reduceLuminanceCs,
            NativeComputeShader toneMapApplyCs)
        {
            _reduceCs = new NativeComputePipeline(reduceLuminanceCs);
            _reduceDs = new NativeComputeDescriptorSet(_reduceCs);
            _applyCs  = new NativeComputePipeline(toneMapApplyCs);
            _applyDs  = new NativeComputeDescriptorSet(_applyCs);

            _avgLuminanceBuffer = new DeviceBuffer(sizeof(uint));
            _applyCb            = new VolatileConstantBuffer(Marshal.SizeOf<ToneMappingConstants>());
        }

        public void Dispose()
        {
            _reduceDs?.Dispose();
            _reduceCs?.Dispose();
            _applyDs?.Dispose();
            _applyCs?.Dispose();
            _avgLuminanceBuffer?.Dispose();
            _applyCb?.Dispose();
        }

        /// <summary>Source = HDR input (e.g. DlssRrOutput); output = LDR target (e.g. ProcessedOutputColor).</summary>
        public void Setup(NativeRtxptPassContext ctx, NriTextureResource source, NriTextureResource output)
        {
            _ctx       = ctx;
            _sourcePtr = source.NativePtr;
            _outputPtr = output.NativePtr;
        }

        // ── Build the per-frame constants from settings (mirrors SetParameters / Update* ) ──
        private static ToneMappingConstants BuildConstants(NativeRtxptSetting s)
        {
            bool auto = s.autoExposure;

            // UpdateColorTransform: m_ColorTransform = whiteBalance * exposureScale * manualExposureScale.
            // White balance defaults to identity; only the scalar gain is reproduced here.
            float exposureScale       = Mathf.Pow(2f, s.exposureCompensation);
            float manualExposureScale = 1f;
            if (!auto)
            {
                // AperturePriority: derive shutter from EV and aperture (UpdateExposureValue).
                float ev      = s.exposureValue;
                float shutter = Mathf.Clamp(Mathf.Pow(2f, ev) / (s.fNumber * s.fNumber), 0.001f, 10000f);
                manualExposureScale = (s.filmSpeed / 100f) / (shutter * s.fNumber * s.fNumber);
            }
            float k = exposureScale * manualExposureScale;

            // autoExposureLumValueMin/Max: clamp range for kExposureKey/avgLuminance (ToneMappingPasses.cpp:330-339).
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
                _padding0               = 0u,
                _padding1               = 0u,
                _padding2               = 0u,
            };
        }

        private class PassData
        {
            internal NativeComputePipeline      ReduceCs, ApplyCs;
            internal NativeComputeDescriptorSet ReduceDs, ApplyDs;
            internal IntPtr                     SourcePtr, OutputPtr;
            internal DeviceBuffer               AvgLumBuffer;
            internal VolatileConstantBuffer     ApplyCb;
            internal ToneMappingConstants       Cb;
            internal int2                       Resolution;
            internal bool                       AutoExposure;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxpt.ToneMapping", out var pd);

            pd.ReduceCs     = _reduceCs;
            pd.ReduceDs     = _reduceDs;
            pd.ApplyCs      = _applyCs;
            pd.ApplyDs      = _applyDs;
            pd.SourcePtr    = _sourcePtr;
            pd.OutputPtr    = _outputPtr;
            pd.AvgLumBuffer = _avgLuminanceBuffer;
            pd.ApplyCb      = _applyCb;
            pd.Cb           = BuildConstants(_ctx.Setting);
            pd.Resolution   = _ctx.DisplayResolution;
            pd.AutoExposure = _ctx.Setting.autoExposure;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            uint gx = ((uint)data.Resolution.x + 7u) / 8u;
            uint gy = ((uint)data.Resolution.y + 7u) / 8u;

            cmd.BeginSample(RenderPassMarkers.ToneMapping);

            if (data.AutoExposure)
            {
                // Average log-luminance over the whole image into the auto-exposure buffer (one group).
                var rds = data.ReduceDs;
                rds.SetTexture("gColorTex", data.SourcePtr);
                rds.SetRWTypedBuffer("u_AvgLuminance", data.AvgLumBuffer, 1, (uint)DXGI_FORMAT.DXGI_FORMAT_R32_FLOAT);
                data.ReduceCs.Dispatch(cmd, rds, 1, 1, 1);
            }

            // Tone-map apply (auto-exposure + color transform + operator).
            data.ApplyCb.UploadDirect(context.cmd, data.Cb);
            var ads = data.ApplyDs;
            ads.SetConstantBuffer("PerImageCB", data.ApplyCb);
            ads.SetTexture("gColorTex", data.SourcePtr);
            ads.SetTypedBuffer("t_AvgLuminance", data.AvgLumBuffer, 1, (uint)DXGI_FORMAT.DXGI_FORMAT_R32_FLOAT);
            ads.SetRWTexture("u_Output", data.OutputPtr);
            data.ApplyCs.Dispatch(cmd, ads, gx, gy, 1);

            cmd.EndSample(RenderPassMarkers.ToneMapping);
        }
    }
}
