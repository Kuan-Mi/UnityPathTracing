using System;
using System.Runtime.InteropServices;
using NativeRender;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Native-compute-shader tone-mapping pass.
    /// Mirrors donut ToneMappingPass (histogram → exposure → tonemap) fully on the GPU,
    /// writing results directly into the Unity command list.
    ///
    /// Pass 1 – histogram.computeshader
    ///   Bindings: c_ToneMapping(b0), t_Source(t0), u_Histogram(u0)
    ///   Dispatch: ceil(renderSize / 16)
    ///
    /// Pass 2 – exposure.computeshader
    ///   Bindings: c_ToneMapping(b0), t_Histogram(t0), u_Exposure(u0)
    ///   Dispatch: 1 x 1 x 1
    ///
    /// Pass 3 – tonemapping.rastershader (donut fullscreen_vs + verbatim tonemapping_ps.hlsl)
    ///   Bindings: c_ToneMapping(b0), s_ColorLUTSampler(s0),
    ///             t_Source(t0), t_Exposure(t1), t_ColorLUT(t2) → SV_Target (color RT)
    ///   Draw: fullscreen quad (4 verts) over the render-resolution viewport
    /// </summary>
    public class NativeToneMappingPass : ScriptableRenderPass, IDisposable
    {
        // ── Constant buffer layout (must match ToneMappingCB.hlsli) ─────────
        [StructLayout(LayoutKind.Sequential)]
        private struct ToneMappingConstants
        {
            public uint2  viewOrigin; // offset  0
            public uint2  viewSize; // offset  8
            public float  logLuminanceScale; // offset 16
            public float  logLuminanceBias; // offset 20
            public float  histogramLowPercentile; // offset 24
            public float  histogramHighPercentile; // offset 28
            public float  eyeAdaptationSpeedUp; // offset 32
            public float  eyeAdaptationSpeedDown; // offset 36
            public float  minAdaptedLuminance; // offset 40
            public float  maxAdaptedLuminance; // offset 44
            public float  frameTime; // offset 48
            public float  exposureScale; // offset 52
            public float  whitePointInvSquared; // offset 56
            public uint   sourceSlice; // offset 60
            public float2 colorLUTTextureSize; // offset 64
            public float2 colorLUTTextureSizeInv; // offset 72
        }

        // ── donut defaults ───────────────────────────────────────────────────
        private const float MinLogLuminance = -10f;
        private const float MaxLogLuminance = 4f;

        // ── Pipelines ────────────────────────────────────────────────────────
        private readonly NativeComputePipeline      _histogramCs;
        private readonly NativeComputeDescriptorSet _histogramDs;

        private readonly NativeComputePipeline      _exposureCs;
        private readonly NativeComputeDescriptorSet _exposureDs;

        // Pass 3 (tone-map apply) is a raster fullscreen draw — the verbatim donut tonemapping_ps.hlsl
        // is a PIXEL shader, so it runs through a graphics pipeline writing a color render target
        // (SV_Target) rather than a compute UAV.
        private readonly NativeRasterPipeline       _tonemapRaster;
        private readonly NativeRasterDescriptorSet  _tonemapDs;

        // Output RT format (LdrColor is RGBA16F in both the NRD and RTXDI features). Used both to bake
        // the pipeline's RTV format and as the per-draw colorFormats entry.
        private const uint kRGBA16F = (uint)Nri.DXGI_FORMAT.DXGI_FORMAT_R16G16B16A16_FLOAT;
        private readonly uint[]   _colorFmt = { kRGBA16F };
        private readonly IntPtr[] _colorRes = new IntPtr[1]; // scratch length-1, refilled per draw

        private readonly DeviceBuffer _histogramBuffer; // 256 x uint (DEFAULT heap, UAV-capable)
        private readonly GraphicsBuffer  _exposureBuffer; // 1 x uint (float bits)

        private IntPtr expPtr;

        private Resource _resource;
        private Settings _settings;

        // ── Construction / Disposal ──────────────────────────────────────────

        public NativeToneMappingPass(
            NativeComputeShader histogramCs,
            NativeComputeShader exposureCs,
            NativeRasterShader  tonemapShader)
        {
            _histogramCs = new NativeComputePipeline(histogramCs);
            _histogramDs = new NativeComputeDescriptorSet(_histogramCs);

            _exposureCs = new NativeComputePipeline(exposureCs);
            _exposureDs = new NativeComputeDescriptorSet(_exposureCs);

            _tonemapRaster = new NativeRasterPipeline(tonemapShader,
                NativeRenderPlugin.RasterPipelineStateDesc.FullscreenOpaque(kRGBA16F));
            _tonemapDs     = new NativeRasterDescriptorSet(_tonemapRaster);

            _histogramBuffer = new DeviceBuffer(256 * sizeof(uint));
            _exposureBuffer   = new GraphicsBuffer(GraphicsBuffer.Target.Raw, 1, sizeof(uint));

            expPtr  = _exposureBuffer.GetNativeBufferPtr();
        }

        public void Dispose()
        {
            _histogramDs?.Dispose();
            _histogramCs?.Dispose();
            _exposureDs?.Dispose();
            _exposureCs?.Dispose();
            _tonemapDs?.Dispose();
            _tonemapRaster?.Dispose();
            _histogramBuffer?.Dispose();
            _exposureBuffer?.Release();
        }

        // ── Per-frame Setup ──────────────────────────────────────────────────

        public void Setup(Resource resource, Settings settings)
        {
            _resource = resource;
            _settings = settings;
        }

        // ── Debug Readback ────────────────────────────────────────────────────

        /// <summary>
        /// 回读 ExposureBuffer，打印当前 adapted luminance 值。
        /// </summary>
        public void ReadbackExposure()
        {
            var req = AsyncGPUReadback.Request(_exposureBuffer);
            req.WaitForCompletion();
            if (req.hasError)
            {
                Debug.LogError("[ToneMapping] ExposureBuffer readback error");
                return;
            }

            var   data = req.GetData<uint>();
            float lum  = System.BitConverter.ToSingle(System.BitConverter.GetBytes(data[0]), 0);
            Debug.Log($"[ToneMapping] Adapted luminance = {lum:F6}  (raw uint = {data[0]})");
        }

        // ── Resource / Settings ──────────────────────────────────────────────

        public class Resource
        {
            /// <summary>HDR source texture (SRV).</summary>
            public IntPtr SourceTexture;

            /// <summary>Tone-mapped output texture. Bound as a color render target (SV_Target) for the
            /// raster apply pass; any Unity RenderTexture qualifies.</summary>
            public IntPtr OutputTexture;

            /// <summary>Optional color LUT texture (SRV). Pass IntPtr.Zero to disable.</summary>
            public IntPtr ColorLUT;

            /// <summary>Color LUT size (height dimension, e.g. 32). 0 = disabled.</summary>
            public float ColorLUTSize;
        }

        [Serializable]
        public class ToneMappingParameters
        {
            [Range(0f, 1f)]
            public float histogramLowPercentile = 0.8f;

            [Range(0f, 1f)]
            public float histogramHighPercentile = 0.95f;

            public float eyeAdaptationSpeedUp   = 1.0f;
            public float eyeAdaptationSpeedDown = 0.5f;

            public float minAdaptedLuminance = 0.02f;

            public float maxAdaptedLuminance = 0.5f;

            [Range(-5f, 5f)]
            public float exposureBias = -0.5f;

            public float whitePoint     = 3.0f;
            public bool  enableColorLUT = true;
        }

        public class Settings
        {
            public int2                  RenderResolution;
            public float                 FrameTime;
            public ToneMappingParameters ToneMappingParams;
        }

        // ── Pass data ─────────────────────────────────────────────────────────

        class PassData
        {
            public NativeComputePipeline      HistogramCs;
            public NativeComputeDescriptorSet HistogramDs;
            public NativeComputePipeline      ExposureCs;
            public NativeComputeDescriptorSet ExposureDs;
            public NativeRasterPipeline       TonemapRaster;
            public NativeRasterDescriptorSet  TonemapDs;
            public uint[]                     ColorFmt;
            public IntPtr[]                   ColorRes;

            public DeviceBuffer          HistogramBuffer;
            public Resource                   Resource;
            public Settings                   Settings;
            public   IntPtr                     expPtr;
        }

        // ── Execution ─────────────────────────────────────────────────────────

        static unsafe void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var res = data.Resource;
            var s   = data.Settings;
            var p   = s.ToneMappingParams;

            cmd.BeginSample(RenderPassMarkers.ToneMapping);

            // ── Build constant buffer ────────────────────────────────────────
            var cb = new ToneMappingConstants
            {
                viewOrigin              = uint2.zero,
                viewSize                = new uint2((uint)s.RenderResolution.x, (uint)s.RenderResolution.y),
                logLuminanceScale       = 1f / (MaxLogLuminance - MinLogLuminance),
                logLuminanceBias        = -MinLogLuminance / (MaxLogLuminance - MinLogLuminance),
                histogramLowPercentile  = math.clamp(p.histogramLowPercentile, 0f, 0.99f),
                histogramHighPercentile = math.clamp(p.histogramHighPercentile, 0f, 1f),
                eyeAdaptationSpeedUp    = p.eyeAdaptationSpeedUp,
                eyeAdaptationSpeedDown  = p.eyeAdaptationSpeedDown,
                minAdaptedLuminance     = p.minAdaptedLuminance,
                maxAdaptedLuminance     = p.maxAdaptedLuminance,
                frameTime               = s.FrameTime,
                exposureScale           = math.exp2(p.exposureBias),
                whitePointInvSquared    = 1f / (p.whitePoint * p.whitePoint),
                sourceSlice             = 0,
                colorLUTTextureSize = res.ColorLUTSize > 0
                    ? new float2(res.ColorLUTSize * res.ColorLUTSize, res.ColorLUTSize)
                    : float2.zero,
                colorLUTTextureSizeInv = res.ColorLUTSize > 0
                    ? new float2(1f / (res.ColorLUTSize * res.ColorLUTSize), 1f / res.ColorLUTSize)
                    : float2.zero,
            };

            var expPtr         = data.expPtr;

            uint renderW = (uint)s.RenderResolution.x;
            uint renderH = (uint)s.RenderResolution.y;

            // ── Pass 1: Clear + Build Histogram ─────────────────────────────
            // Clear (GPU-side zero via ClearUnorderedAccessViewUint on the render thread)
            data.HistogramBuffer.Clear(context.cmd);
            
            // Build
            data.HistogramDs.SetRootConstants("c_ToneMapping", &cb);
            data.HistogramDs.SetTexture("t_Source", res.SourceTexture);
            data.HistogramDs.SetRWTypedBuffer("u_Histogram", data.HistogramBuffer, 256, (uint)Nri.DXGI_FORMAT.DXGI_FORMAT_R32_UINT);

            cb.logLuminanceScale = MaxLogLuminance - MinLogLuminance;
            cb.logLuminanceBias  = MinLogLuminance;

            uint hx = (renderW + 15u) / 16u;
            uint hy = (renderH + 15u) / 16u;
            data.HistogramCs.Dispatch(cmd, data.HistogramDs, hx, hy, 1);

            // ── Pass 2: Compute Exposure ─────────────────────────────────────
            data.ExposureDs.SetRootConstants("c_ToneMapping", &cb);
            data.ExposureDs.SetTypedBuffer("t_Histogram", data.HistogramBuffer, 256, (uint)Nri.DXGI_FORMAT.DXGI_FORMAT_R32_UINT);
            data.ExposureDs.SetRWTypedBuffer("u_Exposure", expPtr, 1, (uint)Nri.DXGI_FORMAT.DXGI_FORMAT_R32_UINT);

            data.ExposureCs.Dispatch(cmd, data.ExposureDs, 1, 1, 1);

            // ── Pass 3: Tone Map (raster fullscreen draw → color RT) ─────────
            // donut's tonemapping_ps.hlsl is a pixel shader: the fullscreen quad (fullscreen_vs)
            // covers the viewport and main() writes SV_Target, indexing t_Source by SV_Position.
            data.TonemapDs.SetRootConstants("c_ToneMapping", &cb);
            data.TonemapDs.SetTexture("t_Source", res.SourceTexture);
            data.TonemapDs.SetTypedBuffer("t_Exposure", expPtr, 1, (uint)Nri.DXGI_FORMAT.DXGI_FORMAT_R32_UINT);
            if (res.ColorLUT != IntPtr.Zero)
                data.TonemapDs.SetTexture("t_ColorLUT", res.ColorLUT);

            data.ColorRes[0] = res.OutputTexture;
            var draw = new RasterDrawDesc
            {
                numRenderTargets = 1,
                colorResources   = data.ColorRes,
                colorFormats     = data.ColorFmt,
                depthResource    = IntPtr.Zero,
                viewport         = new Rect(0, 0, renderW, renderH),
                vertexCount      = 4,
                instanceCount    = 1,
            };
            data.TonemapRaster.Draw(cmd, data.TonemapDs, in draw);

            cmd.EndSample(RenderPassMarkers.ToneMapping);
        }

        // ── RenderGraph ───────────────────────────────────────────────────────

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeToneMappingPass", out var passData);

            passData.HistogramCs      = _histogramCs;
            passData.HistogramDs      = _histogramDs;
            passData.ExposureCs       = _exposureCs;
            passData.ExposureDs       = _exposureDs;
            passData.TonemapRaster    = _tonemapRaster;
            passData.TonemapDs        = _tonemapDs;
            passData.ColorFmt         = _colorFmt;
            passData.ColorRes         = _colorRes;

            passData.HistogramBuffer  = _histogramBuffer;
            passData.Resource         = _resource;
            passData.Settings         = _settings;
            passData.expPtr           = expPtr;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext ctx) => ExecutePass(data, ctx));
        }
    }
}