using System;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace PathTracing
{
    /// <summary>
    /// Owns all per-camera structured GPU buffers for <see cref="NativeRtxptFeature"/>.
    ///
    /// Layout notes (from RTXPT Config.h / RenderTargets.cpp):
    ///   cStablePlaneCount      = 3
    ///   sizeof(StablePlane)    = 80 bytes
    ///   sizeof(PackedPathTracerSurfaceData) = 64 bytes  (TODO: verify from HLSL)
    /// </summary>
    public class NativeRtxptBufferResources : IDisposable
    {
        // sizeof(StablePlane) in bytes — must match Config.h.
        public const int StablePlaneStride = 80;

        // Number of stable planes — must match cStablePlaneCount in Config.h.
        public const int StablePlaneCount = 3;

        // sizeof(PackedPathTracerSurfaceData) — TODO: verify from HLSL struct.
        public const int SurfaceDataStride = 64;

        // Max lights — can be enlarged if scene exceeds this.
        public const int MaxLights = 8192;

        // ── Stable Planes ────────────────────────────────────────────────────
        /// <summary>
        /// Per-pixel stable plane data. W×H×cStablePlaneCount entries, stride = StablePlaneStride.
        /// HLSL: RWByteAddressBuffer u_StablePlanesBuffer (u42).
        /// </summary>
        public GraphicsBuffer StablePlanesBuffer;

        // ── Surface data (GBuffer cache) ─────────────────────────────────────
        /// <summary>
        /// Packed surface GBuffer for current + previous frame. W×H×2 entries, stride = SurfaceDataStride.
        /// HLSL: RWByteAddressBuffer u_SurfaceDataBuffer (u45).
        /// </summary>
        public GraphicsBuffer SurfaceDataBuffer;

        // ── Debug buffers ─────────────────────────────────────────────────────
        /// <summary>
        /// Stub feedback buffer for shader debugging. 1 element, 64B stride.
        /// HLSL: RWStructuredBuffer u_FeedbackBuffer (u51).
        /// </summary>
        public GraphicsBuffer FeedbackBuffer;

        // ── Light system buffers ──────────────────────────────────────────────
        /// <summary>
        /// Light control / management struct. Single-element structured buffer.
        /// HLSL: u_controlBuffer (u0 in BakeEmissiveTriangles).
        /// </summary>
        public GraphicsBuffer LightControlBuffer;

        /// <summary>
        /// Per-light data array. MaxLights elements.
        /// HLSL: u_lightsBuffer (u1).
        /// </summary>
        public GraphicsBuffer LightBuffer;

        /// <summary>
        /// Extended per-light data array. MaxLights elements.
        /// HLSL: u_lightsExBuffer (u2).
        /// </summary>
        public GraphicsBuffer LightExBuffer;

        /// <summary>
        /// Scratch buffer for light processing passes.
        /// HLSL: u_scratchBuffer (u3).
        /// </summary>
        public GraphicsBuffer LightScratchBuffer;

        /// <summary>
        /// Light history remap: current frame index → previous frame index.
        /// MaxLights uint elements.
        /// HLSL: u_historyRemapCurrentToPast (u6).
        /// </summary>
        public GraphicsBuffer HistoryRemapCurrentToPast;

        /// <summary>
        /// Light history remap: previous frame index → current frame index.
        /// MaxLights uint elements.
        /// HLSL: u_historyRemapPastToCurrent (u7).
        /// </summary>
        public GraphicsBuffer HistoryRemapPastToCurrent;

        // ── Light proxy / sampling buffers ────────────────────────────────────
        /// <summary>Per-proxy light counters. Size determined by proxy grid config.</summary>
        public GraphicsBuffer LightProxyCounters;

        /// <summary>Proxy light index list. Size determined by proxy grid config.</summary>
        public GraphicsBuffer LightSamplingProxies;

        /// <summary>Local sampling scratch buffer.</summary>
        public GraphicsBuffer LocalSamplingBuffer;

        /// <summary>
        /// Per-light weight values, ping-pong. 2 × (MaxLights+1) float elements.
        /// HLSL: u_lightWeights (u5). CurrentWeightsBufferOffset / HistoricWeightsBufferOffset select the half.
        /// </summary>
        public GraphicsBuffer LightWeightsBuffer;

        /// <summary>
        /// Typed uint scratch list used by ComputeProxyCounts / CreateProxyJobs.
        /// MaxLights uint elements.
        /// HLSL: u_scratchList (u4).
        /// </summary>
        public GraphicsBuffer ScratchListBuffer;

        // ── Resolved resolution ───────────────────────────────────────────────
        public int2 renderResolution { get; private set; }

        // Scratch buffer size (heuristic: 16 ints per light entry).
        private const int ScratchElementCount = MaxLights * 16;

        // Proxy buffer sizes. ProxyCounterCount must be >= MaxLights (indexed by lightIndex).
        // ProxySamplingCount must hold worst-case SamplingProxyCount from GPU:
        //   HLSL budget = RTXPT_LIGHTING_SAMPLING_PROXY_RATIO * max(TotalLightCount, RTXPT_LIGHTING_MAX_LIGHTS/10)
        //              = 12 * max(N, 524288/10) → max ~629,136 proxies when N is small.
        //   Each light is additionally capped at RTXPT_LIGHTING_MAX_SAMPLING_PROXIES_PER_LIGHT-1=262143,
        //   but the total never exceeds the budget, so 630k is a safe upper bound.
        private const int HlslMaxLights       = 512 * 1024;   // RTXPT_LIGHTING_MAX_LIGHTS compiled into shaders
        private const int HlslProxyRatio      = 12;           // RTXPT_LIGHTING_SAMPLING_PROXY_RATIO
        private const int ProxyCounterCount   = MaxLights + 1; // +1: HLSL uses [TotalLightCount] for invalid-feedback count
        internal const int ProxySamplingCount = HlslProxyRatio * (HlslMaxLights / 10) + 4096; // ~633k
        private const int LocalSamplingCount  = MaxLights;

        // LightWeights ping-pong half-count: mirrors RTXPT_LIGHTING_WEIGHTS_COUNT_HALF = MaxLights+1.
        private const int WeightsCountHalf = MaxLights + 1;

        // ScratchListBuffer must be large enough for both:
        //   - proxy-build passes: MaxLights entries
        //   - env-light backup region: 2 × RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT (88×61×2 = 10736)
        private const int EnvTotalNodeCount  = 5368; // RTXPT_NEEAT_ENVMAP_QT_UNBOOSTED(88) × BOOST_NODES_MULT(61)
        private const int ScratchListCount   = 16384; // max(MaxLights, EnvTotalNodeCount*2) rounded up to power-of-2

        /// <summary>
        /// Allocates or reallocates all resolution-dependent buffers.
        /// Returns true if any allocation occurred.
        /// </summary>
        public bool EnsureResources(int2 renderRes)
        {
            bool sameRes = StablePlanesBuffer != null
                           && renderResolution.x == renderRes.x
                           && renderResolution.y == renderRes.y;
            if (sameRes) return false;

            renderResolution = renderRes;
            int pixelCount = renderRes.x * renderRes.y;

            ReleaseResolutionBuffers();

            // StablePlanesBuffer: W×H×StablePlaneCount structured entries, stride = StablePlaneStride (80).
            // Shader declares: RWStructuredBuffer<StablePlane> u_StablePlanesBuffer (u42).
            StablePlanesBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                pixelCount * StablePlaneCount,
                StablePlaneStride)
            { name = "Rtxpt_StablePlanesBuffer" };

            // SurfaceDataBuffer: W×H×2 structured entries, stride = SurfaceDataStride (64).
            // Shader declares: RWStructuredBuffer<PackedPathTracerSurfaceData> u_SurfaceData (u45).
            SurfaceDataBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                pixelCount * 2,
                SurfaceDataStride)
            { name = "Rtxpt_SurfaceDataBuffer" };

            return true;
        }

        /// <summary>
        /// Allocates light-system buffers. Call once after initialization.
        /// Does nothing if already allocated.
        /// </summary>
        public void EnsureLightBuffers()
        {
            if (LightControlBuffer != null) return;

            // FeedbackBuffer stub — 1 element, 64B stride (matches DebugFeedbackStruct size).
            FeedbackBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                1, 64)
            { name = "Rtxpt_FeedbackBuffer" };

            // LightControlBuffer — single RtxptLightingControlData element (576 bytes).
            LightControlBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                1, Marshal.SizeOf<RtxptLightingControlData>())
            { name = "Rtxpt_LightControlBuffer" };

            // LightBuffer / LightExBuffer — match PolymorphicLightInfo (32B) and PolymorphicLightInfoEx (16B).
            LightBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                MaxLights, Marshal.SizeOf<RtxptPolymorphicLightInfo>())
            { name = "Rtxpt_LightBuffer" };

            LightExBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                MaxLights, Marshal.SizeOf<RtxptPolymorphicLightInfoEx>())
            { name = "Rtxpt_LightExBuffer" };

            LightScratchBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Raw,
                ScratchElementCount, 4)
            { name = "Rtxpt_LightScratchBuffer" };

            HistoryRemapCurrentToPast = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                MaxLights, 4)
            { name = "Rtxpt_HistoryRemapCurrentToPast" };

            HistoryRemapPastToCurrent = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                MaxLights, 4)
            { name = "Rtxpt_HistoryRemapPastToCurrent" };

            LightProxyCounters = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                ProxyCounterCount, 4)
            { name = "Rtxpt_LightProxyCounters" };

            LightSamplingProxies = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                ProxySamplingCount, 4)
            { name = "Rtxpt_LightSamplingProxies" };

            LocalSamplingBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Raw,
                LocalSamplingCount, 4)
            { name = "Rtxpt_LocalSamplingBuffer" };

            // LightWeightsBuffer: 2 halves of (MaxLights+1) floats for ping-pong historic weights.
            LightWeightsBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                2 * WeightsCountHalf, 4)
            { name = "Rtxpt_LightWeightsBuffer" };

            // ScratchListBuffer: uint typed scratch for proxy count prefix-sum and job list.
            // Must hold at least 2 × RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT (=5368×2=10736) entries
            // for the env-light backup history region, plus MaxLights for proxy-build passes.
            ScratchListBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                ScratchListCount, 4)
            { name = "Rtxpt_ScratchListBuffer" };
        }

        private void ReleaseResolutionBuffers()
        {
            StablePlanesBuffer?.Release();
            StablePlanesBuffer = null;
            SurfaceDataBuffer?.Release();
            SurfaceDataBuffer = null;
        }

        private void ReleaseLightBuffers()
        {
            FeedbackBuffer?.Release();           FeedbackBuffer           = null;
            LightControlBuffer?.Release();       LightControlBuffer       = null;
            LightBuffer?.Release();              LightBuffer              = null;
            LightExBuffer?.Release();            LightExBuffer            = null;
            LightScratchBuffer?.Release();       LightScratchBuffer       = null;
            HistoryRemapCurrentToPast?.Release();HistoryRemapCurrentToPast= null;
            HistoryRemapPastToCurrent?.Release();HistoryRemapPastToCurrent= null;
            LightProxyCounters?.Release();       LightProxyCounters       = null;
            LightSamplingProxies?.Release();     LightSamplingProxies     = null;
            LocalSamplingBuffer?.Release();      LocalSamplingBuffer      = null;
            LightWeightsBuffer?.Release();       LightWeightsBuffer       = null;
            ScratchListBuffer?.Release();        ScratchListBuffer        = null;
        }

        public void Dispose()
        {
            ReleaseResolutionBuffers();
            ReleaseLightBuffers();
        }
    }
}
