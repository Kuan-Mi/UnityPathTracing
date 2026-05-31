using System;
using System.Runtime.InteropServices;
using NativeRender;
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
        // sizeof(StablePlane) in bytes — must match HLSL struct StablePlane layout.
        public const int StablePlaneStride = 80;

        // Number of stable planes — mirrors cStablePlaneCount in Config.h.
        public const int StablePlaneCount = PathTracerConfig.cStablePlaneCount;

        // sizeof(PackedPathTracerSurfaceData) — TODO: verify from HLSL struct.
        public const int SurfaceDataStride = 64;

        // Max lights — mirrors RTXPT_LIGHTING_MAX_LIGHTS in LightingConfig.h.
        public const int MaxLights = LightingConfig.RTXPT_LIGHTING_MAX_LIGHTS;

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
        public UploadBuffer LightControlBuffer;

        /// <summary>
        /// Per-light data array. MaxLights elements.
        /// HLSL: u_lightsBuffer (u1).
        /// GPU-resident native buffer: CPU uploads the analytic-light sub-range while compute
        /// passes write env-quad-node and emissive-triangle entries via UAV. Stable native ptr.
        /// </summary>
        public UploadBuffer LightBuffer;

        /// <summary>
        /// Extended per-light data array. MaxLights elements.
        /// HLSL: u_lightsExBuffer (u2). Same CPU-upload + GPU-UAV model as <see cref="LightBuffer"/>.
        /// </summary>
        public UploadBuffer LightExBuffer;

        /// <summary>
        /// Scratch buffer for light processing passes.
        /// HLSL: u_scratchBuffer (u3).
        /// </summary>
        public UploadBuffer LightScratchBuffer;

        /// <summary>
        /// Light history remap: current frame index → previous frame index.
        /// 2 × (MaxLights+1) uint elements (matches original's carried-over byteSize; only [0..MaxLights) used).
        /// HLSL: u_historyRemapCurrentToPast (u6).
        /// </summary>
        public GraphicsBuffer HistoryRemapCurrentToPast;

        /// <summary>
        /// Light history remap: previous frame index → current frame index.
        /// 2 × (MaxLights+1) uint elements (matches original's carried-over byteSize; only [0..MaxLights) used).
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
        /// 2 × (MaxLights+1) uint elements (matches original's carried-over byteSize).
        /// HLSL: u_scratchList (u4).
        /// </summary>
        public GraphicsBuffer ScratchListBuffer;

        // ── Env map baker constant buffers ────────────────────────────────────
        /// <summary>Constant buffer for the base-layer env bake pass. 704 bytes.</summary>
        public VolatileConstantBuffer EnvBakerCb;

        /// <summary>Constant buffer for the importance baker pass. 48 bytes.</summary>
        public VolatileConstantBuffer ImportanceBakerCb;

        // ── Cached native pointers ────────────────────────────────────────────
        // Resolution-dependent buffers — valid after EnsureResources(), cleared on resize
        public IntPtr StablePlanesBufferPtr  { get; private set; }
        public IntPtr SurfaceDataBufferPtr   { get; private set; }
        public IntPtr LocalSamplingBufferPtr { get; private set; }

        // Light system buffers — valid after EnsureLightBuffers(), cleared in ReleaseLightBuffers()
        public IntPtr FeedbackBufferPtr           { get; private set; }
        public IntPtr HistoryRemapCurrToPastPtr   { get; private set; }
        public IntPtr HistoryRemapPastToCurrPtr   { get; private set; }
        public IntPtr LightProxyCountersPtr       { get; private set; }
        public IntPtr LightSamplingProxiesPtr     { get; private set; }
        public IntPtr LightWeightsBufferPtr       { get; private set; }
        public IntPtr ScratchListBufferPtr        { get; private set; }

        // CPU-side copy of the latest control data uploaded into LightControlBuffer.
        // NativeBuffer does not support the old synchronous GraphicsBuffer.GetData path.
        public RtxptLightingControlData LastLightControlData { get; internal set; }

        // ── Resolved resolution ───────────────────────────────────────────────
        public int2 renderResolution { get; private set; }

        // Scratch buffer size (heuristic: 16 ints per light entry).
        private const int ScratchElementCount = MaxLights * 16;

        // LightWeights ping-pong half-count: RTXPT_LIGHTING_WEIGHTS_COUNT_HALF (= MaxLights + 1).
        private const int WeightsCountHalf = LightingConfig.RTXPT_LIGHTING_WEIGHTS_COUNT_HALF;

        // Carryover-group element count = 2 × (MaxLights+1).
        // In the original RTXPT (LightsBaker.cpp) a single bufferDesc.byteSize —
        // 2 × sizeof(float) × RTXPT_LIGHTING_WEIGHTS_COUNT_HALF — is computed once for
        // m_lightWeights and then reused (never reset) for the next four buffers:
        //   m_historyRemapCurrentToPast, m_historyRemapPastToCurrent,
        //   m_perLightProxyCounters, m_scratchList.
        // So all five are actually allocated with 2 × (MaxLights+1) uint/float elements,
        // even though LightsBaker.h documents the intent of the last four as MaxLights.
        // We mirror the actual allocation here to stay byte-for-byte consistent with the
        // reference; the upper half of the four uint buffers is unused but the layout matches.
        private const int CarryoverGroupCount = 2 * WeightsCountHalf;

        // Proxy buffer sizes — derived from LightingConfig.h constants.
        // ProxyCounterCount: HLSL only ever indexes [0 .. TotalLightCount], where the last
        // slot [TotalLightCount] holds the invalid-feedback count, so MaxLights+1 elements
        // would be functionally sufficient. Sized to CarryoverGroupCount to match the original.
        private const int ProxyCounterCount   = CarryoverGroupCount;
        // ProxySamplingCount: worst-case total proxies = RTXPT_LIGHTING_MAX_SAMPLING_PROXIES.
        internal const int ProxySamplingCount = LightingConfig.RTXPT_LIGHTING_MAX_SAMPLING_PROXIES;
        private const int LocalSamplingCount  = LightingConfig.RTXPT_LIGHTING_MAX_LIGHTS; // placeholder; actual buffer is resolution-dependent (see EnsureResources)

        // ScratchListBuffer functional requirement is the larger of:
        //   - proxy-build passes: MaxLights entries
        //   - env-light backup region: 2 × RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT (5368×2 = 10736)
        // Both are well under CarryoverGroupCount, which is what the original actually allocates.
        private const int ScratchListCount = CarryoverGroupCount;

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

            // Tiled-swizzled storage element count for StablePlanesBuffer (TS_TILE_SIZE = 8 in Utils.hlsli).
            // Must match GenericTSComputePlaneStride: ceil(W/8)*8 * ceil(H/8)*8.
            int tsLineStride   = ((renderRes.x + 7) / 8) * 8;
            int tsPlaneStride  = tsLineStride * (((renderRes.y + 7) / 8) * 8);

            ReleaseResolutionBuffers();

            // LocalSamplingBuffer: tileW × tileH × LOCAL_PROXY_COUNT uint elements.
            // Each tile covers RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE × TILE_SIZE pixels.
            // +1 border tile on each axis to accommodate the jitter offset — must match
            // LightsBaker.cpp:340-341 and the LocalSamplingResolution uploaded to the control
            // buffer (NativeRtxptLightingUpdateBeginPass), since that value is the addressing stride.
            int tileW    = (renderRes.x + LightingConfig.RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE - 1) / LightingConfig.RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE + 1;
            int tileH    = (renderRes.y + LightingConfig.RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE - 1) / LightingConfig.RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE + 1;
            int localSampCount = tileW * tileH * LightingConfig.RTXPT_LIGHTING_LOCAL_PROXY_COUNT;
            LocalSamplingBuffer?.Release();
            LocalSamplingBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                localSampCount, 4)
            { name = "NEE_AT_LocalSamplingBuffer" };
            LocalSamplingBufferPtr = LocalSamplingBuffer.GetNativeBufferPtr();

            // StablePlanesBuffer: W×H×StablePlaneCount structured entries, stride = StablePlaneStride (80).
            // Shader declares: RWStructuredBuffer<StablePlane> u_StablePlanesBuffer (u42).
            StablePlanesBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                tsPlaneStride * StablePlaneCount,
                StablePlaneStride)
            { name = "StablePlanesBuffer" };
            StablePlanesBufferPtr = StablePlanesBuffer.GetNativeBufferPtr();

            // SurfaceDataBuffer: W×H×2 structured entries, stride = SurfaceDataStride (64).
            // Shader declares: RWStructuredBuffer<PackedPathTracerSurfaceData> u_SurfaceData (u45).
            SurfaceDataBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                pixelCount * 2,
                SurfaceDataStride)
            { name = "SurfaceData(GBuffer)" };
            SurfaceDataBufferPtr = SurfaceDataBuffer.GetNativeBufferPtr();

            return true;
        }

        /// <summary>
        /// Allocates light-system buffers. Call once after initialization.
        /// Does nothing if already allocated.
        /// </summary>
        public void EnsureLightBuffers()
        {
            if (LightControlBuffer != null) return;

            // Buffer debug names are kept byte-for-byte identical to the original RTXPT
            // RenderTargets.cpp / LightsBaker.cpp debugName strings for PIX parity.

            // FeedbackBuffer stub — 1 element, 64B stride (matches DebugFeedbackStruct size).
            FeedbackBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                1, 64)
            { name = "Feedback_Buffer_Gpu" };
            FeedbackBufferPtr = FeedbackBuffer.GetNativeBufferPtr();

            // LightControlBuffer — single RtxptLightingControlData element (576 bytes).
            LightControlBuffer = new UploadBuffer(
                1, Marshal.SizeOf<RtxptLightingControlData>(),
                UploadBuffer.UploadMode.Whole, allowUAV: true, debugName: "LightingControlData");

            // LightBuffer / LightExBuffer — match PolymorphicLightInfo (32B) and PolymorphicLightInfoEx (16B).
            // Native UAV-capable upload buffers: CPU writes the analytic-light sub-range (Ranges mode),
            // compute passes write env/emissive entries via UAV. Stable pointer (no GetNativeBufferPtr churn).
            LightBuffer = new UploadBuffer(
                MaxLights, Marshal.SizeOf<RtxptPolymorphicLightInfo>(),
                UploadBuffer.UploadMode.Ranges, allowUAV: true, debugName: "LightsBuffer");

            LightExBuffer = new UploadBuffer(
                MaxLights, Marshal.SizeOf<RtxptPolymorphicLightInfoEx>(),
                UploadBuffer.UploadMode.Ranges, allowUAV: true, debugName: "LightsExBuffer");

            LightScratchBuffer = new UploadBuffer(
                ScratchElementCount, 4,
                UploadBuffer.UploadMode.Ranges, allowUAV: true, debugName: "LightsScratchBuffer");

            // HistoryRemap buffers: functional need is MaxLights, but the original reuses the
            // carried-over byteSize (see CarryoverGroupCount) → 2 × (MaxLights+1). Match it.
            HistoryRemapCurrentToPast = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                CarryoverGroupCount, 4)
            { name = "HistoryRemapCurrentToPast" };
            HistoryRemapCurrToPastPtr = HistoryRemapCurrentToPast.GetNativeBufferPtr();

            HistoryRemapPastToCurrent = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                CarryoverGroupCount, 4)
            { name = "HistoryRemapPastToCurrent" };
            HistoryRemapPastToCurrPtr = HistoryRemapPastToCurrent.GetNativeBufferPtr();

            LightProxyCounters = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                ProxyCounterCount, 4)
            { name = "PerLightProxyCounters" };
            LightProxyCountersPtr = LightProxyCounters.GetNativeBufferPtr();

            LightSamplingProxies = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                ProxySamplingCount, 4)
            { name = "LightSamplingProxies" };
            LightSamplingProxiesPtr = LightSamplingProxies.GetNativeBufferPtr();

            // LocalSamplingBuffer is now allocated in EnsureResources() with the correct resolution-dependent size.

            // LightWeightsBuffer: 2 halves of (MaxLights+1) floats for ping-pong historic weights.
            LightWeightsBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                2 * WeightsCountHalf, 4)
            { name = "LightsWeights" };
            LightWeightsBufferPtr = LightWeightsBuffer.GetNativeBufferPtr();

            // ScratchListBuffer: uint typed scratch for proxy count prefix-sum and job list.
            // Functional need is max(MaxLights, 2 × RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT);
            // sized to CarryoverGroupCount (= 2 × (MaxLights+1)) to match the original allocation.
            ScratchListBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                ScratchListCount, 4)
            { name = "ScratchList" };
            ScratchListBufferPtr = ScratchListBuffer.GetNativeBufferPtr();
        }

        /// <summary>
        /// Allocates env map baker constant buffers. Idempotent — safe to call every frame.
        /// </summary>
        public void EnsureEnvBakerBuffers()
        {
            if (EnvBakerCb != null && ImportanceBakerCb != null) return;
            // Volatile CBs (pool suballocations) — names mirror the original RTXPT debugName
            // strings for symmetry, though volatile buffers have no stable PIX resource name.
            EnvBakerCb?.Dispose();
            EnvBakerCb = new VolatileConstantBuffer(NativeRtxptEnvMapBakerPass.EnvBakerCbSize, "EnvMapBakerConstants");
            ImportanceBakerCb?.Dispose();
            ImportanceBakerCb = new VolatileConstantBuffer(NativeRtxptEnvMapBakerPass.ImportanceBakerCbSize, "EnvMapImportanceSamplingBakerConstants");
        }

        private void ReleaseResolutionBuffers()
        {
            StablePlanesBuffer?.Release();
            StablePlanesBuffer  = null;
            StablePlanesBufferPtr = IntPtr.Zero;
            SurfaceDataBuffer?.Release();
            SurfaceDataBuffer  = null;
            SurfaceDataBufferPtr = IntPtr.Zero;
            LocalSamplingBuffer?.Release();
            LocalSamplingBuffer  = null;
            LocalSamplingBufferPtr = IntPtr.Zero;
        }

        private void ReleaseLightBuffers()
        {
            EnvBakerCb?.Dispose();               EnvBakerCb               = null;
            ImportanceBakerCb?.Dispose();         ImportanceBakerCb        = null;
            FeedbackBuffer?.Release();            FeedbackBuffer           = null; FeedbackBufferPtr           = IntPtr.Zero;
            LightControlBuffer?.Dispose();        LightControlBuffer       = null; 
            LightBuffer?.Dispose();               LightBuffer              = null;
            LightExBuffer?.Dispose();             LightExBuffer            = null;
            LightScratchBuffer?.Dispose();        LightScratchBuffer       = null; 
            HistoryRemapCurrentToPast?.Release(); HistoryRemapCurrentToPast= null; HistoryRemapCurrToPastPtr   = IntPtr.Zero;
            HistoryRemapPastToCurrent?.Release(); HistoryRemapPastToCurrent= null; HistoryRemapPastToCurrPtr   = IntPtr.Zero;
            LightProxyCounters?.Release();        LightProxyCounters       = null; LightProxyCountersPtr       = IntPtr.Zero;
            LightSamplingProxies?.Release();      LightSamplingProxies     = null; LightSamplingProxiesPtr     = IntPtr.Zero;
            LightWeightsBuffer?.Release();        LightWeightsBuffer       = null; LightWeightsBufferPtr       = IntPtr.Zero;
            ScratchListBuffer?.Release();         ScratchListBuffer        = null; ScratchListBufferPtr        = IntPtr.Zero;
        }

        public void Dispose()
        {
            ReleaseResolutionBuffers();
            ReleaseLightBuffers();
        }
    }
}
