using System;
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

        // ── Resolved resolution ───────────────────────────────────────────────
        public int2 renderResolution { get; private set; }

        // Scratch buffer size (heuristic: 16 ints per light entry).
        private const int ScratchElementCount = MaxLights * 16;

        // Proxy buffer sizes (heuristic defaults; can be adjusted for scene complexity).
        private const int ProxyCounterCount = 1024;
        private const int ProxySamplingCount = MaxLights * 4;
        private const int LocalSamplingCount = MaxLights;

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

            // LightControlBuffer — single 256-byte element (pad to 256 for CBV alignment).
            LightControlBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                1, 256)
            { name = "Rtxpt_LightControlBuffer" };

            // LightBuffer / LightExBuffer — placeholder stride of 64B; TODO: match LightData struct.
            LightBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                MaxLights, 64)
            { name = "Rtxpt_LightBuffer" };

            LightExBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                MaxLights, 32)
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
        }

        public void Dispose()
        {
            ReleaseResolutionBuffers();
            ReleaseLightBuffers();
        }
    }
}
