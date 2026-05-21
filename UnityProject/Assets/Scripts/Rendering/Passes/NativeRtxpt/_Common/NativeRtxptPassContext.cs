using System;
using NativeRender;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace PathTracing
{
    /// <summary>
    /// Per-frame data carrier shared by all NativeRtxpt compute / RT passes.
    ///
    /// Created by <see cref="NativeRtxptFeature"/> once per camera per frame and
    /// passed to each pass via <c>Setup(ctx)</c> before enqueueing.
    ///
    /// Binding layout follows <c>ShaderResourceBindings.hlsli</c>:
    ///   b0  = g_Const        (SampleConstants)
    ///   b1  = g_MiniConst    (SampleMiniConstants — not yet used, reserved)
    ///   TLAS via NRDSampleResource.AccelerationStructure
    /// </summary>
    public class NativeRtxptPassContext
    {
        // ── Constant buffers ──────────────────────────────────────────────────
        /// <summary>g_Const (b0) — SampleConstants, built each frame by NativeRtxptConstantsBuilder.</summary>
        public GraphicsBuffer ConstantBuffer;

        // ── Scene acceleration structure ──────────────────────────────────────
        /// <summary>Top-level acceleration structure for DXR passes.</summary>
        public NRDSampleResource NrdSampleResource;

        // ── GPU scene (bindless geometry / material arrays) ───────────────────
        /// <summary>Provides BindToShader() for bindless instance/geometry/material arrays.</summary>
        public NativeRtxptGPUScene GpuScene;

        // ── Texture pool (render-res) ─────────────────────────────────────────
        public NativeRtxptTextureResources Textures;

        // ── Buffer pool ───────────────────────────────────────────────────────
        public NativeRtxptBufferResources Buffers;

        // ── Resolution ───────────────────────────────────────────────────────
        public int2 RenderResolution;
        public int2 DisplayResolution;

        // ── Per-frame temporal state ──────────────────────────────────────────
        public CameraFrameState FrameState;

        // ── Inspector settings snapshot ───────────────────────────────────────
        public NativeRtxptSetting Setting;

        // ── Pre-resolved IntPtrs (filled once per frame from Textures pool) ───
        // u0  OutputColor
        public IntPtr OutputColorPtr;

        // u1  ProcessedOutputColor
        public IntPtr ProcessedOutputColorPtr;

        // u4  Throughput
        public IntPtr ThroughputPtr;

        // u5  MotionVectors
        public IntPtr MotionVectorsPtr;

        // u6  Depth
        public IntPtr DepthPtr;

        // u7  SpecularHitT
        public IntPtr SpecularHitTPtr;

        // u8  ScratchFloat1
        public IntPtr ScratchFloat1Ptr;

        // u40 StablePlanesHeader
        public IntPtr StablePlanesHeaderPtr;

        // u44 StableRadiance
        public IntPtr StableRadiancePtr;

        // u126 ShaderDebugVizTextureBuffer
        public IntPtr ShaderDebugVizPtr;

        // DLSS-RR guide buffers
        public IntPtr DlssRrDiffAlbedoPtr;
        public IntPtr DlssRrSpecAlbedoPtr;
        public IntPtr DlssRrNormalRoughnessPtr;
        public IntPtr DlssRrOutputPtr;

        // ── Factory ───────────────────────────────────────────────────────────

        /// <summary>
        /// Resolves all native IntPtrs from the texture pool.
        /// Must be called on the main thread after <see cref="NativeRtxptTextureResources.EnsureResources"/> completes.
        /// </summary>
        public void ResolveNativePtrs()
        {
            OutputColorPtr           = Textures.OutputColor.NativePtr;
            ProcessedOutputColorPtr  = Textures.ProcessedOutputColor.NativePtr;
            ThroughputPtr            = Textures.Throughput.NativePtr;
            MotionVectorsPtr         = Textures.ScreenMotionVectors.NativePtr;
            DepthPtr                 = Textures.Depth.NativePtr;
            SpecularHitTPtr          = Textures.SpecularHitT.NativePtr;
            ScratchFloat1Ptr         = Textures.ScratchFloat1.NativePtr;
            StablePlanesHeaderPtr    = Textures.StablePlanesHeader.NativePtr;
            StableRadiancePtr        = Textures.StableRadiance.NativePtr;
            ShaderDebugVizPtr        = Textures.ShaderDebugViz.NativePtr;
            DlssRrDiffAlbedoPtr      = Textures.DlssRrDiffAlbedo.NativePtr;
            DlssRrSpecAlbedoPtr      = Textures.DlssRrSpecAlbedo.NativePtr;
            DlssRrNormalRoughnessPtr = Textures.DlssRrNormalRoughness.NativePtr;
            DlssRrOutputPtr          = Textures.DlssRrOutput.NativePtr;
        }
    }
}