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
        public NativeRtxdiGPUScene GpuScene;

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
            OutputColorPtr            = Textures.OutputColor.IsCreated          ? Textures.OutputColor.NativePtr          : IntPtr.Zero;
            ProcessedOutputColorPtr   = Textures.ProcessedOutputColor.IsCreated ? Textures.ProcessedOutputColor.NativePtr  : IntPtr.Zero;
            ThroughputPtr             = Textures.Throughput.IsCreated           ? Textures.Throughput.NativePtr            : IntPtr.Zero;
            MotionVectorsPtr          = Textures.ScreenMotionVectors.IsCreated  ? Textures.ScreenMotionVectors.NativePtr   : IntPtr.Zero;
            DepthPtr                  = Textures.Depth.IsCreated                ? Textures.Depth.NativePtr                 : IntPtr.Zero;
            SpecularHitTPtr           = Textures.SpecularHitT.IsCreated         ? Textures.SpecularHitT.NativePtr          : IntPtr.Zero;
            ScratchFloat1Ptr          = Textures.ScratchFloat1.IsCreated        ? Textures.ScratchFloat1.NativePtr         : IntPtr.Zero;
            StablePlanesHeaderPtr     = Textures.StablePlanesHeader.IsCreated   ? Textures.StablePlanesHeader.NativePtr    : IntPtr.Zero;
            StableRadiancePtr         = Textures.StableRadiance.IsCreated       ? Textures.StableRadiance.NativePtr        : IntPtr.Zero;
            DlssRrDiffAlbedoPtr       = Textures.DlssRrDiffAlbedo.IsCreated     ? Textures.DlssRrDiffAlbedo.NativePtr      : IntPtr.Zero;
            DlssRrSpecAlbedoPtr       = Textures.DlssRrSpecAlbedo.IsCreated     ? Textures.DlssRrSpecAlbedo.NativePtr      : IntPtr.Zero;
            DlssRrNormalRoughnessPtr  = Textures.DlssRrNormalRoughness.IsCreated ? Textures.DlssRrNormalRoughness.NativePtr : IntPtr.Zero;
            DlssRrOutputPtr           = Textures.DlssRrOutput.IsCreated         ? Textures.DlssRrOutput.NativePtr          : IntPtr.Zero;
        }
    }
}
