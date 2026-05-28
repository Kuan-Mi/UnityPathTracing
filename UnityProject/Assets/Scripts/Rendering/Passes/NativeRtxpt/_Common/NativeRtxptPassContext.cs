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
    ///   TLAS via GpuScene.AccelerationStructure
    /// </summary>
    public class NativeRtxptPassContext
    {
        // ── Constant buffers ──────────────────────────────────────────────────
        /// <summary>g_Const (b0) — SampleConstants, built each frame by NativeRtxptConstantsBuilder.</summary>
        public NativeBuffer ConstantBuffer;

        // ── Scene acceleration structure ──────────────────────────────────────
        // (TLAS is owned by GpuScene; use GpuScene.AccelerationStructure directly)

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
        public RtxptCameraFrameState FrameState;

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

        // ── Baked env map (filled by NativeRtxptEnvMapBakerPass each frame) ─────
        /// <summary>Baked 256×256 cubemap mip0. Bound as t_EnvironmentMap (TextureCube).</summary>
        public IntPtr BakedEnvCubePtr;
        /// <summary>1024×1024 flat importance map (R32F). Single-channel luminance/importance only.</summary>
        public IntPtr EnvImportanceMapPtr;
        /// <summary>1024×1024 combined radiance+importance map (RGBA16F, rgb=radiance, a=luminance).
        /// This is what LightsBaker shaders expect as t_envRadianceAndImportanceMap.</summary>
        public IntPtr EnvRadianceAndImportanceMapPtr;
        /// <summary>1024×1024 env-light lookup map (R32_UINT). Filled by EnvLightsFillLookupMap. Bound as t_EnvLookupMap (t18).</summary>
        public IntPtr EnvLightLookupMapPtr;

        // ── NEE-AT feedback texture pointers ─────────────────────────────────
        /// <summary>u_feedbackTotalWeight (u11) — main reservoir working surface (R32_FLOAT).</summary>
        public IntPtr FeedbackTotalWeightPtr;
        /// <summary>u_feedbackCandidates (u12) — main reservoir working surface (R32_UINT).</summary>
        public IntPtr FeedbackCandidatesPtr;
        /// <summary>u_feedbackTotalWeightScratch (u13) — scratch reprojection surface (R32_FLOAT).</summary>
        public IntPtr FeedbackTotalWeightScratchPtr;
        /// <summary>u_feedbackCandidatesScratch (u14) — scratch reprojection surface (R32_UINT).</summary>
        public IntPtr FeedbackCandidatesScratchPtr;
        /// <summary>u_feedbackTotalWeightBlended (u15) — blended early-feedback surface (R32_FLOAT, half-res).</summary>
        public IntPtr FeedbackTotalWeightBlendedPtr;
        /// <summary>u_feedbackCandidatesBlended (u16) — blended early-feedback surface (R32_UINT, half-res).</summary>
        public IntPtr FeedbackCandidatesBlendedPtr;
        /// <summary>u_historyDepth (u17) — NEE-AT per-pixel history depth / confidence (R32_FLOAT).</summary>
        public IntPtr NEEATHistoryDepthPtr;

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

            FeedbackTotalWeightPtr        = Textures.LightFeedbackTotalWeight.NativePtr;
            FeedbackCandidatesPtr         = Textures.LightFeedbackCandidates.NativePtr;
            FeedbackTotalWeightScratchPtr = Textures.FeedbackTotalWeightScratch.NativePtr;
            FeedbackCandidatesScratchPtr  = Textures.FeedbackCandidatesScratch.NativePtr;
            FeedbackTotalWeightBlendedPtr = Textures.FeedbackTotalWeightBlended.NativePtr;
            FeedbackCandidatesBlendedPtr  = Textures.FeedbackCandidatesBlended.NativePtr;
            NEEATHistoryDepthPtr          = Textures.NEEATHistoryDepth.NativePtr;
        }
    }
}
