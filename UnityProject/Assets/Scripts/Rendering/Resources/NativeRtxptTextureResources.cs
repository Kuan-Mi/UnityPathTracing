using System;
using Nri;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace PathTracing
{
    /// <summary>
    /// Owns all per-camera render textures for <see cref="NativeRtxptFeature"/>.
    /// Formats mirror RenderTargets.cpp from the RTXPT sample.
    ///
    /// Denoising is performed by DLSS Ray Reconstruction (DLSS-RR) — no NRD.
    /// Stable plane count = 3 (cStablePlaneCount).
    /// </summary>
    public class NativeRtxptTextureResources : IDisposable
    {
        // ── Path tracer primary outputs ───────────────────────────────────────
        /// <summary>Main PT output color. RGBA16_FLOAT. Written by PT shader → NoDenoiserFinalMerge input.</summary>
        public NriTextureResource OutputColor;

        /// <summary>Screen-space depth. R32_FLOAT. Written by PT / ExportVisibilityBuffer.</summary>
        public NriTextureResource Depth;

        /// <summary>Screen-space motion vectors. RGBA16_FLOAT.</summary>
        public NriTextureResource ScreenMotionVectors;

        /// <summary>PSR path throughput packed as fp16×2. R32_UINT.</summary>
        public NriTextureResource Throughput;

        // ── Specular hit-distance filtering ──────────────────────────────────
        /// <summary>Specular hit distance. R32_FLOAT. Bilateral-filtered by DenoiseSpecHitT pass (×2 ping-pong).</summary>
        public NriTextureResource SpecularHitT;

        /// <summary>Scratch buffer for bilateral filter ping-pong. R32_FLOAT.</summary>
        public NriTextureResource ScratchFloat1;

        // ── Stable plane outputs ──────────────────────────────────────────────
        /// <summary>
        /// Stable plane header. R32_UINT, Texture2DArray with 4 slices.
        /// Slices 0-2 = branch IDs per plane, slice 3 = first-hit distance.
        /// </summary>
        public NriTextureResource StablePlanesHeader;

        /// <summary>Per-stable-plane stable radiance (portion not sent to denoiser). RGBA16_FLOAT.</summary>
        public NriTextureResource StableRadiance;

        // ── GBuffer (written by PT shader) ────────────────────────────────────
        /// <summary>Base color. B10G11R11_UFloatPack32.</summary>
        public NriTextureResource BaseColor;

        /// <summary>Specular normal packed. R32_UINT.</summary>
        public NriTextureResource SpecNormal;

        /// <summary>Roughness + metalness. RG16_FLOAT.</summary>
        public NriTextureResource RoughnessMetal;

        /// <summary>Material flags / info. R32_UINT.</summary>
        public NriTextureResource MaterialInfo;

        // ── DLSS-RR guide buffers (prepared by DlssBeforePass equivalent) ────
        /// <summary>Diffuse albedo guide for DLSS-RR. R11G11B10_FLOAT (sl::kBufferTypeAlbedo).</summary>
        public NriTextureResource DlssRrDiffAlbedo;

        /// <summary>Specular albedo guide for DLSS-RR. R11G11B10_FLOAT (sl::kBufferTypeSpecularAlbedo).</summary>
        public NriTextureResource DlssRrSpecAlbedo;

        /// <summary>Specular motion vectors guide for DLSS-RR. RG16_FLOAT (sl::kBufferTypeSpecularMotionVectors).</summary>
        public NriTextureResource DlssRrSpecMotionVectors;

        /// <summary>Normal + roughness guide for DLSS-RR. RGBA16_FLOAT.</summary>
        public NriTextureResource DlssRrNormalRoughness;

        // ── DLSS-RR output ────────────────────────────────────────────────────
        /// <summary>DLSS-RR denoised + upscaled output. RGBA16_FLOAT. Display resolution.</summary>
        public NriTextureResource DlssRrOutput;

        // ── Light feedback (NEE adaptive sampling) ────────────────────────────────
        /// <summary>Per-pixel accumulated NEE weight. R32_FLOAT. Bound as u_LightFeedbackTotalWeight (u20).</summary>
        public NriTextureResource LightFeedbackTotalWeight;

        /// <summary>Per-pixel NEE candidate light index. R32_UINT. Bound as u_LightFeedbackCandidates (u21).</summary>
        public NriTextureResource LightFeedbackCandidates;

        // ── NEE-AT feedback scratch / blended / history ───────────────────────
        /// <summary>Scratch reprojection target for NEE total weight. R32_FLOAT. u_feedbackTotalWeightScratch (u13).</summary>
        public NriTextureResource FeedbackTotalWeightScratch;

        /// <summary>Scratch reprojection target for NEE candidates. R32_UINT. u_feedbackCandidatesScratch (u14).</summary>
        public NriTextureResource FeedbackCandidatesScratch;

        /// <summary>Blended early-feedback total weight. R32_FLOAT. Size = ceil(renderRes/NEEAT_TILE_SIZE). u_feedbackTotalWeightBlended (u15).</summary>
        public NriTextureResource FeedbackTotalWeightBlended;

        /// <summary>Blended early-feedback candidates. R32_UINT. Size = ceil(renderRes/NEEAT_TILE_SIZE). u_feedbackCandidatesBlended (u16).</summary>
        public NriTextureResource FeedbackCandidatesBlended;

        /// <summary>NEE-AT per-pixel history depth / confidence. R32_FLOAT. u_historyDepth (u17).</summary>
        public NriTextureResource NEEATHistoryDepth;

        // ── Debug viz ─────────────────────────────────────────────────────────────
        /// <summary>Shader debug visualisation texture. R32_FLOAT. Bound as u_ShaderDebugVizTextureBuffer (u126).</summary>
        public NriTextureResource ShaderDebugViz;

        /// <summary>Debug output color texture. RGBA16_FLOAT. Mirrors u_OutputColor for debug inspection (written by PathTracer debug code).</summary>
        public NriTextureResource DebugOutputColor;

        // ── Reference mode accumulation ───────────────────────────────────────
        /// <summary>Multi-frame accumulation buffer (reference mode only). RGBA32_FLOAT.</summary>
        public NriTextureResource AccumulatedRadiance;

        /// <summary>Post-accumulation output (reference mode). RGBA16_FLOAT. Display resolution.</summary>
        public NriTextureResource ProcessedOutputColor;

        // ── Env map baking outputs (fixed size, shared across frames) ─────────
        /// <summary>256×256 baked env cubemap mip0. RGBA16F, UAV. Bound as t_EnvironmentMap.</summary>
        public NriTextureResource EnvCubeMip0;

        /// <summary>128×128 baked env cubemap mip1. RGBA16F, UAV.</summary>
        public NriTextureResource EnvCubeMip1;

        /// <summary>1024×1024 importance map. R32F, UAV, mipmapped. Bound as u_ImportanceMap / t_EnvImportanceMap.</summary>
        public NriTextureResource EnvImportanceMap;

        /// <summary>1024×1024 combined radiance+importance map. RGBA16F, UAV, mipmapped. Bound as u_RadianceMap / t_envRadianceAndImportanceMap.</summary>
        public NriTextureResource EnvRadianceMap;

        /// <summary>4×4 dummy cubemap (RGBA8). Used to satisfy shader bindings when the real cube is unneeded.</summary>
        public NriTextureResource EnvDummyCube;
        
        /// <summary>1024×1024 env-light lookup map (R32_UINT). Filled by EnvLightsFillLookupMap. Bound as t_EnvLookupMap (t18).</summary>
        public NriTextureResource EnvLightLookupMap;

        // ── Resolved dimensions ───────────────────────────────────────────────
        public int2 renderResolution  { get; private set; }
        public int2 displayResolution { get; private set; }

        public NativeRtxptTextureResources()
        {
            var srv = new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE,         layout = Layout.SHADER_RESOURCE,         stageBits = 1 << 7  };
            var uav = new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE_STORAGE, layout = Layout.SHADER_RESOURCE_STORAGE, stageBits = 1 << 10 };

            OutputColor           = new NriTextureResource("Rtxpt_OutputColor",           GraphicsFormat.R16G16B16A16_SFloat,     uav);
            Depth                 = new NriTextureResource("Rtxpt_Depth",                 GraphicsFormat.R32_SFloat,               uav);
            ScreenMotionVectors   = new NriTextureResource("Rtxpt_ScreenMotionVectors",   GraphicsFormat.R16G16B16A16_SFloat,     uav);
            Throughput            = new NriTextureResource("Rtxpt_Throughput",            GraphicsFormat.R32_UInt,                 uav);

            SpecularHitT          = new NriTextureResource("Rtxpt_SpecularHitT",          GraphicsFormat.R32_SFloat,               uav);
            ScratchFloat1         = new NriTextureResource("Rtxpt_ScratchFloat1",         GraphicsFormat.R32_SFloat,               uav);

            StablePlanesHeader    = new NriTextureResource("Rtxpt_StablePlanesHeader",    GraphicsFormat.R32_UInt,                 uav);
            StableRadiance        = new NriTextureResource("Rtxpt_StableRadiance",        GraphicsFormat.R16G16B16A16_SFloat,     uav);

            BaseColor             = new NriTextureResource("Rtxpt_BaseColor",             GraphicsFormat.B10G11R11_UFloatPack32,  uav);
            SpecNormal            = new NriTextureResource("Rtxpt_SpecNormal",            GraphicsFormat.R32_UInt,                 uav);
            RoughnessMetal        = new NriTextureResource("Rtxpt_RoughnessMetal",        GraphicsFormat.R16G16_SFloat,            uav);
            MaterialInfo          = new NriTextureResource("Rtxpt_MaterialInfo",          GraphicsFormat.R32_UInt,                 uav);

            DlssRrDiffAlbedo      = new NriTextureResource("Rtxpt_DlssRrDiffAlbedo",      GraphicsFormat.B10G11R11_UFloatPack32,  uav);
            DlssRrSpecAlbedo      = new NriTextureResource("Rtxpt_DlssRrSpecAlbedo",      GraphicsFormat.B10G11R11_UFloatPack32,  uav);
            DlssRrSpecMotionVectors = new NriTextureResource("Rtxpt_DlssRrSpecMotionVectors", GraphicsFormat.R16G16_SFloat,          uav);
            DlssRrNormalRoughness = new NriTextureResource("Rtxpt_DlssRrNormalRoughness", GraphicsFormat.R16G16B16A16_SFloat,     uav);

            DlssRrOutput          = new NriTextureResource("Rtxpt_DlssRrOutput",          GraphicsFormat.R16G16B16A16_SFloat,     uav);

            LightFeedbackTotalWeight = new NriTextureResource("Rtxpt_LightFeedbackTotalWeight", GraphicsFormat.R32_SFloat,  uav);
            LightFeedbackCandidates  = new NriTextureResource("Rtxpt_LightFeedbackCandidates",  GraphicsFormat.R32_UInt,    uav);

            FeedbackTotalWeightScratch  = new NriTextureResource("Rtxpt_FeedbackTotalWeightScratch",  GraphicsFormat.R32_SFloat, uav);
            FeedbackCandidatesScratch   = new NriTextureResource("Rtxpt_FeedbackCandidatesScratch",   GraphicsFormat.R32_UInt,   uav);
            FeedbackTotalWeightBlended  = new NriTextureResource("Rtxpt_FeedbackTotalWeightBlended",  GraphicsFormat.R32_SFloat, uav);
            FeedbackCandidatesBlended   = new NriTextureResource("Rtxpt_FeedbackCandidatesBlended",   GraphicsFormat.R32_UInt,   uav);
            NEEATHistoryDepth           = new NriTextureResource("Rtxpt_NEEATHistoryDepth",           GraphicsFormat.R32_SFloat, uav);

            ShaderDebugViz       = new NriTextureResource("Rtxpt_ShaderDebugViz",        GraphicsFormat.R16G16B16A16_SFloat,               uav);
            DebugOutputColor     = new NriTextureResource("Rtxpt_DebugOutputColor",       GraphicsFormat.R16G16B16A16_SFloat,     uav);
            AccumulatedRadiance  = new NriTextureResource("Rtxpt_AccumulatedRadiance",   GraphicsFormat.R32G32B32A32_SFloat,     uav);
            ProcessedOutputColor = new NriTextureResource("Rtxpt_ProcessedOutputColor",  GraphicsFormat.R16G16B16A16_SFloat,     uav);

            EnvCubeMip0      = new NriTextureResource("Rtxpt_EnvCubeMip0",      GraphicsFormat.R16G16B16A16_SFloat, uav);
            EnvCubeMip1      = new NriTextureResource("Rtxpt_EnvCubeMip1",      GraphicsFormat.R16G16B16A16_SFloat, uav);
            EnvImportanceMap = new NriTextureResource("Rtxpt_EnvImportanceMap", GraphicsFormat.R32_SFloat,           uav);
            EnvRadianceMap   = new NriTextureResource("Rtxpt_EnvRadianceMap",   GraphicsFormat.R16G16B16A16_SFloat, uav);
            EnvDummyCube     = new NriTextureResource("Rtxpt_EnvDummyCube",     GraphicsFormat.R8G8B8A8_UNorm,      srv);
            EnvLightLookupMap     = new NriTextureResource("Rtxpt_EnvLightLookupMap",     GraphicsFormat.R32_UInt,      srv);
        }

        /// <summary>
        /// Allocates env map baking textures (fixed sizes, not resolution-dependent).
        /// Idempotent — safe to call every frame.
        /// </summary>
        public void EnsureEnvMapResources()
        {
            if (EnvCubeMip0.IsCreated) return;
            EnvCubeMip0.AllocateCube(256);
            EnvCubeMip1.AllocateCube(128);
            EnvImportanceMap.Allocate(new int2(1024, 1024), useMipMap: true);
            EnvRadianceMap.Allocate(new int2(1024, 1024), useMipMap: true);
            EnvDummyCube.AllocateCube(4, enableRandomWrite: false);
            EnvLightLookupMap.Allocate(new int2(1024, 1024), useMipMap: false);
        }

        /// <summary>
        /// Allocates or reallocates all textures for the given render and display resolutions.
        /// Returns true if any allocation occurred (callers must re-bind DLSS-RR resources).
        /// </summary>
        public bool EnsureResources(int2 renderRes, int2 displayRes)
        {
            bool sameRender  = OutputColor.IsCreated
                               && renderResolution.x  == renderRes.x
                               && renderResolution.y  == renderRes.y;
            bool sameDisplay = DlssRrOutput.IsCreated
                               && displayResolution.x == displayRes.x
                               && displayResolution.y == displayRes.y;
            if (sameRender && sameDisplay) return false;

            renderResolution  = renderRes;
            displayResolution = displayRes;

            foreach (var tex in RenderResolutionTextures())
                tex.Allocate(renderResolution);

            // StablePlanesHeader is a Texture2DArray with 4 slices
            StablePlanesHeader.Allocate(renderResolution, slices: 4);

            // Blended feedback: half resolution (ceil each axis by NEEAT_TILE_SIZE=2)
            var blendedRes = new int2(
                (renderRes.x + LightingConfig.RTXPT_NEEAT_EARLY_FEEDBACK_TILE_SIZE - 1) / LightingConfig.RTXPT_NEEAT_EARLY_FEEDBACK_TILE_SIZE,
                (renderRes.y + LightingConfig.RTXPT_NEEAT_EARLY_FEEDBACK_TILE_SIZE - 1) / LightingConfig.RTXPT_NEEAT_EARLY_FEEDBACK_TILE_SIZE);
            FeedbackTotalWeightBlended.Allocate(blendedRes);
            FeedbackCandidatesBlended.Allocate(blendedRes);

            // Display-resolution textures
            DlssRrOutput.Allocate(displayResolution);
            ProcessedOutputColor.Allocate(displayResolution);

            return true;
        }

        private NriTextureResource[] RenderResolutionTextures() => new[]
        {
            OutputColor, Depth, ScreenMotionVectors, Throughput,
            SpecularHitT, ScratchFloat1,
            StableRadiance,
            BaseColor, SpecNormal, RoughnessMetal, MaterialInfo,
            DlssRrDiffAlbedo, DlssRrSpecAlbedo, DlssRrSpecMotionVectors, DlssRrNormalRoughness,
            LightFeedbackTotalWeight, LightFeedbackCandidates,
            FeedbackTotalWeightScratch, FeedbackCandidatesScratch,
            NEEATHistoryDepth,
            ShaderDebugViz, DebugOutputColor, AccumulatedRadiance,
        };

        public void Dispose()
        {
            foreach (var tex in AllTextures())
            {
                if (!tex.IsCreated) continue;
                var h = tex.Handle;
                if (h?.rt != null)
                {
                    AsyncGPUReadback.Request(h).WaitForCompletion();
                    break;
                }
            }
            foreach (var tex in AllTextures()) tex.Release();
        }

        private NriTextureResource[] AllTextures() => new[]
        {
            OutputColor, Depth, ScreenMotionVectors, Throughput,
            SpecularHitT, ScratchFloat1,
            StablePlanesHeader, StableRadiance,
            BaseColor, SpecNormal, RoughnessMetal, MaterialInfo,
            DlssRrDiffAlbedo, DlssRrSpecAlbedo, DlssRrSpecMotionVectors, DlssRrNormalRoughness,
            LightFeedbackTotalWeight, LightFeedbackCandidates,
            FeedbackTotalWeightScratch, FeedbackCandidatesScratch,
            FeedbackTotalWeightBlended, FeedbackCandidatesBlended,
            NEEATHistoryDepth,
            ShaderDebugViz, DebugOutputColor, DlssRrOutput, AccumulatedRadiance, ProcessedOutputColor,
            EnvCubeMip0, EnvCubeMip1, EnvImportanceMap, EnvRadianceMap, EnvDummyCube,EnvLightLookupMap
        };
    }
}