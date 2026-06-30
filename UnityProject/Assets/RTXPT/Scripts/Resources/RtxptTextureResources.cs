using System;
using PathTracing.NativeInterop.NRI;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace PathTracing
{
    /// <summary>
    /// Owns all per-camera render textures for <see cref="RtxptFeature"/>.
    /// Formats mirror RenderTargets.cpp from the RTXPT sample.
    ///
    /// Denoising is performed by DLSS Ray Reconstruction (DLSS-RR) — no NRD.
    /// Stable plane count = 3 (cStablePlaneCount).
    /// </summary>
    public class RtxptTextureResources : IDisposable
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

        // ── Bloom scratch (donut BloomPass: downscale ×2 → separable blur ping-pong) ──
        /// <summary>Bloom half-resolution downscale. RGBA16_FLOAT.</summary>
        public NriTextureResource BloomDownscale1;
        /// <summary>Bloom quarter-resolution downscale (blur source). RGBA16_FLOAT.</summary>
        public NriTextureResource BloomDownscale2;
        /// <summary>Bloom horizontal-blur result (quarter res). RGBA16_FLOAT.</summary>
        public NriTextureResource BloomBlurPass1;
        /// <summary>Bloom vertical-blur result (quarter res), composited back into the HDR image. RGBA16_FLOAT.</summary>
        public NriTextureResource BloomBlurPass2;

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
        /// <summary>Baked env cubemap (RtxptEnvMapBakerPass.CubeDim, e.g. 2048²), full solid-angle
        /// mip chain. RGBA16F, UAV. Mirrors the original EnvMapBaker m_cubemap: BaseLayerCS writes mip 0+1,
        /// MIPReduceCS fills mips 2…N. Bound as t_EnvironmentMap / t_EnvMapCube.</summary>
        public NriTextureResource EnvCubemap;

        /// <summary>1024×1024 importance map. R32F, UAV, mipmapped. Bound as u_ImportanceMap / t_EnvImportanceMap.</summary>
        public NriTextureResource EnvImportanceMap;

        /// <summary>1024×1024 combined radiance+importance map. RGBA16F, UAV, mipmapped. Bound as u_RadianceMap / t_envRadianceAndImportanceMap.</summary>
        public NriTextureResource EnvRadianceMap;

        /// <summary>4×4 dummy cubemap (RGBA8). Used to satisfy shader bindings when the real cube is unneeded.</summary>
        public NriTextureResource EnvDummyCube;

        /// <summary>
        /// BC6H compression scratch cube (RGBA32_UINT, UAV, CubeDim/4 base, full mip chain). The
        /// BC6UCompress CS writes one packed 128-bit BC6H block per texel here; <see cref="EnvCubemapBC6H"/>
        /// is then reinterpret-copied from it. Mirrors the original EnvMapBakerMainCubeBC6HScratch.
        /// </summary>
        public NriTextureResource EnvCubemapBC6HScratch;

        /// <summary>
        /// Final BC6H_UFLOAT compressed env cube (plugin-owned ID3D12Resource*, not a Unity
        /// RenderTexture — Unity cannot create BC6H RTs). Sampled as t_EnvironmentMap at trace
        /// time when <see cref="RtxptEnvMapBakerPass.EnableBC6UCompression"/> is on. Mirrors
        /// the original EnvMapBaker m_cubemapBC6H returned by GetEnvMapCube() when m_outputIsCompressed.
        /// </summary>
        public IntPtr EnvCubemapBC6H;
        
        /// <summary>1024×1024 env-light lookup map (R32_UINT). Filled by EnvLightsFillLookupMap. Bound as t_EnvLookupMap (t18).</summary>
        public NriTextureResource EnvLightLookupMap;

        // ── Env map bake state ────────────────────────────────────────────────
        /// <summary>
        /// False until the env cubemap/importance map have been baked at least once for the
        /// current allocation. Reset whenever env textures are (re)allocated so the baker
        /// always produces valid contents before they are sampled. Mirrors the original
        /// EnvMapBaker's force-rebake-on-recreate behaviour.
        /// </summary>
        public bool  EnvBaked;

        /// <summary>
        /// Hash of the inputs used for the last bake: the env baker constants (directional
        /// lights, scale color, background type) plus the sky texture identity and update count.
        /// Tint/intensity are NOT included — they are applied at sample time, not baked. The
        /// baker re-bakes only when this changes — equivalent to the original
        /// EnvMapBaker::Update <c>contentsChanged</c> early-out.
        /// </summary>
        public ulong EnvBakeSignature;

        // ── Resolved dimensions ───────────────────────────────────────────────
        public int2 renderResolution  { get; private set; }
        public int2 displayResolution { get; private set; }

        public RtxptTextureResources()
        {
            var srv = new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE,         layout = Layout.SHADER_RESOURCE,         stageBits = 1 << 7  };
            var uav = new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE_STORAGE, layout = Layout.SHADER_RESOURCE_STORAGE, stageBits = 1 << 10 };

            // Texture debug names are kept byte-for-byte identical to the original RTXPT
            // RenderTargets.cpp / LightsBaker.cpp / EnvMapBaker.cpp / donut BloomPass.cpp
            // debugName strings, so PIX captures of the replica line up with the reference.
            OutputColor           = new NriTextureResource("OutputColor",                 GraphicsFormat.R16G16B16A16_SFloat,     uav);
            Depth                 = new NriTextureResource("Depth",                       GraphicsFormat.R32_SFloat,               uav);
            ScreenMotionVectors   = new NriTextureResource("ScreenMotionVectors",         GraphicsFormat.R16G16B16A16_SFloat,     uav);
            Throughput            = new NriTextureResource("Throughput",                  GraphicsFormat.R32_UInt,                 uav);

            SpecularHitT          = new NriTextureResource("SpecularHitT",                GraphicsFormat.R32_SFloat,               uav);
            ScratchFloat1         = new NriTextureResource("ScratchFloat1",               GraphicsFormat.R32_SFloat,               uav);

            StablePlanesHeader    = new NriTextureResource("StablePlanesHeader",          GraphicsFormat.R32_UInt,                 uav);
            StableRadiance        = new NriTextureResource("StableRadianceBuffer",        GraphicsFormat.R16G16B16A16_SFloat,     uav);

            BaseColor             = new NriTextureResource("GBufferBaseColor",            GraphicsFormat.B10G11R11_UFloatPack32,  uav);
            SpecNormal            = new NriTextureResource("GBufferSpecNormal",           GraphicsFormat.R32_UInt,                 uav);
            RoughnessMetal        = new NriTextureResource("GBufferRoughnessMetal",       GraphicsFormat.R16G16_SFloat,            uav);
            MaterialInfo          = new NriTextureResource("GBufferMaterialInfo",         GraphicsFormat.R32_UInt,                 uav);

            DlssRrDiffAlbedo      = new NriTextureResource("RRDiffuseAlbedo",             GraphicsFormat.B10G11R11_UFloatPack32,  uav);
            DlssRrSpecAlbedo      = new NriTextureResource("RRSpecAlbedo",               GraphicsFormat.B10G11R11_UFloatPack32,  uav);
            DlssRrSpecMotionVectors = new NriTextureResource("RRSpecMotionVectors",       GraphicsFormat.R16G16_SFloat,          uav);
            DlssRrNormalRoughness = new NriTextureResource("RRNormalsAndRoughness",       GraphicsFormat.R16G16B16A16_SFloat,     uav);

            // DLSS-RR evaluate output → original ProcessedOutputColor (display res).
            DlssRrOutput          = new NriTextureResource("ProcessedOutputColor",        GraphicsFormat.R16G16B16A16_SFloat,     uav);

            BloomDownscale1       = new NriTextureResource("bloom src mip1",              GraphicsFormat.R16G16B16A16_SFloat,     uav);
            BloomDownscale2       = new NriTextureResource("bloom src mip2",              GraphicsFormat.R16G16B16A16_SFloat,     uav);
            BloomBlurPass1        = new NriTextureResource("bloom accumulation pass1",    GraphicsFormat.R16G16B16A16_SFloat,     uav);
            BloomBlurPass2        = new NriTextureResource("bloom accumulation pass2",    GraphicsFormat.R16G16B16A16_SFloat,     uav);

            LightFeedbackTotalWeight = new NriTextureResource("NEE_AT_FeedbackTotalWeight", GraphicsFormat.R32_SFloat,  uav);
            LightFeedbackCandidates  = new NriTextureResource("NEE_AT_FeedbackCandidates",  GraphicsFormat.R32_UInt,    uav);

            FeedbackTotalWeightScratch  = new NriTextureResource("NEE_AT_FeedbackTotalWeightScratch",       GraphicsFormat.R32_SFloat, uav);
            FeedbackCandidatesScratch   = new NriTextureResource("NEE_AT_FeedbackCandidatesScratch",        GraphicsFormat.R32_UInt,   uav);
            FeedbackTotalWeightBlended  = new NriTextureResource("NEE_AT_EarlyFeedbackTotalWeightScratch",  GraphicsFormat.R32_SFloat, uav);
            FeedbackCandidatesBlended   = new NriTextureResource("NEE_AT_EarlyFeedbackCandidatesScratch",   GraphicsFormat.R32_UInt,   uav);
            NEEATHistoryDepth           = new NriTextureResource("NEE_AT_HistoryDepth",                     GraphicsFormat.R32_SFloat, uav);

            ShaderDebugViz       = new NriTextureResource("DebugVizOutput",              GraphicsFormat.R16G16B16A16_SFloat,               uav);
            // DebugOutputColor / EnvDummyCube have no original RTXPT counterpart (replica-only); kept unprefixed.
            DebugOutputColor     = new NriTextureResource("DebugOutputColor",            GraphicsFormat.R16G16B16A16_SFloat,     uav);
            AccumulatedRadiance  = new NriTextureResource("AccumulatedRadiance",         GraphicsFormat.R32G32B32A32_SFloat,     uav);
            // Replica's final LDR / tone-map target → original tone-map output (LdrColor).
            ProcessedOutputColor = new NriTextureResource("LdrColor",                    GraphicsFormat.R16G16B16A16_SFloat,     uav);

            EnvCubemap      = new NriTextureResource("EnvMapBakerMainCube",       GraphicsFormat.R16G16B16A16_SFloat, uav);
            EnvImportanceMap = new NriTextureResource("EnvImportanceMap",          GraphicsFormat.R32_SFloat,           uav);
            EnvRadianceMap   = new NriTextureResource("EnvRadianceMap",            GraphicsFormat.R16G16B16A16_SFloat, uav);
            EnvDummyCube     = new NriTextureResource("EnvDummyCube",              GraphicsFormat.R8G8B8A8_UNorm,      srv);
            EnvLightLookupMap     = new NriTextureResource("EnvLightLookupMap",    GraphicsFormat.R32_UInt,      srv);
            EnvCubemapBC6HScratch = new NriTextureResource("EnvMapBakerMainCubeBC6HScratch", GraphicsFormat.R32G32B32A32_UInt, uav);
        }

        /// <summary>
        /// Allocates env map baking textures (fixed sizes, not resolution-dependent).
        /// Idempotent — safe to call every frame.
        /// </summary>
        public bool EnsureEnvMapResources()
        {
            if (EnvCubemap.IsCreated) return false;
            int cubeDim  = RtxptEnvMapBakerPass.CubeDim;
            int cubeMips = RtxptEnvMapBakerPass.CubeMipCount;
            EnvCubemap.AllocateCube(cubeDim, useMipMap: true, mipCount: cubeMips);
            EnvImportanceMap.Allocate(new int2(1024, 1024), useMipMap: true);
            EnvRadianceMap.Allocate(new int2(1024, 1024), useMipMap: true);
            EnvDummyCube.AllocateCube(4, enableRandomWrite: false);
            EnvLightLookupMap.Allocate(new int2(1024, 1024), useMipMap: false);

            // if (RtxptEnvMapBakerPass.EnableBC6UCompression)
            // {
            //     // Scratch base = cubeDim / BC block size (4); same mip count as the cube so each
            //     // BC6H subresource has a matching RGBA32_UINT block grid. Mirrors EnvMapBaker.cpp
            //     // InitBuffers (m_cubemapBC6HScratch / m_cubemapBC6H).
            //     EnvCubemapBC6HScratch.AllocateCube(cubeDim / 4, useMipMap: true, mipCount: cubeMips);
            //     if (EnvCubemapBC6H == IntPtr.Zero)
            //         EnvCubemapBC6H = NativeRender.NativeRenderPlugin.NR_CreateBC6HCube((uint)cubeDim, (uint)cubeMips);
            // } 

            // Freshly allocated cube/importance maps hold garbage — force the baker to re-run.
            EnvBaked = false;
            return true;
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

            // Bloom scratch: half- and quarter-RENDER-resolution (ceil), replicating the original's
            // quirk — Sample.cpp:1291 constructs donut BloomPass with *m_view (the render-res view),
            // so its intermediates are ceil(renderRes/2) even though Render() blits from/to the
            // display-res ProcessedOutputColor. The first "Downscale" is therefore a >2x reduction
            // and the Gaussian sigma lives in quarter-render-res pixels (wider bloom on screen).
            var bloomHalfRes    = new int2((renderRes.x + 1) / 2, (renderRes.y + 1) / 2);
            var bloomQuarterRes = new int2((bloomHalfRes.x + 1) / 2, (bloomHalfRes.y + 1) / 2);
            BloomDownscale1.Allocate(bloomHalfRes);
            BloomDownscale2.Allocate(bloomQuarterRes);
            BloomBlurPass1.Allocate(bloomQuarterRes);
            BloomBlurPass2.Allocate(bloomQuarterRes);

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

            if (EnvCubemapBC6H != IntPtr.Zero)
            {
                NativeRender.NativeRenderPlugin.NR_DestroyBC6HCube(EnvCubemapBC6H);
                EnvCubemapBC6H = IntPtr.Zero;
            }
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
            ShaderDebugViz, DebugOutputColor, DlssRrOutput,
            BloomDownscale1, BloomDownscale2, BloomBlurPass1, BloomBlurPass2,
            AccumulatedRadiance, ProcessedOutputColor,
            EnvCubemap, EnvImportanceMap, EnvRadianceMap, EnvDummyCube, EnvLightLookupMap,
            EnvCubemapBC6HScratch
        };
    }
}