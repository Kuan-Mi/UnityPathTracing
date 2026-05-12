using System;
using Nri;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace PathTracing
{
    /// <summary>
    /// Owns all per-camera NRI render textures for <see cref="NativeNrdFeature"/>.
    /// Formats mirror NRDSample.cpp CreateResourcesAndDescriptors with
    /// SIGMA_TRANSLUCENCY=1, NRD_MODE=NORMAL, USE_LOW_PRECISION_FP_FORMATS=true.
    /// </summary>
    public class NativeNrdTextureResources : IDisposable
    {
        // ── NRD standard inputs (SRV) ──────────────────────────────────────
        public NriTextureResource Viewz;
        public NriTextureResource MV;
        public NriTextureResource NormalRoughness;
        public NriTextureResource BaseColorMetalness;
        public NriTextureResource Unfiltered_Penumbra;
        public NriTextureResource Unfiltered_Diff;
        public NriTextureResource Unfiltered_Spec;
        public NriTextureResource Unfiltered_Translucency;

        // ── NRD outputs (UAV) ───────────────────────────────────────────────
        public NriTextureResource Shadow;       // RGBA8_UNORM because SIGMA_TRANSLUCENCY=1
        public NriTextureResource Diff;
        public NriTextureResource Spec;
        public NriTextureResource Validation;

        // ── SHARC gradient textures — sized by sharcDims, allocated separately ──
        public NriTextureResource Gradient_StoredPing;
        public NriTextureResource Gradient_StoredPong;
        public NriTextureResource Gradient_Ping;
        public NriTextureResource Gradient_Pong;

        // ── DLSS / RR interop (UAV) ─────────────────────────────────────────
        public NriTextureResource DirectLighting;        // B10G11R11 (colorFormat)
        public NriTextureResource Composed;
        public NriTextureResource DlssOutput;            // output resolution
        public NriTextureResource RrGuideDiffAlbedo;
        public NriTextureResource RrGuideSpecAlbedo;
        public NriTextureResource RrGuideSpecHitDistance;
        public NriTextureResource RrGuideNormalRoughness;

        // ── Cross-frame / per-frame (UAV) ───────────────────────────────────
        public NriTextureResource TaaHistory;
        public NriTextureResource TaaHistoryPrev;
        public NriTextureResource PsrThroughput;         // R10_G10_B10_A2_UNORM
        public NriTextureResource Final;                 // output resolution
        public NriTextureResource PreFinal;              // output resolution
        public NriTextureResource DirectEmission;        // B10G11R11 (colorFormat)
        public NriTextureResource ComposedDiff;          // B10G11R11 (colorFormat)
        public NriTextureResource ComposedSpecViewZ;

        public int2 renderResolution { get; private set; }

        public NativeNrdTextureResources()
        {
            var srv = new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE,         layout = Layout.SHADER_RESOURCE,         stageBits = 1 << 7  };
            var uav = new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE_STORAGE, layout = Layout.SHADER_RESOURCE_STORAGE, stageBits = 1 << 10 };

            Viewz               = new NriTextureResource("Viewz",               GraphicsFormat.R32_SFloat,              srv);
            MV                  = new NriTextureResource("MV",                  GraphicsFormat.R16G16B16A16_SFloat,     srv);
            NormalRoughness     = new NriTextureResource("NormalRoughness",     GraphicsFormat.A2B10G10R10_UNormPack32, srv);
            BaseColorMetalness  = new NriTextureResource("BaseColorMetalness",  GraphicsFormat.R8G8B8A8_UNorm,          srv);
            Unfiltered_Penumbra    = new NriTextureResource("Unfiltered_Penumbra",    GraphicsFormat.R16_SFloat,              srv);
            Unfiltered_Diff        = new NriTextureResource("Unfiltered_Diff",        GraphicsFormat.R16G16B16A16_SFloat,     srv);
            Unfiltered_Spec        = new NriTextureResource("Unfiltered_Spec",        GraphicsFormat.R16G16B16A16_SFloat,     srv);
            Unfiltered_Translucency= new NriTextureResource("Unfiltered_Translucency",GraphicsFormat.R8G8B8A8_UNorm,          srv);

            Shadow     = new NriTextureResource("Shadow",     GraphicsFormat.R8G8B8A8_UNorm,          uav);
            Diff       = new NriTextureResource("Diff",       GraphicsFormat.R16G16B16A16_SFloat,     uav);
            Spec       = new NriTextureResource("Spec",       GraphicsFormat.R16G16B16A16_SFloat,     uav);
            Validation = new NriTextureResource("Validation", GraphicsFormat.R8G8B8A8_UNorm,          uav);

            Gradient_StoredPing = new NriTextureResource("Gradient_StoredPing", GraphicsFormat.R16G16B16A16_SFloat, uav);
            Gradient_StoredPong = new NriTextureResource("Gradient_StoredPong", GraphicsFormat.R16G16B16A16_SFloat, uav);
            Gradient_Ping       = new NriTextureResource("Gradient_Ping",       GraphicsFormat.R16G16B16A16_SFloat, uav);
            Gradient_Pong       = new NriTextureResource("Gradient_Pong",       GraphicsFormat.R16G16B16A16_SFloat, uav);

            DirectLighting         = new NriTextureResource("DirectLighting",         GraphicsFormat.B10G11R11_UFloatPack32,  uav);
            Composed               = new NriTextureResource("Composed",               GraphicsFormat.R16G16B16A16_SFloat,     uav);
            DlssOutput             = new NriTextureResource("DlssOutput",             GraphicsFormat.R16G16B16A16_SFloat,     uav);
            RrGuideDiffAlbedo      = new NriTextureResource("RrGuideDiffAlbedo",      GraphicsFormat.A2B10G10R10_UNormPack32, uav);
            RrGuideSpecAlbedo      = new NriTextureResource("RrGuideSpecAlbedo",      GraphicsFormat.A2B10G10R10_UNormPack32, uav);
            RrGuideSpecHitDistance = new NriTextureResource("RrGuideSpecHitDistance", GraphicsFormat.R16_SFloat,              uav);
            RrGuideNormalRoughness = new NriTextureResource("RrGuideNormalRoughness", GraphicsFormat.R16G16B16A16_SFloat,     uav);

            TaaHistory     = new NriTextureResource("TaaHistory",     GraphicsFormat.R16G16B16A16_SFloat,     uav);
            TaaHistoryPrev = new NriTextureResource("TaaHistoryPrev", GraphicsFormat.R16G16B16A16_SFloat,     uav);
            PsrThroughput  = new NriTextureResource("PsrThroughput",  GraphicsFormat.A2B10G10R10_UNormPack32, uav);
            Final          = new NriTextureResource("Final",          GraphicsFormat.R16G16B16A16_SFloat,     uav);
            PreFinal       = new NriTextureResource("PreFinal",       GraphicsFormat.R16G16B16A16_SFloat,     uav);
            DirectEmission = new NriTextureResource("DirectEmission", GraphicsFormat.B10G11R11_UFloatPack32,  uav);
            ComposedDiff   = new NriTextureResource("ComposedDiff",   GraphicsFormat.B10G11R11_UFloatPack32,  uav);
            ComposedSpecViewZ = new NriTextureResource("ComposedSpecViewZ", GraphicsFormat.R16G16B16A16_SFloat, uav);
        }

        public static int2 GetUpscaledResolution(int2 outputRes, UpscalerMode mode)
        {
            float scale = mode switch
            {
                UpscalerMode.NATIVE            => 1.0f,
                UpscalerMode.ULTRA_QUALITY     => 1.3f,
                UpscalerMode.QUALITY           => 1.5f,
                UpscalerMode.BALANCED          => 1.7f,
                UpscalerMode.PERFORMANCE       => 2.0f,
                UpscalerMode.ULTRA_PERFORMANCE => 3.0f,
                _                              => 1.0f
            };
            return new int2((int)(outputRes.x / scale + 0.5f), (int)(outputRes.y / scale + 0.5f));
        }

        /// <summary>
        /// Allocates (or reallocates) all textures (except gradient textures) at the correct resolution.
        /// Gradient textures are sized by sharcDims — call <see cref="EnsureSharcGradientResources"/> separately.
        /// Returns true when resources were (re)allocated — callers must re-snapshot NRD resources.
        /// </summary>
        public bool EnsureResources(int2 outputResolution, UpscalerMode mode)
        {
            bool invalid = !Viewz.IsCreated;
            int2 target  = GetUpscaledResolution(outputResolution, mode);

            if (!invalid && target.x == renderResolution.x && target.y == renderResolution.y)
                return false;

            renderResolution = target;

            foreach (var res in RenderResolutionResources())
                res.Allocate(renderResolution);

            DlssOutput.Allocate(outputResolution);
            PreFinal.Allocate(outputResolution);
            Final.Allocate(outputResolution);

            return true;
        }

        /// <summary>
        /// Allocates (or reallocates) the four SHARC gradient textures at SHARC resolution.
        /// Call after computing sharcDims each frame; is a no-op when already correct.
        /// </summary>
        public void EnsureSharcGradientResources(int2 sharcDims)
        {
            AllocIfNeeded(Gradient_StoredPing, sharcDims);
            AllocIfNeeded(Gradient_StoredPong, sharcDims);
            AllocIfNeeded(Gradient_Ping,       sharcDims);
            AllocIfNeeded(Gradient_Pong,       sharcDims);
        }

        private static void AllocIfNeeded(NriTextureResource res, int2 dims)
        {
            var rt = res.Handle?.rt;
            if (rt != null && rt.width == dims.x && rt.height == dims.y) return;
            res.Allocate(dims);
        }

        private NriTextureResource[] RenderResolutionResources() => new[]
        {
            Viewz, MV, NormalRoughness, BaseColorMetalness,
            Unfiltered_Penumbra, Unfiltered_Diff, Unfiltered_Spec, Unfiltered_Translucency,
            Shadow, Diff, Spec, Validation,
            DirectLighting, Composed,
            RrGuideDiffAlbedo, RrGuideSpecAlbedo, RrGuideSpecHitDistance, RrGuideNormalRoughness,
            TaaHistory, TaaHistoryPrev, PsrThroughput,
            DirectEmission, ComposedDiff, ComposedSpecViewZ,
        };

        public void Dispose()
        {
            var all = AllResources();
            foreach (var res in all)
            {
                if (!res.IsCreated) continue;
                var h = res.Handle;
                if (h != null && (h.externalTexture != null || h.rt != null))
                {
                    AsyncGPUReadback.Request(h).WaitForCompletion();
                    break;
                }
            }
            foreach (var res in all) res.Release();
        }

        private NriTextureResource[] AllResources() => new[]
        {
            Viewz, MV, NormalRoughness, BaseColorMetalness,
            Unfiltered_Penumbra, Unfiltered_Diff, Unfiltered_Spec, Unfiltered_Translucency,
            Shadow, Diff, Spec, Validation,
            Gradient_StoredPing, Gradient_StoredPong, Gradient_Ping, Gradient_Pong,
            DirectLighting, Composed, DlssOutput,
            RrGuideDiffAlbedo, RrGuideSpecAlbedo, RrGuideSpecHitDistance, RrGuideNormalRoughness,
            TaaHistory, TaaHistoryPrev, PsrThroughput,
            Final, PreFinal, DirectEmission, ComposedDiff, ComposedSpecViewZ,
        };
    }
}
