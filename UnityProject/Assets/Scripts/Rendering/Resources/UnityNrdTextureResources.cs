using System;
using Nri;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace PathTracing
{
    /// <summary>
    /// Owns all per-camera NRI render textures for <see cref="UnityNrdFeature"/>.
    /// Fields are named after their purpose; access them directly instead of
    /// going through an enum-keyed dictionary.
    /// </summary>
    public class UnityNrdTextureResources : IDisposable
    {
        // ── NRD standard inputs (SRV) ──────────────────────────────────────
        public NriTextureResource MV;
        public NriTextureResource Viewz;
        public NriTextureResource NormalRoughness;
        public NriTextureResource BaseColorMetalness;
        public NriTextureResource GeoNormal;
        public NriTextureResource Unfiltered_Penumbra;
        public NriTextureResource Unfiltered_Diff;
        public NriTextureResource Unfiltered_Spec;

        // ── NRD outputs (UAV) ───────────────────────────────────────────────
        public NriTextureResource Shadow;
        public NriTextureResource Diff;
        public NriTextureResource Spec;
        public NriTextureResource Validation;

        // ── DLSS / RR interop (UAV) ─────────────────────────────────────────
        public NriTextureResource DirectLighting;
        public NriTextureResource Composed;
        public NriTextureResource DlssOutput;        // output resolution
        public NriTextureResource RrGuideDiffAlbedo;
        public NriTextureResource RrGuideSpecAlbedo;
        public NriTextureResource RrGuideSpecHitDistance;
        public NriTextureResource RrGuideNormalRoughness;

        // ── Cross-frame / per-frame (UAV) ───────────────────────────────────
        public NriTextureResource TaaHistory;
        public NriTextureResource TaaHistoryPrev;
        public NriTextureResource PsrThroughput;
        public NriTextureResource PrevViewZ;
        public NriTextureResource PrevNormalRoughness;
        public NriTextureResource PrevBaseColorMetalness;
        public NriTextureResource PrevGeoNormal;
        public NriTextureResource Final;             // output resolution
        public NriTextureResource PreFinal;          // output resolution
        public NriTextureResource DirectEmission;
        public NriTextureResource ComposedDiff;
        public NriTextureResource ComposedSpecViewZ;

        public int2 renderResolution { get; private set; }

        public UnityNrdTextureResources()
        {
            var srv = new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE,         layout = Layout.SHADER_RESOURCE,         stageBits = 1 << 7  };
            var uav = new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE_STORAGE, layout = Layout.SHADER_RESOURCE_STORAGE, stageBits = 1 << 10 };

            MV                  = new NriTextureResource("MV",                  GraphicsFormat.R16G16B16A16_SFloat,      srv);
            Viewz               = new NriTextureResource("Viewz",               GraphicsFormat.R32_SFloat,               srv);
            NormalRoughness     = new NriTextureResource("NormalRoughness",     GraphicsFormat.A2B10G10R10_UNormPack32,  srv);
            BaseColorMetalness  = new NriTextureResource("BaseColorMetalness",  GraphicsFormat.R8G8B8A8_UNorm,           srv);
            GeoNormal           = new NriTextureResource("GeoNormal",           GraphicsFormat.R32_UInt,                 srv);
            Unfiltered_Penumbra = new NriTextureResource("Unfiltered_Penumbra", GraphicsFormat.R16_SFloat,               srv);
            Unfiltered_Diff     = new NriTextureResource("Unfiltered_Diff",     GraphicsFormat.R16G16B16A16_SFloat,      srv);
            Unfiltered_Spec     = new NriTextureResource("Unfiltered_Spec",     GraphicsFormat.R16G16B16A16_SFloat,      srv);

            Shadow     = new NriTextureResource("Shadow",     GraphicsFormat.R16_SFloat,              uav);
            Diff       = new NriTextureResource("Diff",       GraphicsFormat.R16G16B16A16_SFloat,     uav);
            Spec       = new NriTextureResource("Spec",       GraphicsFormat.R16G16B16A16_SFloat,     uav);
            Validation = new NriTextureResource("Validation", GraphicsFormat.R8G8B8A8_UNorm,          uav);

            DirectLighting         = new NriTextureResource("DirectLighting",         GraphicsFormat.R16G16B16A16_SFloat,     uav);
            Composed               = new NriTextureResource("Composed",               GraphicsFormat.R16G16B16A16_SFloat,     uav);
            DlssOutput             = new NriTextureResource("DlssOutput",             GraphicsFormat.R16G16B16A16_SFloat,     uav);
            RrGuideDiffAlbedo      = new NriTextureResource("RrGuideDiffAlbedo",      GraphicsFormat.A2B10G10R10_UNormPack32, uav);
            RrGuideSpecAlbedo      = new NriTextureResource("RrGuideSpecAlbedo",      GraphicsFormat.A2B10G10R10_UNormPack32, uav);
            RrGuideSpecHitDistance = new NriTextureResource("RrGuideSpecHitDistance", GraphicsFormat.R16_SFloat,              uav);
            RrGuideNormalRoughness = new NriTextureResource("RrGuideNormalRoughness", GraphicsFormat.R16G16B16A16_SFloat,     uav);

            TaaHistory            = new NriTextureResource("TaaHistory",            GraphicsFormat.R16G16B16A16_SFloat,     uav);
            TaaHistoryPrev        = new NriTextureResource("TaaHistoryPrev",        GraphicsFormat.R16G16B16A16_SFloat,     uav);
            PsrThroughput         = new NriTextureResource("PsrThroughput",         GraphicsFormat.R16G16B16A16_SFloat,     uav);
            PrevViewZ             = new NriTextureResource("PrevViewZ",             GraphicsFormat.R32_SFloat,              uav);
            PrevNormalRoughness   = new NriTextureResource("PrevNormalRoughness",   GraphicsFormat.A2B10G10R10_UNormPack32, uav);
            PrevBaseColorMetalness= new NriTextureResource("PrevBaseColorMetalness",GraphicsFormat.R8G8B8A8_UNorm,          uav);
            PrevGeoNormal         = new NriTextureResource("PrevGeoNormal",         GraphicsFormat.R32_UInt,                uav);
            Final                 = new NriTextureResource("Final",                 GraphicsFormat.R16G16B16A16_SFloat,     uav);
            PreFinal              = new NriTextureResource("PreFinal",              GraphicsFormat.R16G16B16A16_SFloat,     uav);
            DirectEmission        = new NriTextureResource("DirectEmission",        GraphicsFormat.B10G11R11_UFloatPack32,  uav);
            ComposedDiff          = new NriTextureResource("ComposedDiff",          GraphicsFormat.R16G16B16A16_SFloat,     uav);
            ComposedSpecViewZ     = new NriTextureResource("ComposedSpecViewZ",     GraphicsFormat.R16G16B16A16_SFloat,     uav);
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
        /// Allocates (or reallocates) all textures at the correct resolution.
        /// Returns true when resources were (re)allocated — callers must re-snapshot NRD resources.
        /// </summary>
        public bool EnsureResources(int2 outputResolution, UpscalerMode mode)
        {
            bool invalid = !MV.IsCreated;
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

        private NriTextureResource[] RenderResolutionResources() => new[]
        {
            MV, Viewz, NormalRoughness, BaseColorMetalness, GeoNormal,
            Unfiltered_Penumbra, Unfiltered_Diff, Unfiltered_Spec,
            Shadow, Diff, Spec, Validation,
            DirectLighting, Composed,
            RrGuideDiffAlbedo, RrGuideSpecAlbedo, RrGuideSpecHitDistance, RrGuideNormalRoughness,
            TaaHistory, TaaHistoryPrev, PsrThroughput,
            PrevViewZ, PrevNormalRoughness, PrevBaseColorMetalness, PrevGeoNormal,
            DirectEmission, ComposedDiff, ComposedSpecViewZ,
        };

        public void Dispose()
        {
            var all = AllResources();
            // Wait for GPU on the first live resource before releasing any
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
            MV, Viewz, NormalRoughness, BaseColorMetalness, GeoNormal,
            Unfiltered_Penumbra, Unfiltered_Diff, Unfiltered_Spec,
            Shadow, Diff, Spec, Validation,
            DirectLighting, Composed, DlssOutput,
            RrGuideDiffAlbedo, RrGuideSpecAlbedo, RrGuideSpecHitDistance, RrGuideNormalRoughness,
            TaaHistory, TaaHistoryPrev, PsrThroughput,
            PrevViewZ, PrevNormalRoughness, PrevBaseColorMetalness, PrevGeoNormal,
            Final, PreFinal, DirectEmission, ComposedDiff, ComposedSpecViewZ,
        };
    }
}
