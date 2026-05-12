using System;
using Nri;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace PathTracing
{
    /// <summary>
    /// Owns all per-camera NRI render textures for <see cref="UnityRtxdiFeature"/> and
    /// <see cref="NativeRtxdiFeature"/>.
    /// Also owns the non-NRI gradient Texture2DArray used by FilterGradientsPass.
    /// </summary>
    public class NativeRtxdiTextureResources : IDisposable
    {
        public NriTextureResource DeviceDepth;
        public NriTextureResource Depth;
        public NriTextureResource PrevDepth;
        public NriTextureResource GBufferDiffuseAlbedo;
        public NriTextureResource GBufferSpecularRough;
        public NriTextureResource GBufferNormals;
        public NriTextureResource GBufferGeoNormals;
        public NriTextureResource GBufferEmissive;
        public NriTextureResource PrevGBufferDiffuseAlbedo;
        public NriTextureResource PrevGBufferSpecularRough;
        public NriTextureResource PrevGBufferNormals;
        public NriTextureResource PrevGBufferGeoNormals;
        public NriTextureResource MotionVectors;
        public NriTextureResource NormalRoughness; // for NRD

        // public NriTextureResource PSRMotionVectors;
        // public NriTextureResource PSRNormalRoughness;
        // public NriTextureResource PSRDepth;
        // public NriTextureResource PSRHitT;
        // // PSR material/direction buffers (packed: R11G11B10 for albedo/F0, oct for view/light dir)
        // public NriTextureResource PSRDiffuseAlbedo;
        // public NriTextureResource PSRSpecularF0;
        // public NriTextureResource PSRLightDir;
        //

        public NriTextureResource HdrColor;
        // public NriTextureResource LdrColor;
        public NriTextureResource DiffuseLighting;
        public NriTextureResource SpecularLighting;
        public NriTextureResource DenoisedDiffuseLighting;
        public NriTextureResource DenoisedSpecularLighting;

        public NriTextureResource NrdValidation;

        // public NriTextureResource      TaaFeedback1;
        // public NriTextureResource      TaaFeedback2;
        // public NriTextureResource      ResolvedColor;
        // public NriTextureResource      AccumulatedColor;
        public NriTextureResource RestirLuminance;
        public NriTextureResource PrevRestirLuminance;

        public NriTextureResource DirectLightingRaw;
        public NriTextureResource IndirectLightingRaw;

        // public NriTextureResource PTSampleIDTexture;
        // public NriTextureResource PTDuplicationMap;

        // ── Non-NRI gradient Texture2DArray (2 slices, FilterGradientsPass) ──
        private RTHandle           _gradientArray;
        public  NriTextureResource TemporalSamplePos;
        public  NriTextureResource DiffuseConfidence;
        public  NriTextureResource SpecularConfidence;
        public  NriTextureResource PrevDiffuseConfidence;
        public  NriTextureResource PrevSpecularConfidence;

        // public NriTextureResource DebugColor;
        // public NriTextureResource ReferenceColor;


        // ── Shared (UAV) ────────────────────────────────────────────────────

        public NriTextureResource DlssOutput; // output resolution
        public NriTextureResource RrGuideDiffAlbedo;
        public NriTextureResource RrGuideSpecAlbedo;
        public NriTextureResource RrGuideSpecHitDistance;
        public NriTextureResource RrGuideNormalRoughness;

        public IntPtr GradientArrayPtr => _gradientArray?.rt != null ? _gradientArray.rt.GetNativeTexturePtr() : IntPtr.Zero;

        public int2 renderResolution { get; private set; }

        public NativeRtxdiTextureResources()
        {
            var uav = new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE_STORAGE, layout = Layout.SHADER_RESOURCE_STORAGE, stageBits = 1 << 10 };

            NrdValidation          = new NriTextureResource("Validation", GraphicsFormat.R8G8B8A8_UNorm, uav);
            HdrColor         = new NriTextureResource("DirectLighting", GraphicsFormat.R16G16B16A16_SFloat, uav);
            DlssOutput             = new NriTextureResource("DlssOutput", GraphicsFormat.R16G16B16A16_SFloat, uav);
            RrGuideDiffAlbedo      = new NriTextureResource("RrGuideDiffAlbedo", GraphicsFormat.A2B10G10R10_UNormPack32, uav);
            RrGuideSpecAlbedo      = new NriTextureResource("RrGuideSpecAlbedo", GraphicsFormat.A2B10G10R10_UNormPack32, uav);
            RrGuideSpecHitDistance = new NriTextureResource("RrGuideSpecHitDistance", GraphicsFormat.R16_SFloat, uav);
            RrGuideNormalRoughness = new NriTextureResource("RrGuideNormalRoughness", GraphicsFormat.R16G16B16A16_SFloat, uav);

            Depth                    = new NriTextureResource("Depth", GraphicsFormat.R32_SFloat, uav);
            PrevDepth                = new NriTextureResource("PrevDepth", GraphicsFormat.R32_SFloat, uav);
            DeviceDepth              = new NriTextureResource("DeviceDepth", GraphicsFormat.R32_SFloat, uav);
            GBufferDiffuseAlbedo     = new NriTextureResource("DiffuseAlbedo", GraphicsFormat.R32_UInt, uav);
            PrevGBufferDiffuseAlbedo = new NriTextureResource("PrevDiffuseAlbedo", GraphicsFormat.R32_UInt, uav);
            GBufferSpecularRough     = new NriTextureResource("SpecularRough", GraphicsFormat.R32_UInt, uav);
            PrevGBufferSpecularRough = new NriTextureResource("PrevSpecularRough", GraphicsFormat.R32_UInt, uav);
            GBufferNormals           = new NriTextureResource("Normals", GraphicsFormat.R32_UInt, uav);
            PrevGBufferNormals       = new NriTextureResource("PrevNormals", GraphicsFormat.R32_UInt, uav);
            GBufferGeoNormals        = new NriTextureResource("GeoNormals", GraphicsFormat.R32_UInt, uav);
            PrevGBufferGeoNormals    = new NriTextureResource("PrevGeoNormals", GraphicsFormat.R32_UInt, uav);
            GBufferEmissive          = new NriTextureResource("Emissive", GraphicsFormat.R16G16B16A16_SFloat, uav);
            MotionVectors            = new NriTextureResource("MotionVectors", GraphicsFormat.R16G16B16A16_SFloat, uav);

            DiffuseLighting     = new NriTextureResource("DiffuseLighting", GraphicsFormat.R16G16B16A16_SFloat, uav);
            SpecularLighting    = new NriTextureResource("SpecularLighting", GraphicsFormat.R16G16B16A16_SFloat, uav);
            TemporalSamplePos   = new NriTextureResource("TemporalSamplePos", GraphicsFormat.R16G16_SInt, uav);
            RestirLuminance     = new NriTextureResource("RestirLuminance", GraphicsFormat.R16G16_SFloat, uav);
            PrevRestirLuminance = new NriTextureResource("PrevRestirLuminance", GraphicsFormat.R16G16_SFloat, uav);
            DirectLightingRaw   = new NriTextureResource("DirectLightingRaw", GraphicsFormat.R16G16B16A16_SFloat, uav);
            IndirectLightingRaw = new NriTextureResource("IndirectLightingRaw", GraphicsFormat.R16G16B16A16_SFloat, uav);
            NormalRoughness     = new NriTextureResource("DenoiserNormalRoughness", GraphicsFormat.R8G8B8A8_UNorm, uav);

            DenoisedDiffuseLighting  = new NriTextureResource("DenoisedDiffuseLighting", GraphicsFormat.R16G16B16A16_SFloat, uav);
            DenoisedSpecularLighting = new NriTextureResource("DenoisedSpecularLighting", GraphicsFormat.R16G16B16A16_SFloat, uav);

            DiffuseConfidence      = new NriTextureResource("DiffuseConfidence", GraphicsFormat.R8_UNorm, uav);
            PrevDiffuseConfidence  = new NriTextureResource("PrevDiffuseConfidence", GraphicsFormat.R8_UNorm, uav);
            SpecularConfidence     = new NriTextureResource("SpecularConfidence", GraphicsFormat.R8_UNorm, uav);
            PrevSpecularConfidence = new NriTextureResource("PrevSpecularConfidence", GraphicsFormat.R8_UNorm, uav);
        }

        public static int2 GetUpscaledResolution(int2 outputRes, UpscalerMode mode)
        {
            float scale = mode switch
            {
                UpscalerMode.NATIVE => 1.0f,
                UpscalerMode.ULTRA_QUALITY => 1.3f,
                UpscalerMode.QUALITY => 1.5f,
                UpscalerMode.BALANCED => 1.7f,
                UpscalerMode.PERFORMANCE => 2.0f,
                UpscalerMode.ULTRA_PERFORMANCE => 3.0f,
                _ => 1.0f
            };
            return new int2((int)(outputRes.x / scale + 0.5f), (int)(outputRes.y / scale + 0.5f));
        }

        /// <summary>
        /// Allocates (or reallocates) all textures at the correct resolution.
        /// Returns true when resources were (re)allocated.
        /// </summary>
        public bool EnsureResources(int2 outputResolution, UpscalerMode mode)
        {
            bool invalid = !NrdValidation.IsCreated;
            int2 target  = GetUpscaledResolution(outputResolution, mode);

            if (!invalid && target.x == renderResolution.x && target.y == renderResolution.y)
                return false;

            renderResolution = target;

            foreach (var res in RenderResolutionResources())
                res.Allocate(renderResolution);

            DlssOutput.Allocate(outputResolution);

            return true;
        }

        /// <summary>
        /// Allocates (or reallocates) the gradient Texture2DArray (2 slices) used by FilterGradientsPass.
        /// Call after computing gradDims = ceil(renderRes / RTXDI_GRAD_FACTOR).
        /// </summary>
        public void EnsureGradientArray(int2 gradDims)
        {
            var rt = _gradientArray?.rt;
            if (rt != null && rt.width == gradDims.x && rt.height == gradDims.y) return;

            _gradientArray?.Release();
            var desc = new RenderTextureDescriptor(gradDims.x, gradDims.y, GraphicsFormat.R16G16B16A16_SFloat, 0)
            {
                dimension         = TextureDimension.Tex2DArray,
                volumeDepth       = 2,
                enableRandomWrite = true,
            };
            _gradientArray = RTHandles.Alloc(desc);
        }

        private NriTextureResource[] RenderResolutionResources() => new[]
        {
            NrdValidation,
            RrGuideDiffAlbedo, RrGuideSpecAlbedo, RrGuideSpecHitDistance, RrGuideNormalRoughness,
            Depth, PrevDepth, DeviceDepth,
            GBufferDiffuseAlbedo, PrevGBufferDiffuseAlbedo,
            GBufferSpecularRough, PrevGBufferSpecularRough,
            GBufferNormals, PrevGBufferNormals,
            GBufferGeoNormals, PrevGBufferGeoNormals,
            GBufferEmissive, MotionVectors,
            HdrColor,
            DiffuseLighting, SpecularLighting,
            TemporalSamplePos,
            RestirLuminance, PrevRestirLuminance,
            DirectLightingRaw, IndirectLightingRaw,
            NormalRoughness,
            DenoisedDiffuseLighting, DenoisedSpecularLighting,
            DiffuseConfidence, PrevDiffuseConfidence,
            SpecularConfidence, PrevSpecularConfidence,
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

            _gradientArray?.Release();
            _gradientArray = null;
        }

        private NriTextureResource[] AllResources() => new[]
        {
            NrdValidation, HdrColor, DlssOutput,
            RrGuideDiffAlbedo, RrGuideSpecAlbedo, RrGuideSpecHitDistance, RrGuideNormalRoughness,
            Depth, PrevDepth, DeviceDepth,
            GBufferDiffuseAlbedo, PrevGBufferDiffuseAlbedo,
            GBufferSpecularRough, PrevGBufferSpecularRough,
            GBufferNormals, PrevGBufferNormals,
            GBufferGeoNormals, PrevGBufferGeoNormals,
            GBufferEmissive, MotionVectors,
            DiffuseLighting, SpecularLighting,
            TemporalSamplePos,
            RestirLuminance, PrevRestirLuminance,
            DirectLightingRaw, IndirectLightingRaw,
            NormalRoughness,
            DenoisedDiffuseLighting, DenoisedSpecularLighting,
            DiffuseConfidence, PrevDiffuseConfidence,
            SpecularConfidence, PrevSpecularConfidence,
        };
    }
}