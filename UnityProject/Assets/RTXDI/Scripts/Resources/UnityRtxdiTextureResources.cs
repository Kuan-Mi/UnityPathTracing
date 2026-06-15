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
    public class UnityRtxdiTextureResources : IDisposable
    {
        // ── Shared (UAV) ────────────────────────────────────────────────────
        public NriTextureResource DirectLighting;
        public NriTextureResource DlssOutput;             // output resolution
        public NriTextureResource RrGuideDiffAlbedo;
        public NriTextureResource RrGuideSpecAlbedo;
        public NriTextureResource RrGuideSpecHitDistance;
        public NriTextureResource RrGuideNormalRoughness;

        // ── RTXDI GBuffer ping-pong (UAV) ────────────────────────────────────
        public NriTextureResource RtxdiViewDepth;
        public NriTextureResource RtxdiPrevViewDepth;

        public NriTextureResource RtxdiDiffuseAlbedo;
        public NriTextureResource RtxdiPrevDiffuseAlbedo;
        public NriTextureResource RtxdiSpecularRough;
        public NriTextureResource RtxdiPrevSpecularRough;
        public NriTextureResource RtxdiNormals;
        public NriTextureResource RtxdiPrevNormals;
        public NriTextureResource RtxdiGeoNormals;
        public NriTextureResource RtxdiPrevGeoNormals;
        public NriTextureResource RtxdiEmissive;
        public NriTextureResource RtxdiMotionVectors;

        // ── RTXDI lighting outputs (UAV) ─────────────────────────────────────
        public NriTextureResource RtxdiDiffuseLighting;
        public NriTextureResource RtxdiSpecularLighting;
        
        
        // ── Non-NRI gradient Texture2DArray (2 slices, FilterGradientsPass) ──
        private RTHandle _gradientArray;
        public IntPtr GradientArrayPtr => _gradientArray?.rt != null ? _gradientArray.rt.GetNativeTexturePtr() : IntPtr.Zero;
        
        public int2 renderResolution { get; private set; }
        
        public UnityRtxdiTextureResources()
        {
            var uav = new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE_STORAGE, layout = Layout.SHADER_RESOURCE_STORAGE, stageBits = 1 << 10 };
        
            DirectLighting         = new NriTextureResource("DirectLighting",         GraphicsFormat.R16G16B16A16_SFloat,     uav);
            DlssOutput             = new NriTextureResource("DlssOutput",             GraphicsFormat.R16G16B16A16_SFloat,     uav);
            RrGuideDiffAlbedo      = new NriTextureResource("RrGuideDiffAlbedo",      GraphicsFormat.A2B10G10R10_UNormPack32, uav);
            RrGuideSpecAlbedo      = new NriTextureResource("RrGuideSpecAlbedo",      GraphicsFormat.A2B10G10R10_UNormPack32, uav);
            RrGuideSpecHitDistance = new NriTextureResource("RrGuideSpecHitDistance", GraphicsFormat.R16_SFloat,              uav);
            RrGuideNormalRoughness = new NriTextureResource("RrGuideNormalRoughness", GraphicsFormat.R16G16B16A16_SFloat,     uav);
        
            RtxdiViewDepth         = new NriTextureResource("RtxdiViewDepth",         GraphicsFormat.R32_SFloat,  uav);
            RtxdiPrevViewDepth     = new NriTextureResource("RtxdiPrevViewDepth",     GraphicsFormat.R32_SFloat,  uav);
            RtxdiDiffuseAlbedo     = new NriTextureResource("RtxdiDiffuseAlbedo",     GraphicsFormat.R32_UInt,    uav);
            RtxdiPrevDiffuseAlbedo = new NriTextureResource("RtxdiPrevDiffuseAlbedo", GraphicsFormat.R32_UInt,    uav);
            RtxdiSpecularRough     = new NriTextureResource("RtxdiSpecularRough",     GraphicsFormat.R32_UInt,    uav);
            RtxdiPrevSpecularRough = new NriTextureResource("RtxdiPrevSpecularRough", GraphicsFormat.R32_UInt,    uav);
            RtxdiNormals           = new NriTextureResource("RtxdiNormals",           GraphicsFormat.R32_UInt,    uav);
            RtxdiPrevNormals       = new NriTextureResource("RtxdiPrevNormals",       GraphicsFormat.R32_UInt,    uav);
            RtxdiGeoNormals        = new NriTextureResource("RtxdiGeoNormals",        GraphicsFormat.R32_UInt,    uav);
            RtxdiPrevGeoNormals    = new NriTextureResource("RtxdiPrevGeoNormals",    GraphicsFormat.R32_UInt,    uav);
            RtxdiEmissive          = new NriTextureResource("RtxdiEmissive",          GraphicsFormat.R16G16B16A16_SFloat, uav);
            RtxdiMotionVectors     = new NriTextureResource("RtxdiMotionVectors",     GraphicsFormat.R16G16B16A16_SFloat, uav);
        
            RtxdiDiffuseLighting         = new NriTextureResource("RtxdiDiffuseLighting",         GraphicsFormat.R16G16B16A16_SFloat, uav);
            RtxdiSpecularLighting        = new NriTextureResource("RtxdiSpecularLighting",        GraphicsFormat.R16G16B16A16_SFloat, uav);
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
        /// Returns true when resources were (re)allocated.
        /// </summary>
        public bool EnsureResources(int2 outputResolution, UpscalerMode mode)
        {
            int2 target  = GetUpscaledResolution(outputResolution, mode);
        
            if (target.x == renderResolution.x && target.y == renderResolution.y)
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
            RrGuideDiffAlbedo, RrGuideSpecAlbedo, RrGuideSpecHitDistance, RrGuideNormalRoughness,
            RtxdiViewDepth, RtxdiPrevViewDepth, RtxdiDiffuseAlbedo, RtxdiPrevDiffuseAlbedo,
            RtxdiSpecularRough, RtxdiPrevSpecularRough,
            RtxdiNormals, RtxdiPrevNormals,
            RtxdiGeoNormals, RtxdiPrevGeoNormals,
            RtxdiEmissive, RtxdiMotionVectors,
            DirectLighting,
            RtxdiDiffuseLighting, RtxdiSpecularLighting,
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
            DirectLighting, DlssOutput,
            RrGuideDiffAlbedo, RrGuideSpecAlbedo, RrGuideSpecHitDistance, RrGuideNormalRoughness,
            RtxdiViewDepth, RtxdiPrevViewDepth, RtxdiDiffuseAlbedo, RtxdiPrevDiffuseAlbedo,
            RtxdiSpecularRough, RtxdiPrevSpecularRough,
            RtxdiNormals, RtxdiPrevNormals,
            RtxdiGeoNormals, RtxdiPrevGeoNormals,
            RtxdiEmissive, RtxdiMotionVectors,
            RtxdiDiffuseLighting, RtxdiSpecularLighting,
        };
    }
}
