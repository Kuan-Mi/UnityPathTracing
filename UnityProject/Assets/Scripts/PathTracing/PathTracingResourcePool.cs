using System.Collections.Generic;
using Nri;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace PathTracing
{
    /// <summary>
    /// Central owner of all per-camera render textures used by the path tracing pipeline.
    ///
    /// Resources are split into two buckets:
    ///   _nriResources  – All textures that require a native NRI/D3D12 pointer
    ///                    (NRD standard I/O + DLSS/RR interop textures).
    ///   _rtResources   – Cross-frame RTHandle-only resources (TAA history, prev GBuffer, etc.)
    ///                    that are only bound as Unity render targets.
    ///
    /// Call InitPathTracingResources() from PathTracingFeature, or
    /// InitRtxdiResources() from RtxdiFeature, to register only the textures each pipeline needs.
    /// </summary>
    public class PathTracingResourcePool : System.IDisposable
    {

        private readonly Dictionary<RenderResourceType, NriTextureResource> _nriResources = new();
        private readonly Dictionary<RenderResourceType, RTHandle>           _rtResources  = new();

        // Deferred RT-only allocations registered by Init* methods and executed in EnsureResources.
        private readonly List<(RenderResourceType type, GraphicsFormat format, bool srgb)> _rtResourceDefs = new();

        public int2 renderResolution { get; private set; }



        // ── Resource initialisation ─────────────────────────────────────────────

        /// <summary>
        /// Registers all textures required by <see cref="PathTracingFeature"/>:
        /// NRD I/O, DLSS/RR interop, TAA history, and prev-frame GBuffer.
        /// </summary>
        public void InitPathTracingResources()
        {
            var srvState = new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE,         layout = Layout.SHADER_RESOURCE,         stageBits = 1 << 7  };
            var uavState = new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE_STORAGE, layout = Layout.SHADER_RESOURCE_STORAGE, stageBits = 1 << 10 };

            // ── NRD standard I/O ────────────────────────────────────────────────
            _nriResources[RenderResourceType.MV]                    = new NriTextureResource(RenderResourceType.MV,                    GraphicsFormat.R16G16B16A16_SFloat,    srvState);
            _nriResources[RenderResourceType.Viewz]                 = new NriTextureResource(RenderResourceType.Viewz,                 GraphicsFormat.R32_SFloat,             srvState);
            _nriResources[RenderResourceType.NormalRoughness]       = new NriTextureResource(RenderResourceType.NormalRoughness,       GraphicsFormat.A2B10G10R10_UNormPack32, srvState);
            _nriResources[RenderResourceType.BasecolorMetalness]    = new NriTextureResource(RenderResourceType.BasecolorMetalness,    GraphicsFormat.B8G8R8A8_SRGB,          srvState, true);
            _nriResources[RenderResourceType.GeoNormal]             = new NriTextureResource(RenderResourceType.GeoNormal,             GraphicsFormat.R32_UInt,               srvState);
            _nriResources[RenderResourceType.Penumbra]              = new NriTextureResource(RenderResourceType.Penumbra,              GraphicsFormat.R16_SFloat,             srvState);
            _nriResources[RenderResourceType.DiffRadianceHitdist]   = new NriTextureResource(RenderResourceType.DiffRadianceHitdist,   GraphicsFormat.R16G16B16A16_SFloat,    srvState);
            _nriResources[RenderResourceType.SpecRadianceHitdist]   = new NriTextureResource(RenderResourceType.SpecRadianceHitdist,   GraphicsFormat.R16G16B16A16_SFloat,    srvState);
            _nriResources[RenderResourceType.OutShadowTranslucency] = new NriTextureResource(RenderResourceType.OutShadowTranslucency, GraphicsFormat.R16_SFloat,             uavState);
            _nriResources[RenderResourceType.OutDiffRadianceHitdist]= new NriTextureResource(RenderResourceType.OutDiffRadianceHitdist,GraphicsFormat.R16G16B16A16_SFloat,    uavState);
            _nriResources[RenderResourceType.OutSpecRadianceHitdist]= new NriTextureResource(RenderResourceType.OutSpecRadianceHitdist,GraphicsFormat.R16G16B16A16_SFloat,    uavState);
            _nriResources[RenderResourceType.Validation]            = new NriTextureResource(RenderResourceType.Validation,            GraphicsFormat.R8G8B8A8_UNorm,         uavState);

            // ── NRI-interop resources (DLSS / composition) ──────────────────────
            _nriResources[RenderResourceType.DirectLighting]        = new NriTextureResource(RenderResourceType.DirectLighting,        GraphicsFormat.R16G16B16A16_SFloat,    uavState);
            _nriResources[RenderResourceType.Composed]              = new NriTextureResource(RenderResourceType.Composed,              GraphicsFormat.R16G16B16A16_SFloat,    uavState);
            _nriResources[RenderResourceType.DlssOutput]            = new NriTextureResource(RenderResourceType.DlssOutput,            GraphicsFormat.R16G16B16A16_SFloat,    uavState);
            _nriResources[RenderResourceType.RrGuideDiffAlbedo]     = new NriTextureResource(RenderResourceType.RrGuideDiffAlbedo,     GraphicsFormat.A2B10G10R10_UNormPack32, uavState);
            _nriResources[RenderResourceType.RrGuideSpecAlbedo]     = new NriTextureResource(RenderResourceType.RrGuideSpecAlbedo,     GraphicsFormat.A2B10G10R10_UNormPack32, uavState);
            _nriResources[RenderResourceType.RrGuideSpecHitDistance]= new NriTextureResource(RenderResourceType.RrGuideSpecHitDistance, GraphicsFormat.R16_SFloat,            uavState);
            _nriResources[RenderResourceType.RrGuideNormalRoughness]= new NriTextureResource(RenderResourceType.RrGuideNormalRoughness, GraphicsFormat.R16G16B16A16_SFloat,   uavState);

            // ── RTHandle-only (cross-frame) ──────────────────────────────────────
            _rtResourceDefs.Add((RenderResourceType.TaaHistory,            GraphicsFormat.R16G16B16A16_SFloat,    false));
            _rtResourceDefs.Add((RenderResourceType.TaaHistoryPrev,        GraphicsFormat.R16G16B16A16_SFloat,    false));
            _rtResourceDefs.Add((RenderResourceType.PsrThroughput,         GraphicsFormat.R16G16B16A16_SFloat,    false));
            _rtResourceDefs.Add((RenderResourceType.PrevViewZ,             GraphicsFormat.R32_SFloat,             false));
            _rtResourceDefs.Add((RenderResourceType.PrevNormalRoughness,   GraphicsFormat.A2B10G10R10_UNormPack32, false));
            _rtResourceDefs.Add((RenderResourceType.PrevBaseColorMetalness,GraphicsFormat.B8G8R8A8_SRGB,          true));
            _rtResourceDefs.Add((RenderResourceType.PrevGeoNormal,         GraphicsFormat.R32_UInt,               false));
        }

        /// <summary>
        /// Registers all textures required by <see cref="RtxdiFeature"/>:
        /// shared NRD I/O subset, DLSS/RR interop, and RTXDI GBuffer ping-pong textures.
        /// </summary>
        public void InitRtxdiResources()
        {
            var srvState = new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE,         layout = Layout.SHADER_RESOURCE,         stageBits = 1 << 7  };
            var uavState = new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE_STORAGE, layout = Layout.SHADER_RESOURCE_STORAGE, stageBits = 1 << 10 };

            // ── NRI-interop resources (DLSS / composition) ──────────────────────
            _nriResources[RenderResourceType.DirectLighting]         = new NriTextureResource(RenderResourceType.DirectLighting,         GraphicsFormat.R16G16B16A16_SFloat,     uavState);
            _nriResources[RenderResourceType.DlssOutput]             = new NriTextureResource(RenderResourceType.DlssOutput,             GraphicsFormat.R16G16B16A16_SFloat,     uavState);
            _nriResources[RenderResourceType.RrGuideDiffAlbedo]      = new NriTextureResource(RenderResourceType.RrGuideDiffAlbedo,      GraphicsFormat.A2B10G10R10_UNormPack32, uavState);
            _nriResources[RenderResourceType.RrGuideSpecAlbedo]      = new NriTextureResource(RenderResourceType.RrGuideSpecAlbedo,      GraphicsFormat.A2B10G10R10_UNormPack32, uavState);
            _nriResources[RenderResourceType.RrGuideSpecHitDistance] = new NriTextureResource(RenderResourceType.RrGuideSpecHitDistance, GraphicsFormat.R16_SFloat,              uavState);
            _nriResources[RenderResourceType.RrGuideNormalRoughness] = new NriTextureResource(RenderResourceType.RrGuideNormalRoughness, GraphicsFormat.R16G16B16A16_SFloat,     uavState);

            // ── RTXDI GBuffer ping-pong textures ────────────────────────────────
            _nriResources[RenderResourceType.RtxdiViewDepth]         = new NriTextureResource(RenderResourceType.RtxdiViewDepth,         GraphicsFormat.R32_SFloat,              uavState);
            _nriResources[RenderResourceType.RtxdiPrevViewDepth]     = new NriTextureResource(RenderResourceType.RtxdiPrevViewDepth,     GraphicsFormat.R32_SFloat,              uavState);
            _nriResources[RenderResourceType.RtxdiDiffuseAlbedo]     = new NriTextureResource(RenderResourceType.RtxdiDiffuseAlbedo,     GraphicsFormat.R32_UInt,                uavState);
            _nriResources[RenderResourceType.RtxdiPrevDiffuseAlbedo] = new NriTextureResource(RenderResourceType.RtxdiPrevDiffuseAlbedo, GraphicsFormat.R32_UInt,                uavState);
            _nriResources[RenderResourceType.RtxdiSpecularRough]     = new NriTextureResource(RenderResourceType.RtxdiSpecularRough,     GraphicsFormat.R32_UInt,                uavState);
            _nriResources[RenderResourceType.RtxdiPrevSpecularRough] = new NriTextureResource(RenderResourceType.RtxdiPrevSpecularRough, GraphicsFormat.R32_UInt,                uavState);
            _nriResources[RenderResourceType.RtxdiNormals]           = new NriTextureResource(RenderResourceType.RtxdiNormals,           GraphicsFormat.R32_UInt,                uavState);
            _nriResources[RenderResourceType.RtxdiPrevNormals]       = new NriTextureResource(RenderResourceType.RtxdiPrevNormals,       GraphicsFormat.R32_UInt,                uavState);
            _nriResources[RenderResourceType.RtxdiGeoNormals]        = new NriTextureResource(RenderResourceType.RtxdiGeoNormals,        GraphicsFormat.R32_UInt,                uavState);
            _nriResources[RenderResourceType.RtxdiPrevGeoNormals]    = new NriTextureResource(RenderResourceType.RtxdiPrevGeoNormals,    GraphicsFormat.R32_UInt,                uavState);
            _nriResources[RenderResourceType.RtxdiEmissive]          = new NriTextureResource(RenderResourceType.RtxdiEmissive,          GraphicsFormat.R16G16B16A16_SFloat,     uavState);
            _nriResources[RenderResourceType.RtxdiMotionVectors]     = new NriTextureResource(RenderResourceType.RtxdiMotionVectors,     GraphicsFormat.R16G16B16A16_SFloat,     uavState);
        }

        // ── Public accessors ────────────────────────────────────────────────────

        /// <summary>Returns the NriTextureResource (RTHandle + NriPtr) for a given resource type.</summary>
        public NriTextureResource GetNriResource(RenderResourceType type) => _nriResources[type];

        /// <summary>Returns the RTHandle for any resource, whether NRI-interop or RTHandle-only.</summary>
        public RTHandle GetRT(RenderResourceType type)
        {
            if (_nriResources.TryGetValue(type, out var nriRes)) return nriRes.Handle;
            return _rtResources[type];
        }

        // ── Resolution/allocation ───────────────────────────────────────────────

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
        /// Ensures all resources are allocated at the correct resolution.
        /// Returns true when resources were (re)allocated — callers must re-snapshot NRD resources.
        /// </summary>
        public bool EnsureResources(int2 outputResolution,UpscalerMode mode)
        {
            bool invalid = false;
            foreach (var res in _nriResources.Values)
                if (res.Handle == null || res.Handle.rt == null)
                {
                    invalid = true;
                    break;
                }

            if (!invalid)
                foreach (var h in _rtResources.Values)
                    if (h == null || h.rt == null)
                    {
                        invalid = true;
                        break;
                    }

            int2 target = GetUpscaledResolution(outputResolution,mode);
            if (!invalid && target.x == renderResolution.x && target.y == renderResolution.y)
                return false;

            renderResolution = target;

            foreach (var kvp in _nriResources)
            {
                int2 res = kvp.Key == RenderResourceType.DlssOutput ? outputResolution : renderResolution;
                kvp.Value.Allocate(res);
            }

            foreach (var (type, format, srgb) in _rtResourceDefs)
                AllocateRT(type, format, renderResolution, srgb);

            return true;
        }

        private void AllocateRT(RenderResourceType type, GraphicsFormat format, int2 resolution, bool srgb = false)
        {
            if (_rtResources.TryGetValue(type, out var existing) && existing != null)
            {
                var oldRt = existing.rt;
                RTHandles.Release(existing);
                if (oldRt != null)
                {
                    if (Application.isPlaying) Object.Destroy(oldRt);
                    else Object.DestroyImmediate(oldRt);
                }
            }

            var desc = new RenderTextureDescriptor(resolution.x, resolution.y, format, 0)
            {
                enableRandomWrite = true,
                useMipMap = false,
                msaaSamples = 1,
                sRGB = srgb
            };
            var rt = new RenderTexture(desc)
            {
                name = type.ToString(),
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };
            rt.Create();
            _rtResources[type] = RTHandles.Alloc(rt);
        }

        public void Dispose()
        {
            // Wait for GPU before releasing any NRI texture
            foreach (var res in _nriResources.Values)
            {
                if (!res.IsCreated) continue;
                var h = res.Handle;
                if (h != null && (h.externalTexture != null || h.rt != null))
                {
                    AsyncGPUReadback.Request(h).WaitForCompletion();
                    break;
                }
            }

            foreach (var res in _nriResources.Values) res.Release();
            _nriResources.Clear();

            foreach (var handle in _rtResources.Values)
            {
                if (handle == null) continue;
                var rt = handle.rt;
                RTHandles.Release(handle);
                if (rt != null)
                {
                    if (Application.isPlaying) Object.Destroy(rt);
                    else Object.DestroyImmediate(rt);
                }
            }

            _rtResources.Clear();
        }
    }
}