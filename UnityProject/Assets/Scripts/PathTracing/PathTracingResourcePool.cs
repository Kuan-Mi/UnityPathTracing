using System.Collections.Generic;
using NRD;
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
    /// </summary>
    public class PathTracingResourcePool : System.IDisposable
    {
        private readonly PathTracingSetting _setting;

        private readonly Dictionary<RenderResourceType, NriTextureResource> _nriResources = new();
        private readonly Dictionary<RenderResourceType, RTHandle> _rtResources = new();

        public int2 renderResolution { get; private set; }

        public PathTracingResourcePool(PathTracingSetting setting)
        {
            _setting = setting;

            var srvState = new Nri.NriResourceState { accessBits = Nri.AccessBits.SHADER_RESOURCE, layout = Nri.Layout.SHADER_RESOURCE, stageBits = 1 << 7 };
            var uavState = new Nri.NriResourceState { accessBits = Nri.AccessBits.SHADER_RESOURCE_STORAGE, layout = Nri.Layout.SHADER_RESOURCE_STORAGE, stageBits = 1 << 10 };

            // ── NRD standard I/O ────────────────────────────────────────────────
            _nriResources[RenderResourceType.IN_MV]                     = new NriTextureResource(RenderResourceType.IN_MV,                     GraphicsFormat.R16G16B16A16_SFloat,    srvState);
            _nriResources[RenderResourceType.IN_VIEWZ]                  = new NriTextureResource(RenderResourceType.IN_VIEWZ,                  GraphicsFormat.R32_SFloat,             srvState);
            _nriResources[RenderResourceType.IN_NORMAL_ROUGHNESS]       = new NriTextureResource(RenderResourceType.IN_NORMAL_ROUGHNESS,       GraphicsFormat.A2B10G10R10_UNormPack32, srvState);
            _nriResources[RenderResourceType.IN_BASECOLOR_METALNESS]    = new NriTextureResource(RenderResourceType.IN_BASECOLOR_METALNESS,    GraphicsFormat.B8G8R8A8_SRGB,          srvState, true);
            _nriResources[RenderResourceType.IN_PENUMBRA]               = new NriTextureResource(RenderResourceType.IN_PENUMBRA,               GraphicsFormat.R16_SFloat,             srvState);
            _nriResources[RenderResourceType.IN_DIFF_RADIANCE_HITDIST]  = new NriTextureResource(RenderResourceType.IN_DIFF_RADIANCE_HITDIST,  GraphicsFormat.R16G16B16A16_SFloat,    srvState);
            _nriResources[RenderResourceType.IN_SPEC_RADIANCE_HITDIST]  = new NriTextureResource(RenderResourceType.IN_SPEC_RADIANCE_HITDIST,  GraphicsFormat.R16G16B16A16_SFloat,    srvState);
            _nriResources[RenderResourceType.OUT_SHADOW_TRANSLUCENCY]   = new NriTextureResource(RenderResourceType.OUT_SHADOW_TRANSLUCENCY,   GraphicsFormat.R16_SFloat,             uavState);
            _nriResources[RenderResourceType.OUT_DIFF_RADIANCE_HITDIST] = new NriTextureResource(RenderResourceType.OUT_DIFF_RADIANCE_HITDIST, GraphicsFormat.R16G16B16A16_SFloat,    uavState);
            _nriResources[RenderResourceType.OUT_SPEC_RADIANCE_HITDIST] = new NriTextureResource(RenderResourceType.OUT_SPEC_RADIANCE_HITDIST, GraphicsFormat.R16G16B16A16_SFloat,    uavState);
            _nriResources[RenderResourceType.OUT_VALIDATION]            = new NriTextureResource(RenderResourceType.OUT_VALIDATION,            GraphicsFormat.R8G8B8A8_UNorm,         uavState);

            // ── NRI-interop resources (DLSS / composition) ──────────────────────
            _nriResources[RenderResourceType.Composed]               = new NriTextureResource(RenderResourceType.Composed,               GraphicsFormat.R16G16B16A16_SFloat,    uavState);
            _nriResources[RenderResourceType.DlssOutput]             = new NriTextureResource(RenderResourceType.DlssOutput,             GraphicsFormat.R16G16B16A16_SFloat,    uavState);
            _nriResources[RenderResourceType.RRGuide_DiffAlbedo]     = new NriTextureResource(RenderResourceType.RRGuide_DiffAlbedo,     GraphicsFormat.A2B10G10R10_UNormPack32, uavState);
            _nriResources[RenderResourceType.RRGuide_SpecAlbedo]     = new NriTextureResource(RenderResourceType.RRGuide_SpecAlbedo,     GraphicsFormat.A2B10G10R10_UNormPack32, uavState);
            _nriResources[RenderResourceType.RRGuide_SpecHitDistance]   = new NriTextureResource(RenderResourceType.RRGuide_SpecHitDistance,   GraphicsFormat.R16_SFloat,          uavState);
            _nriResources[RenderResourceType.RRGuide_Normal_Roughness]  = new NriTextureResource(RenderResourceType.RRGuide_Normal_Roughness,  GraphicsFormat.R16G16B16A16_SFloat, uavState);
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
        /// Ensures all resources are allocated at the correct resolution.
        /// Returns true when resources were (re)allocated — callers must re-snapshot NRD resources.
        /// </summary>
        public bool EnsureResources(int2 outputResolution)
        {
            bool invalid = false;
            foreach (var res in _nriResources.Values)
                if (res.Handle == null || res.Handle.rt == null) { invalid = true; break; }

            if (!invalid)
                foreach (var h in _rtResources.Values)
                    if (h == null || h.rt == null) { invalid = true; break; }

            int2 target = GetUpscaledResolution(outputResolution, _setting.upscalerMode);
            if (!invalid && target.x == renderResolution.x && target.y == renderResolution.y)
                return false;

            renderResolution = target;

            foreach (var kvp in _nriResources)
            {
                int2 res = kvp.Key == RenderResourceType.DlssOutput ? outputResolution : renderResolution;
                kvp.Value.Allocate(res);
            }

            AllocateRT(RenderResourceType.TaaHistory,              GraphicsFormat.R16G16B16A16_SFloat,     renderResolution);
            AllocateRT(RenderResourceType.TaaHistoryPrev,          GraphicsFormat.R16G16B16A16_SFloat,     renderResolution);
            AllocateRT(RenderResourceType.PsrThroughput,           GraphicsFormat.R16G16B16A16_SFloat,     renderResolution);
            AllocateRT(RenderResourceType.Prev_ViewZ,              GraphicsFormat.R32_SFloat,              renderResolution);
            AllocateRT(RenderResourceType.Prev_NormalRoughness,    GraphicsFormat.A2B10G10R10_UNormPack32, renderResolution);
            AllocateRT(RenderResourceType.Prev_BaseColorMetalness, GraphicsFormat.B8G8R8A8_SRGB,           renderResolution, true);

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
            // Wait for GPU before releasing (same guard as was in original NRDDenoiser)
            if (_nriResources.TryGetValue(RenderResourceType.IN_MV, out var mvRes) && mvRes.IsCreated)
            {
                var h = mvRes.Handle;
                if (h != null && (h.externalTexture != null || h.rt != null))
                    AsyncGPUReadback.Request(h).WaitForCompletion();
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


namespace PathTracing
{
    /// <summary>
    /// Central owner of all per-camera render textures used by the path tracing pipeline.
    ///
    /// Resources are split into three buckets:
    ///   _nrdResources  – NRD standard I/O (ResourceType IN_*/OUT_*); passed to the C++ NRD denoiser.
    ///   _nriResources  – Non-NRD resources that still need a native NRI pointer (DLSS/RR interop).
    ///   _rtResources   – Cross-frame RTHandle-only resources (TAA history, previous-frame GBuffer).
    /// </summary>
    public class PathTracingResourcePool : System.IDisposable
    {
        private readonly PathTracingSetting _setting;

        private readonly Dictionary<ResourceType, NrdTextureResource> _nrdResources = new();
        private readonly Dictionary<RenderResourceType, NrdTextureResource> _nriResources = new();
        private readonly Dictionary<RenderResourceType, RTHandle> _rtResources = new();

        public int2 renderResolution { get; private set; }

        public PathTracingResourcePool(PathTracingSetting setting)
        {
            _setting = setting;

            var srvState = new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE, layout = Layout.SHADER_RESOURCE, stageBits = 1 << 7 };
            var uavState = new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE_STORAGE, layout = Layout.SHADER_RESOURCE_STORAGE, stageBits = 1 << 10 };

            // ── NRD standard I/O ────────────────────────────────────────────────
            // Non-noisy inputs
            _nrdResources[ResourceType.IN_MV]                     = new NrdTextureResource(ResourceType.IN_MV,                     GraphicsFormat.R16G16B16A16_SFloat,       srvState);
            _nrdResources[ResourceType.IN_VIEWZ]                  = new NrdTextureResource(ResourceType.IN_VIEWZ,                  GraphicsFormat.R32_SFloat,                 srvState);
            _nrdResources[ResourceType.IN_NORMAL_ROUGHNESS]       = new NrdTextureResource(ResourceType.IN_NORMAL_ROUGHNESS,       GraphicsFormat.A2B10G10R10_UNormPack32,    srvState);
            _nrdResources[ResourceType.IN_BASECOLOR_METALNESS]    = new NrdTextureResource(ResourceType.IN_BASECOLOR_METALNESS,    GraphicsFormat.B8G8R8A8_SRGB,             srvState, true);
            // Noisy inputs
            _nrdResources[ResourceType.IN_PENUMBRA]               = new NrdTextureResource(ResourceType.IN_PENUMBRA,               GraphicsFormat.R16_SFloat,                 srvState);
            _nrdResources[ResourceType.IN_DIFF_RADIANCE_HITDIST]  = new NrdTextureResource(ResourceType.IN_DIFF_RADIANCE_HITDIST,  GraphicsFormat.R16G16B16A16_SFloat,       srvState);
            _nrdResources[ResourceType.IN_SPEC_RADIANCE_HITDIST]  = new NrdTextureResource(ResourceType.IN_SPEC_RADIANCE_HITDIST,  GraphicsFormat.R16G16B16A16_SFloat,       srvState);
            // Outputs
            _nrdResources[ResourceType.OUT_SHADOW_TRANSLUCENCY]   = new NrdTextureResource(ResourceType.OUT_SHADOW_TRANSLUCENCY,   GraphicsFormat.R16_SFloat,                 uavState);
            _nrdResources[ResourceType.OUT_DIFF_RADIANCE_HITDIST] = new NrdTextureResource(ResourceType.OUT_DIFF_RADIANCE_HITDIST, GraphicsFormat.R16G16B16A16_SFloat,       uavState);
            _nrdResources[ResourceType.OUT_SPEC_RADIANCE_HITDIST] = new NrdTextureResource(ResourceType.OUT_SPEC_RADIANCE_HITDIST, GraphicsFormat.R16G16B16A16_SFloat,       uavState);
            _nrdResources[ResourceType.OUT_VALIDATION]            = new NrdTextureResource(ResourceType.OUT_VALIDATION,            GraphicsFormat.R8G8B8A8_UNorm,            uavState);

            // ── NRI-interop resources (DLSS / composition) ──────────────────────
            _nriResources[RenderResourceType.Composed]               = new NrdTextureResource("Composed",               GraphicsFormat.R16G16B16A16_SFloat,    uavState);
            _nriResources[RenderResourceType.DlssOutput]             = new NrdTextureResource("DlssOutput",             GraphicsFormat.R16G16B16A16_SFloat,    uavState);
            _nriResources[RenderResourceType.RRGuide_DiffAlbedo]     = new NrdTextureResource("RRGuide_DiffAlbedo",     GraphicsFormat.A2B10G10R10_UNormPack32, uavState);
            _nriResources[RenderResourceType.RRGuide_SpecAlbedo]     = new NrdTextureResource("RRGuide_SpecAlbedo",     GraphicsFormat.A2B10G10R10_UNormPack32, uavState);
            _nriResources[RenderResourceType.RRGuide_SpecHitDistance]  = new NrdTextureResource("RRGuide_SpecHitDistance",  GraphicsFormat.R16_SFloat,          uavState);
            _nriResources[RenderResourceType.RRGuide_Normal_Roughness] = new NrdTextureResource("RRGuide_Normal_Roughness", GraphicsFormat.R16G16B16A16_SFloat, uavState);
        }

        // ── Public accessors ────────────────────────────────────────────────────

        /// <summary>Returns the NrdTextureResource for a standard NRD I/O slot (for NRI pointer access).</summary>
        public NrdTextureResource GetNrdResource(ResourceType type) => _nrdResources[type];

        /// <summary>Returns the RTHandle for a standard NRD I/O slot (for render-graph assignment).</summary>
        public RTHandle GetNrdRT(ResourceType type) => _nrdResources[type].Handle;

        /// <summary>Returns the NrdTextureResource for an NRI-interop resource (for NRI pointer access).</summary>
        public NrdTextureResource GetNriResource(RenderResourceType type) => _nriResources[type];

        /// <summary>Returns the RTHandle for any non-NRD resource (NRI-interop or cross-frame RTHandle).</summary>
        public RTHandle GetRT(RenderResourceType type)
        {
            if (_nriResources.TryGetValue(type, out var nriRes)) return nriRes.Handle;
            return _rtResources[type];
        }

        /// <summary>Enumerates all NRD standard I/O resources (for snapshotting into the C++ denoiser).</summary>
        public IEnumerable<NrdTextureResource> GetAllNrdResources() => _nrdResources.Values;

        // ── Resolution/allocation ───────────────────────────────────────────────

        public static int2 GetUpscaledResolution(int2 outputRes, UpscalerMode mode)
        {
            float scale = mode switch
            {
                UpscalerMode.NATIVE           => 1.0f,
                UpscalerMode.ULTRA_QUALITY    => 1.3f,
                UpscalerMode.QUALITY          => 1.5f,
                UpscalerMode.BALANCED         => 1.7f,
                UpscalerMode.PERFORMANCE      => 2.0f,
                UpscalerMode.ULTRA_PERFORMANCE => 3.0f,
                _                             => 1.0f
            };
            return new int2((int)(outputRes.x / scale + 0.5f), (int)(outputRes.y / scale + 0.5f));
        }

        /// <summary>
        /// Ensures all resources are allocated at the correct resolution.
        /// Returns true if resources were (re)allocated (i.e. the NRD snapshot must be refreshed).
        /// </summary>
        public bool EnsureResources(int2 outputResolution)
        {
            // Validate existing allocations
            bool isResourceInvalid = false;
            foreach (var res in _nrdResources.Values)
            {
                if (res.Handle == null || res.Handle.rt == null) { isResourceInvalid = true; break; }
            }
            if (!isResourceInvalid)
            {
                foreach (var res in _nriResources.Values)
                {
                    if (res.Handle == null || res.Handle.rt == null) { isResourceInvalid = true; break; }
                }
            }
            if (!isResourceInvalid)
            {
                foreach (var handle in _rtResources.Values)
                {
                    if (handle == null || handle.rt == null) { isResourceInvalid = true; break; }
                }
            }

            int2 targetRes = GetUpscaledResolution(outputResolution, _setting.upscalerMode);
            if (!isResourceInvalid && targetRes.x == renderResolution.x && targetRes.y == renderResolution.y)
                return false;

            renderResolution = targetRes;

            // Allocate NRD I/O at render resolution
            foreach (var res in _nrdResources.Values)
                res.Allocate(renderResolution);

            // Allocate NRI-interop resources; DlssOutput uses full output resolution
            foreach (var kvp in _nriResources)
            {
                int2 allocRes = kvp.Key == RenderResourceType.DlssOutput ? outputResolution : renderResolution;
                kvp.Value.Allocate(allocRes);
            }

            // Allocate cross-frame RTHandle-only resources
            AllocateRT(RenderResourceType.TaaHistory,             GraphicsFormat.R16G16B16A16_SFloat,    renderResolution);
            AllocateRT(RenderResourceType.TaaHistoryPrev,         GraphicsFormat.R16G16B16A16_SFloat,    renderResolution);
            AllocateRT(RenderResourceType.PsrThroughput,          GraphicsFormat.R16G16B16A16_SFloat,    renderResolution);
            AllocateRT(RenderResourceType.Prev_ViewZ,             GraphicsFormat.R32_SFloat,             renderResolution);
            AllocateRT(RenderResourceType.Prev_NormalRoughness,   GraphicsFormat.A2B10G10R10_UNormPack32, renderResolution);
            AllocateRT(RenderResourceType.Prev_BaseColorMetalness, GraphicsFormat.B8G8R8A8_SRGB,        renderResolution, true);

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
            // Wait for the GPU to finish before releasing resources (same guard as original NRDDenoiser)
            if (_nrdResources.TryGetValue(ResourceType.IN_MV, out var mvRes) && mvRes.IsCreated)
            {
                var handle = mvRes.Handle;
                if (handle != null && (handle.externalTexture != null || handle.rt != null))
                {
                    var request = AsyncGPUReadback.Request(handle);
                    request.WaitForCompletion();
                }
            }

            foreach (var res in _nrdResources.Values) res.Release();
            _nrdResources.Clear();

            foreach (var res in _nriResources.Values) res.Release();
            _nriResources.Clear();

            foreach (var handle in _rtResources.Values)
            {
                if (handle != null)
                {
                    var rt = handle.rt;
                    RTHandles.Release(handle);
                    if (rt != null)
                    {
                        if (Application.isPlaying) Object.Destroy(rt);
                        else Object.DestroyImmediate(rt);
                    }
                }
            }
            _rtResources.Clear();
        }
    }
}
