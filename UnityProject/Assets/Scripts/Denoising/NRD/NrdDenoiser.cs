using System;
using System.Runtime.InteropServices;
using Nri;
using PathTracing;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace Nrd
{
    public class NrdDenoiser : IDisposable
    {
        [DllImport("RenderingPlugin")]
        private static extern int CreateDenoiserInstance();

        [DllImport("RenderingPlugin")]
        private static extern void DestroyDenoiserInstance(int id);

        [DllImport("RenderingPlugin")]
        private static extern void UpdateDenoiserResources(int instanceId, IntPtr resources, int count);

        private NativeArray<NrdResourceInput> _resourceCache;

        private readonly int _nrdInstanceId;
        private readonly string _cameraName;

        private NativeArray<NrdFrameData> _buffer;
        private const int BufferCount = 3;

        private readonly PathTracingSetting _setting;

        /// <summary>
        /// Per-frame camera data filled by PathTracingFeature from CameraFrameState.
        /// NRDDenoiser does not depend on CameraFrameState directly.
        /// </summary>
        public struct NrdFrameInput
        {
            public Matrix4x4 worldToView;
            public Matrix4x4 prevWorldToView;
            public Matrix4x4 viewToClip;
            public Matrix4x4 prevViewToClip;
            public float2    viewportJitter;
            public float2    prevViewportJitter;
            public float     resolutionScale;
            public float     prevResolutionScale;
            public int2      renderResolution;
            public uint      frameIndex;
            public float3    lightDirection;
        }

        /// <summary>
        /// NRD-required textures packed by PathTracingFeature and passed on each resource refresh.
        /// </summary>
        public struct NrdResources
        {
            public NriTextureResource inMv;
            public NriTextureResource inViewZ;
            public NriTextureResource inNormalRoughness;
            public NriTextureResource inBaseColorMetalness;
            public NriTextureResource inPenumbra;
            public NriTextureResource inDiffRadianceHitDist;
            public NriTextureResource inSpecRadianceHitDist;

            public NriTextureResource outShadowTranslucency;
            public NriTextureResource outDiffRadianceHitDist;
            public NriTextureResource outSpecRadianceHitDist;
            public NriTextureResource outValidation;
        }

        public NrdDenoiser(PathTracingSetting setting, string camName)
        {
            _setting = setting;
            _nrdInstanceId = CreateDenoiserInstance();
            _cameraName = camName;
            _buffer = new NativeArray<NrdFrameData>(BufferCount, Allocator.Persistent);
            Debug.Log($"[NRD] Created Denoiser Instance {_nrdInstanceId} for Camera {_cameraName}");
        }

        /// <summary>
        /// Pushes the current NRD texture snapshot to the C++ denoiser.
        /// Called by PathTracingFeature whenever textures are reallocated.
        /// </summary>
        public unsafe void UpdateResources(NrdResources res)
        {
            const int count = 11;
            if (!_resourceCache.IsCreated || _resourceCache.Length < count)
            {
                if (_resourceCache.IsCreated) _resourceCache.Dispose();
                _resourceCache = new NativeArray<NrdResourceInput>(count, Allocator.Persistent);
            }

            int idx = 0;
            var ptr = (NrdResourceInput*)_resourceCache.GetUnsafePtr();

            void Add(ResourceType t, NriTextureResource r) =>
                ptr[idx++] = new NrdResourceInput { type = t, texture = r.NriPtr, state = r.ResourceState };

            Add(ResourceType.IN_MV,                    res.inMv);
            Add(ResourceType.IN_VIEWZ,                 res.inViewZ);
            Add(ResourceType.IN_NORMAL_ROUGHNESS,      res.inNormalRoughness);
            Add(ResourceType.IN_BASECOLOR_METALNESS,   res.inBaseColorMetalness);
            Add(ResourceType.IN_PENUMBRA,              res.inPenumbra);
            Add(ResourceType.IN_DIFF_RADIANCE_HITDIST, res.inDiffRadianceHitDist);
            Add(ResourceType.IN_SPEC_RADIANCE_HITDIST, res.inSpecRadianceHitDist);
            Add(ResourceType.OUT_SHADOW_TRANSLUCENCY,  res.outShadowTranslucency);
            Add(ResourceType.OUT_DIFF_RADIANCE_HITDIST,res.outDiffRadianceHitDist);
            Add(ResourceType.OUT_SPEC_RADIANCE_HITDIST,res.outSpecRadianceHitDist);
            Add(ResourceType.OUT_VALIDATION,           res.outValidation);

            UpdateDenoiserResources(_nrdInstanceId, (IntPtr)ptr, idx);
            Debug.Log($"[NRD] Updated Resources for Denoiser Instance {_nrdInstanceId} with {idx} resources.");
        }

        private unsafe NrdFrameData GetData(NrdFrameInput fi)
        {
            NrdFrameData data = NrdFrameData._default;

            // --- 矩阵赋值 ---
            data.commonSettings.viewToClipMatrix     = fi.viewToClip;
            data.commonSettings.viewToClipMatrixPrev = fi.prevViewToClip;
            data.commonSettings.worldToViewMatrix     = fi.worldToView;
            data.commonSettings.worldToViewMatrixPrev = fi.prevWorldToView;

            // --- Jitter ---
            data.commonSettings.cameraJitter     = _setting.cameraJitter ? fi.viewportJitter     : float2.zero;
            data.commonSettings.cameraJitterPrev = _setting.cameraJitter ? fi.prevViewportJitter : float2.zero;

            // --- 分辨率 ---
            ushort rectW     = (ushort)(fi.renderResolution.x * fi.resolutionScale     + 0.5f);
            ushort rectH     = (ushort)(fi.renderResolution.y * fi.resolutionScale     + 0.5f);
            ushort prevRectW = (ushort)(fi.renderResolution.x * fi.prevResolutionScale + 0.5f);
            ushort prevRectH = (ushort)(fi.renderResolution.y * fi.prevResolutionScale + 0.5f);

            data.commonSettings.resourceSize[0]     = (ushort)fi.renderResolution.x;
            data.commonSettings.resourceSize[1]     = (ushort)fi.renderResolution.y;
            data.commonSettings.rectSize[0]         = rectW;
            data.commonSettings.rectSize[1]         = rectH;
            data.commonSettings.resourceSizePrev[0] = (ushort)fi.renderResolution.x;
            data.commonSettings.resourceSizePrev[1] = (ushort)fi.renderResolution.y;
            data.commonSettings.rectSizePrev[0]     = prevRectW;
            data.commonSettings.rectSizePrev[1]     = prevRectH;

            data.commonSettings.motionVectorScale          = new float3(1.0f / rectW, 1.0f / rectH, -1.0f);
            data.commonSettings.isMotionVectorInWorldSpace = false;
            data.commonSettings.accumulationMode           = AccumulationMode.CONTINUE;
            data.commonSettings.frameIndex                 = fi.frameIndex;

            // --- Sigma 设置 (光照) ---
            data.sigmaSettings.lightDirection = fi.lightDirection;

            data.instanceId = _nrdInstanceId;
            data.width      = (ushort)fi.renderResolution.x;
            data.height     = (ushort)fi.renderResolution.y;

            // Common 设置
            data.commonSettings.denoisingRange                = _setting.denoisingRange;
            data.commonSettings.splitScreen                   = _setting.splitScreen;
            data.commonSettings.isBaseColorMetalnessAvailable = _setting.isBaseColorMetalnessAvailable;
            data.commonSettings.enableValidation              = _setting.showValidation;

            // Sigma 设置
            data.sigmaSettings.planeDistanceSensitivity = _setting.planeDistanceSensitivity;
            data.sigmaSettings.maxStabilizedFrameNum    = _setting.maxStabilizedFrameNum;

            // Reblur 设置
            data.reblurSettings.checkerboardMode       = CheckerboardMode.OFF;
            data.reblurSettings.minMaterialForDiffuse  = 0;
            data.reblurSettings.minMaterialForSpecular = 1;

            return data;
        }

        public IntPtr GetInteropDataPtr(NrdFrameInput fi)
        {
            var index = (int)(fi.frameIndex % BufferCount);
            _buffer[index] = GetData(fi);
            unsafe
            {
                return (IntPtr)_buffer.GetUnsafePtr() + index * sizeof(NrdFrameData);
            }
        }

        public void Dispose()
        {
            if (_buffer.IsCreated) _buffer.Dispose();
            if (_resourceCache.IsCreated) _resourceCache.Dispose();
            DestroyDenoiserInstance(_nrdInstanceId);
            Debug.Log($"[NRD] Destroyed Denoiser Instance {_nrdInstanceId} for Camera {_cameraName} - Dispose Complete");
        }
    }
}