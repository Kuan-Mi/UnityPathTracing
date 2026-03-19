using System;
using System.Runtime.InteropServices;
using Nri;
using PathTracing;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering.Universal;

namespace Nrd
{
    public class NRDDenoiser : IDisposable
    {
        [DllImport("RenderingPlugin")]
        private static extern int CreateDenoiserInstance();

        [DllImport("RenderingPlugin")]
        private static extern void DestroyDenoiserInstance(int id);

        [DllImport("RenderingPlugin")]
        private static extern void UpdateDenoiserResources(int instanceId, IntPtr resources, int count);

        private NativeArray<NrdResourceInput> m_ResourceCache;

        private readonly int nrdInstanceId;
        private string cameraName;

        private NativeArray<FrameData> buffer;
        private const int BufferCount = 3;

        private PathTracingSetting setting;

        /// <summary>
        /// Per-frame camera data filled by PathTracingFeature from CameraFrameState.
        /// NRDDenoiser does not depend on CameraFrameState directly.
        /// </summary>
        public struct NrdFrameInput
        {
            public Matrix4x4 worldToView;
            public Matrix4x4 prevWorldToView;
            public Matrix4x4 worldToClip;
            public Matrix4x4 prevWorldToClip;
            public Matrix4x4 viewToClip;
            public Matrix4x4 prevViewToClip;
            public float3    camPos;
            public float3    prevCamPos;
            public float2    ViewportJitter;
            public float2    PrevViewportJitter;
            public float     resolutionScale;
            public float     prevResolutionScale;
            public int2      renderResolution;
            public uint      FrameIndex;
        }

        /// <summary>
        /// NRD-required textures packed by PathTracingFeature and passed on each resource refresh.
        /// </summary>
        public struct NrdResources
        {
            public NriTextureResource InMv;
            public NriTextureResource InViewZ;
            public NriTextureResource InNormalRoughness;
            public NriTextureResource InBaseColorMetalness;
            public NriTextureResource InPenumbra;
            public NriTextureResource InDiffRadianceHitDist;
            public NriTextureResource InSpecRadianceHitDist;

            public NriTextureResource OutShadowTranslucency;
            public NriTextureResource OutDiffRadianceHitDist;
            public NriTextureResource OutSpecRadianceHitDist;
            public NriTextureResource OutValidation;
        }

        public NRDDenoiser(PathTracingSetting setting, string camName)
        {
            this.setting = setting;
            nrdInstanceId = CreateDenoiserInstance();
            cameraName = camName;
            buffer = new NativeArray<FrameData>(BufferCount, Allocator.Persistent);
            Debug.Log($"[NRD] Created Denoiser Instance {nrdInstanceId} for Camera {cameraName}");
        }

        /// <summary>
        /// Pushes the current NRD texture snapshot to the C++ denoiser.
        /// Called by PathTracingFeature whenever textures are reallocated.
        /// </summary>
        public unsafe void UpdateResources(NrdResources res)
        {
            const int count = 11;
            if (!m_ResourceCache.IsCreated || m_ResourceCache.Length < count)
            {
                if (m_ResourceCache.IsCreated) m_ResourceCache.Dispose();
                m_ResourceCache = new NativeArray<NrdResourceInput>(count, Allocator.Persistent);
            }

            int idx = 0;
            var ptr = (NrdResourceInput*)m_ResourceCache.GetUnsafePtr();

            void Add(ResourceType t, NriTextureResource r) =>
                ptr[idx++] = new NrdResourceInput { type = t, texture = r.NriPtr, state = r.ResourceState };

            Add(ResourceType.IN_MV,                    res.InMv);
            Add(ResourceType.IN_VIEWZ,                 res.InViewZ);
            Add(ResourceType.IN_NORMAL_ROUGHNESS,      res.InNormalRoughness);
            Add(ResourceType.IN_BASECOLOR_METALNESS,   res.InBaseColorMetalness);
            Add(ResourceType.IN_PENUMBRA,              res.InPenumbra);
            Add(ResourceType.IN_DIFF_RADIANCE_HITDIST, res.InDiffRadianceHitDist);
            Add(ResourceType.IN_SPEC_RADIANCE_HITDIST, res.InSpecRadianceHitDist);
            Add(ResourceType.OUT_SHADOW_TRANSLUCENCY,  res.OutShadowTranslucency);
            Add(ResourceType.OUT_DIFF_RADIANCE_HITDIST,res.OutDiffRadianceHitDist);
            Add(ResourceType.OUT_SPEC_RADIANCE_HITDIST,res.OutSpecRadianceHitDist);
            Add(ResourceType.OUT_VALIDATION,           res.OutValidation);

            UpdateDenoiserResources(nrdInstanceId, (IntPtr)ptr, idx);
            Debug.Log($"[NRD] Updated Resources for Denoiser Instance {nrdInstanceId} with {idx} resources.");
        }

        private unsafe FrameData GetData(NrdFrameInput fi, RenderingData renderingData)
        {
            var lightData = renderingData.lightData;
            var mainLight = lightData.mainLightIndex >= 0 ? lightData.visibleLights[lightData.mainLightIndex] : default;
            var dirToLight = -(Vector3)mainLight.localToWorldMatrix.GetColumn(2);

            FrameData localData = FrameData._default;

            // --- 矩阵赋值 ---
            localData.commonSettings.viewToClipMatrix     = fi.viewToClip;
            localData.commonSettings.viewToClipMatrixPrev = fi.prevViewToClip;
            localData.commonSettings.worldToViewMatrix     = fi.worldToView;
            localData.commonSettings.worldToViewMatrixPrev = fi.prevWorldToView;

            // --- Jitter ---
            localData.commonSettings.cameraJitter     = setting.cameraJitter ? fi.ViewportJitter     : float2.zero;
            localData.commonSettings.cameraJitterPrev = setting.cameraJitter ? fi.PrevViewportJitter : float2.zero;

            // --- 分辨率 ---
            ushort rectW     = (ushort)(fi.renderResolution.x * fi.resolutionScale     + 0.5f);
            ushort rectH     = (ushort)(fi.renderResolution.y * fi.resolutionScale     + 0.5f);
            ushort prevRectW = (ushort)(fi.renderResolution.x * fi.prevResolutionScale + 0.5f);
            ushort prevRectH = (ushort)(fi.renderResolution.y * fi.prevResolutionScale + 0.5f);

            localData.commonSettings.resourceSize[0]     = (ushort)fi.renderResolution.x;
            localData.commonSettings.resourceSize[1]     = (ushort)fi.renderResolution.y;
            localData.commonSettings.rectSize[0]         = rectW;
            localData.commonSettings.rectSize[1]         = rectH;
            localData.commonSettings.resourceSizePrev[0] = (ushort)fi.renderResolution.x;
            localData.commonSettings.resourceSizePrev[1] = (ushort)fi.renderResolution.y;
            localData.commonSettings.rectSizePrev[0]     = prevRectW;
            localData.commonSettings.rectSizePrev[1]     = prevRectH;

            localData.commonSettings.motionVectorScale          = new float3(1.0f / rectW, 1.0f / rectH, -1.0f);
            localData.commonSettings.isMotionVectorInWorldSpace = false;
            localData.commonSettings.accumulationMode           = AccumulationMode.CONTINUE;
            localData.commonSettings.frameIndex                 = fi.FrameIndex;

            // --- Sigma 设置 (光照) ---
            localData.sigmaSettings.lightDirection = dirToLight;

            localData.instanceId = nrdInstanceId;
            localData.width      = (ushort)fi.renderResolution.x;
            localData.height     = (ushort)fi.renderResolution.y;

            // Common 设置
            localData.commonSettings.denoisingRange                = setting.denoisingRange;
            localData.commonSettings.splitScreen                   = setting.splitScreen;
            localData.commonSettings.isBaseColorMetalnessAvailable = setting.isBaseColorMetalnessAvailable;
            localData.commonSettings.enableValidation              = setting.showValidation;

            // Sigma 设置
            localData.sigmaSettings.planeDistanceSensitivity = setting.planeDistanceSensitivity;
            localData.sigmaSettings.maxStabilizedFrameNum    = setting.maxStabilizedFrameNum;

            // Reblur 设置
            localData.reblurSettings.checkerboardMode       = CheckerboardMode.OFF;
            localData.reblurSettings.minMaterialForDiffuse  = 0;
            localData.reblurSettings.minMaterialForSpecular = 1;

            return localData;
        }

        public IntPtr GetInteropDataPtr(NrdFrameInput fi, RenderingData renderingData)
        {
            var index = (int)(fi.FrameIndex % BufferCount);
            buffer[index] = GetData(fi, renderingData);
            unsafe
            {
                return (IntPtr)buffer.GetUnsafePtr() + index * sizeof(FrameData);
            }
        }

        public void Dispose()
        {
            if (buffer.IsCreated) buffer.Dispose();
            if (m_ResourceCache.IsCreated) m_ResourceCache.Dispose();
            DestroyDenoiserInstance(nrdInstanceId);
            Debug.Log($"[NRD] Destroyed Denoiser Instance {nrdInstanceId} for Camera {cameraName} - Dispose Complete");
        }
    }
}