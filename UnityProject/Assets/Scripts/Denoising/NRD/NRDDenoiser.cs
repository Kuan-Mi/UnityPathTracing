using System;
using System.Runtime.InteropServices;
using Nri;
using PathTracing;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering.Universal;
using static PathTracing.PathTracingUtils;

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

        public uint FrameIndex;
        private readonly int nrdInstanceId;
        private string cameraName;

        public Matrix4x4 worldToView;
        public Matrix4x4 worldToClip;


        public Matrix4x4 prevWorldToView;
        public Matrix4x4 prevWorldToClip;

        public Matrix4x4 viewToClip;
        public Matrix4x4 preViewToClip;


        public float3 camPos;
        public float3 prevCamPos;

        public int2 renderResolution;

        public float resolutionScale;
        public float prevResolutionScale;


        private NativeArray<FrameData> buffer;
        private const int BufferCount = 3;

        private PathTracingSetting setting;

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
            prevResolutionScale = setting.resolutionScale;

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

        public static float Halton(uint n, uint @base)
        {
            float a = 1.0f;
            float b = 0.0f;
            float baseInv = 1.0f / @base;

            while (n != 0)
            {
                a *= baseInv;
                b += a * (n % @base);
                n = (uint)(n * baseInv);
            }

            return b;
        }

        // 32 位反转（等价于 Math::ReverseBits32）
        public static uint ReverseBits32(uint v)
        {
            v = ((v & 0x55555555u) << 1) | ((v >> 1) & 0x55555555u);
            v = ((v & 0x33333333u) << 2) | ((v >> 2) & 0x33333333u);
            v = ((v & 0x0F0F0F0Fu) << 4) | ((v >> 4) & 0x0F0F0F0Fu);
            v = ((v & 0x00FF00FFu) << 8) | ((v >> 8) & 0x00FF00FFu);
            v = (v << 16) | (v >> 16);
            return v;
        }

        // 优化版 Halton(n, 2)
        public static float Halton2(uint n)
        {
            return ReverseBits32(n) * 2.3283064365386963e-10f;
            // 2^-32
        }

        public static float Halton1D(uint n)
        {
            return Halton2(n);
        }

        public static float2 Halton2D(uint n)
        {
            return new float2(
                Halton2(n),
                Halton(n, 3)
            );
        }

        public float2 ViewportJitter;
        public float2 PrevViewportJitter;

        private unsafe FrameData GetData(RenderingData renderingData)
        {

            var cameraData = renderingData.cameraData;
            var lightData = renderingData.lightData;
            var mainLight = lightData.mainLightIndex >= 0 ? lightData.visibleLights[lightData.mainLightIndex] : default;
            var mat = mainLight.localToWorldMatrix;
            Vector3 lightForward = mat.GetColumn(2);
            
            var dirToLight = -lightForward;
            
            
            if (setting.RR)
            {
                setting.resolutionScale = 1.0f;
            }

            prevWorldToView = worldToView;
            prevWorldToClip = worldToClip;
            preViewToClip = viewToClip;
            prevCamPos = camPos;
            prevResolutionScale = resolutionScale;


            var xrPass = cameraData.xr;
            if (xrPass.enabled)
            {
                worldToView = cameraData.xr.GetViewMatrix();
                var proj = GL.GetGPUProjectionMatrix(xrPass.GetProjMatrix(), false);
                worldToClip = proj * worldToView;
                viewToClip = proj;

                Matrix4x4 invView = worldToView.inverse;
                camPos = new float3(invView.m03, invView.m13, invView.m23);
            }
            else
            {
                var mCamera = cameraData.camera;
                camPos = new float3(mCamera.transform.position.x, mCamera.transform.position.y, mCamera.transform.position.z);
                worldToView = mCamera.worldToCameraMatrix;
                worldToClip = GetWorldToClipMatrix(mCamera);
                viewToClip = GL.GetGPUProjectionMatrix(mCamera.projectionMatrix, false);
            }

            resolutionScale = setting.resolutionScale;
            FrameData localData = FrameData._default;

            // --- 矩阵赋值 ---
            localData.commonSettings.viewToClipMatrix = viewToClip;
            localData.commonSettings.viewToClipMatrixPrev = preViewToClip;

            localData.commonSettings.worldToViewMatrix = worldToView;
            localData.commonSettings.worldToViewMatrixPrev = prevWorldToView;

            ViewportJitter = Halton2D(FrameIndex + 1) - new float2(0.5f, 0.5f);

            // --- Jitter ---
            localData.commonSettings.cameraJitter = setting.cameraJitter ? ViewportJitter : float2.zero;
            localData.commonSettings.cameraJitterPrev = setting.cameraJitter ? PrevViewportJitter : float2.zero;

            PrevViewportJitter = ViewportJitter;

            // --- 分辨率与重置逻辑 ---

            ushort rectW = (ushort)(renderResolution.x * setting.resolutionScale + 0.5f);
            ushort rectH = (ushort)(renderResolution.y * setting.resolutionScale + 0.5f);

            ushort prevRectW = (ushort)(renderResolution.x * prevResolutionScale + 0.5f);
            ushort prevRectH = (ushort)(renderResolution.y * prevResolutionScale + 0.5f);

            localData.commonSettings.resourceSize[0] = (ushort)renderResolution.x;
            localData.commonSettings.resourceSize[1] = (ushort)renderResolution.y;
            localData.commonSettings.rectSize[0] = rectW;
            localData.commonSettings.rectSize[1] = rectH;

            localData.commonSettings.resourceSizePrev[0] = (ushort)renderResolution.x;
            localData.commonSettings.resourceSizePrev[1] = (ushort)renderResolution.y;
            localData.commonSettings.rectSizePrev[0] = prevRectW;
            localData.commonSettings.rectSizePrev[1] = prevRectH;

            localData.commonSettings.motionVectorScale = new float3(1.0f / rectW, 1.0f / rectH, -1.0f);
            localData.commonSettings.isMotionVectorInWorldSpace = false;

            localData.commonSettings.accumulationMode = AccumulationMode.CONTINUE;
            localData.commonSettings.frameIndex = FrameIndex;

            // --- Sigma 设置 (光照) ---
            // Sigma 需要指向光源的方向 (normalized)
            localData.sigmaSettings.lightDirection = dirToLight;

            // Debug.Log("Record Frame Index: " + m_FrameIndex);

            // 4. 更新历史状态

            localData.instanceId = nrdInstanceId;

            localData.width = (ushort)renderResolution.x;
            localData.height = (ushort)renderResolution.y;

            //  Common 设置

            localData.commonSettings.denoisingRange = setting.denoisingRange;
            localData.commonSettings.splitScreen = setting.splitScreen;
            localData.commonSettings.isBaseColorMetalnessAvailable = setting.isBaseColorMetalnessAvailable;
            localData.commonSettings.enableValidation = setting.showValidation;


            // Sigma 设置
            localData.sigmaSettings.planeDistanceSensitivity = setting.planeDistanceSensitivity;
            localData.sigmaSettings.maxStabilizedFrameNum = setting.maxStabilizedFrameNum;

            // reblur 设置

            localData.reblurSettings.checkerboardMode = CheckerboardMode.OFF;
            localData.reblurSettings.minMaterialForDiffuse = 0;
            localData.reblurSettings.minMaterialForSpecular = 1;

            return localData;
        }
        
        public IntPtr GetInteropDataPtr(RenderingData renderingData)
        {
            var index = (int)(FrameIndex % BufferCount);
            buffer[index] = GetData(renderingData);
            FrameIndex++;
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