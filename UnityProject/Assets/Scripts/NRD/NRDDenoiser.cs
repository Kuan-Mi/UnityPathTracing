using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using NRD;
using Nri;
using PathTracing;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
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

        private int _prevWidth = -1;
        private int _prevHeight = -1;

        private NativeArray<FrameData> buffer;
        private const int BufferCount = 3;

        private List<NrdTextureResource> allocatedResources = new();

        public NrdTextureResource GetResource(ResourceType type)
        {
            return allocatedResources.Find(res => res.ResourceType == type);
        }

        public RTHandle GetRT(ResourceType type)
        {
            return allocatedResources.Find(res => res.ResourceType == type).Handle;
        }

        private PathTracingSetting setting;

        public NRDDenoiser(PathTracingSetting setting, string camName)
        {
            this.setting = setting;
            nrdInstanceId = CreateDenoiserInstance();
            cameraName = camName;
            buffer = new NativeArray<FrameData>(BufferCount, Allocator.Persistent);

            var srvState = new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE, layout = Layout.SHADER_RESOURCE, stageBits = 1 << 7 };
            var uavState = new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE_STORAGE, layout = Layout.SHADER_RESOURCE_STORAGE, stageBits = 1 << 10 };

            // 无噪声输入
            allocatedResources.Add(new NrdTextureResource(ResourceType.IN_MV, GraphicsFormat.R16G16B16A16_SFloat, srvState));
            allocatedResources.Add(new NrdTextureResource(ResourceType.IN_VIEWZ, GraphicsFormat.R32_SFloat, srvState));
            allocatedResources.Add(new NrdTextureResource(ResourceType.IN_NORMAL_ROUGHNESS, GraphicsFormat.A2B10G10R10_UNormPack32, srvState));
            allocatedResources.Add(new NrdTextureResource(ResourceType.IN_BASECOLOR_METALNESS, GraphicsFormat.B8G8R8A8_SRGB, srvState));

            // 有噪声输入
            allocatedResources.Add(new NrdTextureResource(ResourceType.IN_PENUMBRA, GraphicsFormat.R16_SFloat, srvState));
            allocatedResources.Add(new NrdTextureResource(ResourceType.IN_DIFF_RADIANCE_HITDIST, GraphicsFormat.R16G16B16A16_SFloat, srvState));
            allocatedResources.Add(new NrdTextureResource(ResourceType.IN_SPEC_RADIANCE_HITDIST, GraphicsFormat.R16G16B16A16_SFloat, srvState));

            // 输出
            allocatedResources.Add(new NrdTextureResource(ResourceType.OUT_SHADOW_TRANSLUCENCY, GraphicsFormat.R16_SFloat, uavState));
            allocatedResources.Add(new NrdTextureResource(ResourceType.OUT_DIFF_RADIANCE_HITDIST, GraphicsFormat.R16G16B16A16_SFloat, uavState));
            allocatedResources.Add(new NrdTextureResource(ResourceType.OUT_SPEC_RADIANCE_HITDIST, GraphicsFormat.R16G16B16A16_SFloat, uavState));
            allocatedResources.Add(new NrdTextureResource(ResourceType.OUT_VALIDATION, GraphicsFormat.R8G8B8A8_UNorm, uavState));

            // TAA
            allocatedResources.Add(new NrdTextureResource(ResourceType.TaaHistory, GraphicsFormat.R16G16B16A16_SFloat, uavState));
            allocatedResources.Add(new NrdTextureResource(ResourceType.TaaHistoryPrev, GraphicsFormat.R16G16B16A16_SFloat, uavState));


            Debug.Log($"[NRD] Created Denoiser Instance {nrdInstanceId} for Camera {cameraName}");
        }

        public void EnsureResources(int width, int height)
        {
            // 如果尺寸没变且资源都存在，直接返回
            if (width == _prevWidth && height == _prevHeight)
            {
                return;
            }

            _prevWidth = width;
            _prevHeight = height;

            FrameIndex = 0;

            foreach (var nrdTextureResource in allocatedResources)
            {
                nrdTextureResource.Allocate(width, height);
            }

            UpdateResourceSnapshotInCpp();
        }

        private unsafe void UpdateResourceSnapshotInCpp()
        {
            // 定义需要的资源数量 (Sigma + Reblur 大概 10-15 个)
            int maxResources = 20;
            if (!m_ResourceCache.IsCreated || m_ResourceCache.Length < maxResources)
            {
                if (m_ResourceCache.IsCreated) m_ResourceCache.Dispose();
                m_ResourceCache = new NativeArray<NrdResourceInput>(maxResources, Allocator.Persistent);
            }

            int idx = 0;

            // Reblur/Sigma Inputs
            NrdResourceInput* ptr = (NrdResourceInput*)m_ResourceCache.GetUnsafePtr();

            foreach (var nrdTextureResource in allocatedResources)
            {
                if (nrdTextureResource.ResourceType == ResourceType.TaaHistory ||
                    nrdTextureResource.ResourceType == ResourceType.TaaHistoryPrev)
                    continue; // TAA 资源不传给 NRD)

                ptr[idx++] = new NrdResourceInput { type = nrdTextureResource.ResourceType, texture = nrdTextureResource.NriPtr, state = nrdTextureResource.ResourceState };
            }

            UpdateDenoiserResources(nrdInstanceId, (IntPtr)ptr, idx);

            Debug.Log($"[NRD] Updated Resources for Denoiser Instance {nrdInstanceId} with {idx} resources.");
        }

        private void ReleaseTextures()
        {
            Debug.Log($"[NRD] Releasing Textures for Denoiser Instance {nrdInstanceId}.");
            foreach (var nrdTextureResource in allocatedResources)
            {
                nrdTextureResource.Release();
            }

            allocatedResources.Clear();
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

        private unsafe FrameData GetData(Camera mCamera, Vector3 dirToLight)
        {
            prevWorldToView = worldToView;
            prevWorldToClip = worldToClip;
            preViewToClip = viewToClip;


            worldToView = mCamera.worldToCameraMatrix;
            worldToClip = GetWorldToClipMatrix(mCamera);
            viewToClip = GL.GetGPUProjectionMatrix(mCamera.projectionMatrix, false);

            FrameData localData = FrameData._default;

            // --- 矩阵赋值 ---
            localData.commonSettings.viewToClipMatrix = viewToClip;
            localData.commonSettings.viewToClipMatrixPrev = preViewToClip;

            localData.commonSettings.worldToViewMatrix = worldToView;
            localData.commonSettings.worldToViewMatrixPrev = prevWorldToView;

            ViewportJitter = Halton2D(FrameIndex + 1) - new float2(0.5f, 0.5f);

            // --- Jitter ---
            localData.commonSettings.cameraJitter = ViewportJitter;
            localData.commonSettings.cameraJitterPrev = PrevViewportJitter;

            PrevViewportJitter = ViewportJitter;

            // --- 分辨率与重置逻辑 ---
            ushort w = (ushort)mCamera.pixelWidth;
            ushort h = (ushort)mCamera.pixelHeight;

            localData.commonSettings.resourceSize[0] = w;
            localData.commonSettings.resourceSize[1] = h;
            localData.commonSettings.rectSize[0] = w;
            localData.commonSettings.rectSize[1] = h;

            localData.commonSettings.resourceSizePrev[0] = w;
            localData.commonSettings.resourceSizePrev[1] = h;
            localData.commonSettings.rectSizePrev[0] = w;
            localData.commonSettings.rectSizePrev[1] = h;

            localData.commonSettings.motionVectorScale = new float3(1.0f / w, 1.0f / h, 0.0f);
            localData.commonSettings.isMotionVectorInWorldSpace = false;

            localData.commonSettings.accumulationMode = AccumulationMode.CONTINUE;
            localData.commonSettings.frameIndex = FrameIndex;

            // --- Sigma 设置 (光照) ---
            // Sigma 需要指向光源的方向 (normalized)
            localData.sigmaSettings.lightDirection = dirToLight;

            // Debug.Log("Record Frame Index: " + m_FrameIndex);

            // 4. 更新历史状态

            localData.instanceId = nrdInstanceId;

            localData.width = w;
            localData.height = h;

            //  Common 设置

            // if (setting.useOverriddenCommonSettings)
            // {
            //     localData.commonSettings.viewToClipMatrix = setting.viewToClipMatrix;
            //     localData.commonSettings.viewToClipMatrixPrev = setting.viewToClipMatrixPrev;
            //     localData.commonSettings.worldToViewMatrix = setting.worldToViewMatrix;
            //     localData.commonSettings.worldToViewMatrixPrev = setting.worldToViewMatrixPrev;
            // }

            localData.commonSettings.motionVectorScale.z = 1.0f;
            localData.commonSettings.denoisingRange = setting.denoisingRange;

            // localData.commonSettings.disocclusionThreshold = setting.disocclusionThreshold;
            // localData.commonSettings.disocclusionThresholdAlternate = setting.disocclusionThresholdAlternate;
            localData.commonSettings.splitScreen = setting.splitScreen;

            // localData.commonSettings.isMotionVectorInWorldSpace = setting.isMotionVectorInWorldSpace;
            // localData.commonSettings.isHistoryConfidenceAvailable = setting.isHistoryConfidenceAvailable;
            // localData.commonSettings.isDisocclusionThresholdMixAvailable = setting.isDisocclusionThresholdMixAvailable;
            localData.commonSettings.isBaseColorMetalnessAvailable = setting.isBaseColorMetalnessAvailable;
            localData.commonSettings.enableValidation = true;


            // Sigma 设置

            // if (setting.useOverriddenSigmaValues)
            // {
            //     localData.sigmaSettings.lightDirection = setting.lightDir;
            // }

            localData.sigmaSettings.planeDistanceSensitivity = setting.planeDistanceSensitivity;
            localData.sigmaSettings.maxStabilizedFrameNum = setting.maxStabilizedFrameNum;

            // reblur 设置

            localData.reblurSettings.checkerboardMode = CheckerboardMode.OFF;
            localData.reblurSettings.minMaterialForDiffuse = 0;
            localData.reblurSettings.minMaterialForSpecular = 1;
            // localData.reblurSettings.hitDistanceReconstructionMode = mHitDistanceReconstructionMode::OFF;

            return localData;
        }

        public IntPtr GetInteropDataPtr(Camera mCamera, Vector3 dirToLight)
        {
            var index = (int)(FrameIndex % BufferCount);
            buffer[index] = GetData(mCamera, dirToLight);
            FrameIndex++;
            unsafe
            {
                return (IntPtr)buffer.GetUnsafePtr() + index * sizeof(FrameData);
            }
        }

        public void Dispose()
        {
            if (buffer.IsCreated)
            {
                buffer.Dispose();
            }

            var request = AsyncGPUReadback.Request(allocatedResources[0].Handle);
            request.WaitForCompletion();

            ReleaseTextures();
            DestroyDenoiserInstance(nrdInstanceId);
            Debug.Log($"[NRD] Destroyed Denoiser Instance {nrdInstanceId} for Camera {cameraName} - Dispose Complete");
        }
    }
}