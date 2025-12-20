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

        private Matrix4x4 PrevViewMatrix;
        private Matrix4x4 PrevViewProjMatrix;

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

        public NRDDenoiser(PathTracingSetting setting)
        {
            this.setting = setting;
            int instanceId = CreateDenoiserInstance();
            nrdInstanceId = instanceId;
            buffer = new NativeArray<FrameData>(BufferCount, Allocator.Persistent);
        }

        public void EnsureResources(int width, int height)
        {
            // 如果尺寸没变且资源都存在，直接返回
            if (width == _prevWidth && height == _prevHeight)
            {
                if (FrameIndex == 1)
                {
                    UpdateResourceSnapshotInCpp();
                }

                return;
            }

            _prevWidth = width;
            _prevHeight = height;

            FrameIndex = 0;


            var srvState = new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE, layout = Layout.SHADER_RESOURCE, stageBits = 1 << 7 };
            var uavState = new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE_STORAGE, layout = Layout.SHADER_RESOURCE_STORAGE, stageBits = 1 << 10 };

            allocatedResources.Add(new NrdTextureResource(ResourceType.IN_MV, GraphicsFormat.R16G16B16A16_SFloat, srvState));
            allocatedResources.Add(new NrdTextureResource(ResourceType.IN_VIEWZ, GraphicsFormat.R32_SFloat, srvState));
            allocatedResources.Add(new NrdTextureResource(ResourceType.IN_NORMAL_ROUGHNESS, GraphicsFormat.A2B10G10R10_UNormPack32, srvState));
            allocatedResources.Add(new NrdTextureResource(ResourceType.IN_PENUMBRA, GraphicsFormat.R16_SFloat, srvState));
            allocatedResources.Add(new NrdTextureResource(ResourceType.OUT_SHADOW_TRANSLUCENCY, GraphicsFormat.R16_SFloat, uavState));
            allocatedResources.Add(new NrdTextureResource(ResourceType.IN_DIFF_RADIANCE_HITDIST, GraphicsFormat.R16G16B16A16_SFloat, srvState));
            allocatedResources.Add(new NrdTextureResource(ResourceType.OUT_DIFF_RADIANCE_HITDIST, GraphicsFormat.R16G16B16A16_SFloat, uavState));
            allocatedResources.Add(new NrdTextureResource(ResourceType.OUT_VALIDATION, GraphicsFormat.R8G8B8A8_UNorm, uavState));

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

            var srvState = new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE, layout = Layout.SHADER_RESOURCE, stageBits = 1 << 7 };
            var uavState = new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE_STORAGE, layout = Layout.SHADER_RESOURCE_STORAGE, stageBits = 1 << 10 };

            // Reblur/Sigma Inputs
            NrdResourceInput* ptr = (NrdResourceInput*)m_ResourceCache.GetUnsafePtr();

            foreach (var nrdTextureResource in allocatedResources)
            {
                ptr[idx++] = new NrdResourceInput { type = nrdTextureResource.ResourceType, texture = nrdTextureResource.NriPtr, state = nrdTextureResource.ResourceState };
            }


            UpdateDenoiserResources(nrdInstanceId, (IntPtr)ptr, idx);

            Debug.Log($"[NRD] Updated resources pointer to C++. Count: {idx}");
        }

        private void ReleaseTextures()
        {
            foreach (var nrdTextureResource in allocatedResources)
            {
                nrdTextureResource.Release();
            }

            allocatedResources.Clear();
        }

        private unsafe FrameData GetData(Camera mCamera, Vector3 dirToLight)
        {
            Matrix4x4 proj = GL.GetGPUProjectionMatrix(mCamera.projectionMatrix, false);

            Matrix4x4 worldToView = mCamera.worldToCameraMatrix;

            var viewProj = proj;

            FrameData localData = FrameData._default;

            // --- 矩阵赋值 ---
            localData.commonSettings.viewToClipMatrix = viewProj;
            localData.commonSettings.viewToClipMatrixPrev = PrevViewProjMatrix;

            localData.commonSettings.worldToViewMatrix = worldToView;
            localData.commonSettings.worldToViewMatrixPrev = PrevViewMatrix;

            // --- Jitter ---
            // 填入你实际的 TAA Jitter 值。如果没有 TAA，保持为 0
            localData.commonSettings.cameraJitter[0] = 0.0f;
            localData.commonSettings.cameraJitter[1] = 0.0f;
            localData.commonSettings.cameraJitterPrev[0] = 0.0f;
            localData.commonSettings.cameraJitterPrev[1] = 0.0f;

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
            PrevViewProjMatrix = viewProj;
            PrevViewMatrix = worldToView;

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

            localData.commonSettings.motionVectorScale.z = setting.is2DMotionVector ? 0.0f : 1.0f;


            localData.commonSettings.denoisingRange = setting.denoisingRange;

            // localData.commonSettings.disocclusionThreshold = setting.disocclusionThreshold;
            // localData.commonSettings.disocclusionThresholdAlternate = setting.disocclusionThresholdAlternate;
            localData.commonSettings.splitScreen = setting.splitScreen;

            // localData.commonSettings.isMotionVectorInWorldSpace = setting.isMotionVectorInWorldSpace;
            // localData.commonSettings.isHistoryConfidenceAvailable = setting.isHistoryConfidenceAvailable;
            // localData.commonSettings.isDisocclusionThresholdMixAvailable = setting.isDisocclusionThresholdMixAvailable;
            // localData.commonSettings.isBaseColorMetalnessAvailable = setting.isBaseColorMetalnessAvailable;
            localData.commonSettings.enableValidation = true;


            // Sigma 设置

            if (setting.useOverriddenSigmaValues)
            {
                localData.sigmaSettings.lightDirection = setting.lightDir;
            }

            localData.sigmaSettings.planeDistanceSensitivity = setting.planeDistanceSensitivity;
            localData.sigmaSettings.maxStabilizedFrameNum = setting.maxStabilizedFrameNum;

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

            ReleaseTextures();
            DestroyDenoiserInstance(nrdInstanceId);
        }
    }
}