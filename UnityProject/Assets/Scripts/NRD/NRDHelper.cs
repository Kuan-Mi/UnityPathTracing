using System;
using System.Runtime.InteropServices;
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
    public class NRDHelper : IDisposable
    {
        [DllImport("RenderingPlugin")]
        private static extern int CreateDenoiserInstance();

        [DllImport("RenderingPlugin")]
        private static extern void DestroyDenoiserInstance(int id);

        [DllImport("RenderingPlugin")]
        private static extern IntPtr WrapD3D12Texture(IntPtr resource, DXGI_FORMAT format);

        [DllImport("RenderingPlugin")]
        private static extern void ReleaseTexture(IntPtr nriTex);

        [DllImport("RenderingPlugin")]
        private static extern void UpdateDenoiserResources(int instanceId, IntPtr resources, int count);

        private NativeArray<NrdResourceInput> m_ResourceCache;

        private uint FrameIndex;
        private readonly int nrdInstanceId;

        private Matrix4x4 PrevViewMatrix;
        private Matrix4x4 PrevViewProjMatrix;

        private int _prevWidth = -1;
        private int _prevHeight = -1;

        private NativeArray<FrameData> buffer;
        private const int BufferCount = 3;

        public RTHandle MvHandle;
        public RTHandle NormalRoughnessHandle;
        public RTHandle ViewZHandle;
        public RTHandle PenumbraHandle;
        public RTHandle ShadowTranslucencyHandle;
        public RTHandle DiffRadianceHandle;
        public RTHandle OutDiffRadianceHandle;
        public RTHandle ValidationHandle;


        private IntPtr Ptr_Mv;
        private IntPtr Ptr_NormalRoughness;
        private IntPtr Ptr_ViewZ;
        private IntPtr Ptr_Penumbra;
        private IntPtr Ptr_ShadowTranslucency;
        private IntPtr Ptr_DiffRadiancePointer;
        private IntPtr Ptr_OutDiffRadiancePointer;
        private IntPtr Ptr_ValidationPointer;

        private IntPtr nriMv;
        private IntPtr nriNormal;
        private IntPtr nriViewZ;
        private IntPtr nriPENUMBRA;
        private IntPtr nriSHADOW_TRANSLUCENCY;
        private IntPtr nriDiffRadiance;
        private IntPtr nriOutDiffRadiance;
        private IntPtr nriValidation;

        private PathTracingSetting setting;

        public NRDHelper(PathTracingSetting setting)
        {
            this.setting = setting;
            int instanceId = CreateDenoiserInstance();
            nrdInstanceId = instanceId;
            buffer = new NativeArray<FrameData>(BufferCount, Allocator.Persistent);
        }

        public void EnsureResources(int width, int height)
        {
            // 如果尺寸没变且资源都存在，直接返回
            if (width == _prevWidth && height == _prevHeight &&
                MvHandle != null && ViewZHandle != null)
            {

                if (FrameIndex == 1)
                {
                    UpdateResourceSnapshotInCpp();
                }
                
                return;
            }

            // 尺寸变化或初始化，先释放旧的
            ReleaseTextures();

            _prevWidth = width;
            _prevHeight = height;

            FrameIndex = 0;

            MvHandle = AllocRT("NRD_Mv", width, height, GraphicsFormat.R16G16B16A16_SFloat);
            Ptr_Mv = MvHandle.rt.GetNativeTexturePtr();

            ViewZHandle = AllocRT("NRD_ViewZ", width, height, GraphicsFormat.R32_SFloat);
            Ptr_ViewZ = ViewZHandle.rt.GetNativeTexturePtr();

            NormalRoughnessHandle =
                AllocRT("NRD_NormalRoughness", width, height, GraphicsFormat.A2B10G10R10_UNormPack32);
            Ptr_NormalRoughness = NormalRoughnessHandle.rt.GetNativeTexturePtr();

            PenumbraHandle = AllocRT("NRD_Penumbra", width, height, GraphicsFormat.R16_SFloat);
            Ptr_Penumbra = PenumbraHandle.rt.GetNativeTexturePtr();

            ShadowTranslucencyHandle = AllocRT("NRD_ShadowTranslucency", width, height, GraphicsFormat.R16_SFloat);
            Ptr_ShadowTranslucency = ShadowTranslucencyHandle.rt.GetNativeTexturePtr();

            DiffRadianceHandle = AllocRT("NRD_DiffRadiance", width, height, GraphicsFormat.R16G16B16A16_SFloat);
            Ptr_DiffRadiancePointer = DiffRadianceHandle.rt.GetNativeTexturePtr();

            OutDiffRadianceHandle = AllocRT("NRD_OutDiffRadiance", width, height, GraphicsFormat.R16G16B16A16_SFloat);
            Ptr_OutDiffRadiancePointer = OutDiffRadianceHandle.rt.GetNativeTexturePtr();

            ValidationHandle = AllocRT("NRD_Validation", width, height, GraphicsFormat.R8G8B8A8_UNorm);
            Ptr_ValidationPointer = ValidationHandle.rt.GetNativeTexturePtr();


            nriMv = WrapD3D12Texture(Ptr_Mv, DXGI_FORMAT.DXGI_FORMAT_R16G16B16A16_FLOAT);
            nriNormal = WrapD3D12Texture(Ptr_NormalRoughness, DXGI_FORMAT.DXGI_FORMAT_R10G10B10A2_UNORM);
            nriViewZ = WrapD3D12Texture(Ptr_ViewZ, DXGI_FORMAT.DXGI_FORMAT_R32_FLOAT);
            nriPENUMBRA = WrapD3D12Texture(Ptr_Penumbra, DXGI_FORMAT.DXGI_FORMAT_R16_FLOAT);
            nriSHADOW_TRANSLUCENCY = WrapD3D12Texture(Ptr_ShadowTranslucency, DXGI_FORMAT.DXGI_FORMAT_R16_FLOAT);
            nriDiffRadiance = WrapD3D12Texture(Ptr_DiffRadiancePointer, DXGI_FORMAT.DXGI_FORMAT_R16G16B16A16_FLOAT);
            nriOutDiffRadiance = WrapD3D12Texture(Ptr_OutDiffRadiancePointer, DXGI_FORMAT.DXGI_FORMAT_R16G16B16A16_FLOAT);
            nriValidation = WrapD3D12Texture(Ptr_ValidationPointer, DXGI_FORMAT.DXGI_FORMAT_R8G8B8A8_UNORM);


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

            // 定义状态 (与 C++ 保持一致)
            var srvState = new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE, layout = Layout.SHADER_RESOURCE, stageBits = 1 << 7 };
            var uavState = new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE_STORAGE, layout = Layout.SHADER_RESOURCE_STORAGE, stageBits = 1 << 10 };
            var rtState = new NriResourceState { accessBits = AccessBits.COLOR_ATTACHMENT, layout = Layout.COLOR_ATTACHMENT, stageBits = 1 << 7 };
            var commonState = new NriResourceState { accessBits = AccessBits.NONE, layout = Layout.GENERAL, stageBits = 0 };


            // Reblur/Sigma Inputs
            NrdResourceInput* ptr = (NrdResourceInput*)m_ResourceCache.GetUnsafePtr();

            ptr[idx++] = new NrdResourceInput { type = ResourceType.IN_MV, texture = nriMv, state = FrameIndex == 0 ? uavState : srvState };
            ptr[idx++] = new NrdResourceInput { type = ResourceType.IN_NORMAL_ROUGHNESS, texture = nriNormal, state = FrameIndex == 0 ? uavState : srvState  };
            ptr[idx++] = new NrdResourceInput { type = ResourceType.IN_VIEWZ, texture = nriViewZ, state = FrameIndex == 0 ? uavState : srvState  };
            ptr[idx++] = new NrdResourceInput { type = ResourceType.IN_PENUMBRA, texture = nriPENUMBRA, state = FrameIndex == 0 ? uavState : srvState  };

            ptr[idx++] = new NrdResourceInput { type = ResourceType.OUT_SHADOW_TRANSLUCENCY, texture = nriSHADOW_TRANSLUCENCY, state = uavState };

            ptr[idx++] = new NrdResourceInput { type = ResourceType.IN_DIFF_RADIANCE_HITDIST, texture = nriDiffRadiance, state = FrameIndex == 0 ? rtState : srvState };
            ptr[idx++] = new NrdResourceInput { type = ResourceType.OUT_DIFF_RADIANCE_HITDIST, texture = nriOutDiffRadiance, state = FrameIndex == 0 ? rtState : uavState };

            ptr[idx++] = new NrdResourceInput { type = ResourceType.OUT_VALIDATION, texture = nriValidation, state = commonState };

            // 发送到 C++
            UpdateDenoiserResources(nrdInstanceId, (IntPtr)ptr, idx);

            Debug.Log($"[NRD] Updated resources pointer to C++. Count: {idx}");
        }

        private RTHandle AllocRT(string name, int w, int h, GraphicsFormat format)
        {
            var desc = new RenderTextureDescriptor(w, h, format, 0)
            {
                enableRandomWrite = true,
                useMipMap = false,
                msaaSamples = 1,
                sRGB = false,
            };

            var rt = new RenderTexture(desc)
            {
                name = name,
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };
            rt.Create();

            return RTHandles.Alloc(rt);
        }

        private void ReleaseTextures()
        {
            ReleaseTexture(nriMv);
            ReleaseTexture(nriNormal);
            ReleaseTexture(nriViewZ);
            ReleaseTexture(nriPENUMBRA);
            ReleaseTexture(nriSHADOW_TRANSLUCENCY);
            ReleaseTexture(nriDiffRadiance);
            ReleaseTexture(nriOutDiffRadiance);
            ReleaseTexture(nriValidation);

            nriMv = IntPtr.Zero;
            nriNormal = IntPtr.Zero;
            nriViewZ = IntPtr.Zero;
            nriPENUMBRA = IntPtr.Zero;
            nriSHADOW_TRANSLUCENCY = IntPtr.Zero;
            nriDiffRadiance = IntPtr.Zero;
            nriOutDiffRadiance = IntPtr.Zero;
            nriValidation = IntPtr.Zero;


            RTHandles.Release(MvHandle);
            MvHandle = null;
            RTHandles.Release(ViewZHandle);
            ViewZHandle = null;
            RTHandles.Release(NormalRoughnessHandle);
            NormalRoughnessHandle = null;
            RTHandles.Release(PenumbraHandle);
            PenumbraHandle = null;
            RTHandles.Release(ShadowTranslucencyHandle);
            ShadowTranslucencyHandle = null;
            RTHandles.Release(DiffRadianceHandle);
            DiffRadianceHandle = null;
            RTHandles.Release(OutDiffRadianceHandle);
            OutDiffRadianceHandle = null;
            RTHandles.Release(ValidationHandle);
            ValidationHandle = null;

            // 指针置空
            Ptr_Mv = IntPtr.Zero;
            Ptr_ViewZ = IntPtr.Zero;
            Ptr_NormalRoughness = IntPtr.Zero;
            Ptr_Penumbra = IntPtr.Zero;
            Ptr_ShadowTranslucency = IntPtr.Zero;
            Ptr_DiffRadiancePointer = IntPtr.Zero;
            Ptr_OutDiffRadiancePointer = IntPtr.Zero;
            Ptr_ValidationPointer = IntPtr.Zero;
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

            localData.commonSettings.motionVectorScale = new float3(1.0f / w, 1.0f / h, 1.0f);
            localData.commonSettings.isMotionVectorInWorldSpace = false;

            localData.commonSettings.accumulationMode = AccumulationMode.CONTINUE;
            localData.commonSettings.frameIndex = FrameIndex;

            // --- Sigma 设置 (光照) ---
            // Sigma 需要指向光源的方向 (normalized)
            localData.sigmaSettings.lightDirection = dirToLight;

            // --- 其他设置 ---
            localData.mvPointer = Ptr_Mv;
            localData.normalRoughnessPointer = Ptr_NormalRoughness;
            localData.viewZPointer = Ptr_ViewZ;
            localData.penumbraPointer = Ptr_Penumbra;
            localData.shadowTranslucencyPointer = Ptr_ShadowTranslucency;
            localData.diffRadiancePointer = Ptr_DiffRadiancePointer;
            localData.outDiffRadiancePointer = Ptr_OutDiffRadiancePointer;
            localData.validationPointer = Ptr_ValidationPointer;

            // Debug.Log("Record Frame Index: " + m_FrameIndex);

            // 4. 更新历史状态
            PrevViewProjMatrix = viewProj;
            PrevViewMatrix = worldToView;

            localData.instanceId = nrdInstanceId;

            localData.width = w;
            localData.height = h;

            //  Common 设置

            if (setting.useOverriddenCommonSettings)
            {
                localData.commonSettings.viewToClipMatrix = setting.viewToClipMatrix;
                localData.commonSettings.viewToClipMatrixPrev = setting.viewToClipMatrixPrev;
                localData.commonSettings.worldToViewMatrix = setting.worldToViewMatrix;
                localData.commonSettings.worldToViewMatrixPrev = setting.worldToViewMatrixPrev;
                localData.commonSettings.motionVectorScale = setting.motionVectorScale;
            }

            localData.commonSettings.denoisingRange = setting.denoisingRange;
            localData.commonSettings.disocclusionThreshold = setting.disocclusionThreshold;
            localData.commonSettings.disocclusionThresholdAlternate = setting.disocclusionThresholdAlternate;
            localData.commonSettings.splitScreen = setting.splitScreen;

            // localData.commonSettings.isMotionVectorInWorldSpace = setting.isMotionVectorInWorldSpace;
            localData.commonSettings.isHistoryConfidenceAvailable = setting.isHistoryConfidenceAvailable;
            localData.commonSettings.isDisocclusionThresholdMixAvailable = setting.isDisocclusionThresholdMixAvailable;
            localData.commonSettings.isBaseColorMetalnessAvailable = setting.isBaseColorMetalnessAvailable;
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