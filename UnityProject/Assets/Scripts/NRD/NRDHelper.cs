using System;
using System.Runtime.InteropServices;
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

        private uint FrameIndex;
        private readonly int nrdInstanceId;

        private Matrix4x4 PrevViewMatrix;
        private Matrix4x4 PrevViewProjMatrix;

        private int _prevWidth = -1;
        private int _prevHeight = -1;

        private NativeArray<FrameData> buffer;
        private const int BufferCount = 3;

        public RTHandle MvHandle;
        public RTHandle ViewZHandle;
        public RTHandle NormalRoughnessHandle;
        public RTHandle PenumbraHandle;
        public RTHandle ShadowTranslucencyHandle;
        public RTHandle DiffRadianceHandle;
        public RTHandle OutDiffRadianceHandle;
        public RTHandle ValidationHandle;


        private IntPtr Ptr_Mv;
        private IntPtr Ptr_ViewZ;
        private IntPtr Ptr_NormalRoughness;
        private IntPtr Ptr_Penumbra;
        private IntPtr Ptr_ShadowTranslucency;
        private IntPtr Ptr_DiffRadiancePointer;
        private IntPtr Ptr_OutDiffRadiancePointer;
        private IntPtr Ptr_ValidationPointer;

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
        }

        private RTHandle AllocRT(string name, int w, int h, GraphicsFormat format)
        {
            var desc = new RenderTextureDescriptor(w, h, format, 0)
            {
                enableRandomWrite = true,
                useMipMap = false,
                msaaSamples = 1,
                sRGB = false
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


        private unsafe FrameData GetData(Camera m_Camera, Vector3 dirToLight)
        {
            Matrix4x4 proj = GL.GetGPUProjectionMatrix(m_Camera.projectionMatrix, false);

            Matrix4x4 worldToView = m_Camera.worldToCameraMatrix;
            Matrix4x4 viewProj = m_Camera.projectionMatrix;

            viewProj = proj;

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
            ushort w = (ushort)m_Camera.pixelWidth;
            ushort h = (ushort)m_Camera.pixelHeight;

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
            localData.viewZPointer = Ptr_ViewZ;
            localData.normalRoughnessPointer = Ptr_NormalRoughness;
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

        public IntPtr GetInteropDataPtr(Camera m_Camera, Vector3 dirToLight)
        {
            var index = (int)(FrameIndex % BufferCount);
            buffer[index] = GetData(m_Camera, dirToLight);
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