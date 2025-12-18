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
    enum DXGI_FORMAT : uint
    {
        DXGI_FORMAT_UNKNOWN = 0,
        DXGI_FORMAT_R32G32B32A32_TYPELESS = 1,
        DXGI_FORMAT_R32G32B32A32_FLOAT = 2,
        DXGI_FORMAT_R32G32B32A32_UINT = 3,
        DXGI_FORMAT_R32G32B32A32_SINT = 4,
        DXGI_FORMAT_R32G32B32_TYPELESS = 5,
        DXGI_FORMAT_R32G32B32_FLOAT = 6,
        DXGI_FORMAT_R32G32B32_UINT = 7,
        DXGI_FORMAT_R32G32B32_SINT = 8,
        DXGI_FORMAT_R16G16B16A16_TYPELESS = 9,
        DXGI_FORMAT_R16G16B16A16_FLOAT = 10,
        DXGI_FORMAT_R16G16B16A16_UNORM = 11,
        DXGI_FORMAT_R16G16B16A16_UINT = 12,
        DXGI_FORMAT_R16G16B16A16_SNORM = 13,
        DXGI_FORMAT_R16G16B16A16_SINT = 14,
        DXGI_FORMAT_R32G32_TYPELESS = 15,
        DXGI_FORMAT_R32G32_FLOAT = 16,
        DXGI_FORMAT_R32G32_UINT = 17,
        DXGI_FORMAT_R32G32_SINT = 18,
        DXGI_FORMAT_R32G8X24_TYPELESS = 19,
        DXGI_FORMAT_D32_FLOAT_S8X24_UINT = 20,
        DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS = 21,
        DXGI_FORMAT_X32_TYPELESS_G8X24_UINT = 22,
        DXGI_FORMAT_R10G10B10A2_TYPELESS = 23,
        DXGI_FORMAT_R10G10B10A2_UNORM = 24,
        DXGI_FORMAT_R10G10B10A2_UINT = 25,
        DXGI_FORMAT_R11G11B10_FLOAT = 26,
        DXGI_FORMAT_R8G8B8A8_TYPELESS = 27,
        DXGI_FORMAT_R8G8B8A8_UNORM = 28,
        DXGI_FORMAT_R8G8B8A8_UNORM_SRGB = 29,
        DXGI_FORMAT_R8G8B8A8_UINT = 30,
        DXGI_FORMAT_R8G8B8A8_SNORM = 31,
        DXGI_FORMAT_R8G8B8A8_SINT = 32,
        DXGI_FORMAT_R16G16_TYPELESS = 33,
        DXGI_FORMAT_R16G16_FLOAT = 34,
        DXGI_FORMAT_R16G16_UNORM = 35,
        DXGI_FORMAT_R16G16_UINT = 36,
        DXGI_FORMAT_R16G16_SNORM = 37,
        DXGI_FORMAT_R16G16_SINT = 38,
        DXGI_FORMAT_R32_TYPELESS = 39,
        DXGI_FORMAT_D32_FLOAT = 40,
        DXGI_FORMAT_R32_FLOAT = 41,
        DXGI_FORMAT_R32_UINT = 42,
        DXGI_FORMAT_R32_SINT = 43,
        DXGI_FORMAT_R24G8_TYPELESS = 44,
        DXGI_FORMAT_D24_UNORM_S8_UINT = 45,
        DXGI_FORMAT_R24_UNORM_X8_TYPELESS = 46,
        DXGI_FORMAT_X24_TYPELESS_G8_UINT = 47,
        DXGI_FORMAT_R8G8_TYPELESS = 48,
        DXGI_FORMAT_R8G8_UNORM = 49,
        DXGI_FORMAT_R8G8_UINT = 50,
        DXGI_FORMAT_R8G8_SNORM = 51,
        DXGI_FORMAT_R8G8_SINT = 52,
        DXGI_FORMAT_R16_TYPELESS = 53,
        DXGI_FORMAT_R16_FLOAT = 54,
        DXGI_FORMAT_D16_UNORM = 55,
        DXGI_FORMAT_R16_UNORM = 56,
        DXGI_FORMAT_R16_UINT = 57,
        DXGI_FORMAT_R16_SNORM = 58,
        DXGI_FORMAT_R16_SINT = 59,
        DXGI_FORMAT_R8_TYPELESS = 60,
        DXGI_FORMAT_R8_UNORM = 61,
        DXGI_FORMAT_R8_UINT = 62,
        DXGI_FORMAT_R8_SNORM = 63,
        DXGI_FORMAT_R8_SINT = 64,
        DXGI_FORMAT_A8_UNORM = 65,
        DXGI_FORMAT_R1_UNORM = 66,
        DXGI_FORMAT_R9G9B9E5_SHAREDEXP = 67,
        DXGI_FORMAT_R8G8_B8G8_UNORM = 68,
        DXGI_FORMAT_G8R8_G8B8_UNORM = 69,
        DXGI_FORMAT_BC1_TYPELESS = 70,
        DXGI_FORMAT_BC1_UNORM = 71,
        DXGI_FORMAT_BC1_UNORM_SRGB = 72,
        DXGI_FORMAT_BC2_TYPELESS = 73,
        DXGI_FORMAT_BC2_UNORM = 74,
        DXGI_FORMAT_BC2_UNORM_SRGB = 75,
        DXGI_FORMAT_BC3_TYPELESS = 76,
        DXGI_FORMAT_BC3_UNORM = 77,
        DXGI_FORMAT_BC3_UNORM_SRGB = 78,
        DXGI_FORMAT_BC4_TYPELESS = 79,
        DXGI_FORMAT_BC4_UNORM = 80,
        DXGI_FORMAT_BC4_SNORM = 81,
        DXGI_FORMAT_BC5_TYPELESS = 82,
        DXGI_FORMAT_BC5_UNORM = 83,
        DXGI_FORMAT_BC5_SNORM = 84,
        DXGI_FORMAT_B5G6R5_UNORM = 85,
        DXGI_FORMAT_B5G5R5A1_UNORM = 86,
        DXGI_FORMAT_B8G8R8A8_UNORM = 87,
        DXGI_FORMAT_B8G8R8X8_UNORM = 88,
        DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM = 89,
        DXGI_FORMAT_B8G8R8A8_TYPELESS = 90,
        DXGI_FORMAT_B8G8R8A8_UNORM_SRGB = 91,
        DXGI_FORMAT_B8G8R8X8_TYPELESS = 92,
        DXGI_FORMAT_B8G8R8X8_UNORM_SRGB = 93,
        DXGI_FORMAT_BC6H_TYPELESS = 94,
        DXGI_FORMAT_BC6H_UF16 = 95,
        DXGI_FORMAT_BC6H_SF16 = 96,
        DXGI_FORMAT_BC7_TYPELESS = 97,
        DXGI_FORMAT_BC7_UNORM = 98,
        DXGI_FORMAT_BC7_UNORM_SRGB = 99,
        DXGI_FORMAT_AYUV = 100,
        DXGI_FORMAT_Y410 = 101,
        DXGI_FORMAT_Y416 = 102,
        DXGI_FORMAT_NV12 = 103,
        DXGI_FORMAT_P010 = 104,
        DXGI_FORMAT_P016 = 105,
        DXGI_FORMAT_420_OPAQUE = 106,
        DXGI_FORMAT_YUY2 = 107,
        DXGI_FORMAT_Y210 = 108,
        DXGI_FORMAT_Y216 = 109,
        DXGI_FORMAT_NV11 = 110,
        DXGI_FORMAT_AI44 = 111,
        DXGI_FORMAT_IA44 = 112,
        DXGI_FORMAT_P8 = 113,
        DXGI_FORMAT_A8P8 = 114,
        DXGI_FORMAT_B4G4R4A4_UNORM = 115,

        DXGI_FORMAT_P208 = 130,
        DXGI_FORMAT_V208 = 131,
        DXGI_FORMAT_V408 = 132,


        DXGI_FORMAT_SAMPLER_FEEDBACK_MIN_MIP_OPAQUE = 189,
        DXGI_FORMAT_SAMPLER_FEEDBACK_MIP_REGION_USED_OPAQUE = 190,

        DXGI_FORMAT_A4B4G4R4_UNORM = 191,


        DXGI_FORMAT_FORCE_UINT = 0xffffffff
    }

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
            localData.normalRoughnessPointer = Ptr_NormalRoughness;
            localData.viewZPointer = Ptr_ViewZ;
            localData.penumbraPointer = Ptr_Penumbra;
            localData.shadowTranslucencyPointer = Ptr_ShadowTranslucency;
            localData.diffRadiancePointer = Ptr_DiffRadiancePointer;
            localData.outDiffRadiancePointer = Ptr_OutDiffRadiancePointer;
            localData.validationPointer = Ptr_ValidationPointer;
            
            localData.nriMv = nriMv;
            localData.nriNormalRoughness = nriNormal;
            localData.nriViewZ = nriViewZ;
            localData.nriPenumbra = nriPENUMBRA;
            localData.nriShadowTranslucency = nriSHADOW_TRANSLUCENCY;
            localData.nriDiffRadiance = nriDiffRadiance;
            localData.nriOutDiffRadiance = nriOutDiffRadiance;
            localData.nriValidation = nriValidation;
            

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