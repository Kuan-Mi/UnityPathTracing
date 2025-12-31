using System;
using System.Runtime.InteropServices;
using Nrd;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;
using static PathTracing.ShaderIDs;
using static PathTracing.PathTracingUtils;

namespace PathTracing
{
    public class PathTracingPassSingle : ScriptableRenderPass
    {
        private static readonly int GInOutMv = Shader.PropertyToID("gInOut_Mv");
        public RayTracingShader OpaqueTs;
        public RayTracingShader TransparentTs;
        public ComputeShader CompositionCs;
        public ComputeShader TaaCs;
        public Material BiltMaterial;

        public ComputeShader SharcResolveCs;
        public RayTracingShader SharcUpdateTs;

        public GraphicsBuffer HashEntriesBuffer;
        public GraphicsBuffer AccumulationBuffer;
        public GraphicsBuffer ResolvedBuffer;

        public RayTracingAccelerationStructure AccelerationStructure;
        public NRDDenoiser NrdDenoiser;

        public GraphicsBuffer ScramblingRanking;
        public GraphicsBuffer Sobol;

        private readonly PathTracingSetting m_Settings;
        private readonly GraphicsBuffer _pathTracingSettingsBuffer;

        [DllImport("RenderingPlugin")]
        private static extern IntPtr GetRenderEventAndDataFunc();

        class PassData
        {
            internal TextureHandle CameraTexture;

            internal GraphicsBuffer ScramblingRanking;
            internal GraphicsBuffer Sobol;

            internal TextureHandle OutputTexture;

            internal TextureHandle Mv;
            internal TextureHandle ViewZ;
            internal TextureHandle NormalRoughness;
            internal TextureHandle BaseColorMetalness;

            internal TextureHandle DirectLighting;
            internal TextureHandle DirectEmission;

            internal TextureHandle Penumbra;
            internal TextureHandle Diff;
            internal TextureHandle Spec;

            internal TextureHandle ShadowTranslucency;
            internal TextureHandle DenoisedDiff;
            internal TextureHandle DenoisedSpec;
            internal TextureHandle Validation;

            internal TextureHandle ComposedDiff;
            internal TextureHandle ComposedSpecViewZ;
            internal TextureHandle Composed;

            internal TextureHandle TaaHistory;
            internal TextureHandle TaaHistoryPrev;
            internal TextureHandle PsrThroughput;

            internal RayTracingShader OpaqueTs;
            internal RayTracingShader TransparentTs;
            internal ComputeShader CompositionCs;
            internal ComputeShader TaaCs;
            internal Material BlitMaterial;
            internal uint Width;
            internal uint Height;

            internal GlobalConstants GlobalConstants;
            internal GraphicsBuffer ConstantBuffer;
            internal IntPtr NrdDataPtr;
            internal PathTracingSetting Setting;


            internal ComputeShader SharcResolveCs;
            internal RayTracingShader SharcUpdateTs;

            internal GraphicsBuffer HashEntriesBuffer;
            internal GraphicsBuffer AccumulationBuffer;
            internal GraphicsBuffer ResolvedBuffer;
        }

        public PathTracingPassSingle(PathTracingSetting setting)
        {
            m_Settings = setting;
            _pathTracingSettingsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Constant, 1, Marshal.SizeOf<GlobalConstants>());
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            int threadGroupX = Mathf.CeilToInt(data.Width / 16.0f);
            int threadGroupY = Mathf.CeilToInt(data.Height / 16.0f);
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            natCmd.SetBufferData(data.ConstantBuffer, new[] { data.GlobalConstants });

            // Sharc update
            natCmd.SetRayTracingShaderPass(data.SharcUpdateTs, "Test2");
            natCmd.SetRayTracingConstantBufferParam(data.SharcUpdateTs, paramsID, data.ConstantBuffer, 0, data.ConstantBuffer.stride);

            natCmd.SetRayTracingBufferParam(data.SharcUpdateTs, g_HashEntriesID, data.HashEntriesBuffer);
            natCmd.SetRayTracingBufferParam(data.SharcUpdateTs, g_AccumulationBufferID, data.AccumulationBuffer);
            natCmd.SetRayTracingBufferParam(data.SharcUpdateTs, g_ResolvedBufferID, data.ResolvedBuffer);

            natCmd.DispatchRays(data.SharcUpdateTs, "MainRayGenShader", data.Width, data.Height, 1);

            // Sharc resolve
            natCmd.SetComputeConstantBufferParam(data.SharcResolveCs, paramsID, data.ConstantBuffer, 0, data.ConstantBuffer.stride);
            natCmd.SetComputeBufferParam(data.SharcResolveCs, 0, g_HashEntriesID, data.HashEntriesBuffer);
            natCmd.SetComputeBufferParam(data.SharcResolveCs, 0, g_AccumulationBufferID, data.AccumulationBuffer);
            natCmd.SetComputeBufferParam(data.SharcResolveCs, 0, g_ResolvedBufferID, data.ResolvedBuffer);

            natCmd.DispatchCompute(data.SharcResolveCs, 0, threadGroupX, threadGroupY, 1);

            // 不透明
            natCmd.SetRayTracingShaderPass(data.OpaqueTs, "Test2");
            natCmd.SetRayTracingConstantBufferParam(data.OpaqueTs, paramsID, data.ConstantBuffer, 0, data.ConstantBuffer.stride);

            natCmd.SetRayTracingBufferParam(data.OpaqueTs, g_ScramblingRankingID, data.ScramblingRanking);
            natCmd.SetRayTracingBufferParam(data.OpaqueTs, g_SobolID, data.Sobol);

            natCmd.SetRayTracingBufferParam(data.OpaqueTs, g_HashEntriesID, data.HashEntriesBuffer);
            natCmd.SetRayTracingBufferParam(data.OpaqueTs, g_AccumulationBufferID, data.AccumulationBuffer);
            natCmd.SetRayTracingBufferParam(data.OpaqueTs, g_ResolvedBufferID, data.ResolvedBuffer);

            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_OutputID, data.OutputTexture);

            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_MvID, data.Mv);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_ViewZID, data.ViewZ);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_Normal_RoughnessID, data.NormalRoughness);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_BaseColor_MetalnessID, data.BaseColorMetalness);

            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_DirectLightingID, data.DirectLighting);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_DirectEmissionID, data.DirectEmission);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_PsrThroughputID, data.PsrThroughput);

            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_ShadowDataID, data.Penumbra);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_DiffID, data.Diff);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_SpecID, data.Spec);

            natCmd.SetRayTracingTextureParam(data.OpaqueTs, gIn_PrevComposedDiffID, data.ComposedDiff);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, gIn_PrevComposedSpec_PrevViewZID, data.ComposedSpecViewZ);

            natCmd.DispatchRays(data.OpaqueTs, "MainRayGenShader", data.Width, data.Height, 1);

            // NRD降噪
            natCmd.IssuePluginEventAndData(GetRenderEventAndDataFunc(), 1, data.NrdDataPtr);

            // 合成
            natCmd.SetComputeConstantBufferParam(data.CompositionCs, paramsID, data.ConstantBuffer, 0, data.ConstantBuffer.stride);
            natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_ViewZID, data.ViewZ);
            natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_Normal_RoughnessID, data.NormalRoughness);
            natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_BaseColor_MetalnessID, data.BaseColorMetalness);
            natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_DirectLightingID, data.DirectLighting);
            natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_DirectEmissionID, data.DirectEmission);
            natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_ShadowID, data.ShadowTranslucency);
            natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_DiffID, data.DenoisedDiff);
            natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_SpecID, data.DenoisedSpec);
            natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_PsrThroughputID, data.PsrThroughput);
            natCmd.SetComputeTextureParam(data.CompositionCs, 0, gOut_ComposedDiffID, data.ComposedDiff);
            natCmd.SetComputeTextureParam(data.CompositionCs, 0, gOut_ComposedSpec_ViewZID, data.ComposedSpecViewZ);

            natCmd.DispatchCompute(data.CompositionCs, 0, threadGroupX, threadGroupY, 1);

            // 透明
            natCmd.SetRayTracingShaderPass(data.TransparentTs, "Test2");
            natCmd.SetRayTracingConstantBufferParam(data.TransparentTs, paramsID, data.ConstantBuffer, 0, data.ConstantBuffer.stride);

            natCmd.SetRayTracingBufferParam(data.TransparentTs, g_HashEntriesID, data.HashEntriesBuffer);
            natCmd.SetRayTracingBufferParam(data.TransparentTs, g_AccumulationBufferID, data.AccumulationBuffer);
            natCmd.SetRayTracingBufferParam(data.TransparentTs, g_ResolvedBufferID, data.ResolvedBuffer);


            natCmd.SetRayTracingTextureParam(data.TransparentTs, gIn_ComposedDiffID, data.ComposedDiff);
            natCmd.SetRayTracingTextureParam(data.TransparentTs, gIn_ComposedSpec_ViewZID, data.ComposedSpecViewZ);
            natCmd.SetRayTracingTextureParam(data.TransparentTs, gOut_ComposedID, data.Composed);
            natCmd.SetRayTracingTextureParam(data.TransparentTs, GInOutMv, data.Mv);

            natCmd.DispatchRays(data.TransparentTs, "MainRayGenShader", data.Width, data.Height, 1);

            // TAA
            var isEven = (data.GlobalConstants.gFrameIndex & 1) == 0;
            var taaSrc = isEven ? data.TaaHistoryPrev : data.TaaHistory;
            var taaDst = isEven ? data.TaaHistory : data.TaaHistoryPrev;

            natCmd.SetComputeConstantBufferParam(data.TaaCs, paramsID, data.ConstantBuffer, 0, data.ConstantBuffer.stride);
            natCmd.SetComputeTextureParam(data.TaaCs, 0, gIn_MvID, data.Mv);
            natCmd.SetComputeTextureParam(data.TaaCs, 0, gIn_ComposedID, data.Composed);
            natCmd.SetComputeTextureParam(data.TaaCs, 0, gIn_HistoryID, taaSrc);
            natCmd.SetComputeTextureParam(data.TaaCs, 0, gOut_ResultID, taaDst);
            natCmd.SetComputeTextureParam(data.TaaCs, 0, gOut_DebugID, data.OutputTexture);
            natCmd.DispatchCompute(data.TaaCs, 0, threadGroupX, threadGroupY, 1);

            // 显示输出
            natCmd.SetRenderTarget(data.CameraTexture);

            switch (data.Setting.showMode)
            {
                case ShowMode.None:
                    break;
                case ShowMode.BaseColor:
                    Blitter.BlitTexture(natCmd, data.BaseColorMetalness, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.showOut);
                    break;
                case ShowMode.Metalness:
                    Blitter.BlitTexture(natCmd, data.BaseColorMetalness, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.showAlpha);
                    break;
                case ShowMode.Normal:
                    Blitter.BlitTexture(natCmd, data.NormalRoughness, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.ShowNormal);
                    break;
                case ShowMode.Roughness:
                    Blitter.BlitTexture(natCmd, data.NormalRoughness, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.ShowRoughness);
                    break;
                case ShowMode.Shadow:
                    Blitter.BlitTexture(natCmd, data.ShadowTranslucency, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.showShadow);
                    break;
                case ShowMode.Diffuse:
                    Blitter.BlitTexture(natCmd, data.DenoisedDiff, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.ShowRadiance);
                    break;
                case ShowMode.Specular:
                    Blitter.BlitTexture(natCmd, data.DenoisedSpec, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.ShowRadiance);
                    break;
                case ShowMode.DirectLight:
                    Blitter.BlitTexture(natCmd, data.DirectLighting, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.showOut);
                    break;
                case ShowMode.Emissive:
                    Blitter.BlitTexture(natCmd, data.DirectEmission, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.showOut);
                    break;
                case ShowMode.Out:
                    Blitter.BlitTexture(natCmd, data.OutputTexture, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.showOut);
                    break;
                case ShowMode.ComposedDiff:
                    Blitter.BlitTexture(natCmd, data.ComposedDiff, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.showOut);
                    break;
                case ShowMode.ComposedSpec:
                    Blitter.BlitTexture(natCmd, data.ComposedSpecViewZ, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.showOut);
                    break;
                case ShowMode.Taa:
                    Blitter.BlitTexture(natCmd, taaDst, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.showAlpha);
                    break;
                case ShowMode.Final:
                    if (data.BlitMaterial == null)
                    {
                        Debug.LogError("BlitMaterial is null");
                    }

                    Blitter.BlitTexture(natCmd, taaDst, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.showOut);
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

            if (data.Setting.showMV)
            {
                Blitter.BlitTexture(natCmd, data.Mv, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.showMv);
            }

            if (data.Setting.showValidation)
            {
                Blitter.BlitTexture(natCmd, data.Validation, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.showValidation);
            }
        }

        uint GetMaxAccumulatedFrameNum(float accumulationTime, float fps)
        {
            return (uint)(accumulationTime * fps + 0.5f);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            var cameraData = frameData.Get<UniversalCameraData>();

            // 获取主光源方向
            var universalLightData = frameData.Get<UniversalLightData>();
            var lightData = universalLightData;
            var mainLight = lightData.mainLightIndex >= 0 ? lightData.visibleLights[lightData.mainLightIndex] : default;
            var mat = mainLight.localToWorldMatrix;
            Vector3 lightForward = mat.GetColumn(2);

            if (cameraData.camera.cameraType != CameraType.Game && cameraData.camera.cameraType != CameraType.SceneView)
            {
                return;
            }


            var resourceData = frameData.Get<UniversalResourceData>();

            NrdDenoiser.EnsureResources(cameraData.camera.pixelWidth, cameraData.camera.pixelHeight);

            Shader.SetGlobalRayTracingAccelerationStructure(g_AccelStructID, AccelerationStructure);

            using var builder = renderGraph.AddUnsafePass<PassData>("Path Tracing Pass", out var passData);

            passData.OpaqueTs = OpaqueTs;
            passData.TransparentTs = TransparentTs;
            passData.CompositionCs = CompositionCs;
            passData.TaaCs = TaaCs;
            passData.BlitMaterial = BiltMaterial;

            passData.SharcResolveCs = SharcResolveCs;
            passData.SharcUpdateTs = SharcUpdateTs;
            passData.AccumulationBuffer = AccumulationBuffer;
            passData.HashEntriesBuffer = HashEntriesBuffer;
            passData.ResolvedBuffer = ResolvedBuffer;

            var gSunDirection = -lightForward;
            var up = new Vector3(0, 1, 0);
            var gSunBasisX = math.normalize(math.cross(new float3(up.x, up.y, up.z), new float3(gSunDirection.x, gSunDirection.y, gSunDirection.z)));
            var gSunBasisY = math.normalize(math.cross(new float3(gSunDirection.x, gSunDirection.y, gSunDirection.z), gSunBasisX));

            var cam = cameraData.camera;
            var m11 = cam.projectionMatrix.m11;
            var rectH = cam.pixelHeight;
            var rectW = cam.pixelWidth;
            var rectSize = new float2(rectW, rectH);
            var invRectSize = new float2(1.0f / rectW, 1.0f / rectH);

            passData.NrdDataPtr = NrdDenoiser.GetInteropDataPtr(cam, gSunDirection);

            var verticalFieldOfView = cam.fieldOfView;
            var aspectRatio = (float)rectW / rectH;
            var horizontalFieldOfView = Mathf.Atan(Mathf.Tan(Mathf.Deg2Rad * verticalFieldOfView * 0.5f) * aspectRatio) * 2 * Mathf.Rad2Deg;


            float emissionIntensity = m_Settings.emissionIntensity * (m_Settings.emission ? 1.0f : 0.0f);

            float ACCUMULATION_TIME = 0.5f;
            int MAX_HISTORY_FRAME_NUM = 60;

            float fps = 1000.0f / Mathf.Max(Time.deltaTime * 1000.0f, 0.0001f);
            fps = math.min(fps, 121.0f);

            // Debug.Log(fps);

            float resetHistoryFactor = 1.0f;


            float otherMaxAccumulatedFrameNum = GetMaxAccumulatedFrameNum(ACCUMULATION_TIME, fps);
            otherMaxAccumulatedFrameNum = math.min(otherMaxAccumulatedFrameNum, (MAX_HISTORY_FRAME_NUM));
            otherMaxAccumulatedFrameNum *= resetHistoryFactor;


            uint sharcMaxAccumulatedFrameNum = (uint)(otherMaxAccumulatedFrameNum * (m_Settings.boost ? 0.667f : 1.0f) + 0.5f);
            float taaMaxAccumulatedFrameNum = otherMaxAccumulatedFrameNum * 0.5f;
            float prevFrameMaxAccumulatedFrameNum = otherMaxAccumulatedFrameNum * 0.3f;


            float minProbability = 0.0f;
            if (m_Settings.tracingMode == RESOLUTION.RESOLUTION_FULL_PROBABILISTIC)
            {
                HitDistanceReconstructionMode mode = HitDistanceReconstructionMode.OFF;
                if (m_Settings.denoiser == DenoiserType.DENOISER_REBLUR)
                    mode = HitDistanceReconstructionMode.OFF;
                //     mode = m_ReblurSettings.hitDistanceReconstructionMode;
                // else if (m_Settings.denoiser == DenoiserType.DENOISER_RELAX)
                //     mode = m_RelaxSettings.hitDistanceReconstructionMode;

                // Min / max allowed probability to guarantee a sample in 3x3 or 5x5 area - https://godbolt.org/z/YGYo1rjnM
                if (mode == HitDistanceReconstructionMode.AREA_3X3)
                    minProbability = 1.0f / 4.0f;
                else if (mode == HitDistanceReconstructionMode.AREA_5X5)
                    minProbability = 1.0f / 16.0f;
            }


            var globalConstants = new GlobalConstants
            {
                gViewToWorld = NrdDenoiser.worldToView.inverse,
                gViewToClip = NrdDenoiser.viewToClip,
                gWorldToView = NrdDenoiser.worldToView,
                gWorldToViewPrev = NrdDenoiser.prevWorldToView,
                gWorldToClip = NrdDenoiser.worldToClip,
                gWorldToClipPrev = NrdDenoiser.prevWorldToClip,

                gHitDistParams = new float4(3, 0.1f, 20, -25),
                gCameraFrustum = GetNrdFrustum(cameraData.camera),
                gSunBasisX = new float4(gSunBasisX.x, gSunBasisX.y, gSunBasisX.z, 0),
                gSunBasisY = new float4(gSunBasisY.x, gSunBasisY.y, gSunBasisY.z, 0),
                gSunDirection = new float4(gSunDirection.x, gSunDirection.y, gSunDirection.z, 0),
                gCameraGlobalPos = new float4(NrdDenoiser.camPos, 1),
                gCameraGlobalPosPrev = new float4(NrdDenoiser.prevCamPos, 0),
                gViewDirection = new float4(cam.transform.forward, 0),
                gHairBaseColor = new float4(0.1f, 0.1f, 0.1f, 1.0f),

                gHairBetas = new float2(0.25f, 0.3f),
                gOutputSize = rectSize,
                gRenderSize = rectSize,
                gRectSize = rectSize,
                gInvOutputSize = invRectSize,
                gInvRenderSize = invRectSize,
                gInvRectSize = invRectSize,
                gRectSizePrev = rectSize,
                gJitter = (m_Settings.cameraJitter ? NrdDenoiser.ViewportJitter : 0f) / rectSize,

                gEmissionIntensity = emissionIntensity,
                gNearZ = -cam.nearClipPlane,
                gSeparator = m_Settings.splitScreen,
                gRoughnessOverride = 0,
                gMetalnessOverride = 0,
                gUnitToMetersMultiplier = 1.0f,
                gTanSunAngularRadius = math.tan(math.radians(m_Settings.sunAngularDiameter * 0.5f)),
                gTanPixelAngularRadius = math.tan(0.5f * math.radians(horizontalFieldOfView) / cam.pixelWidth),
                gDebug = 0,
                gPrevFrameConfidence = (m_Settings.usePrevFrame && !m_Settings.RR) ? prevFrameMaxAccumulatedFrameNum / (1.0f + prevFrameMaxAccumulatedFrameNum) : 0.0f,
                gUnproject = 1.0f / (0.5f * rectH * m11),
                gAperture = m_Settings.dofAperture * 0.01f,
                gFocalDistance = m_Settings.dofFocalDistance,
                gFocalLength = (0.5f * (35.0f * 0.001f)) / math.tan(math.radians(horizontalFieldOfView * 0.5f)),
                gTAA = (m_Settings.denoiser != DenoiserType.DENOISER_REFERENCE && m_Settings.TAA) ? 1.0f / (1.0f + taaMaxAccumulatedFrameNum) : 1.0f,
                gHdrScale = 1.0f,
                gExposure = m_Settings.exposure,
                gMipBias = m_Settings.mipBias,
                gOrthoMode = cam.orthographic ? 1.0f : 0f,
                gIndirectDiffuse = m_Settings.indirectDiffuse ? 1.0f : 0.0f,
                gIndirectSpecular = m_Settings.indirectSpecular ? 1.0f : 0.0f,
                gMinProbability = minProbability,

                gSharcMaxAccumulatedFrameNum = sharcMaxAccumulatedFrameNum,
                gDenoiserType = (uint)m_Settings.denoiser,
                gDisableShadowsAndEnableImportanceSampling = m_Settings.importanceSampling ? 1u : 0u,
                gFrameIndex = (uint)Time.frameCount,
                gForcedMaterial = 0,
                gUseNormalMap = 1,
                gBounceNum = m_Settings.bounceNum,
                gResolve = 1,
                gValidation = 1,
                gSR = (m_Settings.SR && !m_Settings.RR) ? 1u : 0u,
                gRR = m_Settings.RR ? 1u : 0,
                gIsSrgb = 0,
                gOnScreen = 0,
                gTracingMode = m_Settings.RR ? (uint)RESOLUTION.RESOLUTION_FULL_PROBABILISTIC : (uint)m_Settings.tracingMode,
                gSampleNum = m_Settings.rpp,
                gPSR = m_Settings.psr ? (uint)1 : 0,
                gSHARC = m_Settings.SHARC ? (uint)1 : 0,
                gTrimLobe = m_Settings.specularLobeTrimming ? 1u : 0,
            };

            // Debug.Log(globalConstants.ToString());

            var textureDesc = resourceData.activeColorTexture.GetDescriptor(renderGraph);
            textureDesc.enableRandomWrite = true;
            textureDesc.depthBufferBits = 0;
            textureDesc.clearBuffer = false;
            textureDesc.discardBuffer = false;
            CreateTextureHandle(renderGraph, passData, textureDesc, builder);

            passData.GlobalConstants = globalConstants;
            passData.CameraTexture = resourceData.activeColorTexture;
            passData.Width = (uint)textureDesc.width;
            passData.Height = (uint)textureDesc.height;
            passData.ConstantBuffer = _pathTracingSettingsBuffer;
            passData.Setting = m_Settings;
            passData.ScramblingRanking = ScramblingRanking;
            passData.Sobol = Sobol;

            builder.UseTexture(passData.CameraTexture, AccessFlags.Write);

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }

        private void CreateTextureHandle(RenderGraph renderGraph, PassData passData, TextureDesc textureDesc, IUnsafeRenderGraphBuilder builder)
        {
            passData.OutputTexture = CreateTex(textureDesc, renderGraph, "PathTracingOutput", GraphicsFormat.R16G16B16A16_SFloat);

            passData.Mv = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_MV));
            passData.ViewZ = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_VIEWZ));
            passData.NormalRoughness = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_NORMAL_ROUGHNESS));

            passData.BaseColorMetalness = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_BASECOLOR_METALNESS));
            passData.DirectLighting = CreateTex(textureDesc, renderGraph, "DirectLighting", GraphicsFormat.B10G11R11_UFloatPack32);
            passData.DirectEmission = CreateTex(textureDesc, renderGraph, "DirectEmission", GraphicsFormat.B10G11R11_UFloatPack32);

            passData.Penumbra = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_PENUMBRA));
            passData.Diff = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_DIFF_RADIANCE_HITDIST));
            passData.Spec = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_SPEC_RADIANCE_HITDIST));

            // 输出
            passData.ShadowTranslucency = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.OUT_SHADOW_TRANSLUCENCY));
            passData.DenoisedDiff = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.OUT_DIFF_RADIANCE_HITDIST));
            passData.DenoisedSpec = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.OUT_SPEC_RADIANCE_HITDIST));
            passData.Validation = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.OUT_VALIDATION));

            passData.ComposedDiff = CreateTex(textureDesc, renderGraph, "ComposedDiff", GraphicsFormat.R16G16B16A16_SFloat);
            passData.ComposedSpecViewZ = CreateTex(textureDesc, renderGraph, "ComposedSpec_ViewZ", GraphicsFormat.R16G16B16A16_SFloat);

            passData.Composed = CreateTex(textureDesc, renderGraph, "Composed", GraphicsFormat.R16G16B16A16_SFloat);

            passData.TaaHistory = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.TaaHistory));
            passData.TaaHistoryPrev = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.TaaHistoryPrev));
            passData.PsrThroughput = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.PsrThroughput));

            builder.UseTexture(passData.OutputTexture, AccessFlags.ReadWrite);

            builder.UseTexture(passData.Mv, AccessFlags.ReadWrite);
            builder.UseTexture(passData.ViewZ, AccessFlags.ReadWrite);
            builder.UseTexture(passData.NormalRoughness, AccessFlags.ReadWrite);
            builder.UseTexture(passData.BaseColorMetalness, AccessFlags.ReadWrite);

            builder.UseTexture(passData.DirectLighting, AccessFlags.ReadWrite);
            builder.UseTexture(passData.DirectEmission, AccessFlags.ReadWrite);

            builder.UseTexture(passData.Penumbra, AccessFlags.ReadWrite);
            builder.UseTexture(passData.Diff, AccessFlags.ReadWrite);
            builder.UseTexture(passData.Spec, AccessFlags.ReadWrite);

            // 输出
            builder.UseTexture(passData.ShadowTranslucency, AccessFlags.ReadWrite);
            builder.UseTexture(passData.DenoisedDiff, AccessFlags.ReadWrite);
            builder.UseTexture(passData.DenoisedSpec, AccessFlags.ReadWrite);
            builder.UseTexture(passData.Validation, AccessFlags.ReadWrite);

            builder.UseTexture(passData.ComposedDiff, AccessFlags.ReadWrite);
            builder.UseTexture(passData.ComposedSpecViewZ, AccessFlags.ReadWrite);
            builder.UseTexture(passData.Composed, AccessFlags.ReadWrite);

            builder.UseTexture(passData.TaaHistory, AccessFlags.ReadWrite);
            builder.UseTexture(passData.TaaHistoryPrev, AccessFlags.ReadWrite);
            builder.UseTexture(passData.PsrThroughput, AccessFlags.ReadWrite);
        }

        private TextureHandle CreateTex(TextureDesc textureDesc, RenderGraph renderGraph, string name, GraphicsFormat format)
        {
            textureDesc.format = format;
            textureDesc.name = name;
            return renderGraph.CreateTexture(textureDesc);
        }

        public void Dispose()
        {
            _pathTracingSettingsBuffer?.Release();
        }
    }
}