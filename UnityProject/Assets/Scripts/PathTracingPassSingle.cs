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
        public RayTracingShader opaqueTracingShader;
        public RayTracingShader transparentTracingShader;
        public ComputeShader compositionComputeShader;
        public ComputeShader taaComputeShader;
        public Material biltMaterial;


        public RayTracingAccelerationStructure accelerationStructure;
        private PathTracingSetting _settings;
        public NRDDenoiser NrdDenoiser;


        public ComputeBuffer scramblingRanking;
        public ComputeBuffer sobol;

        private ComputeBuffer pathTracingSettingsBuffer;

        [DllImport("RenderingPlugin")]
        public static extern IntPtr GetRenderEventAndDataFunc();

        class PassData
        {
            internal TextureHandle cameraTexture;

            internal ComputeBuffer scramblingRanking;
            internal ComputeBuffer sobol;

            internal TextureHandle outputTexture;

            internal TextureHandle Mv;
            internal TextureHandle ViewZ;
            internal TextureHandle Normal_Roughness;
            internal TextureHandle BaseColor_Metalness;

            internal TextureHandle DirectLighting;
            internal TextureHandle DirectEmission;

            internal TextureHandle Penumbra;
            internal TextureHandle Diff;
            internal TextureHandle Spec;

            internal TextureHandle Shadow_Translucency;
            internal TextureHandle DenoisedDiff;
            internal TextureHandle DenoisedSpec;
            internal TextureHandle Validation;

            internal TextureHandle ComposedDiff;
            internal TextureHandle ComposedSpec_ViewZ;
            internal TextureHandle Composed;

            internal TextureHandle TaaHistory;
            internal TextureHandle TaaHistoryPrev;

            internal RayTracingShader opaqueTracingShader;
            internal RayTracingShader transparentTracingShader;
            internal ComputeShader compositionComputeShader;
            internal ComputeShader taaComputeShader;
            internal Material blitMaterial;
            internal Camera cam;
            internal int width;
            internal int height;

            internal GlobalConstants globalConstants;
            internal ComputeBuffer pathTracingSettingsBuffer;
            internal IntPtr dataPtr;

            internal PathTracingSetting _setting;
            internal bool isEven;
        }

        public PathTracingPassSingle(PathTracingSetting setting)
        {
            _settings = setting;
            pathTracingSettingsBuffer = new ComputeBuffer(1, Marshal.SizeOf<GlobalConstants>(), ComputeBufferType.Constant);
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            GlobalConstants[] settingsArray = new GlobalConstants[1];
            settingsArray[0] = data.globalConstants;

            natCmd.SetBufferData(data.pathTracingSettingsBuffer, settingsArray);

            // 不透明
            natCmd.SetRayTracingConstantBufferParam(data.opaqueTracingShader, "PathTracingParams", data.pathTracingSettingsBuffer, 0, data.pathTracingSettingsBuffer.stride);

            data.opaqueTracingShader.SetBuffer(g_ScramblingRankingID, data.scramblingRanking);
            data.opaqueTracingShader.SetBuffer(g_SobolID, data.sobol);

            natCmd.SetRayTracingTextureParam(data.opaqueTracingShader, g_OutputID, data.outputTexture);

            natCmd.SetRayTracingTextureParam(data.opaqueTracingShader, g_MvID, data.Mv);
            natCmd.SetRayTracingTextureParam(data.opaqueTracingShader, g_ViewZID, data.ViewZ);
            natCmd.SetRayTracingTextureParam(data.opaqueTracingShader, g_Normal_RoughnessID, data.Normal_Roughness);
            natCmd.SetRayTracingTextureParam(data.opaqueTracingShader, g_BaseColor_MetalnessID, data.BaseColor_Metalness);

            natCmd.SetRayTracingTextureParam(data.opaqueTracingShader, g_DirectLightingID, data.DirectLighting);
            natCmd.SetRayTracingTextureParam(data.opaqueTracingShader, g_DirectEmissionID, data.DirectEmission);

            natCmd.SetRayTracingTextureParam(data.opaqueTracingShader, g_ShadowDataID, data.Penumbra);
            natCmd.SetRayTracingTextureParam(data.opaqueTracingShader, g_DiffID, data.Diff);
            natCmd.SetRayTracingTextureParam(data.opaqueTracingShader, g_SpecID, data.Spec);

            natCmd.DispatchRays(data.opaqueTracingShader, "MainRayGenShader", (uint)data.width, (uint)data.height, 1, data.cam);

            // NRD降噪
            natCmd.IssuePluginEventAndData(GetRenderEventAndDataFunc(), 1, data.dataPtr);

            // 合成
            natCmd.SetComputeBufferParam(data.compositionComputeShader, 0, "PathTracingParams", data.pathTracingSettingsBuffer);
            natCmd.SetComputeTextureParam(data.compositionComputeShader, 0, gIn_ViewZID, data.ViewZ);
            natCmd.SetComputeTextureParam(data.compositionComputeShader, 0, gIn_Normal_RoughnessID, data.Normal_Roughness);
            natCmd.SetComputeTextureParam(data.compositionComputeShader, 0, gIn_BaseColor_MetalnessID, data.BaseColor_Metalness);
            natCmd.SetComputeTextureParam(data.compositionComputeShader, 0, gIn_DirectLightingID, data.DirectLighting);
            natCmd.SetComputeTextureParam(data.compositionComputeShader, 0, gIn_DirectEmissionID, data.DirectEmission);
            natCmd.SetComputeTextureParam(data.compositionComputeShader, 0, gIn_ShadowID, data.Shadow_Translucency);
            natCmd.SetComputeTextureParam(data.compositionComputeShader, 0, gIn_DiffID, data.DenoisedDiff);
            natCmd.SetComputeTextureParam(data.compositionComputeShader, 0, gIn_SpecID, data.DenoisedSpec);
            natCmd.SetComputeTextureParam(data.compositionComputeShader, 0, gOut_ComposedDiffID, data.ComposedDiff);
            natCmd.SetComputeTextureParam(data.compositionComputeShader, 0, gOut_ComposedSpec_ViewZID, data.ComposedSpec_ViewZ);

            int threadGroupX = Mathf.CeilToInt(data.width / 16.0f);
            int threadGroupY = Mathf.CeilToInt(data.height / 16.0f);
            natCmd.DispatchCompute(data.compositionComputeShader, 0, threadGroupX, threadGroupY, 1);

            // 透明
            natCmd.SetRayTracingConstantBufferParam(data.transparentTracingShader, "PathTracingParams", data.pathTracingSettingsBuffer, 0, data.pathTracingSettingsBuffer.stride);
            natCmd.SetRayTracingTextureParam(data.transparentTracingShader, gIn_ComposedDiffID, data.ComposedDiff);
            natCmd.SetRayTracingTextureParam(data.transparentTracingShader, gIn_ComposedSpec_ViewZID, data.ComposedSpec_ViewZ);
            natCmd.SetRayTracingTextureParam(data.transparentTracingShader, gOut_ComposedID, data.Composed);

            natCmd.DispatchRays(data.transparentTracingShader, "MainRayGenShader", (uint)data.width, (uint)data.height, 1, data.cam);

            // TAA
            TextureHandle taaSrc = data.isEven ? data.TaaHistoryPrev : data.TaaHistory;
            TextureHandle taaDst = data.isEven ? data.TaaHistory : data.TaaHistoryPrev;

            natCmd.SetComputeBufferParam(data.taaComputeShader, 0, "PathTracingParams", data.pathTracingSettingsBuffer);
            natCmd.SetComputeTextureParam(data.taaComputeShader, 0, gIn_MvID, data.Mv);
            natCmd.SetComputeTextureParam(data.taaComputeShader, 0, gIn_ComposedID, data.Composed);
            natCmd.SetComputeTextureParam(data.taaComputeShader, 0, gIn_HistoryID, taaSrc);
            natCmd.SetComputeTextureParam(data.taaComputeShader, 0, gOut_ResultID, taaDst);
            natCmd.SetComputeTextureParam(data.taaComputeShader, 0, gOut_DebugID, data.outputTexture);
            natCmd.DispatchCompute(data.taaComputeShader, 0, threadGroupX, threadGroupY, 1);

            // 显示输出
            natCmd.SetRenderTarget(data.cameraTexture);

            // 0 showValidation     Blend Alpha
            // 1 showShadow         解码后输出阴影
            // 2 showMv             VM
            // 3 ShowNormal         解码后输出法线 转到NRD坐标系
            // 4 showOut            Blend Alpha
            // 5 showAlpha          灰度输出
            // 6 ShowRoughness      解码后输出粗糙度
            // 7 ShowRadiance       解码后RGB输出

            switch (data._setting.showMode)
            {
                case ShowMode.None:
                    break;
                case ShowMode.BaseColor:
                    Blitter.BlitTexture(natCmd, data.BaseColor_Metalness, new Vector4(1, 1, 0, 0), data.blitMaterial, 4);
                    break;
                case ShowMode.Metalness:
                    Blitter.BlitTexture(natCmd, data.BaseColor_Metalness, new Vector4(1, 1, 0, 0), data.blitMaterial, 5);
                    break;
                case ShowMode.Normal:
                    Blitter.BlitTexture(natCmd, data.Normal_Roughness, new Vector4(1, 1, 0, 0), data.blitMaterial, 3);
                    break;
                case ShowMode.Roughness:
                    Blitter.BlitTexture(natCmd, data.Normal_Roughness, new Vector4(1, 1, 0, 0), data.blitMaterial, 6);
                    break;
                case ShowMode.Shadow:
                    Blitter.BlitTexture(natCmd, data.Shadow_Translucency, new Vector4(1, 1, 0, 0), data.blitMaterial, 1);
                    break;
                case ShowMode.Diffuse:
                    Blitter.BlitTexture(natCmd, data.DenoisedDiff, new Vector4(1, 1, 0, 0), data.blitMaterial, 7);
                    break;
                case ShowMode.Specular:
                    Blitter.BlitTexture(natCmd, data.DenoisedSpec, new Vector4(1, 1, 0, 0), data.blitMaterial, 7);
                    break;
                case ShowMode.DirectLight:
                    Blitter.BlitTexture(natCmd, data.DirectLighting, new Vector4(1, 1, 0, 0), data.blitMaterial, 4);
                    break;
                case ShowMode.Emissive:
                    Blitter.BlitTexture(natCmd, data.DirectEmission, new Vector4(1, 1, 0, 0), data.blitMaterial, 4);
                    break;
                case ShowMode.Out:
                    Blitter.BlitTexture(natCmd, data.outputTexture, new Vector4(1, 1, 0, 0), data.blitMaterial, 4);
                    break;
                case ShowMode.ComposedDiff:
                    Blitter.BlitTexture(natCmd, data.ComposedDiff, new Vector4(1, 1, 0, 0), data.blitMaterial, 4);
                    break;
                case ShowMode.ComposedSpec:
                    Blitter.BlitTexture(natCmd, data.ComposedSpec_ViewZ, new Vector4(1, 1, 0, 0), data.blitMaterial, 4);
                    break;
                case ShowMode.Taa:
                    Blitter.BlitTexture(natCmd, taaDst, new Vector4(1, 1, 0, 0), data.blitMaterial, 5);
                    break;
                case ShowMode.Final:
                    Blitter.BlitTexture(natCmd, taaDst, new Vector4(1, 1, 0, 0), data.blitMaterial, 4);
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

            if (data._setting.showMV)
            {
                Blitter.BlitTexture(natCmd, data.Mv, new Vector4(1, 1, 0, 0), data.blitMaterial, 2);
            }

            if (data._setting.showValidation)
            {
                Blitter.BlitTexture(natCmd, data.Validation, new Vector4(1, 1, 0, 0), data.blitMaterial, 0);
            }
        }

        private Matrix4x4 prevWorldToView;
        private Matrix4x4 prevWorldToClip;

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

            Shader.SetGlobalRayTracingAccelerationStructure(g_AccelStructID, accelerationStructure);

            opaqueTracingShader.SetShaderPass("Test2");
            transparentTracingShader.SetShaderPass("Test2");

            var passName = "Path Tracing Pass";

            using var builder = renderGraph.AddUnsafePass<PassData>(passName, out var passData);

            passData.opaqueTracingShader = opaqueTracingShader;
            passData.transparentTracingShader = transparentTracingShader;
            passData.compositionComputeShader = compositionComputeShader;
            passData.taaComputeShader = taaComputeShader;
            passData.blitMaterial = biltMaterial;
            passData.cam = cameraData.camera;


            var worldToView = cameraData.camera.worldToCameraMatrix;
            var worldToClip = GetWorldToClipMatrix(cameraData.camera);
            var viewToWorld = worldToView.inverse;

            // var cameraProjectionMatrix = cameraData.camera.projectionMatrix;
            // var invCameraProjectionMatrix = cameraProjectionMatrix.inverse;

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

            passData.dataPtr = NrdDenoiser.GetInteropDataPtr(cam, gSunDirection);

            var verticalFieldOfView = cam.fieldOfView;
            var aspectRatio = (float)rectW / rectH;
            var horizontalFieldOfView = Mathf.Atan(Mathf.Tan(Mathf.Deg2Rad * verticalFieldOfView * 0.5f) * aspectRatio) * 2 * Mathf.Rad2Deg;

            var frameIndex = (uint)Time.frameCount;
            var isEven = (frameIndex & 1) == 0;

            var globalConstants = new GlobalConstants
            {
                gViewToWorld = viewToWorld,
                gWorldToView = worldToView,
                gWorldToClip = worldToClip,
                gWorldToViewPrev = prevWorldToView,
                gWorldToClipPrev = prevWorldToClip,
                gRectSize = rectSize,
                gInvRectSize = invRectSize,
                gJitter = NrdDenoiser.ViewportJitter / rectSize,
                gRectSizePrev = rectSize,
                gRenderSize = rectSize,
                gInvRenderSize = invRectSize,

                gCameraFrustum = GetNrdFrustum(cameraData.camera),
                gSunBasisX = new float4(gSunBasisX.x, gSunBasisX.y, gSunBasisX.z, 0),
                gSunBasisY = new float4(gSunBasisY.x, gSunBasisY.y, gSunBasisY.z, 0),
                gSunDirection = new float4(gSunDirection.x, gSunDirection.y, gSunDirection.z, 0),

                gTanPixelAngularRadius = math.tan(0.5f * math.radians(horizontalFieldOfView) / cam.pixelWidth),

                gUnproject = 1.0f / (0.5f * rectH * m11),
                gTanSunAngularRadius = math.tan(math.radians(_settings.sunAngularDiameter * 0.5f)),
                gNearZ = -cam.nearClipPlane,
                gAperture = _settings.dofAperture * 0.01f,
                gFocalDistance = _settings.dofFocalDistance,
                gExposure = _settings.exposure,
                gFrameIndex = frameIndex,
                gTAA = _settings.taa,
                gSampleNum = _settings.rpp,
                gBounceNum = _settings.bounceNum
            };

            opaqueTracingShader.SetShaderPass("Test2");
            transparentTracingShader.SetShaderPass("Test2");

            var textureDesc = resourceData.activeColorTexture.GetDescriptor(renderGraph);
            textureDesc.enableRandomWrite = true;
            textureDesc.depthBufferBits = 0;
            textureDesc.clearBuffer = false;
            textureDesc.discardBuffer = false;
            CreateTextureHandle(renderGraph, passData, textureDesc, builder);

            passData.globalConstants = globalConstants;
            passData.cameraTexture = resourceData.activeColorTexture;
            passData.width = textureDesc.width;
            passData.height = textureDesc.height;
            passData.pathTracingSettingsBuffer = pathTracingSettingsBuffer;
            passData._setting = _settings;
            passData.scramblingRanking = scramblingRanking;
            passData.sobol = sobol;
            passData.isEven = isEven;

            Shader.SetGlobalConstantBuffer("PathTracingParams", pathTracingSettingsBuffer, 0, pathTracingSettingsBuffer.stride);


            // compositionComputeShader.SetConstantBuffer("PathTracingParams", pathTracingSettingsBuffer, 0, pathTracingSettingsBuffer.stride);
            // taaComputeShader.SetConstantBuffer("PathTracingParams", pathTracingSettingsBuffer, 0, pathTracingSettingsBuffer.stride);

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });

            prevWorldToView = worldToView;
            prevWorldToClip = worldToClip;
        }

        private void CreateTextureHandle(RenderGraph renderGraph, PassData passData, TextureDesc textureDesc, IUnsafeRenderGraphBuilder builder)
        {
            passData.outputTexture = CreateTex(textureDesc, renderGraph, "PathTracingOutput", GraphicsFormat.R16G16B16A16_SFloat);

            passData.Mv = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_MV));
            passData.ViewZ = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_VIEWZ));
            passData.Normal_Roughness = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_NORMAL_ROUGHNESS));

            passData.BaseColor_Metalness = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_BASECOLOR_METALNESS));
            passData.DirectLighting = CreateTex(textureDesc, renderGraph, "DirectLighting", GraphicsFormat.B10G11R11_UFloatPack32);
            passData.DirectEmission = CreateTex(textureDesc, renderGraph, "DirectEmission", GraphicsFormat.B10G11R11_UFloatPack32);

            passData.Penumbra = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_PENUMBRA));
            passData.Diff = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_DIFF_RADIANCE_HITDIST));
            passData.Spec = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_SPEC_RADIANCE_HITDIST));

            // 输出
            passData.Shadow_Translucency = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.OUT_SHADOW_TRANSLUCENCY));
            passData.DenoisedDiff = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.OUT_DIFF_RADIANCE_HITDIST));
            passData.DenoisedSpec = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.OUT_SPEC_RADIANCE_HITDIST));
            passData.Validation = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.OUT_VALIDATION));

            passData.ComposedDiff = CreateTex(textureDesc, renderGraph, "ComposedDiff", GraphicsFormat.R16G16B16A16_SFloat);
            passData.ComposedSpec_ViewZ = CreateTex(textureDesc, renderGraph, "ComposedSpec_ViewZ", GraphicsFormat.R16G16B16A16_SFloat);

            passData.Composed = CreateTex(textureDesc, renderGraph, "Composed", GraphicsFormat.R16G16B16A16_SFloat);

            passData.TaaHistory = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.TaaHistory));
            passData.TaaHistoryPrev = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.TaaHistoryPrev));

            builder.UseTexture(passData.outputTexture, AccessFlags.ReadWrite);

            builder.UseTexture(passData.Mv, AccessFlags.ReadWrite);
            builder.UseTexture(passData.ViewZ, AccessFlags.ReadWrite);
            builder.UseTexture(passData.Normal_Roughness, AccessFlags.ReadWrite);
            builder.UseTexture(passData.BaseColor_Metalness, AccessFlags.ReadWrite);

            builder.UseTexture(passData.DirectLighting, AccessFlags.ReadWrite);
            builder.UseTexture(passData.DirectEmission, AccessFlags.ReadWrite);

            builder.UseTexture(passData.Penumbra, AccessFlags.ReadWrite);
            builder.UseTexture(passData.Diff, AccessFlags.ReadWrite);
            builder.UseTexture(passData.Spec, AccessFlags.ReadWrite);

            // 输出
            builder.UseTexture(passData.Shadow_Translucency, AccessFlags.ReadWrite);
            builder.UseTexture(passData.DenoisedDiff, AccessFlags.ReadWrite);
            builder.UseTexture(passData.DenoisedSpec, AccessFlags.ReadWrite);
            builder.UseTexture(passData.Validation, AccessFlags.ReadWrite);

            builder.UseTexture(passData.ComposedDiff, AccessFlags.ReadWrite);
            builder.UseTexture(passData.ComposedSpec_ViewZ, AccessFlags.ReadWrite);
            builder.UseTexture(passData.Composed, AccessFlags.ReadWrite);
        }

        private TextureHandle CreateTex(TextureDesc textureDesc, RenderGraph renderGraph, string name, GraphicsFormat format)
        {
            textureDesc.format = format;
            textureDesc.name = name;
            return renderGraph.CreateTexture(textureDesc);
        }

        public void Dispose()
        {
            pathTracingSettingsBuffer?.Release();
        }
    }
}