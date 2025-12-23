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
            internal TextureHandle CameraTexture;

            internal ComputeBuffer ScramblingRanking;
            internal ComputeBuffer Sobol;

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

            internal RayTracingShader OpaqueTracingShader;
            internal RayTracingShader TransparentTracingShader;
            internal ComputeShader CompositionComputeShader;
            internal ComputeShader TaaComputeShader;
            internal Material BlitMaterial;
            internal Camera Cam;
            internal int Width;
            internal int Height;

            internal GlobalConstants GlobalConstants;
            internal ComputeBuffer computeBuffer;

            internal IntPtr NrdDataPtr;

            internal PathTracingSetting Setting;
            internal bool IsEven;
        }

        public PathTracingPassSingle(PathTracingSetting setting)
        {
            _settings = setting;
            pathTracingSettingsBuffer = new ComputeBuffer(1, Marshal.SizeOf<GlobalConstants>(), ComputeBufferType.Constant);
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            natCmd.SetBufferData(data.computeBuffer, new[] { data.GlobalConstants });

            // 不透明
            natCmd.SetRayTracingConstantBufferParam(data.OpaqueTracingShader, paramsID, data.computeBuffer, 0, data.computeBuffer.stride);

            data.OpaqueTracingShader.SetBuffer(g_ScramblingRankingID, data.ScramblingRanking);
            data.OpaqueTracingShader.SetBuffer(g_SobolID, data.Sobol);

            natCmd.SetRayTracingTextureParam(data.OpaqueTracingShader, g_OutputID, data.OutputTexture);

            natCmd.SetRayTracingTextureParam(data.OpaqueTracingShader, g_MvID, data.Mv);
            natCmd.SetRayTracingTextureParam(data.OpaqueTracingShader, g_ViewZID, data.ViewZ);
            natCmd.SetRayTracingTextureParam(data.OpaqueTracingShader, g_Normal_RoughnessID, data.NormalRoughness);
            natCmd.SetRayTracingTextureParam(data.OpaqueTracingShader, g_BaseColor_MetalnessID, data.BaseColorMetalness);

            natCmd.SetRayTracingTextureParam(data.OpaqueTracingShader, g_DirectLightingID, data.DirectLighting);
            natCmd.SetRayTracingTextureParam(data.OpaqueTracingShader, g_DirectEmissionID, data.DirectEmission);

            natCmd.SetRayTracingTextureParam(data.OpaqueTracingShader, g_ShadowDataID, data.Penumbra);
            natCmd.SetRayTracingTextureParam(data.OpaqueTracingShader, g_DiffID, data.Diff);
            natCmd.SetRayTracingTextureParam(data.OpaqueTracingShader, g_SpecID, data.Spec);

            natCmd.SetRayTracingTextureParam(data.OpaqueTracingShader, gIn_PrevComposedDiffID, data.ComposedDiff);
            natCmd.SetRayTracingTextureParam(data.OpaqueTracingShader, gIn_PrevComposedSpec_PrevViewZID, data.ComposedSpecViewZ);

            natCmd.DispatchRays(data.OpaqueTracingShader, "MainRayGenShader", (uint)data.Width, (uint)data.Height, 1, data.Cam);

            // NRD降噪
            natCmd.IssuePluginEventAndData(GetRenderEventAndDataFunc(), 1, data.NrdDataPtr);

            // 合成
            natCmd.SetComputeBufferParam(data.CompositionComputeShader, 0, paramsID, data.computeBuffer);
            natCmd.SetComputeTextureParam(data.CompositionComputeShader, 0, gIn_ViewZID, data.ViewZ);
            natCmd.SetComputeTextureParam(data.CompositionComputeShader, 0, gIn_Normal_RoughnessID, data.NormalRoughness);
            natCmd.SetComputeTextureParam(data.CompositionComputeShader, 0, gIn_BaseColor_MetalnessID, data.BaseColorMetalness);
            natCmd.SetComputeTextureParam(data.CompositionComputeShader, 0, gIn_DirectLightingID, data.DirectLighting);
            natCmd.SetComputeTextureParam(data.CompositionComputeShader, 0, gIn_DirectEmissionID, data.DirectEmission);
            natCmd.SetComputeTextureParam(data.CompositionComputeShader, 0, gIn_ShadowID, data.ShadowTranslucency);
            natCmd.SetComputeTextureParam(data.CompositionComputeShader, 0, gIn_DiffID, data.DenoisedDiff);
            natCmd.SetComputeTextureParam(data.CompositionComputeShader, 0, gIn_SpecID, data.DenoisedSpec);
            natCmd.SetComputeTextureParam(data.CompositionComputeShader, 0, gOut_ComposedDiffID, data.ComposedDiff);
            natCmd.SetComputeTextureParam(data.CompositionComputeShader, 0, gOut_ComposedSpec_ViewZID, data.ComposedSpecViewZ);

            int threadGroupX = Mathf.CeilToInt(data.Width / 16.0f);
            int threadGroupY = Mathf.CeilToInt(data.Height / 16.0f);
            natCmd.DispatchCompute(data.CompositionComputeShader, 0, threadGroupX, threadGroupY, 1);

            // 透明
            natCmd.SetRayTracingConstantBufferParam(data.TransparentTracingShader, paramsID, data.computeBuffer, 0, data.computeBuffer.stride);
            natCmd.SetRayTracingTextureParam(data.TransparentTracingShader, gIn_ComposedDiffID, data.ComposedDiff);
            natCmd.SetRayTracingTextureParam(data.TransparentTracingShader, gIn_ComposedSpec_ViewZID, data.ComposedSpecViewZ);
            natCmd.SetRayTracingTextureParam(data.TransparentTracingShader, gOut_ComposedID, data.Composed);

            natCmd.DispatchRays(data.TransparentTracingShader, "MainRayGenShader", (uint)data.Width, (uint)data.Height, 1, data.Cam);

            // TAA
            TextureHandle taaSrc = data.IsEven ? data.TaaHistoryPrev : data.TaaHistory;
            TextureHandle taaDst = data.IsEven ? data.TaaHistory : data.TaaHistoryPrev;

            natCmd.SetComputeBufferParam(data.TaaComputeShader, 0, paramsID, data.computeBuffer);
            natCmd.SetComputeTextureParam(data.TaaComputeShader, 0, gIn_MvID, data.Mv);
            natCmd.SetComputeTextureParam(data.TaaComputeShader, 0, gIn_ComposedID, data.Composed);
            natCmd.SetComputeTextureParam(data.TaaComputeShader, 0, gIn_HistoryID, taaSrc);
            natCmd.SetComputeTextureParam(data.TaaComputeShader, 0, gOut_ResultID, taaDst);
            natCmd.SetComputeTextureParam(data.TaaComputeShader, 0, gOut_DebugID, data.OutputTexture);
            natCmd.DispatchCompute(data.TaaComputeShader, 0, threadGroupX, threadGroupY, 1);

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

        // private Matrix4x4 prevWorldToView;
        // private Matrix4x4 prevWorldToClip;

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

            using var builder = renderGraph.AddUnsafePass<PassData>("Path Tracing Pass", out var passData);

            passData.OpaqueTracingShader = opaqueTracingShader;
            passData.TransparentTracingShader = transparentTracingShader;
            passData.CompositionComputeShader = compositionComputeShader;
            passData.TaaComputeShader = taaComputeShader;
            passData.BlitMaterial = biltMaterial;
            passData.Cam = cameraData.camera;

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

            var frameIndex = (uint)Time.frameCount;
            var isEven = (frameIndex & 1) == 0;

            var globalConstants = new GlobalConstants
            {
                gViewToWorld = NrdDenoiser.worldToView.inverse,
                gWorldToView = NrdDenoiser.worldToView,
                gWorldToClip = NrdDenoiser.worldToClip,
                gWorldToViewPrev = NrdDenoiser.prevWorldToView,
                gWorldToClipPrev = NrdDenoiser.prevWorldToClip,
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
                gBounceNum = _settings.bounceNum,
                gPrevFrameConfidence = 1
            };

            opaqueTracingShader.SetShaderPass("Test2");
            transparentTracingShader.SetShaderPass("Test2");

            var textureDesc = resourceData.activeColorTexture.GetDescriptor(renderGraph);
            textureDesc.enableRandomWrite = true;
            textureDesc.depthBufferBits = 0;
            textureDesc.clearBuffer = false;
            textureDesc.discardBuffer = false;
            CreateTextureHandle(renderGraph, passData, textureDesc, builder);

            passData.GlobalConstants = globalConstants;
            passData.CameraTexture = resourceData.activeColorTexture;
            passData.Width = textureDesc.width;
            passData.Height = textureDesc.height;
            passData.computeBuffer = pathTracingSettingsBuffer;
            passData.Setting = _settings;
            passData.ScramblingRanking = scramblingRanking;
            passData.Sobol = sobol;
            passData.IsEven = isEven;

            Shader.SetGlobalConstantBuffer(paramsID, pathTracingSettingsBuffer, 0, pathTracingSettingsBuffer.stride);


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