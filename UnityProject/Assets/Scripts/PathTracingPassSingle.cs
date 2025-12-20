using System;
using System.Runtime.InteropServices;
using Nrd;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe; // 新增：用于获取 NativeArray 指针
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    public class PathTracingPassSingle : ScriptableRenderPass
    {
        public RayTracingShader rayTracingShader;

        private PathTracingSetting _settings;
        public RayTracingAccelerationStructure accelerationStructure;

        public ComputeBuffer scramblingRanking;
        public ComputeBuffer sobol;

        // public int convergenceStep = 0;

        #region ID

        private static int g_ZoomID = Shader.PropertyToID("g_Zoom");
        private static int g_OutputID = Shader.PropertyToID("g_Output");


        private static int g_ScramblingRankingID = Shader.PropertyToID("gIn_ScramblingRanking");
        private static int g_SobolID = Shader.PropertyToID("gIn_Sobol");

        private static int g_MvID = Shader.PropertyToID("gOut_Mv");
        private static int g_PenumbraID = Shader.PropertyToID("gOut_Penumbra");
        private static int g_ValidationID = Shader.PropertyToID("gOut_Validation");


        private static int g_ViewZID = Shader.PropertyToID("gOut_ViewZ");
        private static int g_Normal_RoughnessID = Shader.PropertyToID("gOut_Normal_Roughness");
        private static int g_BaseColor_MetalnessID = Shader.PropertyToID("gOut_BaseColor_Metalness");
        private static int g_DirectLightingID = Shader.PropertyToID("gOut_DirectLighting");
        private static int g_DirectEmissionID = Shader.PropertyToID("gOut_DirectEmission");
        private static int g_PsrThroughputID = Shader.PropertyToID("gOut_PsrThroughput");
        private static int g_ShadowDataID = Shader.PropertyToID("gOut_ShadowData");
        private static int g_Shadow_TranslucencyID = Shader.PropertyToID("gOut_Shadow_Translucency");

        private static int g_DiffID = Shader.PropertyToID("gOut_Diff");
        private static int g_SpecID = Shader.PropertyToID("gOut_Spec");


        private static int g_ConvergenceStepID = Shader.PropertyToID("g_ConvergenceStep");
        private static int g_FrameIndexID = Shader.PropertyToID("g_FrameIndex");
        private static int g_SampleCountID = Shader.PropertyToID("g_SampleCount");
        private static int g_EnvTexID = Shader.PropertyToID("g_EnvTex");
        private static int lightOffsetID = Shader.PropertyToID("lightOffset");


        private static int g_BounceCountOpaqueID = Shader.PropertyToID("g_BounceCountOpaque");
        private static int g_BounceCountTransparentID = Shader.PropertyToID("g_BounceCountTransparent");


        private static int g_AccelStructID = Shader.PropertyToID("g_AccelStruct");
        private static int g_DirLightDirectionID = Shader.PropertyToID("g_DirLightDirection");
        private static int g_DirLightColorID = Shader.PropertyToID("g_DirLightColor");

        #endregion

        private ComputeBuffer pathTracingSettingsBuffer;

        struct Settings
        {
            public float g_Zoom;
            public uint g_ConvergenceStep;
            public uint g_FrameIndex;
            public uint g_SampleCount;
            public float lightOffset;

            public float3 _CameraPosition;
            public float4x4 _CCameraToWorld;
            public float4x4 gWorldToView;
            public float4x4 gWorldToClip;
            public float4x4 gWorldToViewPrev;
            public float4x4 gWorldToClipPrev;
            public float2 gRectSize;
            public float2 pad1;
            public float4x4 _CInverseProjection;

            public float4 gSunBasisX;
            public float4 gSunBasisY;
            public float4 gSunDirection;

            public float gTanPixelAngularRadius;
            public float gUnproject;
            public float gTanSunAngularRadius;
            public float pad;
        }

        class PassData
        {
            internal TextureHandle outputTexture;
            internal TextureHandle cameraTexture;

            internal ComputeBuffer scramblingRanking;
            internal ComputeBuffer sobol;

            internal TextureHandle Mv;
            internal TextureHandle ViewZ;
            internal TextureHandle Normal_Roughness;
            internal TextureHandle Penumbra;
            internal TextureHandle DiffRadianceHITDIST;
            internal TextureHandle Validation;

            internal TextureHandle BaseColor_Metalness;
            internal TextureHandle DirectLighting;
            internal TextureHandle DirectEmission;
            internal TextureHandle PsrThroughput;
            internal TextureHandle ShadowData;
            internal TextureHandle Shadow_Translucency;
            internal TextureHandle Diff;
            internal TextureHandle Spec;

            internal RayTracingShader rayTracingShader;
            internal Material blitMaterial;
            internal Camera cam;
            internal int width;
            internal int height;

            internal Settings pathTracingSettings;
            internal ComputeBuffer pathTracingSettingsBuffer;
            internal IntPtr dataPtr;


            internal bool showValidation;
            internal bool showMv;
            internal bool showShadow;
            internal bool showOut;
        }

        public NRDDenoiser NrdDenoiser;
        public Material biltMaterial;

        public PathTracingPassSingle(PathTracingSetting setting)
        {
            _settings = setting;
            pathTracingSettingsBuffer = new ComputeBuffer(1, Marshal.SizeOf<Settings>(), ComputeBufferType.Constant);
        }

        static Matrix4x4 GetWorldToClipMatrix(Camera camera)
        {
            // Unity 的 GPU 投影矩阵（处理平台差异 & Y 翻转）
            Matrix4x4 proj = GL.GetGPUProjectionMatrix(camera.projectionMatrix, false);

            return proj * camera.worldToCameraMatrix;
        }

        [DllImport("RenderingPlugin")]
        public static extern IntPtr GetRenderEventAndDataFunc();

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            Settings[] settingsArray = new Settings[1];
            settingsArray[0] = data.pathTracingSettings;

            natCmd.SetBufferData(data.pathTracingSettingsBuffer, settingsArray);

            natCmd.SetRayTracingConstantBufferParam(data.rayTracingShader, "PathTracingParams",
                data.pathTracingSettingsBuffer, 0, data.pathTracingSettingsBuffer.stride);

            data.rayTracingShader.SetBuffer(g_ScramblingRankingID, data.scramblingRanking);
            data.rayTracingShader.SetBuffer(g_SobolID, data.sobol);
            // natCmd.SetRayTracingTextureParam(data.rayTracingShader, g_SobolID, data.sobol);


            natCmd.SetRayTracingTextureParam(data.rayTracingShader, g_OutputID, data.outputTexture);
            natCmd.SetRayTracingTextureParam(data.rayTracingShader, g_MvID, data.Mv);
            natCmd.SetRayTracingTextureParam(data.rayTracingShader, g_ViewZID, data.ViewZ);
            natCmd.SetRayTracingTextureParam(data.rayTracingShader, g_Normal_RoughnessID, data.Normal_Roughness);
            natCmd.SetRayTracingTextureParam(data.rayTracingShader, g_PenumbraID, data.Penumbra);
            natCmd.SetRayTracingTextureParam(data.rayTracingShader, g_Shadow_TranslucencyID, data.Shadow_Translucency);
            natCmd.SetRayTracingTextureParam(data.rayTracingShader, g_BaseColor_MetalnessID, data.BaseColor_Metalness);
            natCmd.SetRayTracingTextureParam(data.rayTracingShader, g_DirectLightingID, data.DirectLighting);
            natCmd.SetRayTracingTextureParam(data.rayTracingShader, g_DirectEmissionID, data.DirectEmission);
            natCmd.SetRayTracingTextureParam(data.rayTracingShader, g_PsrThroughputID, data.PsrThroughput);
            natCmd.SetRayTracingTextureParam(data.rayTracingShader, g_ShadowDataID, data.ShadowData);
            natCmd.SetRayTracingTextureParam(data.rayTracingShader, g_DiffID, data.Diff);
            natCmd.SetRayTracingTextureParam(data.rayTracingShader, g_SpecID, data.Spec);

            natCmd.DispatchRays(data.rayTracingShader, "MainRayGenShader", (uint)data.width, (uint)data.height, 1,
                data.cam);

            natCmd.IssuePluginEventAndData(GetRenderEventAndDataFunc(), 1, data.dataPtr);


            natCmd.SetRenderTarget(data.cameraTexture);
            if (data.showOut)
            {
                Blitter.BlitTexture(natCmd, data.outputTexture, new Vector4(1, 1, 0, 0), 0, false);
            }

            if (data.showShadow)
            {
                Blitter.BlitTexture(natCmd, data.Shadow_Translucency, new Vector4(1, 1, 0, 0), data.blitMaterial, 1);
            }

            if (data.showValidation)
            {
                Blitter.BlitTexture(natCmd, data.Validation, new Vector4(1, 1, 0, 0), data.blitMaterial, 0);
                // Blitter.BlitTexture(natCmd, data.Validation, new Vector4(1, 1, 0, 0),0,false);
            }

            if (data.showMv)
            {
                Blitter.BlitTexture(natCmd, data.Mv, new Vector4(1, 1, 0, 0), data.blitMaterial, 2);
            }
        }

        private Matrix4x4 prevWorldToView;
        private Matrix4x4 prevWorldToClip;

        // private Matrix4x4 prevCameraMatrix;
        // private int prevBounceCountOpaque;
        // private int prevBounceCountTransparent;

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            var cameraData = frameData.Get<UniversalCameraData>();

            var universalLightData = frameData.Get<UniversalLightData>();
            var lightData = universalLightData;
            var mainLight = lightData.mainLightIndex >= 0 ? lightData.visibleLights[lightData.mainLightIndex] : default;
            var mat = mainLight.localToWorldMatrix;
            Vector3 lightForward = mat.GetColumn(2);

            if (cameraData.camera.cameraType != CameraType.Game && cameraData.camera.cameraType != CameraType.SceneView)
            {
                return;
            }


            NrdDenoiser.EnsureResources(cameraData.camera.pixelWidth, cameraData.camera.pixelHeight);

            string passName = "Path Tracing Pass";

            using var builder = renderGraph.AddUnsafePass<PassData>(passName, out var passData);

            passData.rayTracingShader = rayTracingShader;
            passData.cam = cameraData.camera;

            // if (prevCameraMatrix != cameraData.camera.cameraToWorldMatrix)
            // {
            //     convergenceStep = 0;
            // }
            //
            // if (prevBounceCountOpaque != _settings.bounceCountOpaque)
            // {
            //     convergenceStep = 0;
            // }
            //
            // if (prevBounceCountTransparent != _settings.bounceCountTransparent)
            // {
            //     convergenceStep = 0;
            // }


            var fov = cameraData.camera.fieldOfView;
            if (cameraData.camera.usePhysicalProperties)
            {
                fov = 2.0f * Mathf.Atan(0.5f * cameraData.camera.sensorSize.y / cameraData.camera.focalLength) *
                      Mathf.Rad2Deg;
            }

            var tan = Mathf.Tan(fov * 0.5f * Mathf.Deg2Rad);

            // var worldToView = cameraData.GetViewMatrix();
            var worldToView = cameraData.camera.worldToCameraMatrix;
            var worldToClip = GetWorldToClipMatrix(cameraData.camera);

            var viewToWorld = worldToView.inverse;


            var cameraProjectionMatrix = cameraData.camera.projectionMatrix;

            var invCameraProjectionMatrix = cameraProjectionMatrix.inverse;

            Vector3 gSunDirection = -lightForward;
            // var gSunDirection = new Vector3(-forward.x, -forward.y, -forward.z);
            Vector3 up = new Vector3(0, 1, 0);

            var gSunBasisX = math.normalize(math.cross(new float3(up.x, up.y, up.z),
                new float3(gSunDirection.x, gSunDirection.y, gSunDirection.z)));
            var gSunBasisY = math.normalize(math.cross(new float3(gSunDirection.x, gSunDirection.y, gSunDirection.z),
                gSunBasisX));


            var cam = cameraData.camera;
            float m11 = cam.projectionMatrix.m11;
            var rectH = cam.pixelHeight;

            passData.dataPtr = NrdDenoiser.GetInteropDataPtr(cam, gSunDirection);

            var setting = new Settings
            {
                g_Zoom = tan,
                g_ConvergenceStep = NrdDenoiser.FrameIndex,
                g_FrameIndex = (uint)Time.frameCount,
                g_SampleCount = (uint)_settings.sampleCount,
                lightOffset = _settings.lightOffset,

                _CameraPosition = cameraData.worldSpaceCameraPos,

                _CCameraToWorld = viewToWorld,
                gWorldToView = worldToView,
                gWorldToClip = worldToClip,
                gWorldToViewPrev = prevWorldToView,
                gWorldToClipPrev = prevWorldToClip,
                gRectSize = new float2(cam.pixelWidth, cam.pixelHeight),

                _CInverseProjection = invCameraProjectionMatrix,
                gSunBasisX = new float4(gSunBasisX.x, gSunBasisX.y, gSunBasisX.z, 0),
                gSunBasisY = new float4(gSunBasisY.x, gSunBasisY.y, gSunBasisY.z, 0),
                gSunDirection = new float4(gSunDirection.x, gSunDirection.y, gSunDirection.z, 0),
                gTanPixelAngularRadius = math.tan(0.5f * math.radians(cam.fieldOfView) / cam.pixelWidth),
                gUnproject = 1.0f / (0.5f * rectH * m11),
                gTanSunAngularRadius = _settings.lightOffset
            };
            passData.pathTracingSettings = setting;

            UniversalResourceData resourceData = frameData.Get<UniversalResourceData>();
            passData.cameraTexture = resourceData.activeColorTexture;
            TextureDesc textureDesc = resourceData.activeColorTexture.GetDescriptor(renderGraph);

            textureDesc.format = GraphicsFormat.R16G16B16A16_SFloat;
            textureDesc.enableRandomWrite = true;
            textureDesc.name = "Path";

            rayTracingShader.SetShaderPass("Test2");

            textureDesc.depthBufferBits = 0;
            textureDesc.clearBuffer = false;
            textureDesc.discardBuffer = false;

            passData.width = textureDesc.width;
            passData.height = textureDesc.height;
            passData.pathTracingSettingsBuffer = pathTracingSettingsBuffer;
            passData.blitMaterial = biltMaterial;

            passData.showOut = _settings.showOut;
            passData.showShadow = _settings.showShadow;
            passData.showMv = _settings.showMV;
            passData.showValidation = _settings.showValidation;

            // Debug.Log("Texture Width: " + textureDesc.width + " Height: " + textureDesc.height);


            // var settingsBufferHandle = renderGraph.ImportBuffer(pathTracingSettingsBuffer);

            passData.outputTexture = renderGraph.CreateTexture(textureDesc);

            // passData.scramblingRanking = renderGraph.ImportBuffer(scramblingRanking);
            // passData.sobol = renderGraph.ImportTexture(sobol);
            passData.scramblingRanking = scramblingRanking;
            passData.sobol = sobol;

            passData.Mv = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_MV));
            passData.ViewZ = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_VIEWZ));
            passData.Normal_Roughness = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_NORMAL_ROUGHNESS));
            passData.Penumbra = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_PENUMBRA));
            passData.Shadow_Translucency = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.OUT_SHADOW_TRANSLUCENCY));
            passData.Validation = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.OUT_VALIDATION));

            passData.BaseColor_Metalness = CreateTex(textureDesc, renderGraph, "BaseColor_Metalness",
                GraphicsFormat.R16G16B16A16_SFloat);
            passData.DirectLighting = CreateTex(textureDesc, renderGraph, "DirectLighting",
                GraphicsFormat.B10G11R11_UFloatPack32);
            passData.DirectEmission = CreateTex(textureDesc, renderGraph, "DirectEmission",
                GraphicsFormat.B10G11R11_UFloatPack32);
            passData.PsrThroughput = CreateTex(textureDesc, renderGraph, "PsrThroughput",
                GraphicsFormat.B10G11R11_UFloatPack32);
            passData.ShadowData = CreateTex(textureDesc, renderGraph, "ShadowData", GraphicsFormat.R8G8B8A8_UNorm);

            passData.Diff = CreateTex(textureDesc, renderGraph, "Diff", GraphicsFormat.R16G16B16A16_SFloat);
            passData.Spec = CreateTex(textureDesc, renderGraph, "Spec", GraphicsFormat.R16G16B16A16_SFloat);

            rayTracingShader.SetShaderPass("Test2");

            // rayTracingShader.SetTexture(g_EnvTexID, _settings.envTexture);

            Shader.SetGlobalInt(g_BounceCountOpaqueID, _settings.bounceCountOpaque);
            Shader.SetGlobalInt(g_BounceCountTransparentID, _settings.bounceCountTransparent);


            Shader.SetGlobalVector(g_DirLightDirectionID, gSunDirection);
            Shader.SetGlobalColor(g_DirLightColorID, mainLight.finalColor);

            accelerationStructure.Build();
            Shader.SetGlobalRayTracingAccelerationStructure(g_AccelStructID, accelerationStructure);

            // convergenceStep++;


            builder.UseTexture(passData.outputTexture, AccessFlags.ReadWrite);
            builder.UseTexture(passData.Mv, AccessFlags.ReadWrite);
            builder.UseTexture(passData.ViewZ, AccessFlags.ReadWrite);
            builder.UseTexture(passData.Normal_Roughness, AccessFlags.ReadWrite);
            builder.UseTexture(passData.Penumbra, AccessFlags.ReadWrite);
            builder.UseTexture(passData.Validation, AccessFlags.ReadWrite);


            builder.UseTexture(passData.BaseColor_Metalness, AccessFlags.ReadWrite);
            builder.UseTexture(passData.DirectLighting, AccessFlags.ReadWrite);
            builder.UseTexture(passData.DirectEmission, AccessFlags.ReadWrite);
            builder.UseTexture(passData.PsrThroughput, AccessFlags.ReadWrite);
            builder.UseTexture(passData.ShadowData, AccessFlags.ReadWrite);
            builder.UseTexture(passData.Shadow_Translucency, AccessFlags.ReadWrite);
            // builder.UseTexture(passData.Validation, AccessFlags.ReadWrite);
            builder.UseTexture(passData.Diff, AccessFlags.ReadWrite);
            builder.UseTexture(passData.Spec, AccessFlags.ReadWrite);

            builder.AllowPassCulling(false);

            builder.SetRenderFunc((PassData passData, UnsafeGraphContext context) => { ExecutePass(passData, context); });

            prevWorldToView = worldToView;
            prevWorldToClip = worldToClip;

            // prevCameraMatrix = cameraData.camera.cameraToWorldMatrix;
            // prevBounceCountOpaque = _settings.bounceCountOpaque;
            // prevBounceCountTransparent = _settings.bounceCountTransparent;
        }

        private TextureHandle CreateTex(TextureDesc textureDesc, RenderGraph renderGraph, string name,
            GraphicsFormat format)
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