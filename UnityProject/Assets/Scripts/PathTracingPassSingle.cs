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
        public ComputeShader compositionComputeShader;

        private PathTracingSetting _settings;
        public RayTracingAccelerationStructure accelerationStructure;

        public ComputeBuffer scramblingRanking;
        public ComputeBuffer sobol;

        // public int convergenceStep = 0;

        #region ID

        private static int g_ZoomID = Shader.PropertyToID("g_Zoom");

        private static int g_ScramblingRankingID = Shader.PropertyToID("gIn_ScramblingRanking");
        private static int g_SobolID = Shader.PropertyToID("gIn_Sobol");

        // 测试用
        private static int g_OutputID = Shader.PropertyToID("g_Output");

        //  传入NRD的无噪声资源
        private static int g_MvID = Shader.PropertyToID("gOut_Mv");
        private static int g_ViewZID = Shader.PropertyToID("gOut_ViewZ");
        private static int g_Normal_RoughnessID = Shader.PropertyToID("gOut_Normal_Roughness");
        private static int g_BaseColor_MetalnessID = Shader.PropertyToID("gOut_BaseColor_Metalness");

        // 不传入NRD的资源
        private static int g_DirectLightingID = Shader.PropertyToID("gOut_DirectLighting");
        private static int g_DirectEmissionID = Shader.PropertyToID("gOut_DirectEmission");

        // 传入NRD的有噪声资源
        private static int g_ShadowDataID = Shader.PropertyToID("gOut_ShadowData");
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


        private static int gViewToWorldID = Shader.PropertyToID("gViewToWorld");
        private static int gSunDirectionID = Shader.PropertyToID("gSunDirection");
        private static int gCameraFrustumID = Shader.PropertyToID("gCameraFrustum");
        private static int gInvRectSizeID = Shader.PropertyToID("gInvRectSize");


        private static int gIn_ViewZID = Shader.PropertyToID("gIn_ViewZ");
        private static int gIn_Normal_RoughnessID = Shader.PropertyToID("gIn_Normal_Roughness");
        private static int gIn_BaseColor_MetalnessID = Shader.PropertyToID("gIn_BaseColor_Metalness");
        private static int gIn_DirectLightingID = Shader.PropertyToID("gIn_DirectLighting");
        private static int gIn_DirectEmissionID = Shader.PropertyToID("gIn_DirectEmission");
        private static int gIn_ShadowID = Shader.PropertyToID("gIn_Shadow");
        private static int gIn_DiffID = Shader.PropertyToID("gIn_Diff");
        private static int gIn_SpecID = Shader.PropertyToID("gIn_Spec");
        private static int gOut_ComposedDiffID = Shader.PropertyToID("gOut_ComposedDiff");
        private static int gOut_ComposedSpec_ViewZID = Shader.PropertyToID("gOut_ComposedSpec_ViewZ");

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

            internal RayTracingShader rayTracingShader;
            internal ComputeShader compositionComputeShader;
            internal Material blitMaterial;
            internal Camera cam;
            internal int width;
            internal int height;

            internal Settings pathTracingSettings;
            internal ComputeBuffer pathTracingSettingsBuffer;
            internal IntPtr dataPtr;


            internal PathTracingSetting _setting;
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
            natCmd.SetRayTracingTextureParam(data.rayTracingShader, g_BaseColor_MetalnessID, data.BaseColor_Metalness);

            natCmd.SetRayTracingTextureParam(data.rayTracingShader, g_DirectLightingID, data.DirectLighting);
            natCmd.SetRayTracingTextureParam(data.rayTracingShader, g_DirectEmissionID, data.DirectEmission);

            natCmd.SetRayTracingTextureParam(data.rayTracingShader, g_ShadowDataID, data.Penumbra);
            natCmd.SetRayTracingTextureParam(data.rayTracingShader, g_DiffID, data.Diff);
            natCmd.SetRayTracingTextureParam(data.rayTracingShader, g_SpecID, data.Spec);

            natCmd.DispatchRays(data.rayTracingShader, "MainRayGenShader", (uint)data.width, (uint)data.height, 1,
                data.cam);

            natCmd.IssuePluginEventAndData(GetRenderEventAndDataFunc(), 1, data.dataPtr);


            // 组合通道
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
            

            natCmd.SetRenderTarget(data.cameraTexture);


            // 0 showValidation 1 showShadow 2 showMv 3 ShowNormal 4 showOut 5 showAlpha  6 showR 7 ShowRadiance


            if (data._setting.showBaseColor)
            {
                Blitter.BlitTexture(natCmd, data.BaseColor_Metalness, new Vector4(1, 1, 0, 0), data.blitMaterial, 4);
            }

            if (data._setting.showMetalness)
            {
                Blitter.BlitTexture(natCmd, data.BaseColor_Metalness, new Vector4(1, 1, 0, 0), data.blitMaterial, 5);
            }

            if (data._setting.showNormal)
            {
                Blitter.BlitTexture(natCmd, data.Normal_Roughness, new Vector4(1, 1, 0, 0), data.blitMaterial, 3);
            }

            if (data._setting.showRoughness)
            {
                Blitter.BlitTexture(natCmd, data.Normal_Roughness, new Vector4(1, 1, 0, 0), data.blitMaterial, 6);
            }

            if (data._setting.showShadow)
            {
                Blitter.BlitTexture(natCmd, data.Shadow_Translucency, new Vector4(1, 1, 0, 0), data.blitMaterial, 1);
            }

            if (data._setting.showDiffuse)
            {
                Blitter.BlitTexture(natCmd, data.DenoisedDiff, new Vector4(1, 1, 0, 0), data.blitMaterial, 7);
            }

            if (data._setting.showSpecular)
            {
                Blitter.BlitTexture(natCmd, data.DenoisedSpec, new Vector4(1, 1, 0, 0), data.blitMaterial, 7);
            }

            if (data._setting.showDirectLight)
            {
                Blitter.BlitTexture(natCmd, data.DirectLighting, new Vector4(1, 1, 0, 0), data.blitMaterial, 4);
            }

            if (data._setting.showEmissive)
            {
                Blitter.BlitTexture(natCmd, data.DirectEmission, new Vector4(1, 1, 0, 0), data.blitMaterial, 4);
            }

            if (data._setting.showOut)
            {
                Blitter.BlitTexture(natCmd, data.outputTexture, new Vector4(1, 1, 0, 0), data.blitMaterial, 4);
            }

            if (data._setting.showMV)
            {
                Blitter.BlitTexture(natCmd, data.Mv, new Vector4(1, 1, 0, 0), data.blitMaterial, 2);
            }

            if (data._setting.showValidation)
            {
                Blitter.BlitTexture(natCmd, data.Validation, new Vector4(1, 1, 0, 0), data.blitMaterial, 0);
            }
            
            if (data._setting.showComposedDiff)
            {
                Blitter.BlitTexture(natCmd, data.ComposedDiff, new Vector4(1, 1, 0, 0), data.blitMaterial, 4);
            }
            
            if (data._setting.showComposedSpec)
            {
                Blitter.BlitTexture(natCmd, data.ComposedSpec_ViewZ, new Vector4(1, 1, 0, 0), data.blitMaterial, 4);
            }
        }

        private Matrix4x4 prevWorldToView;
        private Matrix4x4 prevWorldToClip;

        // private Matrix4x4 prevCameraMatrix;
        // private int prevBounceCountOpaque;
        // private int prevBounceCountTransparent;


        public static Vector4 GetNrdFrustum(Camera cam)
        {
            Matrix4x4 p = cam.projectionMatrix;

            // Unity 的投影矩阵 p 的元素索引:
            // [0,0] = 2n/(r-left), [0,2] = (r+left)/(r-left)
            // [1,1] = 2n/(top-bottom), [1,2] = (top+bottom)/(top-bottom)

            float x0, x1, y0, y1;

            if (!cam.orthographic)
            {
                // 透视投影重建 (基于投影矩阵的逆推)
                // 对应 C++ 中的 x0 = vPlane[PLANE_LEFT].z / vPlane[PLANE_LEFT].x
                x0 = (-1.0f - p.m02) / p.m00;
                x1 = (1.0f - p.m02) / p.m00;
                y0 = (-1.0f - p.m12) / p.m11;
                y1 = (1.0f - p.m12) / p.m11;
            }
            else
            {
                // 正交投影
                float halfHeight = cam.orthographicSize;
                float halfWidth = halfHeight * cam.aspect;
                x0 = -halfWidth;
                x1 = halfWidth;
                y0 = -halfHeight;
                y1 = halfHeight;
            }

            // 匹配 C++ 代码逻辑:
            // pfFrustum4[0] = -x0;
            // pfFrustum4[2] = x0 - x1;
            // 针对 D3D 风格 (Unity 在 GPU 端通常使用 D3D 类似约定):
            // pfFrustum4[1] = -y1;
            // pfFrustum4[3] = y1 - y0;

            return new Vector4(-x0, -y1, x0 - x1, y1 - y0);
        }


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
            passData.compositionComputeShader = compositionComputeShader;
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


            float VerticalFieldOfView = cam.fieldOfView;
            float AspectRatio = (float)cam.pixelWidth / (float)cam.pixelHeight;
            float HorizontalFieldOfView =
                Mathf.Atan(Mathf.Tan(Mathf.Deg2Rad * VerticalFieldOfView * 0.5f) * AspectRatio) * 2 * Mathf.Rad2Deg;


            var setting = new Settings
            {
                g_Zoom = tan,
                g_ConvergenceStep = NrdDenoiser.FrameIndex,
                g_FrameIndex = (uint)Time.frameCount,
                // g_SampleCount = (uint)_settings.sampleCount,
                lightOffset = _settings.sunAngularDiameter,

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


                gTanPixelAngularRadius = math.tan(0.5f * math.radians(HorizontalFieldOfView) / cam.pixelWidth),

                gUnproject = 1.0f / (0.5f * rectH * m11),
                gTanSunAngularRadius = math.tan(math.radians(_settings.sunAngularDiameter * 0.5f))
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

            passData._setting = _settings;

            // Debug.Log("Texture Width: " + textureDesc.width + " Height: " + textureDesc.height);


            // var settingsBufferHandle = renderGraph.ImportBuffer(pathTracingSettingsBuffer);


            passData.scramblingRanking = scramblingRanking;
            passData.sobol = sobol;

            passData.outputTexture = renderGraph.CreateTexture(textureDesc);
            
 
            passData.Mv = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_MV));
            passData.ViewZ = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_VIEWZ));
            passData.Normal_Roughness = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_NORMAL_ROUGHNESS));
            var rtHandle = NrdDenoiser.GetRT(ResourceType.IN_BASECOLOR_METALNESS);
            
            passData.BaseColor_Metalness =
                renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_BASECOLOR_METALNESS));

            passData.DirectLighting = CreateTex(textureDesc, renderGraph, "DirectLighting",
                GraphicsFormat.B10G11R11_UFloatPack32);
            passData.DirectEmission = CreateTex(textureDesc, renderGraph, "DirectEmission",
                GraphicsFormat.B10G11R11_UFloatPack32);

            passData.Penumbra = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_PENUMBRA));
            passData.Diff = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_DIFF_RADIANCE_HITDIST));
            passData.Spec = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_SPEC_RADIANCE_HITDIST));

            // 输出
            passData.Shadow_Translucency =
                renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.OUT_SHADOW_TRANSLUCENCY));
            passData.DenoisedDiff =
                renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.OUT_DIFF_RADIANCE_HITDIST));
            passData.DenoisedSpec =
                renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.OUT_SPEC_RADIANCE_HITDIST));
            passData.Validation = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.OUT_VALIDATION));

            
            passData.ComposedDiff = CreateTex(textureDesc, renderGraph, "ComposedDiff",
                GraphicsFormat.R16G16B16A16_SFloat);
            passData.ComposedSpec_ViewZ = CreateTex(textureDesc, renderGraph, "ComposedSpec_ViewZ",
                GraphicsFormat.R16G16B16A16_SFloat);

            rayTracingShader.SetShaderPass("Test2");

            // rayTracingShader.SetTexture(g_EnvTexID, _settings.envTexture);

            // Shader.SetGlobalInt(g_BounceCountOpaqueID, _settings.bounceCountOpaque);
            // Shader.SetGlobalInt(g_BounceCountTransparentID, _settings.bounceCountTransparent);


            Shader.SetGlobalVector(g_DirLightDirectionID, gSunDirection);
            Shader.SetGlobalColor(g_DirLightColorID, mainLight.finalColor);

            // accelerationStructure.Build();
            Shader.SetGlobalRayTracingAccelerationStructure(g_AccelStructID, accelerationStructure);

            compositionComputeShader.SetMatrix(gViewToWorldID, viewToWorld);
            compositionComputeShader.SetVector(gSunDirectionID, gSunDirection);
            compositionComputeShader.SetVector(gInvRectSizeID, new Vector2(1.0f / setting.gRectSize.x, 1.0f / setting.gRectSize.y));
            compositionComputeShader.SetVector(gCameraFrustumID, GetNrdFrustum(cameraData.camera));


            // convergenceStep++;


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

            builder.AllowPassCulling(false);

            builder.SetRenderFunc((PassData passData, UnsafeGraphContext context) =>
            {
                ExecutePass(passData, context);
            });

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