using System;
using System.Runtime.InteropServices;
using DefaultNamespace;
using Nrd;
using Unity.Mathematics;
using Unity.Profiling;
using Unity.Profiling.LowLevel;
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
        public ComputeShader DlssBeforeCs;
        public Material BiltMaterial;

        public ComputeShader SharcResolveCs;
        public RayTracingShader SharcUpdateTs;

        public GraphicsBuffer HashEntriesBuffer;
        public GraphicsBuffer AccumulationBuffer;
        public GraphicsBuffer ResolvedBuffer;
        // public PathTracingDataBuilder _dataBuilder;

        public RayTracingAccelerationStructure AccelerationStructure;
        
        public NRDDenoiser NrdDenoiser;
        public DLRRDenoiser DLRRDenoiser;

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


            internal TextureHandle RRGuide_DiffAlbedo;
            internal TextureHandle RRGuide_SpecAlbedo;
            internal TextureHandle RRGuide_SpecHitDistance;
            internal TextureHandle RRGuide_Normal_Roughness;
            internal TextureHandle DlssOutput;

            internal RayTracingShader OpaqueTs;
            internal RayTracingShader TransparentTs;
            internal ComputeShader CompositionCs;
            internal ComputeShader TaaCs;
            internal ComputeShader DlssBeforeCs;
            internal Material BlitMaterial;
            internal uint outputGridW;
            internal uint outputGridH;
            internal uint rectGridW;
            internal uint rectGridH;
            internal int2 m_RenderResolution;

            internal GlobalConstants GlobalConstants;
            internal GraphicsBuffer ConstantBuffer;
            internal IntPtr NrdDataPtr;
            internal IntPtr RRDataPtr;
            internal PathTracingSetting Setting;
            internal float resolutionScale;


            internal ComputeShader SharcResolveCs;
            internal RayTracingShader SharcUpdateTs;

            internal GraphicsBuffer HashEntriesBuffer;
            internal GraphicsBuffer AccumulationBuffer;
            internal GraphicsBuffer ResolvedBuffer;
            // internal PathTracingDataBuilder _dataBuilder;
        }

        public PathTracingPassSingle(PathTracingSetting setting)
        {
            m_Settings = setting;
            _pathTracingSettingsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Constant, 1, Marshal.SizeOf<GlobalConstants>());
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            natCmd.SetBufferData(data.ConstantBuffer, new[] { data.GlobalConstants });

            var sharcUpdateMarker = new ProfilerMarker(ProfilerCategory.Render, "Sharc Update", MarkerFlags.SampleGPU);
            var sharcResolveMarker = new ProfilerMarker(ProfilerCategory.Render, "Sharc Resolve", MarkerFlags.SampleGPU);
            var opaqueTracingMarker = new ProfilerMarker(ProfilerCategory.Render, "Opaque Tracing", MarkerFlags.SampleGPU);
            var nrdDenoiseMarker = new ProfilerMarker(ProfilerCategory.Render, "NRD Denoise", MarkerFlags.SampleGPU);
            var compositionMarker = new ProfilerMarker(ProfilerCategory.Render, "Composition", MarkerFlags.SampleGPU);
            var transparentTracingMarker = new ProfilerMarker(ProfilerCategory.Render, "Transparent Tracing", MarkerFlags.SampleGPU);
            var taaMarker = new ProfilerMarker(ProfilerCategory.Render, "TAA", MarkerFlags.SampleGPU);
            var dlssBeforeMarker = new ProfilerMarker(ProfilerCategory.Render, "DLSS Before", MarkerFlags.SampleGPU);
            var dlssDenoiseMarker = new ProfilerMarker(ProfilerCategory.Render, "DLSS Denoise", MarkerFlags.SampleGPU);
            var outputBlitMarker = new ProfilerMarker(ProfilerCategory.Render, "Output Blit", MarkerFlags.SampleGPU);


            // Sharc update
            {
                natCmd.BeginSample(sharcUpdateMarker);
                natCmd.SetRayTracingShaderPass(data.SharcUpdateTs, "Test2");
                natCmd.SetRayTracingConstantBufferParam(data.SharcUpdateTs, paramsID, data.ConstantBuffer, 0, data.ConstantBuffer.stride);

                natCmd.SetRayTracingBufferParam(data.SharcUpdateTs, g_HashEntriesID, data.HashEntriesBuffer);
                natCmd.SetRayTracingBufferParam(data.SharcUpdateTs, g_AccumulationBufferID, data.AccumulationBuffer);
                natCmd.SetRayTracingBufferParam(data.SharcUpdateTs, g_ResolvedBufferID, data.ResolvedBuffer);

                natCmd.SetRayTracingTextureParam(data.SharcUpdateTs, g_OutputID, data.OutputTexture);

                int SHARC_DOWNSCALE = 4;

                uint w = (uint)(data.m_RenderResolution.x / SHARC_DOWNSCALE);
                uint h = (uint)(data.m_RenderResolution.y / SHARC_DOWNSCALE);

                natCmd.DispatchRays(data.SharcUpdateTs, "MainRayGenShader", w, h, 1);
                natCmd.EndSample(sharcUpdateMarker);
            }


            // Sharc resolve
            {
                natCmd.BeginSample(sharcResolveMarker);
                natCmd.SetComputeConstantBufferParam(data.SharcResolveCs, paramsID, data.ConstantBuffer, 0, data.ConstantBuffer.stride);
                natCmd.SetComputeBufferParam(data.SharcResolveCs, 0, g_HashEntriesID, data.HashEntriesBuffer);
                natCmd.SetComputeBufferParam(data.SharcResolveCs, 0, g_AccumulationBufferID, data.AccumulationBuffer);
                natCmd.SetComputeBufferParam(data.SharcResolveCs, 0, g_ResolvedBufferID, data.ResolvedBuffer);

                ulong SHARC_CAPACITY = 1 << 22;
                ulong LINEAR_BLOCK_SIZE = 256;
                int x = (int)((SHARC_CAPACITY + LINEAR_BLOCK_SIZE - 1) / LINEAR_BLOCK_SIZE);

                natCmd.DispatchCompute(data.SharcResolveCs, 0, x, 1, 1);

                natCmd.EndSample(sharcResolveMarker);
            }

            // 不透明
            {
                natCmd.BeginSample(opaqueTracingMarker);

                // natCmd.SetGlobalBuffer(gIn_InstanceDataID, data._dataBuilder._instanceBuffer);
                // natCmd.SetGlobalBuffer(gIn_PrimitiveDataID, data._dataBuilder._primitiveBuffer);


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

                // Debug.Log(data.m_RenderResolution);

                uint rectWmod = (uint)(data.m_RenderResolution.x * data.resolutionScale + 0.5f);
                uint rectHmod = (uint)(data.m_RenderResolution.y * data.resolutionScale + 0.5f);

                // Debug.Log($"Dispatch Rays Size: {rectWmod} x {rectHmod}");


                natCmd.DispatchRays(data.OpaqueTs, "MainRayGenShader", rectWmod, rectHmod, 1);

                natCmd.EndSample(opaqueTracingMarker);
            }


            // NRD降噪
            if (!data.Setting.RR)
            {
                natCmd.BeginSample(nrdDenoiseMarker);
                natCmd.IssuePluginEventAndData(GetRenderEventAndDataFunc(), 1, data.NrdDataPtr);
                natCmd.EndSample(nrdDenoiseMarker);
            }


            // 合成
            {
                natCmd.BeginSample(compositionMarker);
                natCmd.SetComputeConstantBufferParam(data.CompositionCs, paramsID, data.ConstantBuffer, 0, data.ConstantBuffer.stride);
                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_ViewZID, data.ViewZ);
                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_Normal_RoughnessID, data.NormalRoughness);
                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_BaseColor_MetalnessID, data.BaseColorMetalness);
                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_DirectLightingID, data.DirectLighting);
                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_DirectEmissionID, data.DirectEmission);
                if (data.Setting.RR)
                {
                    natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_ShadowID, data.Penumbra);
                    natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_DiffID, data.Diff);
                    natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_SpecID, data.Spec);
                }
                else
                {
                    natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_ShadowID, data.ShadowTranslucency);
                    natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_DiffID, data.DenoisedDiff);
                    natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_SpecID, data.DenoisedSpec);
                }

                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_PsrThroughputID, data.PsrThroughput);
                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gOut_ComposedDiffID, data.ComposedDiff);
                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gOut_ComposedSpec_ViewZID, data.ComposedSpecViewZ);

                natCmd.DispatchCompute(data.CompositionCs, 0, (int)data.rectGridW, (int)data.rectGridH, 1);

                natCmd.EndSample(compositionMarker);
            }


            // 透明
            {
                natCmd.BeginSample(transparentTracingMarker);

                natCmd.SetRayTracingShaderPass(data.TransparentTs, "Test2");
                natCmd.SetRayTracingConstantBufferParam(data.TransparentTs, paramsID, data.ConstantBuffer, 0, data.ConstantBuffer.stride);

                natCmd.SetRayTracingBufferParam(data.TransparentTs, g_HashEntriesID, data.HashEntriesBuffer);
                natCmd.SetRayTracingBufferParam(data.TransparentTs, g_AccumulationBufferID, data.AccumulationBuffer);
                natCmd.SetRayTracingBufferParam(data.TransparentTs, g_ResolvedBufferID, data.ResolvedBuffer);


                natCmd.SetRayTracingTextureParam(data.TransparentTs, gIn_ComposedDiffID, data.ComposedDiff);
                natCmd.SetRayTracingTextureParam(data.TransparentTs, gIn_ComposedSpec_ViewZID, data.ComposedSpecViewZ);
                natCmd.SetRayTracingTextureParam(data.TransparentTs, g_Normal_RoughnessID, data.NormalRoughness);
                natCmd.SetRayTracingTextureParam(data.TransparentTs, gOut_ComposedID, data.Composed);
                natCmd.SetRayTracingTextureParam(data.TransparentTs, GInOutMv, data.Mv);

                natCmd.DispatchRays(data.TransparentTs, "MainRayGenShader", (uint)data.m_RenderResolution.x, (uint)data.m_RenderResolution.y, 1);
                natCmd.EndSample(transparentTracingMarker);
            }


            var isEven = (data.GlobalConstants.gFrameIndex & 1) == 0;
            var taaSrc = isEven ? data.TaaHistoryPrev : data.TaaHistory;
            var taaDst = isEven ? data.TaaHistory : data.TaaHistoryPrev;
            if (data.Setting.RR)
            {
                // dlss Before
                natCmd.BeginSample(dlssBeforeMarker);
                natCmd.SetComputeConstantBufferParam(data.DlssBeforeCs, paramsID, data.ConstantBuffer, 0, data.ConstantBuffer.stride);

                natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gIn_Normal_Roughness", data.NormalRoughness);
                natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gIn_BaseColor_Metalness", data.BaseColorMetalness);
                natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gIn_Spec", data.Spec);

                natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gInOut_ViewZ", data.ViewZ);
                natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gOut_DiffAlbedo", data.RRGuide_DiffAlbedo);
                natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gOut_SpecAlbedo", data.RRGuide_SpecAlbedo);
                natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gOut_SpecHitDistance", data.RRGuide_SpecHitDistance);
                natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gOut_Normal_Roughness", data.RRGuide_Normal_Roughness);


                natCmd.DispatchCompute(data.DlssBeforeCs, 0, (int)data.rectGridW, (int)data.rectGridH, 1);
                natCmd.EndSample(dlssBeforeMarker);

                // DLSS调用

                if (!data.Setting.tmpDisableRR)
                {
                    natCmd.BeginSample(dlssDenoiseMarker);
                    natCmd.IssuePluginEventAndData(GetRenderEventAndDataFunc(), 2, data.RRDataPtr);
                    natCmd.EndSample(dlssDenoiseMarker);
                }
            }
            else
            {
                // TAA
                natCmd.BeginSample(taaMarker);

                natCmd.SetComputeConstantBufferParam(data.TaaCs, paramsID, data.ConstantBuffer, 0, data.ConstantBuffer.stride);
                natCmd.SetComputeTextureParam(data.TaaCs, 0, gIn_MvID, data.Mv);
                natCmd.SetComputeTextureParam(data.TaaCs, 0, gIn_ComposedID, data.Composed);
                natCmd.SetComputeTextureParam(data.TaaCs, 0, gIn_HistoryID, taaSrc);
                natCmd.SetComputeTextureParam(data.TaaCs, 0, gOut_ResultID, taaDst);
                natCmd.SetComputeTextureParam(data.TaaCs, 0, gOut_DebugID, data.OutputTexture);
                natCmd.DispatchCompute(data.TaaCs, 0, (int)data.rectGridW, (int)data.rectGridH, 1);
                natCmd.EndSample(taaMarker);
            }


            // 显示输出
            natCmd.BeginSample(outputBlitMarker);

            natCmd.SetRenderTarget(data.CameraTexture);

            Vector4 scaleOffset = new Vector4(data.resolutionScale, data.resolutionScale, 0, 0);
            switch (data.Setting.showMode)
            {
                case ShowMode.None:
                    break;
                case ShowMode.BaseColor:
                    Blitter.BlitTexture(natCmd, data.BaseColorMetalness, scaleOffset, data.BlitMaterial, (int)ShowPass.showOut);
                    break;
                case ShowMode.Metalness:
                    Blitter.BlitTexture(natCmd, data.BaseColorMetalness, scaleOffset, data.BlitMaterial, (int)ShowPass.showAlpha);
                    break;
                case ShowMode.Normal:
                    Blitter.BlitTexture(natCmd, data.NormalRoughness, scaleOffset, data.BlitMaterial, (int)ShowPass.ShowNormal);
                    break;
                case ShowMode.Roughness:
                    Blitter.BlitTexture(natCmd, data.NormalRoughness, scaleOffset, data.BlitMaterial, (int)ShowPass.ShowRoughness);
                    break;
                case ShowMode.NoiseShadow:
                    Blitter.BlitTexture(natCmd, data.Penumbra, scaleOffset, data.BlitMaterial, (int)ShowPass.ShowNoiseShadow);
                    break;
                case ShowMode.Shadow:
                    Blitter.BlitTexture(natCmd, data.ShadowTranslucency, scaleOffset, data.BlitMaterial, (int)ShowPass.showShadow);
                    break;
                case ShowMode.Diffuse:
                    Blitter.BlitTexture(natCmd, data.Diff, scaleOffset, data.BlitMaterial, (int)ShowPass.ShowRadiance);
                    break;
                case ShowMode.Specular:
                    Blitter.BlitTexture(natCmd, data.Spec, scaleOffset, data.BlitMaterial, (int)ShowPass.ShowRadiance);
                    break;
                case ShowMode.DenoisedDiffuse:
                    Blitter.BlitTexture(natCmd, data.DenoisedDiff, scaleOffset, data.BlitMaterial, (int)ShowPass.ShowRadiance);
                    break;
                case ShowMode.DenoisedSpecular:
                    Blitter.BlitTexture(natCmd, data.DenoisedSpec, scaleOffset, data.BlitMaterial, (int)ShowPass.ShowRadiance);
                    break;
                case ShowMode.DirectLight:
                    Blitter.BlitTexture(natCmd, data.DirectLighting, scaleOffset, data.BlitMaterial, (int)ShowPass.showOut);
                    break;
                case ShowMode.Emissive:
                    Blitter.BlitTexture(natCmd, data.DirectEmission, scaleOffset, data.BlitMaterial, (int)ShowPass.showOut);
                    break;
                case ShowMode.Out:
                    Blitter.BlitTexture(natCmd, data.OutputTexture, scaleOffset, data.BlitMaterial, (int)ShowPass.showOut);
                    break;
                case ShowMode.ComposedDiff:
                    Blitter.BlitTexture(natCmd, data.ComposedDiff, scaleOffset, data.BlitMaterial, (int)ShowPass.showOut);
                    break;
                case ShowMode.ComposedSpec:
                    Blitter.BlitTexture(natCmd, data.ComposedSpecViewZ, scaleOffset, data.BlitMaterial, (int)ShowPass.showOut);
                    break;
                case ShowMode.Composed:
                    Blitter.BlitTexture(natCmd, data.Composed, scaleOffset, data.BlitMaterial, (int)ShowPass.showOut);
                    break;
                case ShowMode.Taa:
                    Blitter.BlitTexture(natCmd, taaDst, scaleOffset, data.BlitMaterial, (int)ShowPass.showAlpha);
                    break;
                case ShowMode.Final:

                    if (data.Setting.RR)
                    {
                        Blitter.BlitTexture(natCmd, data.DlssOutput, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.showDlss);
                    }
                    else
                    {
                        Blitter.BlitTexture(natCmd, taaDst, scaleOffset, data.BlitMaterial, (int)ShowPass.showOut);
                    }

                    break;
                case ShowMode.DLSS_DiffuseAlbedo:
                    Blitter.BlitTexture(natCmd, data.RRGuide_DiffAlbedo, scaleOffset, data.BlitMaterial, (int)ShowPass.showOut);
                    break;
                case ShowMode.DLSS_SpecularAlbedo:
                    Blitter.BlitTexture(natCmd, data.RRGuide_SpecAlbedo, scaleOffset, data.BlitMaterial, (int)ShowPass.showOut);
                    break;
                case ShowMode.DLSS_SpecularHitDistance:
                    Blitter.BlitTexture(natCmd, data.RRGuide_SpecHitDistance, scaleOffset, data.BlitMaterial, (int)ShowPass.showOut);
                    break;
                case ShowMode.DLSS_NormalRoughness:
                    Blitter.BlitTexture(natCmd, data.RRGuide_Normal_Roughness, scaleOffset, data.BlitMaterial, (int)ShowPass.showOut);
                    break;
                case ShowMode.DLSS_Output:
                    Blitter.BlitTexture(natCmd, data.DlssOutput, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.showOut);
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

            natCmd.EndSample(outputBlitMarker);
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

            int2 outputResolution = new int2(cameraData.camera.pixelWidth, cameraData.camera.pixelHeight);

            NrdDenoiser.EnsureResources(outputResolution);

            // Shader.SetGlobalRayTracingAccelerationStructure(g_AccelStructID, _dataBuilder.accelerationStructure);
            Shader.SetGlobalRayTracingAccelerationStructure(g_AccelStructID, AccelerationStructure);

            using var builder = renderGraph.AddUnsafePass<PassData>("Path Tracing Pass", out var passData);

            passData.OpaqueTs = OpaqueTs;
            passData.TransparentTs = TransparentTs;
            passData.CompositionCs = CompositionCs;
            passData.TaaCs = TaaCs;
            passData.DlssBeforeCs = DlssBeforeCs;
            passData.BlitMaterial = BiltMaterial;

            passData.SharcResolveCs = SharcResolveCs;
            passData.SharcUpdateTs = SharcUpdateTs;
            passData.AccumulationBuffer = AccumulationBuffer;
            passData.HashEntriesBuffer = HashEntriesBuffer;
            passData.ResolvedBuffer = ResolvedBuffer;
            // passData._dataBuilder = _dataBuilder;

            var gSunDirection = -lightForward;
            var up = new Vector3(0, 1, 0);
            var gSunBasisX = math.normalize(math.cross(new float3(up.x, up.y, up.z), new float3(gSunDirection.x, gSunDirection.y, gSunDirection.z)));
            var gSunBasisY = math.normalize(math.cross(new float3(gSunDirection.x, gSunDirection.y, gSunDirection.z), gSunBasisX));

            var cam = cameraData.camera;
            passData.NrdDataPtr = NrdDenoiser.GetInteropDataPtr(cam, gSunDirection);
            passData.RRDataPtr = DLRRDenoiser.GetInteropDataPtr(cam, NrdDenoiser);

            var m11 = cam.projectionMatrix.m11;

            var renderResolution = NrdDenoiser.renderResolution;

            var rectW = (uint)(renderResolution.x * NrdDenoiser.resolutionScale + 0.5f);
            var rectH = (uint)(renderResolution.y * NrdDenoiser.resolutionScale + 0.5f);

            // todo prev
            var rectWprev = (uint)(renderResolution.x * NrdDenoiser.prevResolutionScale + 0.5f);
            var rectHprev = (uint)(renderResolution.y * NrdDenoiser.prevResolutionScale + 0.5f);


            var renderSize = new float2((renderResolution.x), (renderResolution.y));
            var outputSize = new float2((outputResolution.x), (outputResolution.y));
            var rectSize = new float2(rectW, rectH);

            var rectSizePrev = new float2((rectWprev), (rectHprev));
            var jitter = (m_Settings.cameraJitter ? NrdDenoiser.ViewportJitter : 0f) / rectSize;

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
                gCameraGlobalPos = new float4(NrdDenoiser.camPos, 0),
                gCameraGlobalPosPrev = new float4(NrdDenoiser.prevCamPos, 0),
                gViewDirection = new float4(cam.transform.forward, 0),
                gHairBaseColor = new float4(0.1f, 0.1f, 0.1f, 1.0f),

                gHairBetas = new float2(0.25f, 0.3f),
                gOutputSize = outputSize,
                gRenderSize = renderSize,
                gRectSize = rectSize,
                gInvOutputSize = new float2(1.0f, 1.0f) / outputSize,
                gInvRenderSize = new float2(1.0f, 1.0f) / renderSize,
                gInvRectSize = new float2(1.0f, 1.0f) / rectSize,
                gRectSizePrev = rectSizePrev,
                gJitter = jitter,

                gEmissionIntensity = emissionIntensity,
                gNearZ = -cam.nearClipPlane,
                gSeparator = m_Settings.splitScreen,
                gRoughnessOverride = 0,
                gMetalnessOverride = 0,
                gUnitToMetersMultiplier = 1.0f,
                gTanSunAngularRadius = math.tan(math.radians(m_Settings.sunAngularDiameter * 0.5f)),
                gTanPixelAngularRadius = math.tan(0.5f * math.radians(horizontalFieldOfView) / rectSize.x),
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
            textureDesc.width = renderResolution.x;
            textureDesc.height = renderResolution.y;

            CreateTextureHandle(renderGraph, passData, textureDesc, builder);

            passData.GlobalConstants = globalConstants;
            passData.CameraTexture = resourceData.activeColorTexture;
            passData.outputGridW = (uint)((renderResolution.x + 15) / 16);
            passData.outputGridH = (uint)((renderResolution.y + 15) / 16);
            passData.rectGridW = (uint)((rectW + 15) / 16);
            passData.rectGridH = (uint)((rectH + 15) / 16);
            passData.m_RenderResolution = renderResolution;


            passData.ConstantBuffer = _pathTracingSettingsBuffer;
            passData.Setting = m_Settings;
            passData.resolutionScale = NrdDenoiser.resolutionScale;
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

            passData.Composed = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.Composed));

            passData.TaaHistory = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.TaaHistory));
            passData.TaaHistoryPrev = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.TaaHistoryPrev));
            passData.PsrThroughput = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.PsrThroughput));

            passData.RRGuide_DiffAlbedo = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.RRGuide_DiffAlbedo));
            passData.RRGuide_SpecAlbedo = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.RRGuide_SpecAlbedo));
            passData.RRGuide_SpecHitDistance = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.RRGuide_SpecHitDistance));
            passData.RRGuide_Normal_Roughness = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.RRGuide_Normal_Roughness));
            passData.DlssOutput = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.DlssOutput));


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

            builder.UseTexture(passData.RRGuide_DiffAlbedo, AccessFlags.ReadWrite);
            builder.UseTexture(passData.RRGuide_SpecAlbedo, AccessFlags.ReadWrite);
            builder.UseTexture(passData.RRGuide_SpecHitDistance, AccessFlags.ReadWrite);
            builder.UseTexture(passData.RRGuide_Normal_Roughness, AccessFlags.ReadWrite);
            builder.UseTexture(passData.DlssOutput, AccessFlags.ReadWrite);
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