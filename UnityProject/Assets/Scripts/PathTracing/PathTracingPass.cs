using System;
using System.Runtime.InteropServices;
using DefaultNamespace;
using mini;
using Nrd;
using RTXDI;
using Rtxdi.DI;
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
    public class PathTracingPass : ScriptableRenderPass
    {
        private static readonly int GInOutMv = Shader.PropertyToID("gInOut_Mv");
        
        public RayTracingShader TransparentTs;
        public ComputeShader CompositionCs;
        public ComputeShader TaaCs;
        public ComputeShader DlssBeforeCs;
        public Material BiltMaterial;

        public GraphicsBuffer HashEntriesBuffer;
        public GraphicsBuffer AccumulationBuffer;
        public GraphicsBuffer ResolvedBuffer;


        public NRDDenoiser NrdDenoiser;
        public DLRRDenoiser DLRRDenoiser;


        // Auto-exposure
        public ComputeShader AutoExposureCs;
        public GraphicsBuffer AeHistogramBuffer;
        public GraphicsBuffer AeExposureBuffer;

        private readonly PathTracingSetting m_Settings;
        public  GraphicsBuffer _pathTracingSettingsBuffer;
        
        public GraphicsBuffer m_SpotLightBuffer;
        public GraphicsBuffer m_AreaLightBuffer;
        public GraphicsBuffer m_PointLightBuffer;
        


        [DllImport("RenderingPlugin")]
        private static extern IntPtr GetRenderEventAndDataFunc();

        class PassData
        {
            internal TextureHandle CameraTexture;

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

            // RTXDI：上一帧 GBuffer
            internal TextureHandle PrevViewZ;
            internal TextureHandle PrevNormalRoughness;
            internal TextureHandle PrevBaseColorMetalness;

            internal RayTracingShader TransparentTs;
            internal ComputeShader CompositionCs;
            internal ComputeShader TaaCs;
            internal ComputeShader DlssBeforeCs;
            internal Material BlitMaterial;
            internal uint rectGridW;
            internal uint rectGridH;
            internal int2 m_RenderResolution;

            internal GraphicsBuffer ConstantBuffer;
            // internal IntPtr NrdDataPtr;
            internal IntPtr RRDataPtr;
            internal PathTracingSetting Setting;
            internal float resolutionScale;



            internal GraphicsBuffer HashEntriesBuffer;
            internal GraphicsBuffer AccumulationBuffer;

            internal GraphicsBuffer ResolvedBuffer;

            internal int passIndex;
            // internal PathTracingDataBuilder _dataBuilder;

            // internal TextureHandle SpotDirect;
            internal GraphicsBuffer SpotLightBuffer;
            internal GraphicsBuffer AreaLightBuffer;
            internal GraphicsBuffer PointLightBuffer;

            // ── Auto-exposure ──
            internal ComputeShader AeCs;
            internal GraphicsBuffer AeHistogramBuffer;
            internal GraphicsBuffer AeExposureBuffer;
            internal bool AeEnabled;
            internal float AeEVMin;
            internal float AeEVMax;
            internal float AeLowPercent;
            internal float AeHighPercent;
            internal float AeSpeedUp;
            internal float AeSpeedDown;
            internal float AeDeltaTime;
            internal float AeExposureCompensation;
            internal float AeMinExposure;
            internal float AeMaxExposure;
            internal uint AeTexWidth;
            internal uint AeTexHeight;
            internal float ManualExposure;
            
        }

        public PathTracingPass(PathTracingSetting setting)
        {
            m_Settings = setting;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            if (data.passIndex != 0)
            {
                return;
            }

            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            // Bind the exposure buffer globally so all shaders can read the current EV.
            // When auto-exposure is OFF: seed the buffer with the manual value from settings.
            // When auto-exposure is ON:  the buffer is updated later by ReduceHistogram.
            natCmd.SetGlobalBuffer("_AE_ExposureBuffer", data.AeExposureBuffer);
            if (!data.AeEnabled)
            {
                natCmd.SetBufferData(data.AeExposureBuffer, new[] { data.ManualExposure });
            }

            var transparentTracingMarker = new ProfilerMarker(ProfilerCategory.Render, "Transparent Tracing", MarkerFlags.SampleGPU);
            var taaMarker = new ProfilerMarker(ProfilerCategory.Render, "TAA", MarkerFlags.SampleGPU);
            var dlssBeforeMarker = new ProfilerMarker(ProfilerCategory.Render, "DLSS Before", MarkerFlags.SampleGPU);
            var dlssDenoiseMarker = new ProfilerMarker(ProfilerCategory.Render, "DLSS Denoise", MarkerFlags.SampleGPU);
            var outputBlitMarker = new ProfilerMarker(ProfilerCategory.Render, "Output Blit", MarkerFlags.SampleGPU);
            var aeMarker = new ProfilerMarker(ProfilerCategory.Render, "Auto Exposure", MarkerFlags.SampleGPU);





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

                natCmd.SetRayTracingBufferParam(data.TransparentTs, gIn_SpotLightsID, data.SpotLightBuffer);
                natCmd.SetRayTracingBufferParam(data.TransparentTs, gIn_AreaLightsID, data.AreaLightBuffer);
                natCmd.SetRayTracingBufferParam(data.TransparentTs, gIn_PointLightsID, data.PointLightBuffer);

                natCmd.DispatchRays(data.TransparentTs, "MainRayGenShader", (uint)data.m_RenderResolution.x, (uint)data.m_RenderResolution.y, 1);
                natCmd.EndSample(transparentTracingMarker);
            }


            // ── Auto-exposure: histogram build + reduce (after transparent, before TAA) ──
            if (data.AeEnabled && data.AeCs != null && data.AeHistogramBuffer != null && data.AeExposureBuffer != null)
            {
                natCmd.BeginSample(aeMarker);

                int kernelClear  = data.AeCs.FindKernel("ClearHistogram");
                int kernelBuild  = data.AeCs.FindKernel("BuildHistogram");
                int kernelReduce = data.AeCs.FindKernel("ReduceHistogram");

                // -- Kernel 0: Clear --
                natCmd.SetComputeBufferParam(data.AeCs, kernelClear, "_AE_HistogramBuffer", data.AeHistogramBuffer);
                natCmd.DispatchCompute(data.AeCs, kernelClear, 1, 1, 1);

                // -- Kernel 1: Build --
                natCmd.SetComputeTextureParam(data.AeCs, kernelBuild, "_AE_ComposedTexture", data.Composed);
                natCmd.SetComputeBufferParam(data.AeCs, kernelBuild, "_AE_HistogramBuffer", data.AeHistogramBuffer);
                natCmd.SetComputeIntParam(data.AeCs, "_AE_TexWidth",  (int)data.AeTexWidth);
                natCmd.SetComputeIntParam(data.AeCs, "_AE_TexHeight", (int)data.AeTexHeight);
                natCmd.SetComputeFloatParam(data.AeCs, "_AE_EVMin", data.AeEVMin);
                natCmd.SetComputeFloatParam(data.AeCs, "_AE_EVMax", data.AeEVMax);
                uint buildX = (data.AeTexWidth  + 15u) / 16u;
                uint buildY = (data.AeTexHeight + 15u) / 16u;
                natCmd.DispatchCompute(data.AeCs, kernelBuild, (int)buildX, (int)buildY, 1);

                // -- Kernel 2: Reduce --
                natCmd.SetComputeBufferParam(data.AeCs, kernelReduce, "_AE_HistogramBuffer", data.AeHistogramBuffer);
                natCmd.SetComputeBufferParam(data.AeCs, kernelReduce, "_AE_ExposureBuffer",  data.AeExposureBuffer);
                natCmd.SetComputeFloatParam(data.AeCs, "_AE_EVMin",                data.AeEVMin);
                natCmd.SetComputeFloatParam(data.AeCs, "_AE_EVMax",                data.AeEVMax);
                natCmd.SetComputeFloatParam(data.AeCs, "_AE_LowPercent",           data.AeLowPercent);
                natCmd.SetComputeFloatParam(data.AeCs, "_AE_HighPercent",          data.AeHighPercent);
                natCmd.SetComputeFloatParam(data.AeCs, "_AE_SpeedUp",              data.AeSpeedUp);
                natCmd.SetComputeFloatParam(data.AeCs, "_AE_SpeedDown",            data.AeSpeedDown);
                natCmd.SetComputeFloatParam(data.AeCs, "_AE_DeltaTime",            data.AeDeltaTime);
                natCmd.SetComputeFloatParam(data.AeCs, "_AE_ExposureCompensation", data.AeExposureCompensation);
                natCmd.SetComputeFloatParam(data.AeCs, "_AE_MinExposure",          data.AeMinExposure);
                natCmd.SetComputeFloatParam(data.AeCs, "_AE_MaxExposure",          data.AeMaxExposure);
                natCmd.DispatchCompute(data.AeCs, kernelReduce, 1, 1, 1);

                natCmd.EndSample(aeMarker);
            }


            // var isEven = (data.GlobalConstants.gFrameIndex & 1) == 0;
            var isEven = false;
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
                    Blitter.BlitTexture(natCmd, data.BaseColorMetalness, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.Metalness:
                    Blitter.BlitTexture(natCmd, data.BaseColorMetalness, scaleOffset, data.BlitMaterial, (int)ShowPass.Alpha);
                    break;
                case ShowMode.Normal:
                    Blitter.BlitTexture(natCmd, data.NormalRoughness, scaleOffset, data.BlitMaterial, (int)ShowPass.Normal);
                    break;
                case ShowMode.Roughness:
                    Blitter.BlitTexture(natCmd, data.NormalRoughness, scaleOffset, data.BlitMaterial, (int)ShowPass.Roughness);
                    break;
                case ShowMode.NoiseShadow:
                    Blitter.BlitTexture(natCmd, data.Penumbra, scaleOffset, data.BlitMaterial, (int)ShowPass.NoiseShadow);
                    break;
                case ShowMode.Shadow:
                    Blitter.BlitTexture(natCmd, data.ShadowTranslucency, scaleOffset, data.BlitMaterial, (int)ShowPass.Shadow);
                    break;
                case ShowMode.Diffuse:
                    Blitter.BlitTexture(natCmd, data.Diff, scaleOffset, data.BlitMaterial, (int)ShowPass.Radiance);
                    break;
                case ShowMode.Specular:
                    Blitter.BlitTexture(natCmd, data.Spec, scaleOffset, data.BlitMaterial, (int)ShowPass.Radiance);
                    break;
                case ShowMode.DenoisedDiffuse:
                    Blitter.BlitTexture(natCmd, data.DenoisedDiff, scaleOffset, data.BlitMaterial, (int)ShowPass.Radiance);
                    break;
                case ShowMode.DenoisedSpecular:
                    Blitter.BlitTexture(natCmd, data.DenoisedSpec, scaleOffset, data.BlitMaterial, (int)ShowPass.Radiance);
                    break;
                case ShowMode.DirectLight:
                    Blitter.BlitTexture(natCmd, data.DirectLighting, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.Emissive:
                    Blitter.BlitTexture(natCmd, data.DirectEmission, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.Out:
                    Blitter.BlitTexture(natCmd, data.OutputTexture, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.ComposedDiff:
                    Blitter.BlitTexture(natCmd, data.ComposedDiff, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.ComposedSpec:
                    Blitter.BlitTexture(natCmd, data.ComposedSpecViewZ, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.Composed:
                    Blitter.BlitTexture(natCmd, data.Composed, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.Taa:
                    Blitter.BlitTexture(natCmd, taaDst, scaleOffset, data.BlitMaterial, (int)ShowPass.Alpha);
                    break;
                case ShowMode.Final:

                    if (data.Setting.RR)
                    {
                        Blitter.BlitTexture(natCmd, data.DlssOutput, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.Dlss);
                    }
                    else
                    {
                        Blitter.BlitTexture(natCmd, taaDst, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    }

                    break;
                case ShowMode.DLSS_DiffuseAlbedo:
                    Blitter.BlitTexture(natCmd, data.RRGuide_DiffAlbedo, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.DLSS_SpecularAlbedo:
                    Blitter.BlitTexture(natCmd, data.RRGuide_SpecAlbedo, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.DLSS_SpecularHitDistance:
                    Blitter.BlitTexture(natCmd, data.RRGuide_SpecHitDistance, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.DLSS_NormalRoughness:
                    Blitter.BlitTexture(natCmd, data.RRGuide_Normal_Roughness, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.DLSS_Output:
                    Blitter.BlitTexture(natCmd, data.DlssOutput, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.Out);
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

            if (data.Setting.showMV)
            {
                Blitter.BlitTexture(natCmd, data.Mv, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.Mv);
            }

            if (data.Setting.showValidation)
            {
                Blitter.BlitTexture(natCmd, data.Validation, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.Validation);
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


            var universalLightData = frameData.Get<UniversalLightData>();
            var lightData = universalLightData;
            var mainLight = lightData.mainLightIndex >= 0 ? lightData.visibleLights[lightData.mainLightIndex] : default;
            var mat = mainLight.localToWorldMatrix;
            Vector3 lightForward = mat.GetColumn(2);
            var resourceData = frameData.Get<UniversalResourceData>();
            var xrPass = cameraData.xr;
            var isXr = xrPass.enabled;
            var renderResolution = NrdDenoiser.renderResolution;

            using var builder = renderGraph.AddUnsafePass<PassData>("Path Tracing Pass", out var passData);

            passData.TransparentTs = TransparentTs;
            passData.CompositionCs = CompositionCs;
            passData.TaaCs = TaaCs;
            passData.DlssBeforeCs = DlssBeforeCs;
            passData.BlitMaterial = BiltMaterial;

            passData.AccumulationBuffer = AccumulationBuffer;
            passData.HashEntriesBuffer = HashEntriesBuffer;
            passData.ResolvedBuffer = ResolvedBuffer;
            passData.passIndex = isXr ? xrPass.multipassId : 0;
            // passData._dataBuilder = _dataBuilder;
            passData.SpotLightBuffer  = m_SpotLightBuffer;
            passData.AreaLightBuffer  = m_AreaLightBuffer;
            passData.PointLightBuffer = m_PointLightBuffer;

            // Auto-exposure pass data
            passData.AeCs                  = AutoExposureCs;
            passData.AeHistogramBuffer     = AeHistogramBuffer;
            passData.AeExposureBuffer      = AeExposureBuffer;
            passData.AeEnabled             = m_Settings.enableAutoExposure;
            passData.AeEVMin               = m_Settings.aeEVMin;
            passData.AeEVMax               = m_Settings.aeEVMax;
            passData.AeLowPercent          = m_Settings.aeLowPercent;
            passData.AeHighPercent         = m_Settings.aeHighPercent;
            passData.AeSpeedUp             = m_Settings.aeAdaptationSpeedUp;
            passData.AeSpeedDown           = m_Settings.aeAdaptationSpeedDown;
            passData.AeDeltaTime           = Time.deltaTime;
            passData.AeExposureCompensation = m_Settings.aeExposureCompensation;
            passData.AeMinExposure         = m_Settings.aeMinExposure;
            passData.AeMaxExposure         = m_Settings.aeMaxExposure;
            passData.AeTexWidth            = (uint)renderResolution.x;
            passData.AeTexHeight           = (uint)renderResolution.y;
            passData.ManualExposure        = m_Settings.exposure;


            passData.RRDataPtr = DLRRDenoiser.GetInteropDataPtr(cameraData, NrdDenoiser);



            var textureDesc = resourceData.activeColorTexture.GetDescriptor(renderGraph);
            textureDesc.enableRandomWrite = true;
            textureDesc.depthBufferBits = 0;
            textureDesc.clearBuffer = false;
            textureDesc.discardBuffer = false;
            textureDesc.width = renderResolution.x;
            textureDesc.height = renderResolution.y;

            CreateTextureHandle(renderGraph, passData, textureDesc, builder);
            
            var ptContextItem = frameData.Get<PTContextItem>();


            passData.OutputTexture = ptContextItem.OutputTexture;
            passData.DirectLighting = ptContextItem.DirectLighting;
            passData.DirectEmission = ptContextItem.DirectEmission;
            passData.ComposedDiff = ptContextItem.ComposedDiff;
            passData.ComposedSpecViewZ = ptContextItem.ComposedSpecViewZ;

            builder.UseTexture(passData.OutputTexture,  AccessFlags.ReadWrite);
            builder.UseTexture(passData.DirectLighting,  AccessFlags.ReadWrite);
            builder.UseTexture(passData.DirectEmission,  AccessFlags.ReadWrite);
            builder.UseTexture(passData.ComposedDiff,  AccessFlags.ReadWrite);
            builder.UseTexture(passData.ComposedSpecViewZ,  AccessFlags.ReadWrite);
            
            
            var rectW = (uint)(renderResolution.x * NrdDenoiser.resolutionScale + 0.5f);
            var rectH = (uint)(renderResolution.y * NrdDenoiser.resolutionScale + 0.5f);
            
            
            passData.CameraTexture = resourceData.activeColorTexture;
            passData.rectGridW = (uint)((rectW + 15) / 16);
            passData.rectGridH = (uint)((rectH + 15) / 16);
            passData.m_RenderResolution = renderResolution;


            passData.ConstantBuffer = _pathTracingSettingsBuffer;
            
            passData.Setting = m_Settings;
            passData.resolutionScale = NrdDenoiser.resolutionScale;

            builder.UseTexture(passData.CameraTexture, AccessFlags.Write);

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }

        private void CreateTextureHandle(RenderGraph renderGraph, PassData passData, TextureDesc textureDesc, IUnsafeRenderGraphBuilder builder)
        {

            passData.Mv = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_MV));
            passData.ViewZ = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_VIEWZ));
            passData.NormalRoughness = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_NORMAL_ROUGHNESS));

            passData.BaseColorMetalness = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_BASECOLOR_METALNESS));


            passData.Penumbra = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_PENUMBRA));
            passData.Diff = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_DIFF_RADIANCE_HITDIST));
            passData.Spec = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_SPEC_RADIANCE_HITDIST));

            // 输出
            passData.ShadowTranslucency = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.OUT_SHADOW_TRANSLUCENCY));
            passData.DenoisedDiff = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.OUT_DIFF_RADIANCE_HITDIST));
            passData.DenoisedSpec = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.OUT_SPEC_RADIANCE_HITDIST));
            passData.Validation = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.OUT_VALIDATION));


            passData.Composed = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.Composed));

            passData.TaaHistory = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.TaaHistory));
            passData.TaaHistoryPrev = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.TaaHistoryPrev));
            passData.PsrThroughput = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.PsrThroughput));

            passData.RRGuide_DiffAlbedo = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.RRGuide_DiffAlbedo));
            passData.RRGuide_SpecAlbedo = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.RRGuide_SpecAlbedo));
            passData.RRGuide_SpecHitDistance = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.RRGuide_SpecHitDistance));
            passData.RRGuide_Normal_Roughness = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.RRGuide_Normal_Roughness));
            passData.DlssOutput = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.DlssOutput));

            // RTXDI：上一帧 GBuffer
            passData.PrevViewZ = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.Prev_ViewZ));
            passData.PrevNormalRoughness = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.Prev_NormalRoughness));
            passData.PrevBaseColorMetalness = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.Prev_BaseColorMetalness));



            builder.UseTexture(passData.Mv, AccessFlags.ReadWrite);
            builder.UseTexture(passData.ViewZ, AccessFlags.ReadWrite);
            builder.UseTexture(passData.NormalRoughness, AccessFlags.ReadWrite);
            builder.UseTexture(passData.BaseColorMetalness, AccessFlags.ReadWrite);


            builder.UseTexture(passData.Penumbra, AccessFlags.ReadWrite);
            builder.UseTexture(passData.Diff, AccessFlags.ReadWrite);
            builder.UseTexture(passData.Spec, AccessFlags.ReadWrite);

            // 输出
            builder.UseTexture(passData.ShadowTranslucency, AccessFlags.ReadWrite);
            builder.UseTexture(passData.DenoisedDiff, AccessFlags.ReadWrite);
            builder.UseTexture(passData.DenoisedSpec, AccessFlags.ReadWrite);
            builder.UseTexture(passData.Validation, AccessFlags.ReadWrite);

            builder.UseTexture(passData.Composed, AccessFlags.ReadWrite);

            builder.UseTexture(passData.TaaHistory, AccessFlags.ReadWrite);
            builder.UseTexture(passData.TaaHistoryPrev, AccessFlags.ReadWrite);
            builder.UseTexture(passData.PsrThroughput, AccessFlags.ReadWrite);

            builder.UseTexture(passData.RRGuide_DiffAlbedo, AccessFlags.ReadWrite);
            builder.UseTexture(passData.RRGuide_SpecAlbedo, AccessFlags.ReadWrite);
            builder.UseTexture(passData.RRGuide_SpecHitDistance, AccessFlags.ReadWrite);
            builder.UseTexture(passData.RRGuide_Normal_Roughness, AccessFlags.ReadWrite);
            builder.UseTexture(passData.DlssOutput, AccessFlags.ReadWrite);

            builder.UseTexture(passData.PrevViewZ, AccessFlags.ReadWrite);
            builder.UseTexture(passData.PrevNormalRoughness, AccessFlags.ReadWrite);
            builder.UseTexture(passData.PrevBaseColorMetalness, AccessFlags.ReadWrite);
        }

        public void Setup()
        {
            
        }
    }
}

