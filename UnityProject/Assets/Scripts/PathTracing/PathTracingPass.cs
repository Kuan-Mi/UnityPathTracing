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
        public ComputeShader TaaCs;
        public ComputeShader DlssBeforeCs;
        public Material BiltMaterial;

        public NRDDenoiser NrdDenoiser;
        public DLRRDenoiser DLRRDenoiser;

        private readonly PathTracingSetting m_Settings;
        public  GraphicsBuffer _pathTracingSettingsBuffer;

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

            
            internal ComputeShader TaaCs;
            internal ComputeShader DlssBeforeCs;
            internal Material BlitMaterial;
            internal uint rectGridW;
            internal uint rectGridH;

            internal GraphicsBuffer ConstantBuffer;
            
            internal IntPtr RRDataPtr;
            internal PathTracingSetting Setting;
            internal float resolutionScale;

            internal int passIndex;
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



            // var taaMarker = new ProfilerMarker(ProfilerCategory.Render, "TAA", MarkerFlags.SampleGPU);
            // var dlssBeforeMarker = new ProfilerMarker(ProfilerCategory.Render, "DLSS Before", MarkerFlags.SampleGPU);
            // var dlssDenoiseMarker = new ProfilerMarker(ProfilerCategory.Render, "DLSS Denoise", MarkerFlags.SampleGPU);
            var outputBlitMarker = new ProfilerMarker(ProfilerCategory.Render, "Output Blit", MarkerFlags.SampleGPU);

            //
            // // var isEven = (data.GlobalConstants.gFrameIndex & 1) == 0;
            // var isEven = false;
            // var taaSrc = isEven ? data.TaaHistoryPrev : data.TaaHistory;
            // var taaDst = isEven ? data.TaaHistory : data.TaaHistoryPrev;
            // if (data.Setting.RR)
            // {
            //     // dlss Before
            //     natCmd.BeginSample(dlssBeforeMarker);
            //     natCmd.SetComputeConstantBufferParam(data.DlssBeforeCs, paramsID, data.ConstantBuffer, 0, data.ConstantBuffer.stride);
            //
            //     natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gIn_Normal_Roughness", data.NormalRoughness);
            //     natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gIn_BaseColor_Metalness", data.BaseColorMetalness);
            //     natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gIn_Spec", data.Spec);
            //
            //     natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gInOut_ViewZ", data.ViewZ);
            //     natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gOut_DiffAlbedo", data.RRGuide_DiffAlbedo);
            //     natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gOut_SpecAlbedo", data.RRGuide_SpecAlbedo);
            //     natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gOut_SpecHitDistance", data.RRGuide_SpecHitDistance);
            //     natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gOut_Normal_Roughness", data.RRGuide_Normal_Roughness);
            //
            //
            //     natCmd.DispatchCompute(data.DlssBeforeCs, 0, (int)data.rectGridW, (int)data.rectGridH, 1);
            //     natCmd.EndSample(dlssBeforeMarker);
            //
            //     // DLSS调用
            //
            //     if (!data.Setting.tmpDisableRR)
            //     {
            //         natCmd.BeginSample(dlssDenoiseMarker);
            //         natCmd.IssuePluginEventAndData(GetRenderEventAndDataFunc(), 2, data.RRDataPtr);
            //         natCmd.EndSample(dlssDenoiseMarker);
            //     }
            // }
            // else
            // {
            //     // TAA
            //     natCmd.BeginSample(taaMarker);
            //
            //     natCmd.SetComputeConstantBufferParam(data.TaaCs, paramsID, data.ConstantBuffer, 0, data.ConstantBuffer.stride);
            //     natCmd.SetComputeTextureParam(data.TaaCs, 0, gIn_MvID, data.Mv);
            //     natCmd.SetComputeTextureParam(data.TaaCs, 0, gIn_ComposedID, data.Composed);
            //     natCmd.SetComputeTextureParam(data.TaaCs, 0, gIn_HistoryID, taaSrc);
            //     natCmd.SetComputeTextureParam(data.TaaCs, 0, gOut_ResultID, taaDst);
            //     natCmd.SetComputeTextureParam(data.TaaCs, 0, gOut_DebugID, data.OutputTexture);
            //     natCmd.DispatchCompute(data.TaaCs, 0, (int)data.rectGridW, (int)data.rectGridH, 1);
            //     natCmd.EndSample(taaMarker);
            // }


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
                // case ShowMode.Taa:
                //     Blitter.BlitTexture(natCmd, taaDst, scaleOffset, data.BlitMaterial, (int)ShowPass.Alpha);
                //     break;
                case ShowMode.Final:

                    if (data.Setting.RR)
                    {
                        Blitter.BlitTexture(natCmd, data.DlssOutput, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.Dlss);
                    }
                    else
                    {
                        // Blitter.BlitTexture(natCmd, taaDst, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
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

            passData.TaaCs = TaaCs;
            passData.DlssBeforeCs = DlssBeforeCs;
            passData.BlitMaterial = BiltMaterial;

            passData.passIndex = isXr ? xrPass.multipassId : 0;

            // passData.RRDataPtr = DLRRDenoiser.GetInteropDataPtr(cameraData, NrdDenoiser);

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

