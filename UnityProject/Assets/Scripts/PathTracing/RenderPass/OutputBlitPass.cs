using System;
using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    public class OutputBlitPass : ScriptableRenderPass
    {
        private Material _biltMaterial;

        private Resource _resource;
        private Settings _settings;


        public void Setup(Resource resource, Settings settings)
        {
            _resource = resource;
            _settings = settings;
        }


        public class Resource
        {
            internal RTHandle Mv;
            internal RTHandle NormalRoughness;
            internal RTHandle BaseColorMetalness;

            internal RTHandle Penumbra;
            internal RTHandle Diff;
            internal RTHandle Spec;

            internal RTHandle ShadowTranslucency;
            internal RTHandle DenoisedDiff;
            internal RTHandle DenoisedSpec;
            internal RTHandle Validation;

            internal RTHandle Composed;

            internal RTHandle RRGuide_DiffAlbedo;
            internal RTHandle RRGuide_SpecAlbedo;
            internal RTHandle RRGuide_SpecHitDistance;
            internal RTHandle RRGuide_Normal_Roughness;
            internal RTHandle DlssOutput;

            internal RTHandle taaDst;
        }

        public class Settings
        {
            internal ShowMode showMode;
            internal float resolutionScale;
            internal bool enableDlssRR;
            internal bool showMV;
            internal bool showValidation;
            internal bool showReference;
        }


        class PassData
        {
            internal Material BlitMaterial;

            internal Resource Resource;
            internal Settings Setting;


            internal TextureHandle CameraTexture;

            internal TextureHandle OutputTexture;
            internal TextureHandle DirectLighting;
            internal TextureHandle DirectEmission;
            internal TextureHandle ComposedDiff;

            internal TextureHandle ComposedSpecViewZ;
        }

        public OutputBlitPass(Material biltMaterial)
        {
            _biltMaterial = biltMaterial;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            var outputBlitMarker = new ProfilerMarker(ProfilerCategory.Render, "Output Blit", MarkerFlags.SampleGPU);

            // 显示输出
            natCmd.BeginSample(outputBlitMarker);

            natCmd.SetRenderTarget(data.CameraTexture);

            Vector4 scaleOffset = new Vector4(data.Setting.resolutionScale, data.Setting.resolutionScale, 0, 0);

            switch (data.Setting.showMode)
            {
                case ShowMode.None:
                    break;
                case ShowMode.BaseColor:
                    Blitter.BlitTexture(natCmd, data.Resource.BaseColorMetalness, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.Metalness:
                    Blitter.BlitTexture(natCmd, data.Resource.BaseColorMetalness, scaleOffset, data.BlitMaterial, (int)ShowPass.Alpha);
                    break;
                case ShowMode.Normal:
                    Blitter.BlitTexture(natCmd, data.Resource.NormalRoughness, scaleOffset, data.BlitMaterial, (int)ShowPass.Normal);
                    break;
                case ShowMode.Roughness:
                    Blitter.BlitTexture(natCmd, data.Resource.NormalRoughness, scaleOffset, data.BlitMaterial, (int)ShowPass.Roughness);
                    break;
                case ShowMode.NoiseShadow:
                    Blitter.BlitTexture(natCmd, data.Resource.Penumbra, scaleOffset, data.BlitMaterial, (int)ShowPass.NoiseShadow);
                    break;
                case ShowMode.Shadow:
                    Blitter.BlitTexture(natCmd, data.Resource.ShadowTranslucency, scaleOffset, data.BlitMaterial, (int)ShowPass.Shadow);
                    break;
                case ShowMode.Diffuse:
                    Blitter.BlitTexture(natCmd, data.Resource.Diff, scaleOffset, data.BlitMaterial, (int)ShowPass.Radiance);
                    break;
                case ShowMode.Specular:
                    Blitter.BlitTexture(natCmd, data.Resource.Spec, scaleOffset, data.BlitMaterial, (int)ShowPass.Radiance);
                    break;
                case ShowMode.DenoisedDiffuse:
                    Blitter.BlitTexture(natCmd, data.Resource.DenoisedDiff, scaleOffset, data.BlitMaterial, (int)ShowPass.Radiance);
                    break;
                case ShowMode.DenoisedSpecular:
                    Blitter.BlitTexture(natCmd, data.Resource.DenoisedSpec, scaleOffset, data.BlitMaterial, (int)ShowPass.Radiance);
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
                    Blitter.BlitTexture(natCmd, data.Resource.Composed, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.Taa:
                    Blitter.BlitTexture(natCmd, data.Resource.taaDst, scaleOffset, data.BlitMaterial, (int)ShowPass.Alpha);
                    break;
                case ShowMode.Final:
                    if (data.Setting.enableDlssRR)
                        Blitter.BlitTexture(natCmd, data.Resource.DlssOutput, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.Dlss);
                    else
                        Blitter.BlitTexture(natCmd, data.Resource.taaDst, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.DLSS_DiffuseAlbedo:
                    Blitter.BlitTexture(natCmd, data.Resource.RRGuide_DiffAlbedo, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.DLSS_SpecularAlbedo:
                    Blitter.BlitTexture(natCmd, data.Resource.RRGuide_SpecAlbedo, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.DLSS_SpecularHitDistance:
                    Blitter.BlitTexture(natCmd, data.Resource.RRGuide_SpecHitDistance, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.DLSS_NormalRoughness:
                    Blitter.BlitTexture(natCmd, data.Resource.RRGuide_Normal_Roughness, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.DLSS_Output:
                    Blitter.BlitTexture(natCmd, data.Resource.DlssOutput, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.Out);
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

            if (data.Setting.showMV)
            {
                Blitter.BlitTexture(natCmd, data.Resource.Mv, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.Mv);
            }

            if (data.Setting.showValidation)
            {
                Blitter.BlitTexture(natCmd, data.Resource.Validation, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.Validation);
            }
            
            if (data.Setting.showReference)
            {
                Blitter.BlitTexture(natCmd, data.OutputTexture, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.Validation);
            }

            natCmd.EndSample(outputBlitMarker);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            var resourceData = frameData.Get<UniversalResourceData>();

            using var builder = renderGraph.AddUnsafePass<PassData>("Output Blit", out var passData);

            passData.BlitMaterial = _biltMaterial;

            var ptContextItem = frameData.Get<PTContextItem>();

            passData.OutputTexture = ptContextItem.OutputTexture;
            passData.DirectLighting = ptContextItem.DirectLighting;
            passData.DirectEmission = ptContextItem.DirectEmission;
            passData.ComposedDiff = ptContextItem.ComposedDiff;
            passData.ComposedSpecViewZ = ptContextItem.ComposedSpecViewZ;

            builder.UseTexture(passData.OutputTexture, AccessFlags.ReadWrite);
            builder.UseTexture(passData.DirectLighting, AccessFlags.ReadWrite);
            builder.UseTexture(passData.DirectEmission, AccessFlags.ReadWrite);
            builder.UseTexture(passData.ComposedDiff, AccessFlags.ReadWrite);
            builder.UseTexture(passData.ComposedSpecViewZ, AccessFlags.ReadWrite);

            passData.CameraTexture = resourceData.activeColorTexture;

            passData.Setting = _settings;
            passData.Resource = _resource;

            builder.UseTexture(passData.CameraTexture, AccessFlags.Write);

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}