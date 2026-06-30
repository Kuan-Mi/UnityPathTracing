using System;
using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using PathTracing;
using PathTracing.Profiling;
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
            public RTHandle Mv;
            public RTHandle NormalRoughness;
            public RTHandle BaseColorMetalness;

            public RTHandle Penumbra;
            public RTHandle Diff;
            public RTHandle Spec;

            public RTHandle ShadowTranslucency;
            public RTHandle DenoisedDiff;
            public RTHandle DenoisedSpec;
            public RTHandle Validation;

            public RTHandle Composed;
            public RTHandle DirectLighting;

            public RTHandle RRGuide_DiffAlbedo;
            public RTHandle RRGuide_SpecAlbedo;
            public RTHandle RRGuide_SpecHitDistance;
            public RTHandle RRGuide_Normal_Roughness;
            public RTHandle DlssOutput;

            public RTHandle taaDst;
            public RTHandle ViewZ;
            public RTHandle Gradient;

            public RTHandle Output;
            public RTHandle DirectEmission;
            public RTHandle ComposedDiff;
            public RTHandle ComposedSpecViewZ;

            // ── Rtxdi native GBuffer debug textures ──────────────────────────
            public RTHandle RtxdiViewDepth;
            public RTHandle RtxdiDiffuseAlbedo;   // R32_UINT  R11G11B10_UFLOAT
            public RTHandle RtxdiSpecularRough;   // R32_UINT  R8G8B8A8_Gamma_UFLOAT
            public RTHandle RtxdiNormals;         // R32_UINT  oct32
            public RTHandle RtxdiGeoNormals;      // R32_UINT  oct32            // ── Rtxdi PDF debug textures (R32_Float mip chain) ──────────────────
            public RTHandle RtxdiDirectLightingRaw;
            public RTHandle LocalLightPdfTexture;
            public RTHandle EnvironmentPdfTexture;        }

        public class Settings
        {
            public ShowMode showMode;
            public float resolutionScale;
            public bool enableDlssRR;
            public bool tmpDisableRR;
            public bool showMV;
            public bool showValidation;
            public bool showReference;
            /// <summary>Mip level to display for Rtxdi_LocalLightPdf / Rtxdi_EnvironmentPdf modes.</summary>
            public int   pdfMipLevel;
            /// <summary>Exposure in stops for the PDF heat-map visualisation.</summary>
            public float pdfExposureStops;
        }


        class PassData
        {
            public Material BlitMaterial;

            public Resource Resource;
            public Settings Setting;


            public TextureHandle CameraTexture;

            public TextureHandle OutputTexture;
            public TextureHandle DirectEmission;
            public TextureHandle ComposedDiff;

            public TextureHandle ComposedSpecViewZ;
        }

        public OutputBlitPass(Material biltMaterial)
        {
            _biltMaterial = biltMaterial;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            var outputBlitMarker = RenderPassMarkers.OutputBlit;

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
                    Blitter.BlitTexture(natCmd, data.Resource.DirectLighting, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
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
                        if (data.Setting.tmpDisableRR)
                        {
                            Blitter.BlitTexture(natCmd, data.Resource.DirectLighting, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                        }
                        else
                        {
                            Blitter.BlitTexture(natCmd, data.Resource.DlssOutput, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.Dlss);
                        }
                    else if (data.Resource.DirectLighting != null)
                        // NRD path: CompositingPass wrote the composited result into DirectLighting
                        Blitter.BlitTexture(natCmd, data.Resource.DirectLighting, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    else if (data.Resource.taaDst != null)
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
                case ShowMode.ViewZ:
                    Blitter.BlitTexture(natCmd, data.Resource.ViewZ, scaleOffset, data.BlitMaterial, (int)ShowPass.ViewZ);
                    break;
                
                case ShowMode.Gradient:
                    Blitter.BlitTexture(natCmd, data.Resource.Gradient, scaleOffset, data.BlitMaterial, (int)ShowPass.Gradient);
                    break;
                case ShowMode.Rtxdi_ViewDepth:
                    Blitter.BlitTexture(natCmd, data.Resource.RtxdiViewDepth, scaleOffset, data.BlitMaterial, (int)ShowPass.RtxdiViewDepth);
                    break;
                case ShowMode.Rtxdi_DiffuseAlbedo:
                    Blitter.BlitTexture(natCmd, data.Resource.RtxdiDiffuseAlbedo, scaleOffset, data.BlitMaterial, (int)ShowPass.RtxdiDiffuseAlbedo);
                    break;
                case ShowMode.Rtxdi_SpecularF0:
                    Blitter.BlitTexture(natCmd, data.Resource.RtxdiSpecularRough, scaleOffset, data.BlitMaterial, (int)ShowPass.RtxdiSpecularF0);
                    break;
                case ShowMode.Rtxdi_Roughness:
                    Blitter.BlitTexture(natCmd, data.Resource.RtxdiSpecularRough, scaleOffset, data.BlitMaterial, (int)ShowPass.RtxdiRoughness);
                    break;
                case ShowMode.Rtxdi_Normal:
                    Blitter.BlitTexture(natCmd, data.Resource.RtxdiNormals, scaleOffset, data.BlitMaterial, (int)ShowPass.RtxdiNormal);
                    break;
                case ShowMode.Rtxdi_GeoNormal:
                    Blitter.BlitTexture(natCmd, data.Resource.RtxdiGeoNormals, scaleOffset, data.BlitMaterial, (int)ShowPass.RtxdiGeoNormal);
                    break;
                case ShowMode.Rtxdi_DiffuseLighting:
                    Blitter.BlitTexture(natCmd, data.Resource.Diff, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.Rtxdi_SpecularLighting:
                    Blitter.BlitTexture(natCmd, data.Resource.Spec, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.Rtxdi_LocalLightPdf:
                    if (data.Resource.LocalLightPdfTexture != null)
                    {
                        data.BlitMaterial.SetInt("_PdfMipLevel", data.Setting.pdfMipLevel);
                        data.BlitMaterial.SetFloat("_PdfExposureStops", data.Setting.pdfExposureStops);
                        Blitter.BlitTexture(natCmd, data.Resource.LocalLightPdfTexture, scaleOffset, data.BlitMaterial, (int)ShowPass.PdfTextureMip);
                    }
                    break;
                case ShowMode.Rtxdi_EnvironmentPdf:
                    if (data.Resource.EnvironmentPdfTexture != null)
                    {
                        data.BlitMaterial.SetInt("_PdfMipLevel", data.Setting.pdfMipLevel);
                        data.BlitMaterial.SetFloat("_PdfExposureStops", data.Setting.pdfExposureStops);
                        Blitter.BlitTexture(natCmd, data.Resource.EnvironmentPdfTexture, scaleOffset, data.BlitMaterial, (int)ShowPass.PdfTextureMip);
                    }
                    break;
                case ShowMode.Rtxdi_DirectLightingRaw:
                    Blitter.BlitTexture(natCmd, data.Resource.RtxdiDirectLightingRaw, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
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

            passData.OutputTexture     = _resource.Output != null ? renderGraph.ImportTexture(_resource.Output) : TextureHandle.nullHandle;
            passData.DirectEmission    = _resource.DirectEmission != null ? renderGraph.ImportTexture(_resource.DirectEmission) : TextureHandle.nullHandle;
            passData.ComposedDiff      = _resource.ComposedDiff != null ? renderGraph.ImportTexture(_resource.ComposedDiff) : TextureHandle.nullHandle;
            passData.ComposedSpecViewZ = _resource.ComposedSpecViewZ != null ? renderGraph.ImportTexture(_resource.ComposedSpecViewZ) : TextureHandle.nullHandle;


            if (passData.DirectEmission.IsValid())
                builder.UseTexture(passData.DirectEmission, AccessFlags.ReadWrite);

            if (passData.ComposedDiff.IsValid())
                builder.UseTexture(passData.ComposedDiff, AccessFlags.ReadWrite);

            if (passData.ComposedSpecViewZ.IsValid())
                builder.UseTexture(passData.ComposedSpecViewZ, AccessFlags.ReadWrite);

            if (passData.OutputTexture.IsValid())
                builder.UseTexture(passData.OutputTexture, AccessFlags.ReadWrite);

            passData.CameraTexture = resourceData.activeColorTexture;

            passData.Setting = _settings;
            passData.Resource = _resource;

            builder.UseTexture(passData.CameraTexture, AccessFlags.Write);

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}
