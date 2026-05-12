using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// OutputBlit pass dedicated to <see cref="NativeRtxdiFeature"/>.
    /// Uses <see cref="NativeRtxdiShowMode"/> so NativeRtxdi debug views are kept separate
    /// from the shared <see cref="ShowMode"/> / <see cref="OutputBlitPass"/> used by other features.
    /// </summary>
    public class NativeRtxdiOutputBlitPass : ScriptableRenderPass
    {
        private readonly Material _blitMaterial;
        private Resource  _resource;
        private Settings  _settings;

        public NativeRtxdiOutputBlitPass(Material blitMaterial)
        {
            _blitMaterial = blitMaterial;
        }

        public void Setup(Resource resource, Settings settings)
        {
            _resource = resource;
            _settings = settings;
        }

        // ──────────────────────────────────────────────────────────────────
        // Resource / Settings
        // ──────────────────────────────────────────────────────────────────

        public class Resource
        {
            // ── Main outputs ───────────────────────────────────────────────
            /// <summary>Composited HDR lighting (CompositingPass output = original HdrColor).</summary>
            internal RTHandle HdrColor;
            internal RTHandle LdrColor;
            /// <summary>DLSS-SR upscaled image (display resolution). Null when SR is disabled.</summary>
            internal RTHandle DlssOutput;

            // ── Denoiser buffers ───────────────────────────────────────────
            internal RTHandle DiffuseLighting;
            internal RTHandle SpecularLighting;
            internal RTHandle DenoisedDiffuse;
            internal RTHandle DenoisedSpecular;
            internal RTHandle DirectLightingRaw;
            internal RTHandle IndirectLightingRaw;

            // ── NRD validation ─────────────────────────────────────────────
            internal RTHandle NrdValidation;

            // ── Motion vectors (optional overlay) ─────────────────────────
            internal RTHandle MotionVectors;

            // ── GBuffer debug ──────────────────────────────────────────────
            internal RTHandle ViewDepth;      // R32_SFloat
            internal RTHandle DiffuseAlbedo;  // R32_UINT  R11G11B10_UFLOAT
            internal RTHandle SpecularRough;  // R32_UINT  R8G8B8A8_Gamma
            internal RTHandle Normals;        // R32_UINT  oct32
            internal RTHandle GeoNormals;     // R32_UINT  oct32

            // ── Light PDF debug ────────────────────────────────────────────
            internal RTHandle LocalLightPdfTexture;
            internal RTHandle EnvironmentPdfTexture;
        }

        public class Settings
        {
            internal NativeRtxdiShowMode showMode;
            internal float               resolutionScale;
            internal bool                showMv;
            internal bool                showValidation;
            /// <summary>Mip level for LocalLightPdf / EnvironmentPdf visualisations.</summary>
            internal int   pdfMipLevel;
            /// <summary>Exposure in stops for the PDF heat-map (positive = brighter).</summary>
            internal float pdfExposureStops;
        }

        // ──────────────────────────────────────────────────────────────────
        // RenderGraph
        // ──────────────────────────────────────────────────────────────────

        class PassData
        {
            internal Material             BlitMaterial;
            internal Resource             Resource;
            internal Settings             Settings;
            internal TextureHandle        CameraTexture;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd   = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var res   = data.Resource;
            var set   = data.Settings;
            var mat   = data.BlitMaterial;

            cmd.BeginSample(RenderPassMarkers.OutputBlit);
            cmd.SetRenderTarget(data.CameraTexture);

            var scaleOffset    = new Vector4(set.resolutionScale, set.resolutionScale, 0, 0);
            var fullScaleOffset = new Vector4(1f, 1f, 0, 0);

            switch (set.showMode)
            {
                case NativeRtxdiShowMode.Final:
                    // Mirror FullSample SceneRenderer::RenderWaitFrame:
                    //   DlssOutput when SR is on,
                    //   LdrColor when ToneMapping is on,
                    //   HdrColor otherwise.
                    if (res.DlssOutput != null)
                        Blitter.BlitTexture(cmd, res.DlssOutput, fullScaleOffset, mat, (int)ShowPass.Out);
                    else if (res.LdrColor != null)
                        Blitter.BlitTexture(cmd, res.LdrColor, scaleOffset, mat, (int)ShowPass.Out);
                    else if (res.HdrColor != null)
                        Blitter.BlitTexture(cmd, res.HdrColor, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeRtxdiShowMode.HdrColor:
                    if (res.HdrColor != null)
                        Blitter.BlitTexture(cmd, res.HdrColor, scaleOffset, mat, (int)ShowPass.Out);
                    break;
                case NativeRtxdiShowMode.LdrColor:
                    if (res.LdrColor != null)
                        Blitter.BlitTexture(cmd, res.LdrColor, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeRtxdiShowMode.DlssOutput:
                    if (res.DlssOutput != null)
                        Blitter.BlitTexture(cmd, res.DlssOutput, fullScaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeRtxdiShowMode.DiffuseLighting:
                    if (res.DiffuseLighting != null)
                        Blitter.BlitTexture(cmd, res.DiffuseLighting, scaleOffset, mat, (int)ShowPass.Radiance);
                    break;

                case NativeRtxdiShowMode.SpecularLighting:
                    if (res.SpecularLighting != null)
                        Blitter.BlitTexture(cmd, res.SpecularLighting, scaleOffset, mat, (int)ShowPass.Radiance);
                    break;

                case NativeRtxdiShowMode.DenoisedDiffuse:
                    if (res.DenoisedDiffuse != null)
                        Blitter.BlitTexture(cmd, res.DenoisedDiffuse, scaleOffset, mat, (int)ShowPass.Radiance);
                    break;

                case NativeRtxdiShowMode.DenoisedSpecular:
                    if (res.DenoisedSpecular != null)
                        Blitter.BlitTexture(cmd, res.DenoisedSpecular, scaleOffset, mat, (int)ShowPass.Radiance);
                    break;

                case NativeRtxdiShowMode.DirectLightingRaw:
                    if (res.DirectLightingRaw != null)
                        Blitter.BlitTexture(cmd, res.DirectLightingRaw, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeRtxdiShowMode.IndirectLightingRaw:
                    if (res.IndirectLightingRaw != null)
                        Blitter.BlitTexture(cmd, res.IndirectLightingRaw, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeRtxdiShowMode.NrdValidation:
                    if (res.NrdValidation != null)
                        Blitter.BlitTexture(cmd, res.NrdValidation, scaleOffset, mat, (int)ShowPass.Validation);
                    break;

                // ── GBuffer ────────────────────────────────────────────────
                case NativeRtxdiShowMode.ViewDepth:
                    if (res.ViewDepth != null)
                        Blitter.BlitTexture(cmd, res.ViewDepth, scaleOffset, mat, (int)ShowPass.RtxdiViewDepth);
                    break;

                case NativeRtxdiShowMode.DiffuseAlbedo:
                    if (res.DiffuseAlbedo != null)
                        Blitter.BlitTexture(cmd, res.DiffuseAlbedo, scaleOffset, mat, (int)ShowPass.RtxdiDiffuseAlbedo);
                    break;

                case NativeRtxdiShowMode.SpecularF0:
                    if (res.SpecularRough != null)
                        Blitter.BlitTexture(cmd, res.SpecularRough, scaleOffset, mat, (int)ShowPass.RtxdiSpecularF0);
                    break;

                case NativeRtxdiShowMode.Roughness:
                    if (res.SpecularRough != null)
                        Blitter.BlitTexture(cmd, res.SpecularRough, scaleOffset, mat, (int)ShowPass.RtxdiRoughness);
                    break;

                case NativeRtxdiShowMode.Normal:
                    if (res.Normals != null)
                        Blitter.BlitTexture(cmd, res.Normals, scaleOffset, mat, (int)ShowPass.RtxdiNormal);
                    break;

                case NativeRtxdiShowMode.GeoNormal:
                    if (res.GeoNormals != null)
                        Blitter.BlitTexture(cmd, res.GeoNormals, scaleOffset, mat, (int)ShowPass.RtxdiGeoNormal);
                    break;

                // ── PDF debug ──────────────────────────────────────────────
                case NativeRtxdiShowMode.LocalLightPdf:
                    if (res.LocalLightPdfTexture != null)
                    {
                        mat.SetInt("_PdfMipLevel", set.pdfMipLevel);
                        mat.SetFloat("_PdfExposureStops", set.pdfExposureStops);
                        Blitter.BlitTexture(cmd, res.LocalLightPdfTexture, scaleOffset, mat, (int)ShowPass.PdfTextureMip);
                    }
                    break;

                case NativeRtxdiShowMode.EnvironmentPdf:
                    if (res.EnvironmentPdfTexture != null)
                    {
                        mat.SetInt("_PdfMipLevel", set.pdfMipLevel);
                        mat.SetFloat("_PdfExposureStops", set.pdfExposureStops);
                        Blitter.BlitTexture(cmd, res.EnvironmentPdfTexture, scaleOffset, mat, (int)ShowPass.PdfTextureMip);
                    }
                    break;
            }

            // ── Overlays ───────────────────────────────────────────────────
            if (set.showMv && res.MotionVectors != null)
                Blitter.BlitTexture(cmd, res.MotionVectors, fullScaleOffset, mat, (int)ShowPass.Mv);

            if (set.showValidation && res.NrdValidation != null)
                Blitter.BlitTexture(cmd, res.NrdValidation, fullScaleOffset, mat, (int)ShowPass.Validation);

            cmd.EndSample(RenderPassMarkers.OutputBlit);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            var resourceData = frameData.Get<UniversalResourceData>();

            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxdi Output Blit", out var passData);

            passData.BlitMaterial  = _blitMaterial;
            passData.Resource      = _resource;
            passData.Settings      = _settings;
            passData.CameraTexture = resourceData.activeColorTexture;

            builder.UseTexture(passData.CameraTexture, AccessFlags.Write);
            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }
    }
}
