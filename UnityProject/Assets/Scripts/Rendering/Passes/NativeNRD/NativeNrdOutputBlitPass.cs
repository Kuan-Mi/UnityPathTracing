using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// OutputBlit pass dedicated to <see cref="NativeNrdFeature"/>.
    /// Uses <see cref="NativeNrdShowMode"/> so NativeNrd debug views are kept separate
    /// from the shared <see cref="ShowMode"/> / <see cref="OutputBlitPass"/> used by other features.
    /// </summary>
    public class NativeNrdOutputBlitPass : ScriptableRenderPass
    {
        private readonly Material _blitMaterial;
        private Resource  _resource;
        private Settings  _settings;

        public NativeNrdOutputBlitPass(Material blitMaterial)
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
            // ── GBuffer ────────────────────────────────────────────────────
            internal RTHandle Mv;
            internal RTHandle NormalRoughness;
            internal RTHandle BaseColorMetalness;
            internal RTHandle ViewZ;

            // ── Denoiser inputs ────────────────────────────────────────────
            internal RTHandle Penumbra;
            internal RTHandle NoiseDiff;
            internal RTHandle NoiseSpec;

            // ── Denoiser outputs ───────────────────────────────────────────
            internal RTHandle Shadow;
            internal RTHandle DenoisedDiff;
            internal RTHandle DenoisedSpec;
            internal RTHandle Validation;

            // ── Intermediate lighting ──────────────────────────────────────
            internal RTHandle DirectLighting;
            internal RTHandle DirectEmission;
            internal RTHandle ComposedDiff;
            internal RTHandle ComposedSpecViewZ;
            internal RTHandle Composed;

            // ── TAA output ─────────────────────────────────────────────────
            internal RTHandle TaaDst;

            // ── DLSS / RR guide buffers ────────────────────────────────────
            internal RTHandle RRGuide_DiffAlbedo;
            internal RTHandle RRGuide_SpecAlbedo;
            internal RTHandle RRGuide_SpecHitDistance;
            internal RTHandle RRGuide_Normal_Roughness;
            internal RTHandle DlssOutput;

            // ── SHARC confidence gradient ──────────────────────────────────
            internal RTHandle Gradient;

            // ── Final output ───────────────────────────────────────────────
            internal RTHandle Output;
        }

        public class Settings
        {
            internal NativeNrdShowMode showMode;
            internal float             resolutionScale;
            internal bool              enableDlssRR;
            internal bool              tmpDisableRR;
            internal bool              showMV;
            internal bool              showValidation;
        }

        // ──────────────────────────────────────────────────────────────────
        // RenderGraph
        // ──────────────────────────────────────────────────────────────────

        class PassData
        {
            internal Material      BlitMaterial;
            internal Resource      Resource;
            internal Settings      Settings;
            internal TextureHandle CameraTexture;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var res = data.Resource;
            var set = data.Settings;
            var mat = data.BlitMaterial;

            cmd.BeginSample(RenderPassMarkers.OutputBlit);
            cmd.SetRenderTarget(data.CameraTexture);

            var scaleOffset     = new Vector4(set.resolutionScale, set.resolutionScale, 0, 0);
            var fullScaleOffset = new Vector4(1f, 1f, 0, 0);

            switch (set.showMode)
            {
                case NativeNrdShowMode.Final:
                    if (set.enableDlssRR)
                    {
                        if (set.tmpDisableRR)
                        {
                            if (res.DirectLighting != null)
                                Blitter.BlitTexture(cmd, res.DirectLighting, scaleOffset, mat, (int)ShowPass.Out);
                        }
                        else if (res.DlssOutput != null)
                        {
                            Blitter.BlitTexture(cmd, res.DlssOutput, fullScaleOffset, mat, (int)ShowPass.Dlss);
                        }
                    }
                    else if (res.Output != null)
                    {
                        Blitter.BlitTexture(cmd, res.Output, scaleOffset, mat, (int)ShowPass.Out);
                    }
                    break;

                // ── GBuffer ────────────────────────────────────────────────
                case NativeNrdShowMode.BaseColor:
                    if (res.BaseColorMetalness != null)
                        Blitter.BlitTexture(cmd, res.BaseColorMetalness, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeNrdShowMode.Metalness:
                    if (res.BaseColorMetalness != null)
                        Blitter.BlitTexture(cmd, res.BaseColorMetalness, scaleOffset, mat, (int)ShowPass.Alpha);
                    break;

                case NativeNrdShowMode.Normal:
                    if (res.NormalRoughness != null)
                        Blitter.BlitTexture(cmd, res.NormalRoughness, scaleOffset, mat, (int)ShowPass.Normal);
                    break;

                case NativeNrdShowMode.Roughness:
                    if (res.NormalRoughness != null)
                        Blitter.BlitTexture(cmd, res.NormalRoughness, scaleOffset, mat, (int)ShowPass.Roughness);
                    break;

                case NativeNrdShowMode.ViewZ:
                    if (res.ViewZ != null)
                        Blitter.BlitTexture(cmd, res.ViewZ, scaleOffset, mat, (int)ShowPass.ViewZ);
                    break;

                // ── Denoiser inputs ────────────────────────────────────────
                case NativeNrdShowMode.NoiseShadow:
                    if (res.Penumbra != null)
                        Blitter.BlitTexture(cmd, res.Penumbra, scaleOffset, mat, (int)ShowPass.NoiseShadow);
                    break;

                case NativeNrdShowMode.NoiseDiffuse:
                    if (res.NoiseDiff != null)
                        Blitter.BlitTexture(cmd, res.NoiseDiff, scaleOffset, mat, (int)ShowPass.Radiance);
                    break;

                case NativeNrdShowMode.NoiseSpecular:
                    if (res.NoiseSpec != null)
                        Blitter.BlitTexture(cmd, res.NoiseSpec, scaleOffset, mat, (int)ShowPass.Radiance);
                    break;

                // ── Denoiser outputs ───────────────────────────────────────
                case NativeNrdShowMode.DenoisedDiffuse:
                    if (res.DenoisedDiff != null)
                        Blitter.BlitTexture(cmd, res.DenoisedDiff, scaleOffset, mat, (int)ShowPass.Radiance);
                    break;

                case NativeNrdShowMode.DenoisedSpecular:
                    if (res.DenoisedSpec != null)
                        Blitter.BlitTexture(cmd, res.DenoisedSpec, scaleOffset, mat, (int)ShowPass.Radiance);
                    break;

                case NativeNrdShowMode.Shadow:
                    if (res.Shadow != null)
                        Blitter.BlitTexture(cmd, res.Shadow, scaleOffset, mat, (int)ShowPass.Shadow);
                    break;

                // ── Intermediate lighting ──────────────────────────────────
                case NativeNrdShowMode.DirectLight:
                    if (res.DirectLighting != null)
                        Blitter.BlitTexture(cmd, res.DirectLighting, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeNrdShowMode.Emissive:
                    if (res.DirectEmission != null)
                        Blitter.BlitTexture(cmd, res.DirectEmission, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeNrdShowMode.ComposedDiff:
                    if (res.ComposedDiff != null)
                        Blitter.BlitTexture(cmd, res.ComposedDiff, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeNrdShowMode.ComposedSpec:
                    if (res.ComposedSpecViewZ != null)
                        Blitter.BlitTexture(cmd, res.ComposedSpecViewZ, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeNrdShowMode.Composed:
                    if (res.Composed != null)
                        Blitter.BlitTexture(cmd, res.Composed, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                // ── TAA output ─────────────────────────────────────────────
                case NativeNrdShowMode.Taa:
                    if (res.TaaDst != null)
                        Blitter.BlitTexture(cmd, res.TaaDst, scaleOffset, mat, (int)ShowPass.Alpha);
                    break;

                // ── DLSS / RR guide buffers ────────────────────────────────
                case NativeNrdShowMode.DLSS_DiffuseAlbedo:
                    if (res.RRGuide_DiffAlbedo != null)
                        Blitter.BlitTexture(cmd, res.RRGuide_DiffAlbedo, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeNrdShowMode.DLSS_SpecularAlbedo:
                    if (res.RRGuide_SpecAlbedo != null)
                        Blitter.BlitTexture(cmd, res.RRGuide_SpecAlbedo, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeNrdShowMode.DLSS_SpecularHitDistance:
                    if (res.RRGuide_SpecHitDistance != null)
                        Blitter.BlitTexture(cmd, res.RRGuide_SpecHitDistance, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeNrdShowMode.DLSS_NormalRoughness:
                    if (res.RRGuide_Normal_Roughness != null)
                        Blitter.BlitTexture(cmd, res.RRGuide_Normal_Roughness, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeNrdShowMode.DLSS_Output:
                    if (res.DlssOutput != null)
                        Blitter.BlitTexture(cmd, res.DlssOutput, fullScaleOffset, mat, (int)ShowPass.Out);
                    break;

                // ── SHARC confidence gradient ──────────────────────────────
                case NativeNrdShowMode.Gradient:
                    if (res.Gradient != null)
                        Blitter.BlitTexture(cmd, res.Gradient, scaleOffset, mat, (int)ShowPass.Gradient);
                    break;
            }

            // ── Overlays ───────────────────────────────────────────────────
            if (set.showMV && res.Mv != null)
                Blitter.BlitTexture(cmd, res.Mv, fullScaleOffset, mat, (int)ShowPass.Mv);

            if (set.showValidation && res.Validation != null)
                Blitter.BlitTexture(cmd, res.Validation, fullScaleOffset, mat, (int)ShowPass.Validation);

            cmd.EndSample(RenderPassMarkers.OutputBlit);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            var resourceData = frameData.Get<UniversalResourceData>();

            using var builder = renderGraph.AddUnsafePass<PassData>("NativeNrd Output Blit", out var passData);

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
