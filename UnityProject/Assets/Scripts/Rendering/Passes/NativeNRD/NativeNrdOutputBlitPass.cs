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
        private readonly Material                  _blitMaterial;
        private          NativeNrdTextureResources _resource;
        private          Settings                  _settings;

        public NativeNrdOutputBlitPass(Material blitMaterial)
        {
            _blitMaterial = blitMaterial;
        }

        public void Setup(NativeNrdTextureResources resource, Settings settings)
        {
            _resource = resource;
            _settings = settings;
        }

        public class Settings
        {
            internal NativeNrdShowMode ShowMode;
            internal float             ResolutionScale;
            internal bool              ShowMv;
            internal bool              ShowValidation;
            internal bool              IsEven;
        }

        // ──────────────────────────────────────────────────────────────────
        // RenderGraph
        // ──────────────────────────────────────────────────────────────────

        class PassData
        {
            internal Material                  BlitMaterial;
            internal NativeNrdTextureResources Resource;
            internal Settings                  Settings;
            internal TextureHandle             CameraTexture;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var res = data.Resource;
            var set = data.Settings;
            var mat = data.BlitMaterial;

            cmd.BeginSample(RenderPassMarkers.OutputBlit);
            cmd.SetRenderTarget(data.CameraTexture);

            var scaleOffset     = new Vector4(set.ResolutionScale, set.ResolutionScale, 0, 0);
            var fullScaleOffset = new Vector4(1f, 1f, 0, 0);

            switch (set.ShowMode)
            {
                case NativeNrdShowMode.Final:
                    Blitter.BlitTexture(cmd, res.Final.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                // ── GBuffer ────────────────────────────────────────────────
                case NativeNrdShowMode.BaseColor:
                    Blitter.BlitTexture(cmd, res.BaseColorMetalness.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeNrdShowMode.Metalness:
                    Blitter.BlitTexture(cmd, res.BaseColorMetalness.Handle, scaleOffset, mat, (int)ShowPass.Alpha);
                    break;

                case NativeNrdShowMode.Normal:
                    Blitter.BlitTexture(cmd, res.NormalRoughness.Handle, scaleOffset, mat, (int)ShowPass.Normal);
                    break;

                case NativeNrdShowMode.Roughness:
                    Blitter.BlitTexture(cmd, res.NormalRoughness.Handle, scaleOffset, mat, (int)ShowPass.Roughness);
                    break;

                case NativeNrdShowMode.ViewZ:
                    Blitter.BlitTexture(cmd, res.ViewZ.Handle, scaleOffset, mat, (int)ShowPass.ViewZ);
                    break;

                // ── Denoiser inputs ────────────────────────────────────────
                case NativeNrdShowMode.NoiseShadow:
                    Blitter.BlitTexture(cmd, res.Unfiltered_Translucency.Handle, scaleOffset, mat, (int)ShowPass.NoiseShadow);
                    break;

                case NativeNrdShowMode.NoiseDiffuse:
                    Blitter.BlitTexture(cmd, res.Unfiltered_Diff.Handle, scaleOffset, mat, (int)ShowPass.Radiance);
                    break;

                case NativeNrdShowMode.NoiseSpecular:
                    Blitter.BlitTexture(cmd, res.Unfiltered_Spec.Handle, scaleOffset, mat, (int)ShowPass.Radiance);
                    break;

                // ── Denoiser outputs ───────────────────────────────────────
                case NativeNrdShowMode.DenoisedDiffuse:
                    Blitter.BlitTexture(cmd, res.Diff.Handle, scaleOffset, mat, (int)ShowPass.Radiance);
                    break;

                case NativeNrdShowMode.DenoisedSpecular:
                    Blitter.BlitTexture(cmd, res.Spec.Handle, scaleOffset, mat, (int)ShowPass.Radiance);
                    break;

                case NativeNrdShowMode.DenoisedShadow:
                    Blitter.BlitTexture(cmd, res.Shadow.Handle, scaleOffset, mat, (int)ShowPass.Shadow);
                    break;

                // ── Intermediate lighting ──────────────────────────────────
                case NativeNrdShowMode.DirectLight:
                    Blitter.BlitTexture(cmd, res.DirectLighting.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeNrdShowMode.Emissive:
                    Blitter.BlitTexture(cmd, res.DirectEmission.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeNrdShowMode.ComposedDiff:
                    Blitter.BlitTexture(cmd, res.ComposedDiff.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeNrdShowMode.ComposedSpec:
                    Blitter.BlitTexture(cmd, res.ComposedSpecViewZ.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeNrdShowMode.Composed:
                    Blitter.BlitTexture(cmd, res.Composed.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                // ── TAA output ─────────────────────────────────────────────
                case NativeNrdShowMode.Taa:

                    var taaDst = set.IsEven ? res.TaaHistoryPing.Handle : res.TaaHistoryPong.Handle;
                    Blitter.BlitTexture(cmd, taaDst, scaleOffset, mat, (int)ShowPass.Alpha);
                    break;

                // ── DLSS / RR guide buffers ────────────────────────────────
                case NativeNrdShowMode.DLSS_DiffuseAlbedo:
                    Blitter.BlitTexture(cmd, res.RrGuideDiffAlbedo.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeNrdShowMode.DLSS_SpecularAlbedo:
                    Blitter.BlitTexture(cmd, res.RrGuideSpecAlbedo.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeNrdShowMode.DLSS_SpecularHitDistance:
                    Blitter.BlitTexture(cmd, res.RrGuideSpecHitDistance.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeNrdShowMode.DLSS_NormalRoughness:
                    Blitter.BlitTexture(cmd, res.RrGuideNormalRoughness.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeNrdShowMode.DLSS_Output:
                    Blitter.BlitTexture(cmd, res.DlssOutput.Handle, fullScaleOffset, mat, (int)ShowPass.Out);
                    break;

                // ── SHARC confidence gradient ──────────────────────────────
                case NativeNrdShowMode.Gradient:
                    Blitter.BlitTexture(cmd, res.Gradient_Pong.Handle, scaleOffset, mat, (int)ShowPass.Gradient);
                    break;
            }

            // ── Overlays ───────────────────────────────────────────────────
            if (set.ShowMv)
                Blitter.BlitTexture(cmd, res.Mv.Handle, fullScaleOffset, mat, (int)ShowPass.Mv);

            if (set.ShowValidation)
                Blitter.BlitTexture(cmd, res.Validation.Handle, fullScaleOffset, mat, (int)ShowPass.Validation);

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