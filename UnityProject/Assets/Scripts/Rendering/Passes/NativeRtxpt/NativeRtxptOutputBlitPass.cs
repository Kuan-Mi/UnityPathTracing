using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Debug / display blit pass for <see cref="NativeRtxptFeature"/>.
    /// Blits one of the RTXPT render targets onto the active camera texture.
    /// The target is selected via <see cref="NativeRtxptShowMode"/> in <see cref="NativeRtxptSetting"/>.
    ///
    /// Uses the same <c>KM_Final</c> material and <see cref="ShowPass"/> shader variants
    /// as the rest of the pipeline (analogous to <see cref="NativeNrdOutputBlitPass"/>).
    /// </summary>
    public class NativeRtxptOutputBlitPass : ScriptableRenderPass
    {
        private readonly Material _blitMaterial;
        private NativeRtxptTextureResources _resources;
        private NativeRtxptShowMode         _showMode;
        private float                        _renderScale; // renderRes / displayRes

        public NativeRtxptOutputBlitPass(Material blitMaterial)
        {
            _blitMaterial = blitMaterial;
        }

        public void Setup(NativeRtxptTextureResources resources, NativeRtxptShowMode showMode, float renderScale)
        {
            _resources   = resources;
            _showMode    = showMode;
            _renderScale = renderScale;
        }

        // ──────────────────────────────────────────────────────────────────
        // RenderGraph
        // ──────────────────────────────────────────────────────────────────

        class PassData
        {
            internal Material                   BlitMaterial;
            internal NativeRtxptTextureResources Resources;
            internal NativeRtxptShowMode         ShowMode;
            internal float                       RenderScale;
            internal TextureHandle               CameraTexture;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd  = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var res  = data.Resources;
            var mat  = data.BlitMaterial;
            var mode = data.ShowMode;

            // scaleOffset fills the render-resolution portion of the display target
            var scaleOffset     = new Vector4(data.RenderScale, data.RenderScale, 0f, 0f);
            var fullScaleOffset = new Vector4(1f, 1f, 0f, 0f);

            cmd.BeginSample("Rtxpt.OutputBlit");
            cmd.SetRenderTarget(data.CameraTexture);

            switch (mode)
            {
                // ── Final outputs ──────────────────────────────────────────
                case NativeRtxptShowMode.DlssRrOutput:
                    Blitter.BlitTexture(cmd, res.DlssRrOutput.Handle, fullScaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeRtxptShowMode.ProcessedOutput:
                    Blitter.BlitTexture(cmd, res.ProcessedOutputColor.Handle, fullScaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeRtxptShowMode.OutputColor:
                    Blitter.BlitTexture(cmd, res.OutputColor.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                // ── GBuffer ────────────────────────────────────────────────
                case NativeRtxptShowMode.BaseColor:
                    Blitter.BlitTexture(cmd, res.BaseColor.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeRtxptShowMode.RoughnessMetal:
                    Blitter.BlitTexture(cmd, res.RoughnessMetal.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeRtxptShowMode.SpecNormal:
                    Blitter.BlitTexture(cmd, res.SpecNormal.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                // ── Depth / motion ─────────────────────────────────────────
                case NativeRtxptShowMode.Depth:
                    Blitter.BlitTexture(cmd, res.Depth.Handle, scaleOffset, mat, (int)ShowPass.ViewZ);
                    break;

                case NativeRtxptShowMode.MotionVectors:
                    Blitter.BlitTexture(cmd, res.ScreenMotionVectors.Handle, scaleOffset, mat, (int)ShowPass.Mv);
                    break;

                // ── Stable planes ──────────────────────────────────────────
                case NativeRtxptShowMode.SpecularHitT:
                    Blitter.BlitTexture(cmd, res.SpecularHitT.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeRtxptShowMode.StableRadiance:
                    Blitter.BlitTexture(cmd, res.StableRadiance.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                // ── DLSS-RR guide buffers ──────────────────────────────────
                case NativeRtxptShowMode.DlssDiffuseAlbedo:
                    Blitter.BlitTexture(cmd, res.DlssRrDiffAlbedo.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeRtxptShowMode.DlssSpecularAlbedo:
                    Blitter.BlitTexture(cmd, res.DlssRrSpecAlbedo.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case NativeRtxptShowMode.DlssNormalRoughness:
                    Blitter.BlitTexture(cmd, res.DlssRrNormalRoughness.Handle, scaleOffset, mat, (int)ShowPass.Normal);
                    break;

                case NativeRtxptShowMode.DlssSpecMotionVectors:
                    Blitter.BlitTexture(cmd, res.DlssRrSpecMotionVectors.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                // ── Debug ──────────────────────────────────────────────────
                case NativeRtxptShowMode.ShaderDebugViz:
                    Blitter.BlitTexture(cmd, res.ShaderDebugViz.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;
            }

            cmd.EndSample("Rtxpt.OutputBlit");
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            var resourceData = frameData.Get<UniversalResourceData>();

            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxpt Output Blit", out var passData);

            passData.BlitMaterial  = _blitMaterial;
            passData.Resources     = _resources;
            passData.ShowMode      = _showMode;
            passData.RenderScale   = _renderScale;
            passData.CameraTexture = resourceData.activeColorTexture;

            builder.UseTexture(passData.CameraTexture, AccessFlags.Write);
            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }
    }
}
