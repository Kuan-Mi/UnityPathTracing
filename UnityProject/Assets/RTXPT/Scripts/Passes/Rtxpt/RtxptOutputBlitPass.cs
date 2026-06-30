using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Debug / display blit pass for <see cref="RtxptFeature"/>.
    /// Blits one of the RTXPT render targets onto the active camera texture.
    /// The target is selected via <see cref="RtxptShowMode"/> in <see cref="RtxptSetting"/>.
    ///
    /// Uses the same <c>KM_Final</c> material and <see cref="ShowPass"/> shader variants
    /// as the rest of the pipeline (analogous to <see cref="NativeNrdOutputBlitPass"/>).
    /// </summary>
    public class RtxptOutputBlitPass : ScriptableRenderPass
    {
        private readonly Material              _blitMaterial;
        private          RtxptTextureResources _resources;
        private          RtxptShowMode         _showMode;
        private          float                 _renderScale; // renderRes / displayRes
        private          RtxptDebugViewType    _debugViewType;

        public RtxptOutputBlitPass(Material blitMaterial)
        {
            _blitMaterial = blitMaterial;
        }

        public void Setup(RtxptTextureResources resources, RtxptShowMode showMode, float renderScale,
            RtxptDebugViewType debugViewType = RtxptDebugViewType.Disabled)
        {
            _resources     = resources;
            _showMode      = showMode;
            _renderScale   = renderScale;
            _debugViewType = debugViewType;
        }

        // ──────────────────────────────────────────────────────────────────
        // RenderGraph
        // ──────────────────────────────────────────────────────────────────

        class PassData
        {
            internal Material              BlitMaterial;
            internal RtxptTextureResources Resources;
            internal RtxptShowMode         ShowMode;
            internal float                 RenderScale;
            internal TextureHandle         CameraTexture;
            internal RtxptDebugViewType    DebugViewType;
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

            cmd.BeginSample("OutputBlit");
            cmd.SetRenderTarget(data.CameraTexture);

            // When a debug view is active, override showMode and display the debug viz texture.
            if (data.DebugViewType != RtxptDebugViewType.Disabled)
            {
                Blitter.BlitTexture(cmd, res.ShaderDebugViz.Handle, scaleOffset, mat, (int)ShowPass.Out);
                cmd.EndSample("OutputBlit");
                return;
            }

            switch (mode)
            {
                // ── Final outputs ──────────────────────────────────────────
                case RtxptShowMode.DlssRrOutput:
                    Blitter.BlitTexture(cmd, res.DlssRrOutput.Handle, fullScaleOffset, mat, (int)ShowPass.Out);
                    break;

                case RtxptShowMode.ProcessedOutput:
                    Blitter.BlitTexture(cmd, res.ProcessedOutputColor.Handle, fullScaleOffset, mat, (int)ShowPass.Out);
                    break;

                case RtxptShowMode.OutputColor:
                    Blitter.BlitTexture(cmd, res.OutputColor.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                // ── GBuffer ────────────────────────────────────────────────
                case RtxptShowMode.BaseColor:
                    Blitter.BlitTexture(cmd, res.BaseColor.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case RtxptShowMode.RoughnessMetal:
                    Blitter.BlitTexture(cmd, res.RoughnessMetal.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case RtxptShowMode.SpecNormal:
                    Blitter.BlitTexture(cmd, res.SpecNormal.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                // ── Depth / motion ─────────────────────────────────────────
                case RtxptShowMode.Depth:
                    Blitter.BlitTexture(cmd, res.Depth.Handle, scaleOffset, mat, (int)ShowPass.ViewZ);
                    break;

                case RtxptShowMode.MotionVectors:
                    Blitter.BlitTexture(cmd, res.ScreenMotionVectors.Handle, scaleOffset, mat, (int)ShowPass.Mv);
                    break;

                // ── Stable planes ──────────────────────────────────────────
                case RtxptShowMode.SpecularHitT:
                    Blitter.BlitTexture(cmd, res.SpecularHitT.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case RtxptShowMode.StableRadiance:
                    Blitter.BlitTexture(cmd, res.StableRadiance.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                // ── DLSS-RR guide buffers ──────────────────────────────────
                case RtxptShowMode.DlssDiffuseAlbedo:
                    Blitter.BlitTexture(cmd, res.DlssRrDiffAlbedo.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case RtxptShowMode.DlssSpecularAlbedo:
                    Blitter.BlitTexture(cmd, res.DlssRrSpecAlbedo.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case RtxptShowMode.DlssNormal:
                    Blitter.BlitTexture(cmd, res.DlssRrNormalRoughness.Handle, scaleOffset, mat, (int)ShowPass.Normal);
                    break;

                case RtxptShowMode.DlssRoughness:
                    Blitter.BlitTexture(cmd, res.DlssRrNormalRoughness.Handle, scaleOffset, mat, (int)ShowPass.Roughness);
                    break;

                case RtxptShowMode.DlssSpecMotionVectors:
                    Blitter.BlitTexture(cmd, res.DlssRrSpecMotionVectors.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                // ── Debug ──────────────────────────────────────────────────
                case RtxptShowMode.ShaderDebugViz:
                case RtxptShowMode.NEELightColor:
                    Blitter.BlitTexture(cmd, res.ShaderDebugViz.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;

                case RtxptShowMode.DebugOutputColor:
                    Blitter.BlitTexture(cmd, res.DebugOutputColor.Handle, scaleOffset, mat, (int)ShowPass.Out);
                    break;
            }

            cmd.EndSample("OutputBlit");
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            var resourceData = frameData.Get<UniversalResourceData>();

            using var builder = renderGraph.AddUnsafePass<PassData>("Rtxpt Output Blit", out var passData);

            passData.BlitMaterial  = _blitMaterial;
            passData.Resources     = _resources;
            passData.ShowMode      = _showMode;
            passData.RenderScale   = _renderScale;
            passData.CameraTexture = resourceData.activeColorTexture;
            passData.DebugViewType = _debugViewType;

            builder.UseTexture(passData.CameraTexture, AccessFlags.Write);
            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }
    }
}