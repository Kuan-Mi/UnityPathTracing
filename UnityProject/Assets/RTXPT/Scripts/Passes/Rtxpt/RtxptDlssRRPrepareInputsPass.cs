using System;
using NativeRender;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Phase 6: Prepares DLSS-RR guide buffers from PT GBuffer data.
    ///
    /// Shader: PostProcess_DenoiserPrepareInputsDlssRR.computeshader  numthreads [8,8,1]
    ///
    /// Bindings (reflection JSON):
    ///   b0  g_Const
    ///   u0  u_OutputColor
    ///   u5  u_MotionVectors
    ///   u6  u_Depth
    ///   u7  u_SpecularHitT
    ///   u40 u_StablePlanesHeader
    ///   u42 u_StablePlanesBuffer
    ///   u44 u_StableRadiance
    ///   u70 u_RRDiffuseAlbedo
    ///   u71 u_RRSpecAlbedo
    ///   u72 u_RRNormalsAndRoughness
    ///   u73 u_RRSpecMotionVectors
    /// </summary>
    public class RtxptDlssRRPrepareInputsPass : ScriptableRenderPass, IDisposable
    {
        private readonly NativeComputePipeline      _cs;
        private readonly NativeComputeDescriptorSet _ds;
        private          RtxptPassContext     _ctx;

        public RtxptDlssRRPrepareInputsPass(NativeComputeShader shader)
        {
            _cs = new NativeComputePipeline(shader);
            _ds = new NativeComputeDescriptorSet(_cs);
        }

        public void Dispose()
        {
            _ds?.Dispose();
            _cs?.Dispose();
        }

        public void Setup(RtxptPassContext ctx) => _ctx = ctx;

        // ── Pass data ─────────────────────────────────────────────────────────

        private class PassData
        {
            internal NativeComputePipeline      Cs;
            internal NativeComputeDescriptorSet Ds;
            internal RtxptPassContext     Ctx;
        }

        // ── RenderGraph ───────────────────────────────────────────────────────

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("DLSS-RR", out var passData);
            passData.Cs  = _cs;
            passData.Ds  = _ds;
            passData.Ctx = _ctx;
            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData d, UnsafeGraphContext c) => ExecutePass(d, c));
        }

        // ── Execute ───────────────────────────────────────────────────────────

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var ctx = data.Ctx;
            var ds  = data.Ds;
            var res = ctx.Textures;
            var buf = ctx.Buffers;

            cmd.BeginSample("DLSSRR_PrepareInputs");

            ds.SetConstantBuffer("g_Const", ctx.ConstantBuffer);

            // Input UAVs
            ds.SetRWTexture("u_OutputColor", res.OutputColor.NativePtr);
            ds.SetRWTexture("u_MotionVectors", res.ScreenMotionVectors.NativePtr);
            ds.SetRWTexture("u_Depth", res.Depth.NativePtr);
            ds.SetRWTexture("u_SpecularHitT", res.SpecularHitT.NativePtr);
            ds.SetRWTexture("u_StablePlanesHeader", res.StablePlanesHeader.NativePtr);
            ds.SetRWTexture("u_StableRadiance", res.StableRadiance.NativePtr);

            ds.SetRWStructuredBuffer("u_StablePlanesBuffer",
                buf.StablePlanesBufferPtr,
                buf.StablePlanesBuffer.count, buf.StablePlanesBuffer.stride);

            ds.SetRWTexture("u_ShaderDebugVizTextureBuffer", res.ShaderDebugViz.NativePtr);

            // Output guide UAVs
            ds.SetRWTexture("u_RRDiffuseAlbedo", res.DlssRrDiffAlbedo.NativePtr);
            ds.SetRWTexture("u_RRSpecAlbedo", res.DlssRrSpecAlbedo.NativePtr);
            ds.SetRWTexture("u_RRNormalsAndRoughness", res.DlssRrNormalRoughness.NativePtr);
            ds.SetRWTexture("u_RRSpecMotionVectors", res.DlssRrSpecMotionVectors.NativePtr);

            uint gx = ((uint)ctx.RenderResolution.x + 7u) / 8u;
            uint gy = ((uint)ctx.RenderResolution.y + 7u) / 8u;
            data.Cs.Dispatch(cmd, ds, gx, gy, 1);

            cmd.EndSample("DLSSRR_PrepareInputs");
        }
    }
}
