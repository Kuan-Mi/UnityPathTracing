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
    public class NativeRtxptDlssBeforePass : ScriptableRenderPass, IDisposable
    {
        private readonly NativeComputePipeline      _cs;
        private readonly NativeComputeDescriptorSet _ds;
        private          NativeRtxptPassContext     _ctx;

        public NativeRtxptDlssBeforePass(NativeComputeShader shader)
        {
            _cs = new NativeComputePipeline(shader);
            _ds = new NativeComputeDescriptorSet(_cs);
        }

        public void Dispose()
        {
            _ds?.Dispose();
            _cs?.Dispose();
        }

        public void Setup(NativeRtxptPassContext ctx) => _ctx = ctx;

        // ── Pass data ─────────────────────────────────────────────────────────

        private class PassData
        {
            internal NativeComputePipeline      Cs;
            internal NativeComputeDescriptorSet Ds;
            internal NativeRtxptPassContext     Ctx;
        }

        // ── RenderGraph ───────────────────────────────────────────────────────

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxpt.DlssBefore", out var passData);
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

            cmd.BeginSample("Rtxpt.DlssBefore");

            if (ctx.ConstantBuffer != null)
                ds.SetConstantBuffer("g_Const", ctx.ConstantBuffer.GetNativeBufferPtr());

            // Input UAVs
            ds.SetRWTexture("u_OutputColor", res.OutputColor.NativePtr);
            ds.SetRWTexture("u_MotionVectors", res.ScreenMotionVectors.NativePtr);
            ds.SetRWTexture("u_Depth", res.Depth.NativePtr);
            ds.SetRWTexture("u_SpecularHitT", res.SpecularHitT.NativePtr);
            ds.SetRWTexture("u_StablePlanesHeader", res.StablePlanesHeader.NativePtr);
            ds.SetRWTexture("u_StableRadiance", res.StableRadiance.NativePtr);

            if (buf?.StablePlanesBuffer != null)
                ds.SetRWStructuredBuffer("u_StablePlanesBuffer",
                    buf.StablePlanesBuffer.GetNativeBufferPtr(),
                    buf.StablePlanesBuffer.count, buf.StablePlanesBuffer.stride);

            // Output guide UAVs
            ds.SetRWTexture("u_RRDiffuseAlbedo", res.DlssRrDiffAlbedo.NativePtr);
            ds.SetRWTexture("u_RRSpecAlbedo", res.DlssRrSpecAlbedo.NativePtr);
            ds.SetRWTexture("u_RRNormalsAndRoughness", res.DlssRrNormalRoughness.NativePtr);
            // u_RRSpecMotionVectors reuses ScreenMotionVectors (or can be left to MotionVectors)
            ds.SetRWTexture("u_RRSpecMotionVectors", res.ScreenMotionVectors.NativePtr);

            uint gx = ((uint)ctx.RenderResolution.x + 7u) / 8u;
            uint gy = ((uint)ctx.RenderResolution.y + 7u) / 8u;
            data.Cs.Dispatch(cmd, ds, gx, gy, 1);

            cmd.EndSample("Rtxpt.DlssBefore");
        }
    }
}