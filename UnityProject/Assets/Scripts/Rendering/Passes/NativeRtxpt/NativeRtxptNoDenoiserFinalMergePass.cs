using System;
using NativeRender;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Phase 5: Merges stable planes into OutputColor without NRD denoising.
    ///
    /// Shader: PostProcess_NoDenoiserFinalMerge.computeshader  numthreads [8,8,1]
    ///
    /// Bindings (reflection JSON):
    ///   b0  g_Const
    ///   u0  u_OutputColor
    ///   u40 u_StablePlanesHeader  (Texture2DArray)
    ///   u42 u_StablePlanesBuffer  (RWByteAddressBuffer)
    ///   u44 u_StableRadiance
    /// </summary>
    public class NativeRtxptNoDenoiserFinalMergePass : ScriptableRenderPass, IDisposable
    {
        private readonly NativeComputePipeline      _cs;
        private readonly NativeComputeDescriptorSet _ds;
        private          NativeRtxptPassContext     _ctx;

        public NativeRtxptNoDenoiserFinalMergePass(NativeComputeShader shader)
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
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxpt.NoDenoiserFinalMerge", out var passData);
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

            cmd.BeginSample("Rtxpt.NoDenoiserFinalMerge");

            if (ctx.ConstantBuffer != null)
                ds.SetConstantBuffer("g_Const", ctx.ConstantBuffer.GetNativeBufferPtr());

            if (res.OutputColor.IsCreated)       ds.SetRWTexture("u_OutputColor",       res.OutputColor.NativePtr);
            if (res.StablePlanesHeader.IsCreated) ds.SetRWTexture("u_StablePlanesHeader", res.StablePlanesHeader.NativePtr);
            if (res.StableRadiance.IsCreated)    ds.SetRWTexture("u_StableRadiance",    res.StableRadiance.NativePtr);

            if (buf?.StablePlanesBuffer != null)
                ds.SetRWStructuredBuffer("u_StablePlanesBuffer",
                    buf.StablePlanesBuffer.GetNativeBufferPtr(),
                    buf.StablePlanesBuffer.count, buf.StablePlanesBuffer.stride);

            uint gx = ((uint)ctx.RenderResolution.x + 7u) / 8u;
            uint gy = ((uint)ctx.RenderResolution.y + 7u) / 8u;
            data.Cs.Dispatch(cmd, ds, gx, gy, 1);

            cmd.EndSample("Rtxpt.NoDenoiserFinalMerge");
        }
    }
}
