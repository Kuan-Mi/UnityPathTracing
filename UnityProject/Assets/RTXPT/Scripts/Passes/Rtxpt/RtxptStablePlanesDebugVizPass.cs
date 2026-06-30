using System;
using NativeRender;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Debug pass: runs the StablePlanesDebugViz compute shader and writes the
    /// result into the ShaderDebugViz texture, which <see cref="RtxptOutputBlitPass"/>
    /// then blits to screen when a debug view is active.
    ///
    /// Shader: PostProcess_StablePlanesDebugViz.computeshader  numthreads [8,8,1]
    ///
    /// Bindings (reflection JSON):
    ///   b0   g_Const
    ///   u40  u_StablePlanesHeader
    ///   u42  u_StablePlanesBuffer
    ///   u44  u_StableRadiance
    ///   u126 u_ShaderDebugVizTextureBuffer
    /// </summary>
    public class RtxptStablePlanesDebugVizPass : ScriptableRenderPass, IDisposable
    {
        private readonly NativeComputePipeline      _cs;
        private readonly NativeComputeDescriptorSet _ds;
        private          RtxptPassContext     _ctx;

        public RtxptStablePlanesDebugVizPass(NativeComputeShader shader)
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
            using var builder = renderGraph.AddUnsafePass<PassData>("StablePlanesDebugViz", out var passData);
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

            cmd.BeginSample("StablePlanesDebugViz");

            if (ctx.ConstantBuffer != null)
                ds.SetConstantBuffer("g_Const", ctx.ConstantBuffer);

            ds.SetRWTexture("u_StablePlanesHeader", res.StablePlanesHeader.NativePtr);
            ds.SetRWTexture("u_StableRadiance",     res.StableRadiance.NativePtr);

            if (buf?.StablePlanesBufferPtr != IntPtr.Zero)
                ds.SetRWStructuredBuffer("u_StablePlanesBuffer",
                    buf.StablePlanesBufferPtr,
                    buf.StablePlanesBuffer.count, buf.StablePlanesBuffer.stride);

            ds.SetRWTexture("u_ShaderDebugVizTextureBuffer", res.ShaderDebugViz.NativePtr);

            uint gx = ((uint)ctx.RenderResolution.x + 7u) / 8u;
            uint gy = ((uint)ctx.RenderResolution.y + 7u) / 8u;
            data.Cs.Dispatch(cmd, ds, gx, gy, 1);

            cmd.EndSample("StablePlanesDebugViz");
        }
    }
}
