using System;
using System.Runtime.InteropServices;
using NativeRender;
using Unity.Mathematics;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Phase 8 (reference mode only): Multi-frame accumulation pass.
    ///
    /// Shader: AccumulationPass.computeshader  numthreads [8,8,1]
    ///
    /// Bindings (reflection JSON):
    ///   b0  g_Const
    ///   t0  t_InputColor          (SRV — reads from OutputColor)
    ///   u0  u_AccumulatedColor    (UAV RW — AccumulatedRadiance)
    ///   u1  u_OutputColor         (UAV write — ProcessedOutputColor)
    ///   s0  s_Sampler
    /// </summary>
    public class NativeRtxptAccumulationPass : ScriptableRenderPass, IDisposable
    {
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        private struct AccumulationConstants
        {
            public float2 outputSize;
            public float2 inputSize;
            public float2 inputTextureSizeInv;
            public float2 pixelOffset;
            public float  blendFactor;
        }

        private readonly NativeComputePipeline      _cs;
        private readonly NativeComputeDescriptorSet _ds;
        private          NativeRtxptPassContext     _ctx;

        public NativeRtxptAccumulationPass(NativeComputeShader shader)
        {
            // Root-constant hints (g_Const) live on the .computeshader asset (the importer).
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
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxpt.Accumulation", out var passData);
            passData.Cs  = _cs;
            passData.Ds  = _ds;
            passData.Ctx = _ctx;
            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData d, UnsafeGraphContext c) => ExecutePass(d, c));
        }

        // ── Execute ───────────────────────────────────────────────────────────

        private static unsafe void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var ctx = data.Ctx;
            var ds  = data.Ds;
            var res = ctx.Textures;

            cmd.BeginSample("Rtxpt.Accumulation");

            var inputSize  = new float2(ctx.RenderResolution.x, ctx.RenderResolution.y);
            var outputSize = new float2(ctx.DisplayResolution.x, ctx.DisplayResolution.y);
            uint sampleIndex = math.max(1u, ctx.FrameState.frameIndex);
            float blendFactor = sampleIndex <= (uint)math.max(1, ctx.Setting.accumulationTarget)
                ? 1.0f / sampleIndex
                : 0.0f;
            var constants = new AccumulationConstants
            {
                outputSize          = outputSize,
                inputSize           = inputSize,
                inputTextureSizeInv = 1.0f / inputSize,
                pixelOffset         = float2.zero,
                blendFactor         = blendFactor,
            };
            ds.SetRootConstants("g_Const", &constants);

            // SRV input: noisy PT output
            ds.SetTexture("t_InputColor", res.OutputColor.NativePtr);

            // UAV accumulation buffer (read-modify-write)
            ds.SetRWTexture("u_AccumulatedColor", res.AccumulatedRadiance.NativePtr);

            // UAV output: post-processed result
            ds.SetRWTexture("u_OutputColor", res.ProcessedOutputColor.NativePtr);

            uint gx = ((uint)ctx.DisplayResolution.x + 7u) / 8u;
            uint gy = ((uint)ctx.DisplayResolution.y + 7u) / 8u;
            data.Cs.Dispatch(cmd, ds, gx, gy, 1);

            cmd.EndSample("Rtxpt.Accumulation");
        }
    }
}
