using System;
using NativeRender;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// ReSTIR DI final shading pass: evaluates the chosen reservoir light, traces the visibility
    /// ray (inline RayQuery), and writes Diffuse/Specular lighting + RestirLuminance.
    /// Source: <c>UnityProject/Assets/RTXDI/Shaders/LightingPasses/DI/ShadeSamples.hlsl</c>
    /// (compiled from <c>ShadeSamples.computeshader</c>).
    /// numthreads = [RTXDI_SCREEN_SPACE_GROUP_SIZE, RTXDI_SCREEN_SPACE_GROUP_SIZE, 1] (8x8).
    /// Bindings come from RAB_Buffers.hlsli — bound through <see cref="NativeRtxdiBindings.BindRabCommon"/>.
    /// </summary>
    public class NativeRtxdiShadeSamplesPass : ScriptableRenderPass, IDisposable
    {
        private const uint GroupSize = 8;

        private readonly NativeComputePipeline      _cs;
        private readonly NativeComputeDescriptorSet _ds;

        private NativeRtxdiPassContext _context;

        public NativeRtxdiShadeSamplesPass(NativeComputeShader shader)
        {
            _cs = new NativeComputePipeline(shader);
            _ds = new NativeComputeDescriptorSet(_cs);
        }

        public void Dispose()
        {
            _ds?.Dispose();
            _cs?.Dispose();
        }

        public void Setup(NativeRtxdiPassContext ctx) => _context = ctx;

        class PassData
        {
            internal NativeComputePipeline      Cs;
            internal NativeComputeDescriptorSet Ds;
            internal NativeRtxdiPassContext     Context;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxdi.DI.ShadeSamples", out var pd);
            pd.Cs = _cs; pd.Ds = _ds; pd.Context = _context;
            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData d, UnsafeGraphContext c) => ExecutePass(d, c));
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var marker = RenderPassMarkers.DiShadeSamplesCompute;
            cmd.BeginSample(marker);

            NativeRtxdiBindings.BindRabCommon(data.Ds, data.Context);

            uint w = (uint)(data.Context.RenderResolution.x * data.Context.ResolutionScale + 0.5f);
            uint h = (uint)(data.Context.RenderResolution.y * data.Context.ResolutionScale + 0.5f);
            uint gx = (w + GroupSize - 1u) / GroupSize;
            uint gy = (h + GroupSize - 1u) / GroupSize;

            data.Cs.Dispatch(cmd, data.Ds, gx, gy, 1);

            cmd.EndSample(marker);
        }
    }
}
