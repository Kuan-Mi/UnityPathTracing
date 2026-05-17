using System;
using NativeRender;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// ReSTIR DI temporal resampling pass.
    /// Source: <c>UnityProject/Assets/RTXDI/Shaders/LightingPasses/DI/TemporalResampling.hlsl</c>
    /// (compiled from <c>TemporalResampling.computeshader</c>).
    /// numthreads = [RTXDI_SCREEN_SPACE_GROUP_SIZE, RTXDI_SCREEN_SPACE_GROUP_SIZE, 1] (8x8).
    /// Bindings come from RAB_Buffers.hlsli — bound through <see cref="NativeRtxdiBindings.BindRabCommon"/>.
    /// </summary>
    public class NativeRtxdiTemporalResamplingPass : ScriptableRenderPass, IDisposable
    {
        private const uint GroupSize = 8;

        private readonly NativeComputePipeline      _cs;
        private readonly NativeComputeDescriptorSet _ds;
        private readonly RayTracePipeline            _rs;
        private readonly NativeRayTraceDescriptorSet _rds;

        private NativeRtxdiPassContext _context;
        private bool                   _useRayTracing;

        public NativeRtxdiTemporalResamplingPass(NativeComputeShader shader, RayTraceShader rs)
        {
            _cs = new NativeComputePipeline(shader);
            _ds = new NativeComputeDescriptorSet(_cs);
            if (rs != null)
            {
                _rs  = new RayTracePipeline(rs);
                _rds = new NativeRayTraceDescriptorSet(_rs);
            }
        }

        public void Dispose()
        {
            _ds?.Dispose();
            _cs?.Dispose();
            _rds?.Dispose();
            _rs?.Dispose();
        }

        public void Setup(NativeRtxdiPassContext ctx, bool useRayTracing = false)
        {
            _context       = ctx;
            _useRayTracing = useRayTracing && _rs != null;
        }

        class PassData
        {
            internal NativeComputePipeline      Cs;
            internal NativeComputeDescriptorSet Ds;
            internal RayTracePipeline            Rs;
            internal NativeRayTraceDescriptorSet Rds;
            internal NativeRtxdiPassContext     Context;
            internal bool                       UseRayTracing;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxdi.DI.TemporalResampling", out var pd);
            pd.Cs = _cs; pd.Ds = _ds; pd.Rs = _rs; pd.Rds = _rds; pd.Context = _context; pd.UseRayTracing = _useRayTracing;
            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData d, UnsafeGraphContext c) => ExecutePass(d, c));
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            uint w  = (uint)(data.Context.RenderResolution.x * data.Context.ResolutionScale + 0.5f);
            uint h  = (uint)(data.Context.RenderResolution.y * data.Context.ResolutionScale + 0.5f);

            if (data.UseRayTracing && data.Rs != null && data.Rds != null)
            {
                cmd.BeginSample(RenderPassMarkers.DiTemporalResampling);
                NativeRtxdiBindings.BindRabCommon(data.Rds, data.Context);
                data.Rs.Dispatch(cmd, data.Rds, w, h);
                cmd.EndSample(RenderPassMarkers.DiTemporalResampling);
            }
            else
            {
                cmd.BeginSample(RenderPassMarkers.DiTemporalResamplingCompute);
                NativeRtxdiBindings.BindRabCommon(data.Ds, data.Context);
                uint gx = (w + GroupSize - 1u) / GroupSize;
                uint gy = (h + GroupSize - 1u) / GroupSize;
                data.Cs.Dispatch(cmd, data.Ds, gx, gy, 1);
                cmd.EndSample(RenderPassMarkers.DiTemporalResamplingCompute);
            }
        }
    }
}
