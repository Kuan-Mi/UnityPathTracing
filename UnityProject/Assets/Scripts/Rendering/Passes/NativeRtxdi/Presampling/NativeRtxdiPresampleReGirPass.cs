using System;
using NativeRender;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Native compute pass — presample local lights into the ReGIR RIS buffer.
    /// Source: <c>UnityProject/Assets/RTXDI/Shaders/LightingPasses/Presampling/PresampleReGIR.computeshader</c>.
    ///
    /// Mirrors the managed <see cref="PresampleReGirLightsPass"/> but uses
    /// <see cref="NativeComputePipeline"/> / <see cref="NativeComputeDescriptorSet"/> so the shader
    /// runs as a native CS 6.x dispatch.
    ///
    /// Dispatch mirrors FullSample <c>SceneRenderer::RenderFrame</c>:
    ///   groupsX = ceil(regirContext.GetReGIRLightSlotCount() / ReGIR_TILE_SIZE),  groupsY = 1.
    /// </summary>
    // === Shader Reflection: PresampleReGIR.computeshader ===
    // numthreads  [256, 1, 1]
    //
    // -- CBV (1) --
    //   g_Const                          ConstantBuffer<ResamplingConstants>   space0:b0
    //
    // -- SRV (2) --
    //   t_LocalLightPdfTexture           Texture2D<float>                      space0:t24
    //   t_LightDataBuffer                Buffer<mixed>                         space0:t20
    //
    // -- UAV (2) --
    //   u_RisBuffer                      RWBuffer<uint>                        space0:u10
    //   u_RisLightDataBuffer             RWBuffer<uint>                        space0:u11
    public class NativeRtxdiPresampleReGirPass : ScriptableRenderPass, IDisposable
    {
        private readonly NativeComputePipeline      _cs;
        private readonly NativeComputeDescriptorSet _ds;

        private NativeRtxdiPassContext _context;
        private uint                   _groupsX;

        public NativeRtxdiPresampleReGirPass(NativeComputeShader shader)
        {
            _cs = new NativeComputePipeline(shader);
            _ds = new NativeComputeDescriptorSet(_cs);
        }

        public void Dispose()
        {
            _ds?.Dispose();
            _cs?.Dispose();
        }

        /// <summary>
        /// Setup the pass.
        ///   groupsX = ceil(regirContext.GetReGIRLightSlotCount() / 256).
        /// </summary>
        public void Setup(NativeRtxdiPassContext ctx, uint groupsX)
        {
            _context = ctx;
            _groupsX = groupsX;
        }

        // -------------------------------------------------------------------------
        // RenderGraph
        // -------------------------------------------------------------------------

        class PassData
        {
            internal NativeComputePipeline      Cs;
            internal NativeComputeDescriptorSet Ds;
            internal NativeRtxdiPassContext     Context;
            internal uint                       GroupsX;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxdi.PresampleReGIR", out var passData);

            passData.Cs      = _cs;
            passData.Ds      = _ds;
            passData.Context = _context;
            passData.GroupsX = _groupsX;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            cmd.BeginSample(RenderPassMarkers.ReGir);

            var cs  = data.Cs;
            var ds  = data.Ds;
            var ctx = data.Context;

            // Same RAB superset as PresampleLights: g_Const, t_LocalLightPdfTexture,
            // t_LightDataBuffer, u_RisBuffer, u_RisLightDataBuffer.
            NativeRtxdiBindings.BindRabCommon(ds, ctx);
            
            Debug.Log($"Dispatching NativeRtxdiPresampleReGirPass with groupsX={data.GroupsX}, groupsY=1");

            cs.Dispatch(cmd, ds, data.GroupsX, 1, 1);

            cmd.EndSample(RenderPassMarkers.ReGir);
        }
    }
}
