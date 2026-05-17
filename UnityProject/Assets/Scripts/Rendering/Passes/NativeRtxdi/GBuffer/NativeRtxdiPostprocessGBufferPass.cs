using System;
using NativeRender;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Native compute pass that post-processes the G-Buffer to compute NRD-compatible
    /// <c>NormalRoughness</c> from the raw normals + view depth.
    /// Mirrors <c>PostprocessGBufferPass::Render</c> from
    /// <c>RTXDI/Samples/FullSample/Source/RenderPasses/GBufferPass.cpp</c>.
    ///
    /// Source shader: <c>Assets/RTXDI/Shaders/PostprocessGBuffer.computeshader</c>.
    ///
    /// Must run immediately AFTER <see cref="NativeRtxdiRaytracedGBufferPass"/>.
    /// Writes to <c>u_NormalRoughness</c> which is consumed by NRD denoising passes.
    /// Also modifies <c>u_SpecularRough</c> in-place (roughness variance from normal map).
    /// </summary>
    // === Shader Reflection: PostprocessGBuffer.computeshader ===
    // numthreads [16, 16, 1]
    //
    // -- UAV (2) --
    //   u_SpecularRough    RWTexture2D<uint>    space0:u0
    //   u_NormalRoughness  RWTexture2D<float4>  space0:u1
    //
    // -- SRV (2) --
    //   t_Normals          Texture2D<uint>      space0:t0
    //   t_ViewDepth        Texture2D<float>     space0:t1
    public class NativeRtxdiPostprocessGBufferPass : ScriptableRenderPass, IDisposable
    {
        private const uint GroupSize = 16;

        private readonly NativeComputePipeline      _cs;
        private readonly NativeComputeDescriptorSet _ds;

        private NativeRtxdiPassContext _context;

        public NativeRtxdiPostprocessGBufferPass(NativeComputeShader shader)
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
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxdi.PostprocessGBuffer", out var passData);
            passData.Cs      = _cs;
            passData.Ds      = _ds;
            passData.Context = _context;
            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            cmd.BeginSample(RenderPassMarkers.PostprocessGBufferCompute);

            var cs  = data.Cs;
            var ds  = data.Ds;
            var ctx = data.Context;

            // UAVs
            if (ctx.SpecularRoughPtr          != IntPtr.Zero) ds.SetRWTexture("u_SpecularRough",   ctx.SpecularRoughPtr);
            if (ctx.DenoiserNormalRoughnessPtr != IntPtr.Zero) ds.SetRWTexture("u_NormalRoughness", ctx.DenoiserNormalRoughnessPtr);

            // SRVs
            if (ctx.NormalsPtr   != IntPtr.Zero) ds.SetTexture("t_Normals",   ctx.NormalsPtr);
            if (ctx.ViewDepthPtr != IntPtr.Zero) ds.SetTexture("t_ViewDepth", ctx.ViewDepthPtr);

            uint w       = (uint)(ctx.RenderResolution.x * ctx.ResolutionScale + 0.5f);
            uint h       = (uint)(ctx.RenderResolution.y * ctx.ResolutionScale + 0.5f);
            uint groupsX = (w + GroupSize - 1u) / GroupSize;
            uint groupsY = (h + GroupSize - 1u) / GroupSize;

            cs.Dispatch(cmd, ds, groupsX, groupsY, 1);

            cmd.EndSample(RenderPassMarkers.PostprocessGBufferCompute);
        }
    }
}
