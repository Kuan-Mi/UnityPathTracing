using System;
using NativeRender;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Native compute pass that traces primary rays and fills the G-Buffer.
    /// Mirrors <c>RaytracedGBufferPass::Render</c> from
    /// <c>RTXDI/Samples/FullSample/Source/RenderPasses/GBufferPass.cpp</c>.
    ///
    /// Source shader: <c>Assets/RTXDI/Shaders/RaytracedGBuffer.computeshader</c>
    /// (USE_RAY_QUERY baked into DXIL — no separate RayTracingShader required).
    ///
    /// Must run AFTER <see cref="NativeRtxdiBuildAccelerationStructurePass"/> so the TLAS is ready.
    /// Must run BEFORE <see cref="NativeRtxdiPostprocessGBufferPass"/>.
    /// </summary>
    // === Shader Reflection: RaytracedGBuffer.computeshader ===
    // numthreads [16, 16, 1]
    //
    // -- CBV (2) --
    //   g_Const               ConstantBuffer<GBufferConstants>      space0:b0
    //   g_PerPassConstants    ConstantBuffer<PerPassConstants>       space0:b1
    //
    // -- TLAS (1) --
    //   SceneBVH              RaytracingAccelerationStructure        space0:t0
    //
    // -- SRV (3) --
    //   t_InstanceData        StructuredBuffer<InstanceData>         space0:t1
    //   t_GeometryData        StructuredBuffer<GeometryData>         space0:t2
    //   t_MaterialConstants   StructuredBuffer<MaterialConstants>    space0:t3
    //
    // -- Sampler (1) --
    //   s_MaterialSampler     SamplerState                           space0:s0
    //
    // -- UAV (9) --
    //   u_ViewDepth           RWTexture2D<float>                     space0:u0
    //   u_DiffuseAlbedo       RWTexture2D<uint>                      space0:u1
    //   u_SpecularRough       RWTexture2D<uint>                      space0:u2
    //   u_Normals             RWTexture2D<uint>                      space0:u3
    //   u_GeoNormals          RWTexture2D<uint>                      space0:u4
    //   u_Emissive            RWTexture2D<float4>                    space0:u5
    //   u_MotionVectors       RWTexture2D<float4>                    space0:u6
    //   u_DeviceDepth         RWTexture2D<float>                     space0:u7
    //   u_RayCountBuffer      RWBuffer<uint>                         space0:u8
    public class NativeRtxdiRaytracedGBufferPass : ScriptableRenderPass, IDisposable
    {
        private const uint GroupSize = 16;

        private readonly NativeComputePipeline      _cs;
        private readonly NativeComputeDescriptorSet _ds;

        private NativeRtxdiPassContext _context;

        public NativeRtxdiRaytracedGBufferPass(NativeComputeShader shader)
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
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxdi.RaytracedGBuffer", out var passData);
            passData.Cs      = _cs;
            passData.Ds      = _ds;
            passData.Context = _context;
            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            cmd.BeginSample(RenderPassMarkers.RaytracedGBufferCompute);

            var cs  = data.Cs;
            var ds  = data.Ds;
            var ctx = data.Context;

            // Constant buffers
            // g_Const must be GBufferConstants — use the dedicated buffer, NOT ctx.ConstantBuffer (GlobalConstants)
            if (ctx.GBufferConstantBuffer != null)
                ds.SetConstantBuffer("g_Const", ctx.GBufferConstantBuffer.GetNativeBufferPtr());
            if (ctx.PerPassConstantBuffer != null)
                ds.SetConstantBuffer("g_PerPassConstants", ctx.PerPassConstantBuffer.GetNativeBufferPtr());

            // TLAS + scene buffers in donut-compatible layout
       
            ds.SetAccelerationStructure("SceneBVH", ctx.RtxdiGpuScene.AccelerationStructure);
            ctx.RtxdiGpuScene?.BindToShader(ds);

            // GBuffer output UAVs
            if (ctx.ViewDepthPtr     != IntPtr.Zero) ds.SetRWTexture("u_ViewDepth",     ctx.ViewDepthPtr);
            if (ctx.DiffuseAlbedoPtr != IntPtr.Zero) ds.SetRWTexture("u_DiffuseAlbedo", ctx.DiffuseAlbedoPtr);
            if (ctx.SpecularRoughPtr != IntPtr.Zero) ds.SetRWTexture("u_SpecularRough",  ctx.SpecularRoughPtr);
            if (ctx.NormalsPtr       != IntPtr.Zero) ds.SetRWTexture("u_Normals",        ctx.NormalsPtr);
            if (ctx.GeoNormalsPtr    != IntPtr.Zero) ds.SetRWTexture("u_GeoNormals",     ctx.GeoNormalsPtr);
            if (ctx.EmissivePtr      != IntPtr.Zero) ds.SetRWTexture("u_Emissive",       ctx.EmissivePtr);
            if (ctx.MotionVectorsPtr != IntPtr.Zero) ds.SetRWTexture("u_MotionVectors",  ctx.MotionVectorsPtr);
            if (ctx.DeviceDepthPtr   != IntPtr.Zero) ds.SetRWTexture("u_DeviceDepth",    ctx.DeviceDepthPtr);
            if (ctx.RayCountBuffer   != null)        ds.SetRWBuffer("u_RayCountBuffer",  ctx.RayCountBuffer.GetNativeBufferPtr());

            uint w       = (uint)(ctx.RenderResolution.x * ctx.ResolutionScale + 0.5f);
            uint h       = (uint)(ctx.RenderResolution.y * ctx.ResolutionScale + 0.5f);
            uint groupsX = (w + GroupSize - 1u) / GroupSize;
            uint groupsY = (h + GroupSize - 1u) / GroupSize;

            cs.Dispatch(cmd, ds, groupsX, groupsY, 1);

            cmd.EndSample(RenderPassMarkers.RaytracedGBufferCompute);
        }
    }
}
