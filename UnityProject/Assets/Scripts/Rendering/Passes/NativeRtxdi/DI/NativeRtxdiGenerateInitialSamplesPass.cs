using System;
using NativeRender;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Reference Native-RTXDI compute pass — initial sample generation for ReSTIR DI.
    /// Source: <c>UnityProject/Assets/RTXDI/Shaders/LightingPasses/DI/GenerateInitialSamples.computeshader</c>.
    /// Mirrors <see cref="NRDOpaquePass"/> shape: owns one <see cref="NativeComputePipeline"/> +
    /// one <see cref="NativeComputeDescriptorSet"/>; binds reflected resources then dispatches.
    /// </summary>
    // === Shader Reflection: GenerateInitialSamples.computeshader ===
    // numthreads  [8, 8, 1]
    //
    // -- CBV (2) --
    //   g_Const                          ConstantBuffer<ResamplingConstants>   space0:b0
    //   g_PerPassConstants               ConstantBuffer<PerPassConstants>      space0:b1
    //
    // -- Sampler (1) --
    //   s_EnvironmentSampler             SamplerState                          space0:s1
    //
    // -- SRV (10) --
    //   t_BindlessTextures               Texture2D[]                           space2:t0
    //   t_GBufferDepth                   Texture2D<float>                      space0:t0
    //   t_GBufferNormals                 Texture2D<uint>                       space0:t1
    //   t_GBufferGeoNormals              Texture2D<uint>                       space0:t2
    //   t_GBufferDiffuseAlbedo           Texture2D<uint>                       space0:t3
    //   t_GBufferSpecularRough           Texture2D<uint>                       space0:t4
    //   t_InstanceData                   Buffer<mixed>                         space0:t32
    //   t_LightDataBuffer                Buffer<mixed>                         space0:t20
    //   t_EnvironmentPdfTexture          Texture2D<float>                      space0:t23
    //   t_LocalLightPdfTexture           Texture2D<float>                      space0:t24
    //   t_GeometryInstanceToLight        Buffer<mixed>                         space0:t25
    //
    // -- TLAS (1) --
    //   SceneBVH                         RaytracingAccelerationStructure       space0:t30
    //
    // -- UAV (4) --
    //   u_LightReservoirs                RWBuffer<mixed>                       space0:u0
    //   u_RisBuffer                      RWBuffer<uint>                        space0:u10
    //   u_RisLightDataBuffer             RWBuffer<uint>                        space0:u11
    //   u_RayCountBuffer                 RWBuffer<uint>                        space0:u12
    public class NativeRtxdiGenerateInitialSamplesPass : ScriptableRenderPass, IDisposable
    {
        private const uint GroupSize = 8;

        // Compute shader path
        private readonly NativeComputePipeline      _cs;
        private readonly NativeComputeDescriptorSet _ds;

        // Ray tracing shader path
        private readonly RayTracePipeline            _rs;
        private readonly NativeRayTraceDescriptorSet _rds;

        private NativeRtxdiPassContext _context;
        private bool                   _useRayTracing;

        public NativeRtxdiGenerateInitialSamplesPass(NativeComputeShader shader, RayTraceShader rs)
        {
            _cs  = new NativeComputePipeline(shader);
            _ds  = new NativeComputeDescriptorSet(_cs);

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

        // -------------------------------------------------------------------------
        // RenderGraph
        // -------------------------------------------------------------------------

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
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxdi.GenerateInitialSamples", out var passData);

            passData.Cs           = _cs;
            passData.Ds           = _ds;
            passData.Rs           = _rs;
            passData.Rds          = _rds;
            passData.Context      = _context;
            passData.UseRayTracing = _useRayTracing;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var ctx = data.Context;

            uint w       = (uint)(ctx.RenderResolution.x * ctx.ResolutionScale + 0.5f);
            uint h       = (uint)(ctx.RenderResolution.y * ctx.ResolutionScale + 0.5f);

            if (data.UseRayTracing && data.Rs != null && data.Rds != null)
            {
                cmd.BeginSample(RenderPassMarkers.GenInitialSamples);
                NativeRtxdiBindings.BindRabCommon(data.Rds, ctx);
                data.Rs.Dispatch(cmd, data.Rds, w, h);
                cmd.EndSample(RenderPassMarkers.GenInitialSamples);
            }
            else
            {
                cmd.BeginSample(RenderPassMarkers.GenInitialSamplesCompute);
                NativeRtxdiBindings.BindRabCommon(data.Ds, ctx);

                uint groupsX = (w + GroupSize - 1u) / GroupSize;
                uint groupsY = (h + GroupSize - 1u) / GroupSize;
                data.Cs.Dispatch(cmd, data.Ds, groupsX, groupsY, 1);
                cmd.EndSample(RenderPassMarkers.GenInitialSamplesCompute);
            }
        }
    }
}
