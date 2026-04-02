using mini;
using Unity.Mathematics;
using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;
using static PathTracing.ShaderIDs;

namespace PathTracing
{
    public class GISpatialResamplingPass : ScriptableRenderPass
    {
        private const int GroupSize = 8;

        private readonly RayTracingShader _rtShader;
        private readonly ComputeShader _computeShader;
        private RtxdiPassContext _context;
        private bool _useCompute;

        public GISpatialResamplingPass(RayTracingShader rtShader, ComputeShader computeShader)
        {
            _rtShader = rtShader;
            _computeShader = computeShader;
        }

        public void Setup(RtxdiPassContext ctx, bool useCompute)
        {
            _context = ctx;
            _useCompute = useCompute;
        }

        class PassData
        {
            internal RayTracingShader RtShader;
            internal ComputeShader ComputeShader;
            internal RtxdiPassContext Context;
            internal bool UseCompute;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var ctx = data.Context;

            if (data.UseCompute)
            {
                var marker = new ProfilerMarker(ProfilerCategory.Render, "GISpatialResampling_Compute", MarkerFlags.SampleGPU);
                natCmd.BeginSample(marker);

                var cs = data.ComputeShader;
                int kernel = cs.FindKernel("main");

                natCmd.SetComputeConstantBufferParam(cs, paramsID, ctx.ConstantBuffer, 0, ctx.ConstantBuffer.stride);
                natCmd.SetComputeConstantBufferParam(cs, "g_Const", ctx.ResamplingConstantBuffer, 0, ctx.ResamplingConstantBuffer.stride);

                natCmd.SetComputeBufferParam(cs, kernel, "u_GIReservoirs", ctx.RtxdiResources.GIReservoirBuffer);
                natCmd.SetComputeBufferParam(cs, kernel, t_NeighborOffsetsID, ctx.RtxdiResources.NeighborOffsetsBuffer);

                natCmd.SetComputeTextureParam(cs, kernel, "t_GBufferDepth", ctx.ViewDepth);
                natCmd.SetComputeTextureParam(cs, kernel, "t_GBufferDiffuseAlbedo", ctx.DiffuseAlbedo);
                natCmd.SetComputeTextureParam(cs, kernel, "t_GBufferSpecularRough", ctx.SpecularRough);
                natCmd.SetComputeTextureParam(cs, kernel, "t_GBufferNormals", ctx.Normals);
                natCmd.SetComputeTextureParam(cs, kernel, "t_GBufferGeoNormals", ctx.GeoNormals);

                int rectW = (int)(ctx.RenderResolution.x * ctx.ResolutionScale + 0.5f);
                int rectH = (int)(ctx.RenderResolution.y * ctx.ResolutionScale + 0.5f);
                int groupsX = (rectW + GroupSize - 1) / GroupSize;
                int groupsY = (rectH + GroupSize - 1) / GroupSize;
                natCmd.DispatchCompute(cs, kernel, groupsX, groupsY, 1);

                natCmd.EndSample(marker);
            }
            else
            {
                var marker = new ProfilerMarker(ProfilerCategory.Render, "GISpatialResampling", MarkerFlags.SampleGPU);
                natCmd.BeginSample(marker);

                var shader = data.RtShader;

                natCmd.SetRayTracingShaderPass(shader, "RTXDI");
                natCmd.SetRayTracingConstantBufferParam(shader, paramsID, ctx.ConstantBuffer, 0, ctx.ConstantBuffer.stride);
                natCmd.SetRayTracingBufferParam(shader, "ResampleConstants", ctx.ResamplingConstantBuffer);

                natCmd.SetRayTracingBufferParam(shader, "u_GIReservoirs", ctx.RtxdiResources.GIReservoirBuffer);
                natCmd.SetRayTracingBufferParam(shader, t_NeighborOffsetsID, ctx.RtxdiResources.NeighborOffsetsBuffer);

                natCmd.SetRayTracingTextureParam(shader, "t_GBufferDepth", ctx.ViewDepth);
                natCmd.SetRayTracingTextureParam(shader, "t_GBufferDiffuseAlbedo", ctx.DiffuseAlbedo);
                natCmd.SetRayTracingTextureParam(shader, "t_GBufferSpecularRough", ctx.SpecularRough);
                natCmd.SetRayTracingTextureParam(shader, "t_GBufferNormals", ctx.Normals);
                natCmd.SetRayTracingTextureParam(shader, "t_GBufferGeoNormals", ctx.GeoNormals);

                uint rectW = (uint)(ctx.RenderResolution.x * ctx.ResolutionScale + 0.5f);
                uint rectH = (uint)(ctx.RenderResolution.y * ctx.ResolutionScale + 0.5f);
                natCmd.DispatchRays(shader, "MainRayGenShader", rectW, rectH, 1);

                natCmd.EndSample(marker);
            }
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            string passName = _useCompute ? "GISpatialResampling_Compute" : "GISpatialResampling";
            using var builder = renderGraph.AddUnsafePass<PassData>(passName, out var passData);

            passData.RtShader = _rtShader;
            passData.ComputeShader = _computeShader;
            passData.Context = _context;
            passData.UseCompute = _useCompute;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}
