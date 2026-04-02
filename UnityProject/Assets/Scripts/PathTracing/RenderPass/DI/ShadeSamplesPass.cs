using mini;
using Unity.Mathematics;
using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;
using static PathTracing.ShaderIDs;

namespace PathTracing
{
    public class ShadeSamplesPass : ScriptableRenderPass
    {
        private const int GroupSize = 8;

        private readonly RayTracingShader _rtShader;
        private readonly ComputeShader _computeShader;
        private RtxdiPassContext _context;
        private bool _useCompute;
        private bool _shading;

        public ShadeSamplesPass(RayTracingShader rtShader, ComputeShader computeShader)
        {
            _rtShader = rtShader;
            _computeShader = computeShader;
        }

        public void Setup(RtxdiPassContext ctx, bool shading, bool useCompute)
        {
            _context = ctx;
            _shading = shading;
            _useCompute = useCompute;
        }

        class PassData
        {
            internal RayTracingShader RtShader;
            internal ComputeShader ComputeShader;
            internal RtxdiPassContext Context;
            internal bool UseCompute;
            internal bool Shading;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var ctx = data.Context;

            if (data.UseCompute)
            {
                var marker = new ProfilerMarker(ProfilerCategory.Render, "ShadeSamples_Compute", MarkerFlags.SampleGPU);
                natCmd.BeginSample(marker);

                var cs = data.ComputeShader;
                int kernel = cs.FindKernel("main");

                natCmd.SetComputeConstantBufferParam(cs, paramsID, ctx.ConstantBuffer, 0, ctx.ConstantBuffer.stride);
                natCmd.SetComputeConstantBufferParam(cs, "g_Const", ctx.ResamplingConstantBuffer, 0, ctx.ResamplingConstantBuffer.stride);
                natCmd.SetComputeBufferParam(cs, kernel, "t_GeometryInstanceToLight", ctx.GeometryInstanceToLight);

                natCmd.SetComputeBufferParam(cs, kernel, t_LightDataBufferID, ctx.RtxdiResources.LightDataBuffer);
                natCmd.SetComputeBufferParam(cs, kernel, t_NeighborOffsetsID, ctx.RtxdiResources.NeighborOffsetsBuffer);
                natCmd.SetComputeBufferParam(cs, kernel, u_LightReservoirsID, ctx.RtxdiResources.LightReservoirBuffer);
                natCmd.SetComputeBufferParam(cs, kernel, "u_RisBuffer", ctx.RtxdiResources.RisBuffer);

                natCmd.SetComputeTextureParam(cs, kernel, g_DirectLightingID, ctx.DirectLighting);

                natCmd.SetComputeTextureParam(cs, kernel, "t_GBufferDepth", ctx.ViewDepth);
                natCmd.SetComputeTextureParam(cs, kernel, "t_GBufferDiffuseAlbedo", ctx.DiffuseAlbedo);
                natCmd.SetComputeTextureParam(cs, kernel, "t_GBufferSpecularRough", ctx.SpecularRough);
                natCmd.SetComputeTextureParam(cs, kernel, "t_GBufferNormals", ctx.Normals);
                natCmd.SetComputeTextureParam(cs, kernel, "t_GBufferGeoNormals", ctx.GeoNormals);

                natCmd.SetComputeTextureParam(cs, kernel, "gIn_EmissiveLighting", ctx.Emissive);

                if (data.Shading)
                {
                    int rectW = (int)(ctx.RenderResolution.x * ctx.ResolutionScale + 0.5f);
                    int rectH = (int)(ctx.RenderResolution.y * ctx.ResolutionScale + 0.5f);
                    int groupsX = (rectW + GroupSize - 1) / GroupSize;
                    int groupsY = (rectH + GroupSize - 1) / GroupSize;
                    natCmd.DispatchCompute(cs, kernel, groupsX, groupsY, 1);
                }

                natCmd.EndSample(marker);
            }
            else
            {
                var marker = new ProfilerMarker(ProfilerCategory.Render, "ShadeSamples", MarkerFlags.SampleGPU);
                natCmd.BeginSample(marker);

                natCmd.SetRayTracingShaderPass(data.RtShader, "RTXDI");
                natCmd.SetRayTracingConstantBufferParam(data.RtShader, paramsID, ctx.ConstantBuffer, 0, ctx.ConstantBuffer.stride);
                natCmd.SetRayTracingBufferParam(data.RtShader, "ResampleConstants", ctx.ResamplingConstantBuffer);
                natCmd.SetRayTracingBufferParam(data.RtShader, "t_GeometryInstanceToLight", ctx.GeometryInstanceToLight);

                natCmd.SetRayTracingBufferParam(data.RtShader, t_LightDataBufferID, ctx.RtxdiResources.LightDataBuffer);
                natCmd.SetRayTracingBufferParam(data.RtShader, t_NeighborOffsetsID, ctx.RtxdiResources.NeighborOffsetsBuffer);
                natCmd.SetRayTracingBufferParam(data.RtShader, u_LightReservoirsID, ctx.RtxdiResources.LightReservoirBuffer);

                natCmd.SetRayTracingTextureParam(data.RtShader, g_DirectLightingID, ctx.DirectLighting);

                natCmd.SetRayTracingTextureParam(data.RtShader, "t_GBufferDepth", ctx.ViewDepth);
                natCmd.SetRayTracingTextureParam(data.RtShader, "t_GBufferDiffuseAlbedo", ctx.DiffuseAlbedo);
                natCmd.SetRayTracingTextureParam(data.RtShader, "t_GBufferSpecularRough", ctx.SpecularRough);
                natCmd.SetRayTracingTextureParam(data.RtShader, "t_GBufferNormals", ctx.Normals);
                natCmd.SetRayTracingTextureParam(data.RtShader, "t_GBufferGeoNormals", ctx.GeoNormals);

                natCmd.SetRayTracingTextureParam(data.RtShader, "gIn_EmissiveLighting", ctx.Emissive);

                natCmd.SetRayTracingBufferParam(data.RtShader, "u_RisBuffer", ctx.RtxdiResources.RisBuffer);

                if (data.Shading)
                {
                    uint rectWmod = (uint)(ctx.RenderResolution.x * ctx.ResolutionScale + 0.5f);
                    uint rectHmod = (uint)(ctx.RenderResolution.y * ctx.ResolutionScale + 0.5f);
                    natCmd.DispatchRays(data.RtShader, "MainRayGenShader", rectWmod, rectHmod, 1);
                }

                natCmd.EndSample(marker);
            }
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            string passName = _useCompute ? "ShadeSamples_Compute" : "ShadeSamples";
            using var builder = renderGraph.AddUnsafePass<PassData>(passName, out var passData);

            passData.RtShader = _rtShader;
            passData.ComputeShader = _computeShader;
            passData.Context = _context;
            passData.UseCompute = _useCompute;
            passData.Shading = _shading;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}