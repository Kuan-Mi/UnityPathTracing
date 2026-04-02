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
    public class TemporalResamplingPass : ScriptableRenderPass
    {
        private const int GroupSize = 8;

        private readonly RayTracingShader _rtShader;
        private readonly ComputeShader _computeShader;
        private RtxdiPassContext _context;
        private bool _useCompute;

        public TemporalResamplingPass(RayTracingShader rtShader, ComputeShader computeShader)
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
                var marker = RenderPassMarkers.DiTemporalResamplingCompute;
                natCmd.BeginSample(marker);

                var cs = data.ComputeShader;
                int kernel = cs.FindKernel("main");

                natCmd.SetComputeConstantBufferParam(cs, paramsID, ctx.ConstantBuffer, 0, ctx.ConstantBuffer.stride);
                natCmd.SetComputeConstantBufferParam(cs, g_ConstID, ctx.ResamplingConstantBuffer, 0, ctx.ResamplingConstantBuffer.stride);

                natCmd.SetComputeBufferParam(cs, kernel, t_LightDataBufferID, ctx.RtxdiResources.LightDataBuffer);
                natCmd.SetComputeBufferParam(cs, kernel, t_NeighborOffsetsID, ctx.RtxdiResources.NeighborOffsetsBuffer);
                natCmd.SetComputeBufferParam(cs, kernel, u_LightReservoirsID, ctx.RtxdiResources.LightReservoirBuffer);

                natCmd.SetComputeTextureParam(cs, kernel, g_MvID, ctx.MotionVectors);
                natCmd.SetComputeTextureParam(cs, kernel, g_DirectLightingID, ctx.DirectLighting);

                natCmd.SetComputeTextureParam(cs, kernel, t_GBufferDepthID, ctx.ViewDepth);
                natCmd.SetComputeTextureParam(cs, kernel, t_GBufferDiffuseAlbedoID, ctx.DiffuseAlbedo);
                natCmd.SetComputeTextureParam(cs, kernel, t_GBufferSpecularRoughID, ctx.SpecularRough);
                natCmd.SetComputeTextureParam(cs, kernel, t_GBufferNormalsID, ctx.Normals);
                natCmd.SetComputeTextureParam(cs, kernel, t_GBufferGeoNormalsID, ctx.GeoNormals);

                natCmd.SetComputeTextureParam(cs, kernel, t_PrevGBufferDepthID, ctx.PrevViewDepth);
                natCmd.SetComputeTextureParam(cs, kernel, t_PrevGBufferDiffuseAlbedoID, ctx.PrevDiffuseAlbedo);
                natCmd.SetComputeTextureParam(cs, kernel, t_PrevGBufferSpecularRoughID, ctx.PrevSpecularRough);
                natCmd.SetComputeTextureParam(cs, kernel, t_PrevGBufferNormalsID, ctx.PrevNormals);
                natCmd.SetComputeTextureParam(cs, kernel, t_PrevGBufferGeoNormalsID, ctx.PrevGeoNormals);
                natCmd.SetComputeTextureParam(cs, kernel, t_MotionVectorsID, ctx.MotionVectors);

                int rectW = (int)(ctx.RenderResolution.x * ctx.ResolutionScale + 0.5f);
                int rectH = (int)(ctx.RenderResolution.y * ctx.ResolutionScale + 0.5f);
                int groupsX = (rectW + GroupSize - 1) / GroupSize;
                int groupsY = (rectH + GroupSize - 1) / GroupSize;
                natCmd.DispatchCompute(cs, kernel, groupsX, groupsY, 1);

                natCmd.EndSample(marker);
            }
            else
            {
                var marker = RenderPassMarkers.DiTemporalResampling;
                natCmd.BeginSample(marker);

                natCmd.SetRayTracingShaderPass(data.RtShader, "RTXDI");
                natCmd.SetRayTracingConstantBufferParam(data.RtShader, paramsID, ctx.ConstantBuffer, 0, ctx.ConstantBuffer.stride);
                natCmd.SetRayTracingBufferParam(data.RtShader, ResampleConstantsID, ctx.ResamplingConstantBuffer);

                natCmd.SetRayTracingBufferParam(data.RtShader, t_LightDataBufferID, ctx.RtxdiResources.LightDataBuffer);
                natCmd.SetRayTracingBufferParam(data.RtShader, t_NeighborOffsetsID, ctx.RtxdiResources.NeighborOffsetsBuffer);
                natCmd.SetRayTracingBufferParam(data.RtShader, u_LightReservoirsID, ctx.RtxdiResources.LightReservoirBuffer);

                natCmd.SetRayTracingTextureParam(data.RtShader, g_MvID, ctx.MotionVectors);
                natCmd.SetRayTracingTextureParam(data.RtShader, g_DirectLightingID, ctx.DirectLighting);

                natCmd.SetRayTracingTextureParam(data.RtShader, t_GBufferDepthID, ctx.ViewDepth);
                natCmd.SetRayTracingTextureParam(data.RtShader, t_GBufferDiffuseAlbedoID, ctx.DiffuseAlbedo);
                natCmd.SetRayTracingTextureParam(data.RtShader, t_GBufferSpecularRoughID, ctx.SpecularRough);
                natCmd.SetRayTracingTextureParam(data.RtShader, t_GBufferNormalsID, ctx.Normals);
                natCmd.SetRayTracingTextureParam(data.RtShader, t_GBufferGeoNormalsID, ctx.GeoNormals);

                natCmd.SetRayTracingTextureParam(data.RtShader, t_PrevGBufferDepthID, ctx.PrevViewDepth);
                natCmd.SetRayTracingTextureParam(data.RtShader, t_PrevGBufferDiffuseAlbedoID, ctx.PrevDiffuseAlbedo);
                natCmd.SetRayTracingTextureParam(data.RtShader, t_PrevGBufferSpecularRoughID, ctx.PrevSpecularRough);
                natCmd.SetRayTracingTextureParam(data.RtShader, t_PrevGBufferNormalsID, ctx.PrevNormals);
                natCmd.SetRayTracingTextureParam(data.RtShader, t_PrevGBufferGeoNormalsID, ctx.PrevGeoNormals);
                natCmd.SetRayTracingTextureParam(data.RtShader, t_MotionVectorsID, ctx.MotionVectors);

                uint rectWmod = (uint)(ctx.RenderResolution.x * ctx.ResolutionScale + 0.5f);
                uint rectHmod = (uint)(ctx.RenderResolution.y * ctx.ResolutionScale + 0.5f);
                natCmd.DispatchRays(data.RtShader, "MainRayGenShader", rectWmod, rectHmod, 1);

                natCmd.EndSample(marker);
            }
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            string passName = _useCompute ? "TemporalResampling_Compute" : "TemporalResampling";
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