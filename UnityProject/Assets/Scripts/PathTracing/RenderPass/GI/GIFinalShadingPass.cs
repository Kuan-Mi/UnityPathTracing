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
    public class GIFinalShadingPass : ScriptableRenderPass
    {
        private const int GroupSize = 8;

        private readonly RayTracingShader _rtShader;
        private readonly ComputeShader _computeShader;
        private RtxdiPassContext _context;
        private bool _useCompute;
        private bool _shading;

        public GIFinalShadingPass(RayTracingShader rtShader, ComputeShader computeShader)
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
                var marker = RenderPassMarkers.GiFinalShadingCompute;
                natCmd.BeginSample(marker);

                var cs = data.ComputeShader;
                int kernel = cs.FindKernel("main");

                natCmd.SetComputeConstantBufferParam(cs, paramsID, ctx.ConstantBuffer, 0, ctx.ConstantBuffer.stride);
                natCmd.SetComputeConstantBufferParam(cs, g_ConstID, ctx.ResamplingConstantBuffer, 0, ctx.ResamplingConstantBuffer.stride);

                natCmd.SetComputeBufferParam(cs, kernel, u_GIReservoirsID, ctx.RtxdiResources.GIReservoirBuffer);
                natCmd.SetComputeBufferParam(cs, kernel, u_SecondaryGBufferID, ctx.RtxdiResources.SecondaryGBuffer);

                natCmd.SetComputeTextureParam(cs, kernel, t_GBufferDepthID, ctx.ViewDepth);
                natCmd.SetComputeTextureParam(cs, kernel, t_GBufferDiffuseAlbedoID, ctx.DiffuseAlbedo);
                natCmd.SetComputeTextureParam(cs, kernel, t_GBufferSpecularRoughID, ctx.SpecularRough);
                natCmd.SetComputeTextureParam(cs, kernel, t_GBufferNormalsID, ctx.Normals);
                natCmd.SetComputeTextureParam(cs, kernel, t_GBufferGeoNormalsID, ctx.GeoNormals);

                natCmd.SetComputeTextureParam(cs, kernel, g_DirectLightingID, ctx.DirectLighting);

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
                var marker = RenderPassMarkers.GiFinalShading;
                natCmd.BeginSample(marker);

                var shader = data.RtShader;

                natCmd.SetRayTracingShaderPass(shader, "RTXDI");
                natCmd.SetRayTracingConstantBufferParam(shader, paramsID, ctx.ConstantBuffer, 0, ctx.ConstantBuffer.stride);
                natCmd.SetRayTracingBufferParam(shader, ResampleConstantsID, ctx.ResamplingConstantBuffer);

                natCmd.SetRayTracingBufferParam(shader, u_GIReservoirsID, ctx.RtxdiResources.GIReservoirBuffer);
                natCmd.SetRayTracingBufferParam(shader, u_SecondaryGBufferID, ctx.RtxdiResources.SecondaryGBuffer);

                natCmd.SetRayTracingTextureParam(shader, t_GBufferDepthID, ctx.ViewDepth);
                natCmd.SetRayTracingTextureParam(shader, t_GBufferDiffuseAlbedoID, ctx.DiffuseAlbedo);
                natCmd.SetRayTracingTextureParam(shader, t_GBufferSpecularRoughID, ctx.SpecularRough);
                natCmd.SetRayTracingTextureParam(shader, t_GBufferNormalsID, ctx.Normals);
                natCmd.SetRayTracingTextureParam(shader, t_GBufferGeoNormalsID, ctx.GeoNormals);

                natCmd.SetRayTracingTextureParam(shader, g_DirectLightingID, ctx.DirectLighting);

                if (data.Shading)
                {
                    uint rectW = (uint)(ctx.RenderResolution.x * ctx.ResolutionScale + 0.5f);
                    uint rectH = (uint)(ctx.RenderResolution.y * ctx.ResolutionScale + 0.5f);
                    natCmd.DispatchRays(shader, "MainRayGenShader", rectW, rectH, 1);
                }

                natCmd.EndSample(marker);
            }
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            string passName = _useCompute ? "GIFinalShading_Compute" : "GIFinalShading";
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
