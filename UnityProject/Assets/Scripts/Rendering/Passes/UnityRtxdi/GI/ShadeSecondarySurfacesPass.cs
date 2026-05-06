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
    public class ShadeSecondarySurfacesPass : ScriptableRenderPass
    {
        private const int GroupSize = 8;

        private readonly RayTracingShader _gBufferTs;
        private readonly ComputeShader _computeShader;
        private RtxdiPassContext _context;
        private bool _useCompute;

        public ShadeSecondarySurfacesPass(RayTracingShader gBufferTs, ComputeShader computeShader)
        {
            _gBufferTs = gBufferTs;
            _computeShader = computeShader;
        }

        public void Setup(RtxdiPassContext ctx, bool useCompute = false)
        {
            _context = ctx;
            _useCompute = useCompute;
        }

        class PassData
        {
            internal RayTracingShader gBufferTs;
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
                var marker = RenderPassMarkers.ShadeSecondarySurfacesCompute;
                natCmd.BeginSample(marker);

                var cs = data.ComputeShader;
                int kernel = cs.FindKernel("main");

                natCmd.SetComputeConstantBufferParam(cs, paramsID, ctx.ConstantBuffer, 0, ctx.ConstantBuffer.stride);
                natCmd.SetComputeConstantBufferParam(cs, g_ConstID, ctx.ResamplingConstantBuffer, 0, ctx.ResamplingConstantBuffer.stride);

                natCmd.SetComputeTextureParam(cs, kernel, t_GBufferDepthID, ctx.ViewDepth);
                natCmd.SetComputeTextureParam(cs, kernel, t_GBufferDiffuseAlbedoID, ctx.DiffuseAlbedo);
                natCmd.SetComputeTextureParam(cs, kernel, t_GBufferSpecularRoughID, ctx.SpecularRough);
                natCmd.SetComputeTextureParam(cs, kernel, t_GBufferNormalsID, ctx.Normals);
                natCmd.SetComputeTextureParam(cs, kernel, t_GBufferGeoNormalsID, ctx.GeoNormals);

                natCmd.SetComputeTextureParam(cs, kernel, g_DirectLightingID, ctx.DirectLighting);
                natCmd.SetComputeTextureParam(cs, kernel, t_LocalLightPdfTextureID, ctx.LocalLightPdfTexture);

                natCmd.SetComputeBufferParam(cs, kernel, u_SecondaryGBufferID, ctx.RtxdiResources.SecondaryGBuffer);
                natCmd.SetComputeBufferParam(cs, kernel, u_GIReservoirsID, ctx.RtxdiResources.GIReservoirBuffer);
                natCmd.SetComputeBufferParam(cs, kernel, t_LightDataBufferID, ctx.RtxdiResources.LightDataBuffer);
                natCmd.SetComputeBufferParam(cs, kernel, u_RisBufferID, ctx.RtxdiResources.RisBuffer);
                natCmd.SetComputeBufferParam(cs, kernel, u_RisLightDataBufferID, ctx.RtxdiResources.RisLightDataBuffer);

                natCmd.SetComputeBufferParam(cs, kernel, t_NeighborOffsetsID, ctx.RtxdiResources.NeighborOffsetsBuffer);
                natCmd.SetComputeBufferParam(cs, kernel, u_LightReservoirsID, ctx.RtxdiResources.LightReservoirBuffer);
                natCmd.SetComputeBufferParam(cs, kernel, t_GeometryInstanceToLightID, ctx.GeometryInstanceToLight);

                int rectW = (int)(ctx.RenderResolution.x * ctx.ResolutionScale + 0.5f);
                int rectH = (int)(ctx.RenderResolution.y * ctx.ResolutionScale + 0.5f);
                int groupsX = (rectW + GroupSize - 1) / GroupSize;
                int groupsY = (rectH + GroupSize - 1) / GroupSize;
                natCmd.DispatchCompute(cs, kernel, groupsX, groupsY, 1);

                natCmd.EndSample(marker);
            }
            else
            {
                var marker = RenderPassMarkers.ShadeSecondarySurfaces;
                natCmd.BeginSample(marker);

                natCmd.SetRayTracingShaderPass(data.gBufferTs, "RTXDI");
                natCmd.SetRayTracingConstantBufferParam(data.gBufferTs, paramsID, ctx.ConstantBuffer, 0, ctx.ConstantBuffer.stride);
                natCmd.SetRayTracingBufferParam(data.gBufferTs, ResampleConstantsID, ctx.ResamplingConstantBuffer);

                natCmd.SetRayTracingTextureParam(data.gBufferTs, t_GBufferDepthID, ctx.ViewDepth);
                natCmd.SetRayTracingTextureParam(data.gBufferTs, t_GBufferDiffuseAlbedoID, ctx.DiffuseAlbedo);
                natCmd.SetRayTracingTextureParam(data.gBufferTs, t_GBufferSpecularRoughID, ctx.SpecularRough);
                natCmd.SetRayTracingTextureParam(data.gBufferTs, t_GBufferNormalsID, ctx.Normals);
                natCmd.SetRayTracingTextureParam(data.gBufferTs, t_GBufferGeoNormalsID, ctx.GeoNormals);

                natCmd.SetRayTracingTextureParam(data.gBufferTs, g_DirectLightingID, ctx.DirectLighting);
                natCmd.SetRayTracingTextureParam(data.gBufferTs, t_LocalLightPdfTextureID, ctx.LocalLightPdfTexture);

                natCmd.SetRayTracingBufferParam(data.gBufferTs, u_SecondaryGBufferID, ctx.RtxdiResources.SecondaryGBuffer);
                natCmd.SetRayTracingBufferParam(data.gBufferTs, u_GIReservoirsID, ctx.RtxdiResources.GIReservoirBuffer);
                natCmd.SetRayTracingBufferParam(data.gBufferTs, t_LightDataBufferID, ctx.RtxdiResources.LightDataBuffer);
                natCmd.SetRayTracingBufferParam(data.gBufferTs, u_RisBufferID, ctx.RtxdiResources.RisBuffer);
                natCmd.SetRayTracingBufferParam(data.gBufferTs, u_RisLightDataBufferID, ctx.RtxdiResources.RisLightDataBuffer);

                natCmd.SetRayTracingBufferParam(data.gBufferTs, t_NeighborOffsetsID, ctx.RtxdiResources.NeighborOffsetsBuffer);
                natCmd.SetRayTracingBufferParam(data.gBufferTs, u_LightReservoirsID, ctx.RtxdiResources.LightReservoirBuffer);
                natCmd.SetRayTracingBufferParam(data.gBufferTs, t_GeometryInstanceToLightID, ctx.GeometryInstanceToLight);

                uint rectWmod = (uint)(ctx.RenderResolution.x * ctx.ResolutionScale + 0.5f);
                uint rectHmod = (uint)(ctx.RenderResolution.y * ctx.ResolutionScale + 0.5f);

                natCmd.DispatchRays(data.gBufferTs, "MainRayGenShader", rectWmod, rectHmod, 1);

                natCmd.EndSample(marker);
            }
        }


        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            string passName = _useCompute ? "ShadeSecondarySurfaces_Compute" : "ShadeSecondarySurfaces";
            using var builder = renderGraph.AddUnsafePass<PassData>(passName, out var passData);

            passData.gBufferTs = _gBufferTs;
            passData.ComputeShader = _computeShader;
            passData.Context = _context;
            passData.UseCompute = _useCompute;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}