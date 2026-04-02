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
        private readonly RayTracingShader _gBufferTs;
        private RtxdiPassContext _context;

        public ShadeSecondarySurfacesPass(RayTracingShader gBufferTs)
        {
            _gBufferTs = gBufferTs;
        }

        public void Setup(RtxdiPassContext ctx)
        {
            _context = ctx;
        }

        class PassData
        {
            internal RayTracingShader gBufferTs;
            internal RtxdiPassContext Context;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            var gBufferTracingMarker = new ProfilerMarker(ProfilerCategory.Render, "ShadeSecondarySurfaces", MarkerFlags.SampleGPU);

            natCmd.BeginSample(gBufferTracingMarker);

            var ctx = data.Context;

            natCmd.SetRayTracingShaderPass(data.gBufferTs, "RTXDI");
            natCmd.SetRayTracingConstantBufferParam(data.gBufferTs, paramsID, ctx.ConstantBuffer, 0, ctx.ConstantBuffer.stride);
            natCmd.SetRayTracingBufferParam(data.gBufferTs, ResampleConstantsID, ctx.ResamplingConstantBuffer);

            natCmd.SetRayTracingTextureParam(data.gBufferTs, t_GBufferDepthID, ctx.ViewDepth);
            natCmd.SetRayTracingTextureParam(data.gBufferTs, t_GBufferDiffuseAlbedoID, ctx.DiffuseAlbedo);
            natCmd.SetRayTracingTextureParam(data.gBufferTs, t_GBufferSpecularRoughID, ctx.SpecularRough);
            natCmd.SetRayTracingTextureParam(data.gBufferTs, t_GBufferNormalsID, ctx.Normals);
            natCmd.SetRayTracingTextureParam(data.gBufferTs, t_GBufferGeoNormalsID, ctx.GeoNormals);

            natCmd.SetRayTracingTextureParam(data.gBufferTs, g_DirectLightingID, ctx.DirectLighting);

            natCmd.SetRayTracingBufferParam(data.gBufferTs, u_SecondaryGBufferID, ctx.RtxdiResources.SecondaryGBuffer);
            natCmd.SetRayTracingBufferParam(data.gBufferTs, u_GIReservoirsID, ctx.RtxdiResources.GIReservoirBuffer);
            natCmd.SetRayTracingBufferParam(data.gBufferTs, t_LightDataBufferID, ctx.RtxdiResources.LightDataBuffer);
            natCmd.SetRayTracingBufferParam(data.gBufferTs, u_RisBufferID, ctx.RtxdiResources.RisBuffer);
            natCmd.SetRayTracingBufferParam(data.gBufferTs, u_RisLightDataBufferID, ctx.RtxdiResources.RisLightDataBuffer);

            natCmd.SetRayTracingBufferParam(data.gBufferTs, t_NeighborOffsetsID, ctx.RtxdiResources.NeighborOffsetsBuffer);
            natCmd.SetRayTracingBufferParam(data.gBufferTs, u_LightReservoirsID, ctx.RtxdiResources.LightReservoirBuffer);

            uint rectWmod = (uint)(ctx.RenderResolution.x * ctx.ResolutionScale + 0.5f);
            uint rectHmod = (uint)(ctx.RenderResolution.y * ctx.ResolutionScale + 0.5f);

            // Debug.Log($"Dispatch Rays Size: {rectWmod} x {rectHmod}");


            natCmd.DispatchRays(data.gBufferTs, "MainRayGenShader", rectWmod, rectHmod, 1);

            natCmd.EndSample(gBufferTracingMarker);
        }


        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("ShadeSecondarySurfaces", out var passData);

            passData.gBufferTs = _gBufferTs;
            passData.Context = _context;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}