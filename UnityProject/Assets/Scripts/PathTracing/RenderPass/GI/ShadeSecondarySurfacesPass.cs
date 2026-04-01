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
        private Resource _resource;
        private Settings _settings;


        public ShadeSecondarySurfacesPass(RayTracingShader gBufferTs)
        {
            _gBufferTs = gBufferTs;
        }

        public void Setup(Resource sharcResource, Settings sharcSettings)
        {
            _resource = sharcResource;
            _settings = sharcSettings;
        }

        public class Resource
        {
            internal GraphicsBuffer ConstantBuffer;

            internal RTHandle ViewDepth;
            internal RTHandle DiffuseAlbedo;
            internal RTHandle SpecularRough;
            internal RTHandle Normals;
            internal RTHandle GeoNormals;

            internal RTHandle DirectLighting;

            internal GraphicsBuffer ResamplingConstantBuffer;
            internal RtxdiResources RtxdiResources;
        }

        public class Settings
        {
            internal int2 m_RenderResolution;
            internal float resolutionScale;
            internal int convergenceStep;
        }

        class PassData
        {
            internal RayTracingShader gBufferTs;
            internal Resource Resource;
            internal Settings Settings;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            var gBufferTracingMarker = new ProfilerMarker(ProfilerCategory.Render, "ShadeSecondarySurfaces", MarkerFlags.SampleGPU);

            natCmd.BeginSample(gBufferTracingMarker);

            var resource = data.Resource;
            var settings = data.Settings;

            natCmd.SetRayTracingShaderPass(data.gBufferTs, "RTXDI");
            natCmd.SetRayTracingConstantBufferParam(data.gBufferTs, paramsID, resource.ConstantBuffer, 0, resource.ConstantBuffer.stride);
            natCmd.SetRayTracingBufferParam(data.gBufferTs, "ResampleConstants", resource.ResamplingConstantBuffer);


            natCmd.SetRayTracingTextureParam(data.gBufferTs, "t_GBufferDepth", resource.ViewDepth);
            natCmd.SetRayTracingTextureParam(data.gBufferTs, "t_GBufferDiffuseAlbedo", resource.DiffuseAlbedo);
            natCmd.SetRayTracingTextureParam(data.gBufferTs, "t_GBufferSpecularRough", resource.SpecularRough);
            natCmd.SetRayTracingTextureParam(data.gBufferTs, "t_GBufferNormals", resource.Normals);
            natCmd.SetRayTracingTextureParam(data.gBufferTs, "t_GBufferGeoNormals", resource.GeoNormals);


            natCmd.SetRayTracingTextureParam(data.gBufferTs, g_DirectLightingID, resource.DirectLighting);

            natCmd.SetRayTracingBufferParam(data.gBufferTs, "u_SecondaryGBuffer", resource.RtxdiResources.SecondaryGBuffer);
            natCmd.SetRayTracingBufferParam(data.gBufferTs, "u_GIReservoirs", resource.RtxdiResources.GIReservoirBuffer);
            natCmd.SetRayTracingBufferParam(data.gBufferTs, "t_LightDataBuffer", resource.RtxdiResources.LightDataBuffer);
            natCmd.SetRayTracingBufferParam(data.gBufferTs, "u_RisBuffer", resource.RtxdiResources.RisBuffer);
            natCmd.SetRayTracingBufferParam(data.gBufferTs, "u_RisLightDataBuffer", resource.RtxdiResources.RisLightDataBuffer);
            
            natCmd.SetRayTracingBufferParam(data.gBufferTs, "t_NeighborOffsets", resource.RtxdiResources.NeighborOffsetsBuffer);
            natCmd.SetRayTracingBufferParam(data.gBufferTs, "u_LightReservoirs", resource.RtxdiResources.LightReservoirBuffer);

            uint rectWmod = (uint)(settings.m_RenderResolution.x * settings.resolutionScale + 0.5f);
            uint rectHmod = (uint)(settings.m_RenderResolution.y * settings.resolutionScale + 0.5f);

            // Debug.Log($"Dispatch Rays Size: {rectWmod} x {rectHmod}");


            natCmd.DispatchRays(data.gBufferTs, "MainRayGenShader", rectWmod, rectHmod, 1);

            natCmd.EndSample(gBufferTracingMarker);
        }


        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("ShadeSecondarySurfaces", out var passData);

            passData.gBufferTs = _gBufferTs;

            passData.Resource = _resource;
            passData.Settings = _settings;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}