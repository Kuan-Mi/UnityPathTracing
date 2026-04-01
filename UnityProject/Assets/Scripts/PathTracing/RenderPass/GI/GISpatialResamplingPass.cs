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
        private readonly RayTracingShader _shader;
        private Resource _resource;
        private Settings _settings;

        public GISpatialResamplingPass(RayTracingShader shader)
        {
            _shader = shader;
        }

        public void Setup(Resource resource, Settings settings)
        {
            _resource = resource;
            _settings = settings;
        }

        public class Resource
        {
            internal GraphicsBuffer ConstantBuffer;
            internal GraphicsBuffer ResamplingConstantBuffer;

            internal RTHandle ViewDepth;
            internal RTHandle DiffuseAlbedo;
            internal RTHandle SpecularRough;
            internal RTHandle Normals;
            internal RTHandle GeoNormals;

            internal RtxdiResources RtxdiResources;
        }

        public class Settings
        {
            internal int2 m_RenderResolution;
            internal float resolutionScale;
        }

        class PassData
        {
            internal RayTracingShader Shader;
            internal Resource Resource;
            internal Settings Settings;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var marker = new ProfilerMarker(ProfilerCategory.Render, "GISpatialResampling", MarkerFlags.SampleGPU);
            natCmd.BeginSample(marker);

            var shader = data.Shader;
            var resource = data.Resource;
            var settings = data.Settings;

            natCmd.SetRayTracingShaderPass(shader, "RTXDI");
            natCmd.SetRayTracingConstantBufferParam(shader, paramsID, resource.ConstantBuffer, 0, resource.ConstantBuffer.stride);
            natCmd.SetRayTracingBufferParam(shader, "ResampleConstants", resource.ResamplingConstantBuffer);

            natCmd.SetRayTracingBufferParam(shader, "u_GIReservoirs", resource.RtxdiResources.GIReservoirBuffer);
            natCmd.SetRayTracingBufferParam(shader, t_NeighborOffsetsID, resource.RtxdiResources.NeighborOffsetsBuffer);

            natCmd.SetRayTracingTextureParam(shader, "t_GBufferDepth", resource.ViewDepth);
            natCmd.SetRayTracingTextureParam(shader, "t_GBufferDiffuseAlbedo", resource.DiffuseAlbedo);
            natCmd.SetRayTracingTextureParam(shader, "t_GBufferSpecularRough", resource.SpecularRough);
            natCmd.SetRayTracingTextureParam(shader, "t_GBufferNormals", resource.Normals);
            natCmd.SetRayTracingTextureParam(shader, "t_GBufferGeoNormals", resource.GeoNormals);

            uint rectW = (uint)(settings.m_RenderResolution.x * settings.resolutionScale + 0.5f);
            uint rectH = (uint)(settings.m_RenderResolution.y * settings.resolutionScale + 0.5f);

            natCmd.DispatchRays(shader, "MainRayGenShader", rectW, rectH, 1);

            natCmd.EndSample(marker);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("GISpatialResampling", out var passData);

            passData.Shader = _shader;
            passData.Resource = _resource;
            passData.Settings = _settings;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}
