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
        private Resource _resource;
        private Settings _settings;

        public GISpatialResamplingPass(RayTracingShader rtShader, ComputeShader computeShader)
        {
            _rtShader = rtShader;
            _computeShader = computeShader;
        }

        public void Setup(Resource resource, Settings settings)
        {
            _resource = resource;
            _settings = settings;
        }

        public void Setup(
            GraphicsBuffer constantBuffer,
            GraphicsBuffer resamplingConstantBuffer,
            RTHandle viewDepth,
            RTHandle diffuseAlbedo,
            RTHandle specularRough,
            RTHandle normals,
            RTHandle geoNormals,
            RtxdiResources rtxdiResources,
            int2 renderResolution,
            float resolutionScale,
            bool useCompute)
        {
            _resource = new Resource
            {
                ConstantBuffer = constantBuffer,
                ResamplingConstantBuffer = resamplingConstantBuffer,
                ViewDepth = viewDepth,
                DiffuseAlbedo = diffuseAlbedo,
                SpecularRough = specularRough,
                Normals = normals,
                GeoNormals = geoNormals,
                RtxdiResources = rtxdiResources,
            };
            _settings = new Settings
            {
                m_RenderResolution = renderResolution,
                resolutionScale = resolutionScale,
                useCompute = useCompute,
            };
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
            internal bool useCompute;
        }

        class PassData
        {
            internal RayTracingShader RtShader;
            internal ComputeShader ComputeShader;
            internal Resource Resource;
            internal Settings Settings;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var resource = data.Resource;
            var settings = data.Settings;

            if (settings.useCompute)
            {
                var marker = new ProfilerMarker(ProfilerCategory.Render, "GISpatialResampling_Compute", MarkerFlags.SampleGPU);
                natCmd.BeginSample(marker);

                var cs = data.ComputeShader;
                int kernel = cs.FindKernel("main");

                natCmd.SetComputeConstantBufferParam(cs, paramsID, resource.ConstantBuffer, 0, resource.ConstantBuffer.stride);
                natCmd.SetComputeConstantBufferParam(cs, "g_Const", resource.ResamplingConstantBuffer, 0, resource.ResamplingConstantBuffer.stride);

                natCmd.SetComputeBufferParam(cs, kernel, "u_GIReservoirs", resource.RtxdiResources.GIReservoirBuffer);
                natCmd.SetComputeBufferParam(cs, kernel, t_NeighborOffsetsID, resource.RtxdiResources.NeighborOffsetsBuffer);

                natCmd.SetComputeTextureParam(cs, kernel, "t_GBufferDepth", resource.ViewDepth);
                natCmd.SetComputeTextureParam(cs, kernel, "t_GBufferDiffuseAlbedo", resource.DiffuseAlbedo);
                natCmd.SetComputeTextureParam(cs, kernel, "t_GBufferSpecularRough", resource.SpecularRough);
                natCmd.SetComputeTextureParam(cs, kernel, "t_GBufferNormals", resource.Normals);
                natCmd.SetComputeTextureParam(cs, kernel, "t_GBufferGeoNormals", resource.GeoNormals);

                int rectW = (int)(settings.m_RenderResolution.x * settings.resolutionScale + 0.5f);
                int rectH = (int)(settings.m_RenderResolution.y * settings.resolutionScale + 0.5f);
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
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            string passName = _settings.useCompute ? "GISpatialResampling_Compute" : "GISpatialResampling";
            using var builder = renderGraph.AddUnsafePass<PassData>(passName, out var passData);

            passData.RtShader = _rtShader;
            passData.ComputeShader = _computeShader;
            passData.Resource = _resource;
            passData.Settings = _settings;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}
