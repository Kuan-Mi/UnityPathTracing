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
        private Resource _resource;
        private Settings _settings;

        public ShadeSamplesPass(RayTracingShader rtShader, ComputeShader computeShader)
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
            GraphicsBuffer geometryInstanceToLight,
            RTHandle viewDepth,
            RTHandle diffuseAlbedo,
            RTHandle specularRough,
            RTHandle normals,
            RTHandle geoNormals,
            RTHandle directLighting,
            RTHandle emissive,
            RtxdiResources rtxdiResources,
            int2 renderResolution,
            float resolutionScale,
            bool shading,
            bool useCompute)
        {
            _resource = new Resource
            {
                ConstantBuffer = constantBuffer,
                ResamplingConstantBuffer = resamplingConstantBuffer,
                t_GeometryInstanceToLight = geometryInstanceToLight,
                ViewDepth = viewDepth,
                DiffuseAlbedo = diffuseAlbedo,
                SpecularRough = specularRough,
                Normals = normals,
                GeoNormals = geoNormals,
                DirectLighting = directLighting,
                Emissive = emissive,
                RtxdiResources = rtxdiResources,
            };
            _settings = new Settings
            {
                m_RenderResolution = renderResolution,
                resolutionScale = resolutionScale,
                shading = shading,
                useCompute = useCompute,
            };
        }

        public class Resource
        {
            internal GraphicsBuffer ConstantBuffer;
            internal GraphicsBuffer ResamplingConstantBuffer;
            internal GraphicsBuffer t_GeometryInstanceToLight;

            internal RTHandle Emissive;

            internal RTHandle ViewDepth;
            internal RTHandle DiffuseAlbedo;
            internal RTHandle SpecularRough;
            internal RTHandle Normals;
            internal RTHandle GeoNormals;

            internal RTHandle DirectLighting;

            internal RtxdiResources RtxdiResources;
        }

        public class Settings
        {
            internal int2 m_RenderResolution;
            internal float resolutionScale;
            internal bool shading;
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
                var marker = new ProfilerMarker(ProfilerCategory.Render, "ShadeSamples_Compute", MarkerFlags.SampleGPU);
                natCmd.BeginSample(marker);

                var cs = data.ComputeShader;
                int kernel = cs.FindKernel("main");

                natCmd.SetComputeConstantBufferParam(cs, paramsID, resource.ConstantBuffer, 0, resource.ConstantBuffer.stride);
                natCmd.SetComputeConstantBufferParam(cs, "g_Const", resource.ResamplingConstantBuffer, 0, resource.ResamplingConstantBuffer.stride);
                natCmd.SetComputeBufferParam(cs, kernel, "t_GeometryInstanceToLight", resource.t_GeometryInstanceToLight);

                natCmd.SetComputeBufferParam(cs, kernel, t_LightDataBufferID, resource.RtxdiResources.LightDataBuffer);
                natCmd.SetComputeBufferParam(cs, kernel, t_NeighborOffsetsID, resource.RtxdiResources.NeighborOffsetsBuffer);
                natCmd.SetComputeBufferParam(cs, kernel, u_LightReservoirsID, resource.RtxdiResources.LightReservoirBuffer);
                natCmd.SetComputeBufferParam(cs, kernel, "u_RisBuffer", resource.RtxdiResources.RisBuffer);

                natCmd.SetComputeTextureParam(cs, kernel, g_DirectLightingID, resource.DirectLighting);

                natCmd.SetComputeTextureParam(cs, kernel, "t_GBufferDepth", resource.ViewDepth);
                natCmd.SetComputeTextureParam(cs, kernel, "t_GBufferDiffuseAlbedo", resource.DiffuseAlbedo);
                natCmd.SetComputeTextureParam(cs, kernel, "t_GBufferSpecularRough", resource.SpecularRough);
                natCmd.SetComputeTextureParam(cs, kernel, "t_GBufferNormals", resource.Normals);
                natCmd.SetComputeTextureParam(cs, kernel, "t_GBufferGeoNormals", resource.GeoNormals);

                natCmd.SetComputeTextureParam(cs, kernel, "gIn_EmissiveLighting", resource.Emissive);

                if (settings.shading)
                {
                    int rectW = (int)(settings.m_RenderResolution.x * settings.resolutionScale + 0.5f);
                    int rectH = (int)(settings.m_RenderResolution.y * settings.resolutionScale + 0.5f);
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
                natCmd.SetRayTracingConstantBufferParam(data.RtShader, paramsID, resource.ConstantBuffer, 0, resource.ConstantBuffer.stride);
                natCmd.SetRayTracingBufferParam(data.RtShader, "ResampleConstants", resource.ResamplingConstantBuffer);
                natCmd.SetRayTracingBufferParam(data.RtShader, "t_GeometryInstanceToLight", resource.t_GeometryInstanceToLight);

                natCmd.SetRayTracingBufferParam(data.RtShader, t_LightDataBufferID, resource.RtxdiResources.LightDataBuffer);
                natCmd.SetRayTracingBufferParam(data.RtShader, t_NeighborOffsetsID, resource.RtxdiResources.NeighborOffsetsBuffer);
                natCmd.SetRayTracingBufferParam(data.RtShader, u_LightReservoirsID, resource.RtxdiResources.LightReservoirBuffer);

                natCmd.SetRayTracingTextureParam(data.RtShader, g_DirectLightingID, resource.DirectLighting);

                natCmd.SetRayTracingTextureParam(data.RtShader, "t_GBufferDepth", resource.ViewDepth);
                natCmd.SetRayTracingTextureParam(data.RtShader, "t_GBufferDiffuseAlbedo", resource.DiffuseAlbedo);
                natCmd.SetRayTracingTextureParam(data.RtShader, "t_GBufferSpecularRough", resource.SpecularRough);
                natCmd.SetRayTracingTextureParam(data.RtShader, "t_GBufferNormals", resource.Normals);
                natCmd.SetRayTracingTextureParam(data.RtShader, "t_GBufferGeoNormals", resource.GeoNormals);

                natCmd.SetRayTracingTextureParam(data.RtShader, "gIn_EmissiveLighting", resource.Emissive);

                natCmd.SetRayTracingBufferParam(data.RtShader, "u_RisBuffer", resource.RtxdiResources.RisBuffer);

                if (settings.shading)
                {
                    uint rectWmod = (uint)(settings.m_RenderResolution.x * settings.resolutionScale + 0.5f);
                    uint rectHmod = (uint)(settings.m_RenderResolution.y * settings.resolutionScale + 0.5f);
                    natCmd.DispatchRays(data.RtShader, "MainRayGenShader", rectWmod, rectHmod, 1);
                }

                natCmd.EndSample(marker);
            }
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            string passName = _settings.useCompute ? "ShadeSamples_Compute" : "ShadeSamples";
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