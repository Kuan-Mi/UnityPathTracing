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
    public class TemporalResamplingComputePass : ScriptableRenderPass
    {
        private const int GroupSize = 8; // RTXDI_SCREEN_SPACE_GROUP_SIZE

        private readonly ComputeShader _computeShader;
        private Resource _resource;
        private Settings _settings;

        public TemporalResamplingComputePass(ComputeShader computeShader)
        {
            _computeShader = computeShader;
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

            internal RTHandle Mv;
            internal RTHandle DirectLighting;

            internal RTHandle ViewDepth;
            internal RTHandle DiffuseAlbedo;
            internal RTHandle SpecularRough;
            internal RTHandle Normals;
            internal RTHandle GeoNormals;

            internal RTHandle PrevViewDepth;
            internal RTHandle PrevDiffuseAlbedo;
            internal RTHandle PrevSpecularRough;
            internal RTHandle PrevNormals;
            internal RTHandle PrevGeoNormals;

            internal RtxdiResources RtxdiResources;
        }

        public class Settings
        {
            internal int2 m_RenderResolution;
            internal float resolutionScale;
        }

        class PassData
        {
            internal ComputeShader ComputeShader;
            internal Resource Resource;
            internal Settings Settings;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            var marker = new ProfilerMarker(ProfilerCategory.Render, "TemporalResampling_Compute", MarkerFlags.SampleGPU);
            natCmd.BeginSample(marker);

            var cs = data.ComputeShader;
            int kernel = cs.FindKernel("main");
            var resource = data.Resource;
            var settings = data.Settings;

            natCmd.SetComputeConstantBufferParam(cs, paramsID, resource.ConstantBuffer, 0, resource.ConstantBuffer.stride);
            natCmd.SetComputeConstantBufferParam(cs, "g_Const", resource.ResamplingConstantBuffer, 0, resource.ResamplingConstantBuffer.stride);

            natCmd.SetComputeBufferParam(cs, kernel, t_LightDataBufferID, resource.RtxdiResources.LightDataBuffer);
            natCmd.SetComputeBufferParam(cs, kernel, t_NeighborOffsetsID, resource.RtxdiResources.NeighborOffsetsBuffer);
            natCmd.SetComputeBufferParam(cs, kernel, u_LightReservoirsID, resource.RtxdiResources.LightReservoirBuffer);

            natCmd.SetComputeTextureParam(cs, kernel, g_MvID, resource.Mv);
            natCmd.SetComputeTextureParam(cs, kernel, g_DirectLightingID, resource.DirectLighting);

            natCmd.SetComputeTextureParam(cs, kernel, "t_GBufferDepth", resource.ViewDepth);
            natCmd.SetComputeTextureParam(cs, kernel, "t_GBufferDiffuseAlbedo", resource.DiffuseAlbedo);
            natCmd.SetComputeTextureParam(cs, kernel, "t_GBufferSpecularRough", resource.SpecularRough);
            natCmd.SetComputeTextureParam(cs, kernel, "t_GBufferNormals", resource.Normals);
            natCmd.SetComputeTextureParam(cs, kernel, "t_GBufferGeoNormals", resource.GeoNormals);

            natCmd.SetComputeTextureParam(cs, kernel, "t_PrevGBufferDepth", resource.PrevViewDepth);
            natCmd.SetComputeTextureParam(cs, kernel, "t_PrevGBufferDiffuseAlbedo", resource.PrevDiffuseAlbedo);
            natCmd.SetComputeTextureParam(cs, kernel, "t_PrevGBufferSpecularRough", resource.PrevSpecularRough);
            natCmd.SetComputeTextureParam(cs, kernel, "t_PrevGBufferNormals", resource.PrevNormals);
            natCmd.SetComputeTextureParam(cs, kernel, "t_PrevGBufferGeoNormals", resource.PrevGeoNormals);

            natCmd.SetComputeTextureParam(cs, kernel, "t_MotionVectors", resource.Mv);

            int rectW = (int)(settings.m_RenderResolution.x * settings.resolutionScale + 0.5f);
            int rectH = (int)(settings.m_RenderResolution.y * settings.resolutionScale + 0.5f);

            int groupsX = (rectW + GroupSize - 1) / GroupSize;
            int groupsY = (rectH + GroupSize - 1) / GroupSize;

            natCmd.DispatchCompute(cs, kernel, groupsX, groupsY, 1);

            natCmd.EndSample(marker);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("TemporalResampling_Compute", out var passData);

            passData.ComputeShader = _computeShader;
            passData.Resource = _resource;
            passData.Settings = _settings;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}
