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
    public class SpatialResamplingPass: ScriptableRenderPass
    {
        private readonly RayTracingShader _opaqueTs;
        private Resource _resource;
        private Settings _settings;


        public SpatialResamplingPass(RayTracingShader opaqueTs)
        {
            _opaqueTs = opaqueTs;
        }

        public void Setup(Resource sharcResource, Settings sharcSettings)
        {
            _resource = sharcResource;
            _settings = sharcSettings;
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
            internal RayTracingShader OpaqueTs;
            internal Resource Resource;
            internal Settings Settings;

        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            var opaqueTracingMarker = new ProfilerMarker(ProfilerCategory.Render, "SpatialResampling", MarkerFlags.SampleGPU);

            natCmd.BeginSample(opaqueTracingMarker);

            var resource = data.Resource;
            var settings = data.Settings;

            natCmd.SetRayTracingShaderPass(data.OpaqueTs, "RTXDI");
            natCmd.SetRayTracingConstantBufferParam(data.OpaqueTs, paramsID, resource.ConstantBuffer, 0, resource.ConstantBuffer.stride);
            natCmd.SetRayTracingBufferParam(data.OpaqueTs, "ResampleConstants", resource.ResamplingConstantBuffer);

            natCmd.SetRayTracingBufferParam(data.OpaqueTs, t_LightDataBufferID, resource.RtxdiResources.LightDataBuffer);
            natCmd.SetRayTracingBufferParam(data.OpaqueTs, t_NeighborOffsetsID, resource.RtxdiResources.NeighborOffsetsBuffer);
            natCmd.SetRayTracingBufferParam(data.OpaqueTs, u_LightReservoirsID, resource.RtxdiResources.LightReservoirBuffer);

            natCmd.SetRayTracingTextureParam(data.OpaqueTs, "t_GBufferDepth", resource.ViewDepth);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, "t_GBufferDiffuseAlbedo", resource.DiffuseAlbedo);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, "t_GBufferSpecularRough", resource.SpecularRough);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, "t_GBufferNormals", resource.Normals);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, "t_GBufferGeoNormals", resource.GeoNormals);


            uint rectWmod = (uint)(settings.m_RenderResolution.x * settings.resolutionScale + 0.5f);
            uint rectHmod = (uint)(settings.m_RenderResolution.y * settings.resolutionScale + 0.5f);

            // Debug.Log($"Dispatch Rays Size: {rectWmod} x {rectHmod}");


            natCmd.DispatchRays(data.OpaqueTs, "MainRayGenShader", rectWmod, rectHmod, 1);

            natCmd.EndSample(opaqueTracingMarker);

        }


        private TextureHandle CreateTex(TextureDesc textureDesc, RenderGraph renderGraph, string name, GraphicsFormat format)
        {
            textureDesc.format = format;
            textureDesc.name = name;
            return renderGraph.CreateTexture(textureDesc);
        }


        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("SpatialResampling", out var passData);

            passData.OpaqueTs = _opaqueTs;

            passData.Resource = _resource;
            passData.Settings = _settings;

            var resourceData = frameData.Get<UniversalResourceData>();

            var textureDesc = resourceData.activeColorTexture.GetDescriptor(renderGraph);
            textureDesc.enableRandomWrite = true;
            textureDesc.depthBufferBits = 0;
            textureDesc.clearBuffer = false;
            textureDesc.discardBuffer = false;
            textureDesc.width = _settings.m_RenderResolution.x;
            textureDesc.height = _settings.m_RenderResolution.y;

            

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}