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
    public class GBufferPass : ScriptableRenderPass
    {
        private readonly RayTracingShader _gBufferTs;
        private Resource _resource;
        private Settings _settings;


        public GBufferPass(RayTracingShader gBufferTs)
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
            internal RTHandle Emissive;
            internal RTHandle MotionVectors;
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

            var gBufferTracingMarker = new ProfilerMarker(ProfilerCategory.Render, "GBuffer", MarkerFlags.SampleGPU);

            natCmd.BeginSample(gBufferTracingMarker);

            var resource = data.Resource;
            var settings = data.Settings;

            natCmd.SetRayTracingShaderPass(data.gBufferTs, "Test2");
            natCmd.SetRayTracingConstantBufferParam(data.gBufferTs, paramsID, resource.ConstantBuffer, 0, resource.ConstantBuffer.stride);


            natCmd.SetRayTracingTextureParam(data.gBufferTs, "u_ViewDepth", resource.ViewDepth);
            natCmd.SetRayTracingTextureParam(data.gBufferTs, "u_DiffuseAlbedo", resource.DiffuseAlbedo);
            natCmd.SetRayTracingTextureParam(data.gBufferTs, "u_SpecularRough", resource.SpecularRough);
            natCmd.SetRayTracingTextureParam(data.gBufferTs, "u_Normals", resource.Normals);
            natCmd.SetRayTracingTextureParam(data.gBufferTs, "u_GeoNormals", resource.GeoNormals);
            natCmd.SetRayTracingTextureParam(data.gBufferTs, "u_Emissive", resource.Emissive);
            natCmd.SetRayTracingTextureParam(data.gBufferTs, "u_MotionVectors", resource.MotionVectors);

            uint rectWmod = (uint)(settings.m_RenderResolution.x * settings.resolutionScale + 0.5f);
            uint rectHmod = (uint)(settings.m_RenderResolution.y * settings.resolutionScale + 0.5f);

            // Debug.Log($"Dispatch Rays Size: {rectWmod} x {rectHmod}");


            natCmd.DispatchRays(data.gBufferTs, "MainRayGenShader", rectWmod, rectHmod, 1);

            natCmd.EndSample(gBufferTracingMarker);
        }


        private TextureHandle CreateTex(TextureDesc textureDesc, RenderGraph renderGraph, string name, GraphicsFormat format)
        {
            textureDesc.format = format;
            textureDesc.name = name;
            return renderGraph.CreateTexture(textureDesc);
        }


        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("GBuffer", out var passData);

            passData.gBufferTs = _gBufferTs;

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