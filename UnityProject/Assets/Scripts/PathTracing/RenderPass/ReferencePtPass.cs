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
    public class ReferencePtPass: ScriptableRenderPass
    {
        private readonly RayTracingShader _ReferencePtTs;
        private Resource _resource;
        private Settings _settings;


        public ReferencePtPass(RayTracingShader ReferencePtTs)
        {
            _ReferencePtTs = ReferencePtTs;
        }

        public void Setup(Resource sharcResource, Settings sharcSettings)
        {
            _resource = sharcResource;
            _settings = sharcSettings;
        }

        public class Resource
        {
            internal GraphicsBuffer ConstantBuffer;
            
            internal GraphicsBuffer SpotLightBuffer;
            internal GraphicsBuffer AreaLightBuffer;
            internal GraphicsBuffer PointLightBuffer;
            
            internal GraphicsBuffer AeExposureBuffer;
            
        }

        public class Settings
        {
            internal int2 m_RenderResolution;
            internal float resolutionScale;
            internal int referenceBounceNum;
            internal int convergenceStep;
            internal float split;
        }

        class PassData
        {
            internal RayTracingShader ReferencePtTs;
            internal Resource Resource;
            internal Settings Settings;

            internal TextureHandle OutputTexture;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            var referencePtTracingMarker = new ProfilerMarker(ProfilerCategory.Render, "Reference Pt Tracing", MarkerFlags.SampleGPU);

            natCmd.BeginSample(referencePtTracingMarker);

            var resource = data.Resource;
            var settings = data.Settings;

            natCmd.SetRayTracingShaderPass(data.ReferencePtTs, "Test2");
            natCmd.SetRayTracingConstantBufferParam(data.ReferencePtTs, paramsID, resource.ConstantBuffer, 0, resource.ConstantBuffer.stride);

            natCmd.SetRayTracingBufferParam(data.ReferencePtTs, "_AE_ExposureBuffer", data.Resource.AeExposureBuffer);




            natCmd.SetRayTracingTextureParam(data.ReferencePtTs, g_OutputID, data.OutputTexture);


            natCmd.SetRayTracingBufferParam(data.ReferencePtTs, gIn_SpotLightsID, resource.SpotLightBuffer);
            natCmd.SetRayTracingBufferParam(data.ReferencePtTs, gIn_AreaLightsID, resource.AreaLightBuffer);
            natCmd.SetRayTracingBufferParam(data.ReferencePtTs, gIn_PointLightsID, resource.PointLightBuffer);

            natCmd.SetRayTracingIntParam(data.ReferencePtTs, "_ReferenceBounceNum", settings.referenceBounceNum);
            natCmd.SetRayTracingIntParam(data.ReferencePtTs, "g_ConvergenceStep", settings.convergenceStep);
            natCmd.SetRayTracingFloatParam(data.ReferencePtTs, "g_split", settings.split);

            uint rectWmod = (uint)(settings.m_RenderResolution.x * settings.resolutionScale + 0.5f);
            uint rectHmod = (uint)(settings.m_RenderResolution.y * settings.resolutionScale + 0.5f);

            // Debug.Log($"Dispatch Rays Size: {rectWmod} x {rectHmod}");


            natCmd.DispatchRays(data.ReferencePtTs, "MainRayGenShader", rectWmod, rectHmod, 1);

            natCmd.EndSample(referencePtTracingMarker);

        }


        private TextureHandle CreateTex(TextureDesc textureDesc, RenderGraph renderGraph, string name, GraphicsFormat format)
        {
            textureDesc.format = format;
            textureDesc.name = name;
            return renderGraph.CreateTexture(textureDesc);
        }


        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("Reference Pt", out var passData);

            passData.ReferencePtTs = _ReferencePtTs;

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

            
            var ptContextItem = frameData.Get<PTContextItem>();

            passData.OutputTexture = ptContextItem.OutputTexture;
            
            builder.UseTexture(passData.OutputTexture,  AccessFlags.ReadWrite);

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}