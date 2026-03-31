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
    public class BrdfRayTracingPass : ScriptableRenderPass
    {
        private readonly RayTracingShader _gBufferTs;
        private Resource _resource;
        private Settings _settings;


        public BrdfRayTracingPass(RayTracingShader gBufferTs)
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

            var gBufferTracingMarker = new ProfilerMarker(ProfilerCategory.Render, "BrdfRayTracing", MarkerFlags.SampleGPU);

            natCmd.BeginSample(gBufferTracingMarker);

            var resource = data.Resource;
            var settings = data.Settings;

            natCmd.SetRayTracingShaderPass(data.gBufferTs, "Test2");
            natCmd.SetRayTracingConstantBufferParam(data.gBufferTs, paramsID, resource.ConstantBuffer, 0, resource.ConstantBuffer.stride);

            uint rectWmod = (uint)(settings.m_RenderResolution.x * settings.resolutionScale + 0.5f);
            uint rectHmod = (uint)(settings.m_RenderResolution.y * settings.resolutionScale + 0.5f);

            // Debug.Log($"Dispatch Rays Size: {rectWmod} x {rectHmod}");


            natCmd.DispatchRays(data.gBufferTs, "MainRayGenShader", rectWmod, rectHmod, 1);

            natCmd.EndSample(gBufferTracingMarker);
        }


        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("BrdfRayTracing", out var passData);

            passData.gBufferTs = _gBufferTs;

            passData.Resource = _resource;
            passData.Settings = _settings;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}