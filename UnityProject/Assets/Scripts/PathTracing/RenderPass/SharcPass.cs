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
    public class SharcPass : ScriptableRenderPass
    {
        private readonly ComputeShader _sharcResolveCs;
        private readonly RayTracingShader _sharcUpdateTs;
        private Resource _sharcResource;
        private Settings _sharcSettings;
        
        
        public SharcPass( ComputeShader sharcResolveCs, RayTracingShader sharcUpdateTs)
        {
            _sharcResolveCs = sharcResolveCs;
            _sharcUpdateTs = sharcUpdateTs;
        }

        public void Setup( Resource sharcResource, Settings sharcSettings)
        {
            _sharcResource = sharcResource;
            _sharcSettings = sharcSettings;
        }

        public class Resource
        {
            internal GraphicsBuffer ConstantBuffer;
            
            internal GraphicsBuffer HashEntriesBuffer;
            internal GraphicsBuffer AccumulationBuffer;
            internal GraphicsBuffer ResolvedBuffer;
            
            internal GraphicsBuffer SpotLightBuffer;
            internal GraphicsBuffer AreaLightBuffer;
            internal GraphicsBuffer PointLightBuffer;
        }

        public class Settings
        {
            internal int2 RenderResolution;
        }
        
        class SharcPassData
        {
            internal ComputeShader SharcResolveCs;
            internal RayTracingShader SharcUpdateTs;

            internal Resource Resource;
            internal Settings Settings;
        }

        static void ExecutePass(SharcPassData data, UnsafeGraphContext context)
        {
            
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var sharcUpdateMarker = new ProfilerMarker(ProfilerCategory.Render, "Sharc Update", MarkerFlags.SampleGPU);
            var sharcResolveMarker = new ProfilerMarker(ProfilerCategory.Render, "Sharc Resolve", MarkerFlags.SampleGPU);
            
            // Sharc update
            // if (data.passIndex == 0)
            {
                natCmd.BeginSample(sharcUpdateMarker);
                natCmd.SetRayTracingShaderPass(data.SharcUpdateTs, "Test2");
                natCmd.SetRayTracingConstantBufferParam(data.SharcUpdateTs, paramsID, data.Resource.ConstantBuffer, 0, data.Resource.ConstantBuffer.stride);

                natCmd.SetRayTracingBufferParam(data.SharcUpdateTs, g_HashEntriesID, data.Resource.HashEntriesBuffer);
                natCmd.SetRayTracingBufferParam(data.SharcUpdateTs, g_AccumulationBufferID, data.Resource.AccumulationBuffer);
                natCmd.SetRayTracingBufferParam(data.SharcUpdateTs, g_ResolvedBufferID, data.Resource.ResolvedBuffer);

                // natCmd.SetRayTracingTextureParam(data.SharcUpdateTs, g_OutputID, data.OutputTexture);
                natCmd.SetRayTracingBufferParam(data.SharcUpdateTs, gIn_SpotLightsID, data.Resource.SpotLightBuffer);
                natCmd.SetRayTracingBufferParam(data.SharcUpdateTs, gIn_AreaLightsID, data.Resource.AreaLightBuffer);
                natCmd.SetRayTracingBufferParam(data.SharcUpdateTs, gIn_PointLightsID, data.Resource.PointLightBuffer);

                const int sharcDownscale = 4;

                var w = (uint)(data.Settings.RenderResolution.x / sharcDownscale);
                var h = (uint)(data.Settings.RenderResolution.y / sharcDownscale);

                natCmd.DispatchRays(data.SharcUpdateTs, "MainRayGenShader", w, h, 1);
                natCmd.EndSample(sharcUpdateMarker);
            }
            
            
            // Sharc resolve
            // if (data.passIndex == 0)
            {
                natCmd.BeginSample(sharcResolveMarker);
                natCmd.SetComputeConstantBufferParam(data.SharcResolveCs, paramsID, data.Resource.ConstantBuffer, 0, data.Resource.ConstantBuffer.stride);
                natCmd.SetComputeBufferParam(data.SharcResolveCs, 0, g_HashEntriesID, data.Resource.HashEntriesBuffer);
                natCmd.SetComputeBufferParam(data.SharcResolveCs, 0, g_AccumulationBufferID, data.Resource.AccumulationBuffer);
                natCmd.SetComputeBufferParam(data.SharcResolveCs, 0, g_ResolvedBufferID, data.Resource.ResolvedBuffer);
 
                int LINEAR_BLOCK_SIZE = 256;
                int x = (PathTracingFeature.Capacity + LINEAR_BLOCK_SIZE - 1) / LINEAR_BLOCK_SIZE;

                natCmd.DispatchCompute(data.SharcResolveCs, 0, x, 1, 1);

                natCmd.EndSample(sharcResolveMarker);
            }
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<SharcPassData>("Sharc", out var passData);
            
            passData.SharcResolveCs = _sharcResolveCs;
            passData.SharcUpdateTs = _sharcUpdateTs;

            passData.Resource = _sharcResource;
            passData.Settings = _sharcSettings;
            
            builder.AllowPassCulling(false);
            builder.SetRenderFunc((SharcPassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}