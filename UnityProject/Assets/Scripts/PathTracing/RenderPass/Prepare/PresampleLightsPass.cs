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
    public class PresamplePass : ScriptableRenderPass
    {
        private readonly ComputeShader _opaqueTs;
        private Resource _resource;
        private Settings _settings;


        public PresamplePass(ComputeShader opaqueTs)
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
            
            internal RTHandle u_LocalLightPdfTexture;

            internal RtxdiResources RtxdiResources;
        }

        public class Settings
        {
            internal int x;
            internal int y;
            internal int z;
        }

        class PassData
        {
            internal ComputeShader OpaqueTs;
            internal Resource Resource;
            internal Settings Settings;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            var opaqueTracingMarker = new ProfilerMarker(ProfilerCategory.Render, "PresamplePass", MarkerFlags.SampleGPU);

            natCmd.BeginSample(opaqueTracingMarker);

            var resource = data.Resource;
            var settings = data.Settings;

            natCmd.SetComputeConstantBufferParam(data.OpaqueTs, paramsID, resource.ConstantBuffer, 0, resource.ConstantBuffer.stride);
            natCmd.SetComputeConstantBufferParam(data.OpaqueTs, "g_Const", resource.ResamplingConstantBuffer, 0, resource.ResamplingConstantBuffer.stride);

            
            natCmd.SetComputeBufferParam(data.OpaqueTs, 0,"u_RisBuffer", resource.RtxdiResources.RisBuffer);
            natCmd.SetComputeBufferParam(data.OpaqueTs, 0,"t_LightDataBuffer", resource.RtxdiResources.LightDataBuffer);
            natCmd.SetComputeBufferParam(data.OpaqueTs, 0,"u_RisLightDataBuffer", resource.RtxdiResources.RisLightDataBuffer);


            natCmd.SetComputeTextureParam(data.OpaqueTs, 0,"t_LocalLightPdfTexture", resource.u_LocalLightPdfTexture);
 


            natCmd.DispatchCompute(data.OpaqueTs, 0, settings.x, settings.y, 1);

            natCmd.EndSample(opaqueTracingMarker);
        }
        

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("PresamplePass", out var passData);

            passData.OpaqueTs = _opaqueTs;

            passData.Resource = _resource;
            passData.Settings = _settings;


            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}