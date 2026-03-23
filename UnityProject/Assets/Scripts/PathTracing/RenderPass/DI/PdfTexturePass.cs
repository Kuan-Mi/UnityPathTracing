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
    public class PdfTexturePass: ScriptableRenderPass
    {
        private readonly ComputeShader _opaqueTs;
        private Resource _resource;
        private Settings _settings;


        public PdfTexturePass(ComputeShader opaqueTs)
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
            internal GraphicsBuffer ResamplingConstantBuffer;

            internal RTHandle u_LocalLightPdfTexture;
            
            internal RtxdiResources RtxdiResources;
        }

        public class Settings
        {
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

            var opaqueTracingMarker = new ProfilerMarker(ProfilerCategory.Render, "pdfTexture", MarkerFlags.SampleGPU);

            natCmd.BeginSample(opaqueTracingMarker);

            var resource = data.Resource;
            
            
            natCmd.SetComputeBufferParam(data.OpaqueTs,0, "ResampleConstants", resource.ResamplingConstantBuffer);
            

            natCmd.SetComputeBufferParam(data.OpaqueTs,0, t_LightDataBufferID, resource.RtxdiResources.LightDataBuffer);


            natCmd.SetComputeTextureParam(data.OpaqueTs, 0,"u_LocalLightPdfTexture", resource.u_LocalLightPdfTexture);

            var all = resource.RtxdiResources.Scene.emissiveTriangleCount;
            
            var X = (int) (all + 255) / 256;

            natCmd.DispatchCompute(data.OpaqueTs, 0, X, 1, 1);

            
            // gen mip
            natCmd.GenerateMips(resource.u_LocalLightPdfTexture);
            
            natCmd.EndSample(opaqueTracingMarker);
        }
 



        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("pdfTexture", out var passData);

            passData.OpaqueTs = _opaqueTs;
            passData.Resource = _resource;
            passData.Settings = _settings;


            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}