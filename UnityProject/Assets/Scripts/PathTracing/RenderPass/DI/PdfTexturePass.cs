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
    public class PdfTexturePass : ScriptableRenderPass
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
            internal TextureHandle u_LocalLightPdfTextureHandle;

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

        static void ExecutePass(PassData data, ComputeGraphContext context)
        {
            var natCmd = context.cmd;

            var opaqueTracingMarker = new ProfilerMarker(ProfilerCategory.Render, "pdfTexture", MarkerFlags.SampleGPU);

            natCmd.BeginSample(opaqueTracingMarker);

            var resource = data.Resource;

            natCmd.SetComputeConstantBufferParam(data.OpaqueTs, "g_Const", resource.ResamplingConstantBuffer, 0, resource.ResamplingConstantBuffer.stride);

            natCmd.SetComputeBufferParam(data.OpaqueTs, 0, t_LightDataBufferID, resource.RtxdiResources.LightDataBuffer);

            natCmd.SetComputeTextureParam(data.OpaqueTs, 0, "u_LocalLightPdfTexture", resource.u_LocalLightPdfTextureHandle);
 
            // var all = resource.RtxdiResources.Scene.numLights;
            var all = resource.RtxdiResources.Scene.localLightPdfTextureSize.x * resource.RtxdiResources.Scene.localLightPdfTextureSize.y;

            var X = (int)(all + 255) / 256;

            
            if (X > 0)
            {
                natCmd.DispatchCompute(data.OpaqueTs, 0, X, 1, 1);
            }


            natCmd.EndSample(opaqueTracingMarker);
        }


        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddComputePass<PassData>("pdfTexture", out var passData);

            passData.OpaqueTs = _opaqueTs;
            passData.Resource = _resource;
            passData.Settings = _settings;


            var pdfTexHandle = renderGraph.ImportTexture(_resource.u_LocalLightPdfTexture);

            _resource.u_LocalLightPdfTextureHandle = pdfTexHandle;

            builder.UseTexture(_resource.u_LocalLightPdfTextureHandle, AccessFlags.Write);

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, ComputeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}