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
        private RtxdiPassContext _context;

        public PdfTexturePass(ComputeShader opaqueTs)
        {
            _opaqueTs = opaqueTs;
        }

        public void Setup(RtxdiPassContext ctx)
        {
            _context = ctx;
        }

        class PassData
        {
            internal ComputeShader OpaqueTs;
            internal RtxdiPassContext Context;
            internal TextureHandle LocalLightPdfTextureHandle;
        }

        static void ExecutePass(PassData data, ComputeGraphContext context)
        {
            var natCmd = context.cmd;

            var opaqueTracingMarker = RenderPassMarkers.PdfTexture;

            natCmd.BeginSample(opaqueTracingMarker);

            var ctx = data.Context;

            natCmd.SetComputeConstantBufferParam(data.OpaqueTs, g_ConstID, ctx.ResamplingConstantBuffer, 0, ctx.ResamplingConstantBuffer.stride);

            natCmd.SetComputeBufferParam(data.OpaqueTs, 0, t_LightDataBufferID, ctx.RtxdiResources.LightDataBuffer);

            natCmd.SetComputeTextureParam(data.OpaqueTs, 0, u_LocalLightPdfTextureID, data.LocalLightPdfTextureHandle);

            var all = ctx.RtxdiResources.Scene.localLightPdfTextureSize.x * ctx.RtxdiResources.Scene.localLightPdfTextureSize.y;

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
            passData.Context = _context;

            var pdfTexHandle = renderGraph.ImportTexture(_context.LocalLightPdfTexture);
            passData.LocalLightPdfTextureHandle = pdfTexHandle;
            builder.UseTexture(pdfTexHandle, AccessFlags.Write);

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, ComputeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}