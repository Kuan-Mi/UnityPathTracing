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
        private RtxdiPassContext _context;
        private int _dispatchX;
        private int _dispatchY;

        public PresamplePass(ComputeShader opaqueTs)
        {
            _opaqueTs = opaqueTs;
        }

        public void Setup(RtxdiPassContext ctx, int x, int y)
        {
            _context = ctx;
            _dispatchX = x;
            _dispatchY = y;
        }

        class PassData
        {
            internal ComputeShader OpaqueTs;
            internal RtxdiPassContext Context;
            internal int DispatchX;
            internal int DispatchY;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            var opaqueTracingMarker = RenderPassMarkers.Presample;

            natCmd.BeginSample(opaqueTracingMarker);

            var ctx = data.Context;

            natCmd.SetComputeConstantBufferParam(data.OpaqueTs, paramsID, ctx.ConstantBuffer, 0, ctx.ConstantBuffer.stride);
            natCmd.SetComputeConstantBufferParam(data.OpaqueTs, g_ConstID, ctx.ResamplingConstantBuffer, 0, ctx.ResamplingConstantBuffer.stride);

            natCmd.SetComputeBufferParam(data.OpaqueTs, 0, u_RisBufferID, ctx.RtxdiResources.RisBuffer);
            natCmd.SetComputeBufferParam(data.OpaqueTs, 0, t_LightDataBufferID, ctx.RtxdiResources.LightDataBuffer);
            natCmd.SetComputeBufferParam(data.OpaqueTs, 0, u_RisLightDataBufferID, ctx.RtxdiResources.RisLightDataBuffer);

            natCmd.SetComputeTextureParam(data.OpaqueTs, 0, t_LocalLightPdfTextureID, ctx.LocalLightPdfTexture);

            natCmd.DispatchCompute(data.OpaqueTs, 0, data.DispatchX, data.DispatchY, 1);

            natCmd.EndSample(opaqueTracingMarker);
        }
        

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("PresamplePass", out var passData);

            passData.OpaqueTs = _opaqueTs;
            passData.Context = _context;
            passData.DispatchX = _dispatchX;
            passData.DispatchY = _dispatchY;


            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}