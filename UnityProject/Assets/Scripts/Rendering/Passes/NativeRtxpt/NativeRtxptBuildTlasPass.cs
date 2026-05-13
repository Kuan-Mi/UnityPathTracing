using NativeRender;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Builds/updates the <see cref="NRDSampleResource"/> TLASes once per frame,
    /// before any RTXPT pass that needs the acceleration structure.
    ///
    /// Mirrors <see cref="NRDTlasUpdatePass"/> from the NRD pipeline.
    /// </summary>
    public class NativeRtxptBuildTlasPass : ScriptableRenderPass
    {
        private NRDSampleResource _nrdSampleResource;
        public  ComputeShader     updateSkinnedPrimitivesCS;

        public void SetNRDSampleResource(NRDSampleResource resource)
        {
            _nrdSampleResource = resource;
        }

        private class PassData
        {
            internal NRDSampleResource Resource;
            internal ComputeShader     UpdateSkinnedPrimitivesCS;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxpt.BuildTlas", out var passData);

            passData.Resource                  = _nrdSampleResource;
            passData.UpdateSkinnedPrimitivesCS = updateSkinnedPrimitivesCS;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            if (data.Resource == null) return;
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            cmd.BeginSample(RenderPassMarkers.Streamer);
            data.Resource.FlushPendingCopies(cmd);
            cmd.EndSample(RenderPassMarkers.Streamer);

            cmd.BeginSample(RenderPassMarkers.TLAS);
            data.Resource.BuildAccelerationStructures(cmd);
            data.Resource.RecordSkinnedMorphUpdate(cmd, data.UpdateSkinnedPrimitivesCS);
            cmd.EndSample(RenderPassMarkers.TLAS);
        }
    }
}
