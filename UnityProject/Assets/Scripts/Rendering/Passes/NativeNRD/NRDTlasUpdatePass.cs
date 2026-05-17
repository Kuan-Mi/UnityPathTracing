using NativeRender;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Lightweight pass that builds/updates the NRDSampleResource TLASes exactly once
    /// per frame, before any pass that requires them (NRDSharcPass, NRDOpaquePass, NRDTransparentPass).
    /// </summary>
    public class NRDTlasUpdatePass : ScriptableRenderPass
    {
        private NRDSampleResource _nrdResource;
        public  ComputeShader     updateSkinnedPrimitivesCS;

        public void SetNRDSampleResource(NRDSampleResource nrdResource)
        {
            _nrdResource = nrdResource;
        }

        // -------------------------------------------------------------------------
        // Pass data (RenderGraph)
        // -------------------------------------------------------------------------

        class PassData
        {
            internal NRDSampleResource NrdResource;
            internal ComputeShader     updateSkinnedPrimitivesCS;
        }

        // -------------------------------------------------------------------------
        // Execution
        // -------------------------------------------------------------------------

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            cmd.BeginSample(RenderPassMarkers.Streamer);
            data.NrdResource.FlushPendingCopies(cmd);
            cmd.EndSample(RenderPassMarkers.Streamer);
            
            
            
            cmd.BeginSample(RenderPassMarkers.TLAS);
            data.NrdResource.BuildAccelerationStructures(cmd);
            cmd.EndSample(RenderPassMarkers.TLAS);
            
            cmd.BeginSample(RenderPassMarkers.RecordSkinnedMorphUpdate);
            data.NrdResource.RecordSkinnedMorphUpdate(cmd, data.updateSkinnedPrimitivesCS);
            cmd.EndSample(RenderPassMarkers.RecordSkinnedMorphUpdate);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NRDTlasUpdatePass", out var passData);

            passData.NrdResource               = _nrdResource;
            passData.updateSkinnedPrimitivesCS = updateSkinnedPrimitivesCS;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }
    }
}