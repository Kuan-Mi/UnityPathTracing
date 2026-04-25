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
        public ComputeShader updateSkinnedPrimitivesCS;

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
            internal ComputeShader updateSkinnedPrimitivesCS;
        }

        // -------------------------------------------------------------------------
        // Execution
        // -------------------------------------------------------------------------

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            data.NrdResource.BuildAccelerationStructures(cmd);
            data.NrdResource.RecordSkinnedMorphUpdate(cmd, data.updateSkinnedPrimitivesCS);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NRDTlasUpdatePass", out var passData);

            passData.NrdResource = _nrdResource;
            passData.updateSkinnedPrimitivesCS = updateSkinnedPrimitivesCS;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }
    }
}
