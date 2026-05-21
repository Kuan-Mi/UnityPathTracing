using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Builds/updates the <see cref="NativeRtxptGPUScene"/> TLAS once per frame,
    /// before any RTXPT pass that needs the acceleration structure.
    /// </summary>
    public class NativeRtxptBuildTlasPass : ScriptableRenderPass
    {
        private NativeRtxptGPUScene _gpuScene;

        public void Setup(NativeRtxptGPUScene gpuScene)
        {
            _gpuScene = gpuScene;
        }

        private class PassData
        {
            internal NativeRtxptGPUScene GpuScene;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxpt.BuildTlas", out var passData);

            passData.GpuScene = _gpuScene;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            cmd.BeginSample(RenderPassMarkers.TLAS);
            data.GpuScene.BuildAccelerationStructure(cmd);
            cmd.EndSample(RenderPassMarkers.TLAS);
        }
    }
}
