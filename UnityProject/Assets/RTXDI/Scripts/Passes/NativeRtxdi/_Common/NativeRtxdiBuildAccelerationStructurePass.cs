using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Issues a build/update of the <see cref="NativeRtxdiGPUScene"/>'s TLAS at the start of the
    /// NativeRtxdi pipeline so subsequent compute passes can bind <c>SceneBVH</c> safely.
    /// </summary>
    public class NativeRtxdiBuildAccelerationStructurePass : ScriptableRenderPass
    {
        private NativeRtxdiGPUScene _gpuScene;

        public void Setup(NativeRtxdiGPUScene gpuScene)
        {
            _gpuScene = gpuScene;
        }

        private class PassData
        {
            internal NativeRtxdiGPUScene GpuScene;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxdi.BuildAccelerationStructure", out var passData);

            passData.GpuScene = _gpuScene;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            if (data.GpuScene == null) return;
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            data.GpuScene.BuildAccelerationStructure(cmd);
        }
    }
}
