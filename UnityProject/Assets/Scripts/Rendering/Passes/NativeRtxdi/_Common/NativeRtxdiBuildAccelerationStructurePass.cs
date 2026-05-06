using NativeRender;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Issues a build/update of the <see cref="GPUScene"/>'s native TLAS at the start of the
    /// NativeRtxdi pipeline so subsequent compute passes can bind <c>SceneBVH</c> safely.
    /// </summary>
    public class NativeRtxdiBuildAccelerationStructurePass : ScriptableRenderPass
    {
        private GPUScene _gpuScene;

        public void Setup(GPUScene gpuScene)
        {
            _gpuScene = gpuScene;
        }

        private class PassData
        {
            internal GPUScene GpuScene;
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
