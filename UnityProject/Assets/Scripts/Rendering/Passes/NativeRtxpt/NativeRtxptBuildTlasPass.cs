using NativeRender;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Builds/updates the <see cref="NativeRtxptGPUScene"/> TLAS once per frame,
    /// before any RTXPT pass that needs the acceleration structure.
    /// Also rebuilds the fill-shader hit-group table when fillPipeline is provided.
    /// </summary>
    public class NativeRtxptBuildTlasPass : ScriptableRenderPass
    {
        private NativeRtxptGPUScene _gpuScene;
        private RayTracePipeline    _fillPipeline;

        public void Setup(NativeRtxptGPUScene gpuScene, RayTracePipeline fillPipeline = null)
        {
            _gpuScene     = gpuScene;
            _fillPipeline = fillPipeline;
        }

        private class PassData
        {
            internal NativeRtxptGPUScene GpuScene;
            internal RayTracePipeline    FillPipeline;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxpt.BuildTlas", out var passData);

            passData.GpuScene     = _gpuScene;
            passData.FillPipeline = _fillPipeline;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            cmd.BeginSample(RenderPassMarkers.TLAS);
            data.GpuScene.BuildAccelerationStructure(cmd);
            data.GpuScene.RebuildFillShaderTable(cmd, data.FillPipeline);
            cmd.EndSample(RenderPassMarkers.TLAS);
        }
    }
}
