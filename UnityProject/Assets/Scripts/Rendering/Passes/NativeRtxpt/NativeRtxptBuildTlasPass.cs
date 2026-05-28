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
        private RayTracePipeline    _buildPipeline;
        private RayTracePipeline    _fillPipeline;
        private RayTracePipeline    _refPipeline;

        public void Setup(NativeRtxptGPUScene gpuScene,
                          RayTracePipeline buildPipeline = null,
                          RayTracePipeline fillPipeline  = null,
                          RayTracePipeline refPipeline   = null)
        {
            _gpuScene      = gpuScene;
            _buildPipeline = buildPipeline;
            _fillPipeline  = fillPipeline;
            _refPipeline   = refPipeline;
        }

        private class PassData
        {
            internal NativeRtxptGPUScene GpuScene;
            internal RayTracePipeline    BuildPipeline;
            internal RayTracePipeline    FillPipeline;
            internal RayTracePipeline    RefPipeline;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxpt.BuildTlas", out var passData);

            passData.GpuScene      = _gpuScene;
            passData.BuildPipeline = _buildPipeline;
            passData.FillPipeline  = _fillPipeline;
            passData.RefPipeline   = _refPipeline;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            cmd.BeginSample(RenderPassMarkers.TLAS);
            // Record the deferred t_InstanceData upload (transforms updated this frame) before
            // the TLAS build, so the structured buffer is current for downstream RTXPT passes.
            data.GpuScene.FlushInstanceBuffer(cmd);
            data.GpuScene.BuildAccelerationStructure(cmd);
            // Rebuild each pipeline's hit-group table only when the scene topology changed,
            // not every frame (no-op while the scene is static).
            if (data.GpuScene.ShaderTableDirty)
            {
                data.GpuScene.RebuildShaderTable(cmd, data.BuildPipeline);
                data.GpuScene.RebuildShaderTable(cmd, data.FillPipeline);
                data.GpuScene.RebuildShaderTable(cmd, data.RefPipeline);
                data.GpuScene.MarkShaderTableClean();
            }
            cmd.EndSample(RenderPassMarkers.TLAS);
        }
    }
}
