using PathTracing.Profiling;
using System;
using NativeRender;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Builds/updates the <see cref="RtxptGPUScene"/> TLAS once per frame,
    /// before any RTXPT pass that needs the acceleration structure.
    ///
    /// Also records, ahead of the TLAS/BLAS build in the same CommandBuffer:
    ///   1. the skinned-repack compute for every SkinnedMeshRenderer instance — converts
    ///      Unity's GPU-skinned vertex buffer into the instance's donut SoA buffer and
    ///      maintains its PrevPosition stream (donut SkinningPass model); the dynamic BLAS
    ///      refit inside the build then consumes the fresh positions;
    ///   2. the deferred t_InstanceData upload (transforms updated this frame);
    ///   3. each pipeline's hit-group table rebuild when the scene topology changed.
    ///
    /// The repack runs as a plain Unity <see cref="ComputeShader"/> (SkinnedRepack.compute):
    /// both the source (SkinnedMeshRenderer.GetVertexBuffer) and the destination donut SoA
    /// buffer are Unity GraphicsBuffers, so they bind directly via SetComputeBufferParam —
    /// the former native-compute path required a per-frame GetNativeBufferPtr() on each.
    /// </summary>
    public class RtxptBuildTlasPass : ScriptableRenderPass, IDisposable
    {
        // Shader property IDs for the SkinnedRepack.compute uniforms/resources.
        private static readonly int _idVertexCount      = Shader.PropertyToID("g_VertexCount");
        private static readonly int _idSrcStride        = Shader.PropertyToID("g_SrcStride");
        private static readonly int _idSrcPosOffset     = Shader.PropertyToID("g_SrcPosOffset");
        private static readonly int _idSrcNormalOffset  = Shader.PropertyToID("g_SrcNormalOffset");
        private static readonly int _idSrcTangentOffset = Shader.PropertyToID("g_SrcTangentOffset");
        private static readonly int _idDstPosOffset     = Shader.PropertyToID("g_DstPosOffset");
        private static readonly int _idDstPrevPosOffset = Shader.PropertyToID("g_DstPrevPosOffset");
        private static readonly int _idDstNormalOffset  = Shader.PropertyToID("g_DstNormalOffset");
        private static readonly int _idDstTangentOffset = Shader.PropertyToID("g_DstTangentOffset");
        private static readonly int _idFlags            = Shader.PropertyToID("g_Flags");
        private static readonly int _idSrcVertexBuffer  = Shader.PropertyToID("t_SrcVertexBuffer");
        private static readonly int _idDstVertexBuffer  = Shader.PropertyToID("u_DstVertexBuffer");

        private RtxptGPUScene    _gpuScene;
        private RayTracePipeline _buildPipeline;
        private RayTracePipeline _fillPipeline;
        private RayTracePipeline _refPipeline;

        private readonly ComputeShader _repackCs; // null when no shader assigned
        private readonly int           _repackKernel;

        public RtxptBuildTlasPass(ComputeShader skinnedRepackShader = null)
        {
            _repackCs = skinnedRepackShader;
            if (_repackCs != null)
                _repackKernel = _repackCs.FindKernel("SkinnedRepack");
        }

        public void Dispose()
        {
        }

        public void Setup(RtxptGPUScene gpuScene,
            RayTracePipeline buildPipeline = null,
            RayTracePipeline fillPipeline = null,
            RayTracePipeline refPipeline = null)
        {
            _gpuScene      = gpuScene;
            _buildPipeline = buildPipeline;
            _fillPipeline  = fillPipeline;
            _refPipeline   = refPipeline;
        }

        private class PassData
        {
            internal RtxptGPUScene    GpuScene;
            internal RayTracePipeline BuildPipeline;
            internal RayTracePipeline FillPipeline;
            internal RayTracePipeline RefPipeline;
            internal ComputeShader    RepackCs;
            internal int              RepackKernel;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("BuildTlas", out var passData);

            passData.GpuScene      = _gpuScene;
            passData.BuildPipeline = _buildPipeline;
            passData.FillPipeline  = _fillPipeline;
            passData.RefPipeline   = _refPipeline;
            passData.RepackCs      = _repackCs;
            passData.RepackKernel  = _repackKernel;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            cmd.BeginSample(RenderPassMarkers.TLAS);
            // Skinned repack must precede the AS build: the dynamic BLAS refit reads the SoA
            // positions this compute writes.
            RecordSkinnedRepack(data, cmd);
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

        private static void RecordSkinnedRepack(PassData data, CommandBuffer cmd)
        {
            var dispatches = data.GpuScene.SkinnedDispatches;
            if (dispatches.Count == 0) return;
            if (data.RepackCs == null)
            {
                Debug.LogWarning("[RtxptBuildTlasPass] Scene has skinned renderers but no SkinnedRepack compute shader is assigned — skinned geometry stays in rest pose.");
                return;
            }

            var cs     = data.RepackCs;
            int kernel = data.RepackKernel;
            foreach (var d in dispatches)
            {
                if (d.Smr == null) continue;

                // Current-frame GPU-skinned vertex buffer (a Unity GraphicsBuffer wrapping the
                // SMR-owned resource). Not disposed here: the CommandBuffer keeps it alive until
                // deferred execution, and the wrapper is reclaimed by the GC after the frame —
                // mirroring NRDSampleResource.RecordSkinnedMorphUpdate.
                var srcVb = d.Smr.GetVertexBuffer();
                if (srcVb == null) continue; // not skinned yet this frame; rest pose is in the SoA buffer

                uint flags = d.BaseFlags | (d.Geometry.PendingFirstFrame ? RtxptSkinnedDispatch.FlagFirstFrame : 0u);

                cmd.SetComputeIntParam(cs, _idVertexCount, d.VertexCount);
                cmd.SetComputeIntParam(cs, _idSrcStride, (int)d.SrcStride);
                cmd.SetComputeIntParam(cs, _idSrcPosOffset, (int)d.SrcPosOffset);
                cmd.SetComputeIntParam(cs, _idSrcNormalOffset, (int)d.SrcNormalOffset);
                cmd.SetComputeIntParam(cs, _idSrcTangentOffset, (int)d.SrcTangentOffset);
                cmd.SetComputeIntParam(cs, _idDstPosOffset, (int)d.Streams.Pos);
                cmd.SetComputeIntParam(cs, _idDstPrevPosOffset, (int)d.Streams.PrevPos);
                cmd.SetComputeIntParam(cs, _idDstNormalOffset, (int)d.Streams.Normal);
                cmd.SetComputeIntParam(cs, _idDstTangentOffset, (int)d.Streams.Tangent);
                cmd.SetComputeIntParam(cs, _idFlags, (int)flags);

                cmd.SetComputeBufferParam(cs, kernel, _idSrcVertexBuffer, srcVb);
                cmd.SetComputeBufferParam(cs, kernel, _idDstVertexBuffer, d.Geometry.Vb);

                cmd.DispatchCompute(cs, kernel, ((int)d.VertexCount + 63) / 64, 1, 1);
                d.Geometry.PendingFirstFrame = false;
            }
        }
    }
}