using System;
using System.Runtime.InteropServices;
using NativeRender;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Builds/updates the <see cref="NativeRtxptGPUScene"/> TLAS once per frame,
    /// before any RTXPT pass that needs the acceleration structure.
    ///
    /// Also records, ahead of the TLAS/BLAS build in the same CommandBuffer:
    ///   1. the skinned-repack compute for every SkinnedMeshRenderer instance — converts
    ///      Unity's GPU-skinned vertex buffer into the instance's donut SoA buffer and
    ///      maintains its PrevPosition stream (donut SkinningPass model); the dynamic BLAS
    ///      refit inside the build then consumes the fresh positions;
    ///   2. the deferred t_InstanceData upload (transforms updated this frame);
    ///   3. each pipeline's hit-group table rebuild when the scene topology changed.
    /// </summary>
    public class NativeRtxptBuildTlasPass : ScriptableRenderPass, IDisposable
    {
        // Mirrors SkinnedRepackConstants in SkinnedRepack.computeshader (12 uints, root constants).
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        private struct SkinnedRepackConstants
        {
            public uint VertexCount;
            public uint SrcStride;
            public uint SrcPosOffset;
            public uint SrcNormalOffset;
            public uint SrcTangentOffset;
            public uint DstPosOffset;
            public uint DstPrevPosOffset;
            public uint DstNormalOffset;
            public uint DstTangentOffset;
            public uint Flags;
            public uint _pad0;
            public uint _pad1;
        }

        private NativeRtxptGPUScene _gpuScene;
        private RayTracePipeline    _buildPipeline;
        private RayTracePipeline    _fillPipeline;
        private RayTracePipeline    _refPipeline;

        private readonly NativeComputePipeline      _repackCs; // null when no shader assigned
        private readonly NativeComputeDescriptorSet _repackDs;

        public NativeRtxptBuildTlasPass(NativeComputeShader skinnedRepackShader = null)
        {
            if (skinnedRepackShader != null)
            {
                _repackCs = new NativeComputePipeline(skinnedRepackShader);
                _repackDs = new NativeComputeDescriptorSet(_repackCs);
            }
        }

        public void Dispose()
        {
            _repackDs?.Dispose();
            _repackCs?.Dispose();
        }

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
            internal NativeRtxptGPUScene        GpuScene;
            internal RayTracePipeline           BuildPipeline;
            internal RayTracePipeline           FillPipeline;
            internal RayTracePipeline           RefPipeline;
            internal NativeComputePipeline      RepackCs;
            internal NativeComputeDescriptorSet RepackDs;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("BuildTlas", out var passData);

            passData.GpuScene      = _gpuScene;
            passData.BuildPipeline = _buildPipeline;
            passData.FillPipeline  = _fillPipeline;
            passData.RefPipeline   = _refPipeline;
            passData.RepackCs      = _repackCs;
            passData.RepackDs      = _repackDs;

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

        private static unsafe void RecordSkinnedRepack(PassData data, CommandBuffer cmd)
        {
            var dispatches = data.GpuScene.SkinnedDispatches;
            if (dispatches.Count == 0) return;
            if (data.RepackCs == null || data.RepackDs == null)
            {
                Debug.LogWarning("[NativeRtxptBuildTlasPass] Scene has skinned renderers but no SkinnedRepack compute shader is assigned — skinned geometry stays in rest pose.");
                return;
            }

            var ds = data.RepackDs;
            foreach (var d in dispatches)
            {
                if (d.Smr == null) continue;

                // Current-frame GPU-skinned vertex buffer. The wrapper is disposed immediately —
                // the underlying resource is owned (and kept alive) by the SkinnedMeshRenderer.
                var vb = d.Smr.GetVertexBuffer();
                if (vb == null) continue; // not skinned yet this frame; rest pose is in the SoA buffer
                IntPtr srcPtr = vb.GetNativeBufferPtr();
                vb.Dispose();
                if (srcPtr == IntPtr.Zero) continue;

                var constants = new SkinnedRepackConstants
                {
                    VertexCount      = (uint)d.VertexCount,
                    SrcStride        = d.SrcStride,
                    SrcPosOffset     = d.SrcPosOffset,
                    SrcNormalOffset  = d.SrcNormalOffset,
                    SrcTangentOffset = d.SrcTangentOffset,
                    DstPosOffset     = d.Streams.Pos,
                    DstPrevPosOffset = d.Streams.PrevPos,
                    DstNormalOffset  = d.Streams.Normal,
                    DstTangentOffset = d.Streams.Tangent,
                    Flags            = d.BaseFlags | (d.Geometry.PendingFirstFrame ? RtxptSkinnedDispatch.FlagFirstFrame : 0u),
                };
                ds.SetRootConstants("g_Const", &constants);
                ds.SetBuffer("t_SrcVertexBuffer", srcPtr);
                ds.SetRWBuffer("u_DstVertexBuffer", d.DstVbPtr);

                data.RepackCs.Dispatch(cmd, ds, ((uint)d.VertexCount + 63u) / 64u, 1, 1);
                d.Geometry.PendingFirstFrame = false;
            }
        }
    }
}
