using System;
using NativeRender;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;
using RayTracingAccelerationStructure = NativeRender.RayTracingAccelerationStructure;

namespace PathTracing
{
    /// <summary>
    /// Phase 2a: BuildStablePlanes RT pass (PathTracePrePass in original RTXPT).
    ///
    /// Dispatches the BuildStablePlanes ray-tracing shader to build per-pixel
    /// StablePlane geometry (depth, motion vectors, throughput, specular hit-T).
    ///
    /// Must run BEFORE ExportVisibilityBuffer and LightingUpdateEnd.
    /// FillStablePlanes (Phase 2b) runs after LightingUpdateEnd.
    /// </summary>
    public class RtxptBuildStablePlanesPass : ScriptableRenderPass, IDisposable
    {
        private readonly RayTracePipeline            _buildSP;
        private readonly NativeRayTraceDescriptorSet _buildDs;

        private RtxptPassContext _ctx;

        private static readonly RootConstantsHint[] MiniConstRootConstantsHints =
        {
            new RootConstantsHint { Name = "g_MiniConst", Count = 16 }
        };

        /// <summary>Pipeline handle exposed for RtxptBuildTlasPass hit-table rebuilds.</summary>
        public RayTracePipeline BuildPipeline => _buildSP;

        public RtxptBuildStablePlanesPass(
            RayTraceShader buildStablePlanes,
            HitGroupShader[] buildHitGroups = null)
        {
            _buildSP = buildHitGroups is { Length: > 0 }
                ? new RayTracePipeline(buildStablePlanes, buildHitGroups, MiniConstRootConstantsHints)
                : new RayTracePipeline(buildStablePlanes, MiniConstRootConstantsHints);
            _buildDs = new NativeRayTraceDescriptorSet(_buildSP);
        }

        public void Dispose()
        {
            _buildDs?.Dispose();
            _buildSP?.Dispose();
        }

        public void Setup(RtxptPassContext ctx) => _ctx = ctx;

        // ── Pass data ──────────────────────────────────────────────────────────

        private class PassData
        {
            internal RayTracePipeline            BuildSP;
            internal NativeRayTraceDescriptorSet BuildDs;
            internal RtxptPassContext      Ctx;
            internal int2                        RenderRes;
        }

        // ── RenderGraph ────────────────────────────────────────────────────────

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            // PIX marker matches the original RTXPT (AdvancedSample PathTracePrePass / RayGen_BUILD).
            using var builder = renderGraph.AddUnsafePass<PassData>("PathTracePrePass", out var passData);

            passData.BuildSP   = _buildSP;
            passData.BuildDs   = _buildDs;
            passData.Ctx       = _ctx;
            passData.RenderRes = _ctx.RenderResolution;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }

        // ── Execute ────────────────────────────────────────────────────────────

        private static unsafe void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var ctx = data.Ctx;
            var res = ctx.Textures;
            var buf = ctx.Buffers;

            var miniConst = new SampleMiniConstants
            {
                params0x = ctx.FrameState.frameIndex,
                params0y = (ctx.FrameState.frameIndex & 1),
            };

            var tlas = ctx.GpuScene?.AccelerationStructure;

            cmd.BeginSample("PathTracePrePass");
            {
                var ds = data.BuildDs;
                BindCommonRT(ds, ctx, &miniConst, tlas);

                ds.SetRWTexture("u_Throughput", res.Throughput.NativePtr);
                ds.SetRWTexture("u_MotionVectors", res.ScreenMotionVectors.NativePtr);
                ds.SetRWTexture("u_Depth", res.Depth.NativePtr);
                ds.SetRWTexture("u_StablePlanesHeader", res.StablePlanesHeader.NativePtr);
                ds.SetRWTexture("u_StableRadiance", res.StableRadiance.NativePtr);
                ds.SetRWTexture("u_SpecularHitT", res.SpecularHitT.NativePtr);
                ds.SetRWTexture("u_ShaderDebugVizTextureBuffer", res.ShaderDebugViz.NativePtr);

                ds.SetStructuredBuffer("t_Lights", buf.LightBuffer, buf.LightBuffer.count, buf.LightBuffer.stride);
                ds.SetStructuredBuffer("t_LightsEx", buf.LightExBuffer, buf.LightExBuffer.count, buf.LightExBuffer.stride);
                ds.SetRWStructuredBuffer("u_StablePlanesBuffer", buf.StablePlanesBufferPtr, buf.StablePlanesBuffer.count, buf.StablePlanesBuffer.stride);

                data.BuildSP.Dispatch(cmd, ds, (uint)data.RenderRes.x, (uint)data.RenderRes.y);
            }
            cmd.EndSample("PathTracePrePass");
        }

        // ── Binding helpers ────────────────────────────────────────────────────

        private static unsafe void BindCommonRT(
            NativeRayTraceDescriptorSet ds,
            RtxptPassContext ctx,
            SampleMiniConstants* miniConst,
            RayTracingAccelerationStructure tlas)
        {
            ds.SetConstantBuffer("g_Const", ctx.ConstantBuffer);
            ds.SetRootConstants("g_MiniConst", miniConst);
            ds.SetAccelerationStructure("SceneBVH", tlas);

            ctx.GpuScene.BindToShader(ds);
            ds.SetTexture("t_EnvironmentMap", ctx.BakedEnvCubePtr);
            ds.SetTexture("t_EnvLookupMap", ctx.EnvLightLookupMapPtr);

            ds.SetRWStructuredBuffer("u_FeedbackBuffer",
                ctx.Buffers.FeedbackBufferPtr,
                ctx.Buffers.FeedbackBuffer.count,
                ctx.Buffers.FeedbackBuffer.stride);

            // ShaderDebug raw buffer (u125, DebugPrint/DebugLine/DebugTriangle) and the picked-pixel
            // line buffer (u52, only referenced when compiled with ENABLE_DEBUG_LINES_VIZ=1).
            // Both no-op when absent from the shader reflection or not allocated.
            if (ctx.Buffers.ShaderDebugBufferPtr != IntPtr.Zero)
                ds.SetRWBuffer("u_ShaderDebugBuffer", ctx.Buffers.ShaderDebugBufferPtr);
            if (ctx.Buffers.DebugLinesBufferPtr != IntPtr.Zero)
                ds.SetRWStructuredBuffer("u_DebugLinesBuffer", ctx.Buffers.DebugLinesBufferPtr,
                    RtxptBufferResources.MaxDebugLines, RtxptBufferResources.DebugLineStructSize);
        }
    }
}