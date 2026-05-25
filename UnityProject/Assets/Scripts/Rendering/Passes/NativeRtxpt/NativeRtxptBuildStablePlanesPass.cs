using System;
using System.Runtime.InteropServices;
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
    public class NativeRtxptBuildStablePlanesPass : ScriptableRenderPass, IDisposable
    {
        private readonly RayTracePipeline            _buildSP;
        private readonly NativeRayTraceDescriptorSet _buildDs;

        private          NativeRtxptPassContext _ctx;
        private readonly SampleMiniConstants[]  _miniConstArray = new SampleMiniConstants[1];
        private          GraphicsBuffer         _miniConstBuffer;

        /// <summary>Pipeline handle exposed for NativeRtxptBuildTlasPass hit-table rebuilds.</summary>
        public RayTracePipeline BuildPipeline => _buildSP;

        public NativeRtxptBuildStablePlanesPass(
            RayTraceShader buildStablePlanes,
            HitGroupShader[] buildHitGroups = null)
        {
            _buildSP = buildHitGroups is { Length: > 0 }
                ? new RayTracePipeline(buildStablePlanes, buildHitGroups)
                : new RayTracePipeline(buildStablePlanes);
            _buildDs = new NativeRayTraceDescriptorSet(_buildSP);

            _miniConstBuffer = new GraphicsBuffer(
                    GraphicsBuffer.Target.Constant, 1,
                    Marshal.SizeOf<SampleMiniConstants>())
                { name = "Rtxpt_MiniConst_Build" };
        }

        public void Dispose()
        {
            _buildDs?.Dispose();
            _buildSP?.Dispose();
            _miniConstBuffer?.Dispose();
            _miniConstBuffer = null;
        }

        public void Setup(NativeRtxptPassContext ctx) => _ctx = ctx;

        // ── Pass data ──────────────────────────────────────────────────────────

        private class PassData
        {
            internal RayTracePipeline            BuildSP;
            internal NativeRayTraceDescriptorSet BuildDs;
            internal NativeRtxptPassContext      Ctx;
            internal GraphicsBuffer              MiniConstBuffer;
            internal int2                        RenderRes;
        }

        // ── RenderGraph ────────────────────────────────────────────────────────

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxpt.BuildStablePlanes", out var passData);

            passData.BuildSP         = _buildSP;
            passData.BuildDs         = _buildDs;
            passData.Ctx             = _ctx;
            passData.MiniConstBuffer = _miniConstBuffer;
            passData.RenderRes       = _ctx.RenderResolution;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }

        // ── Execute ────────────────────────────────────────────────────────────

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var ctx = data.Ctx;
            var res = ctx.Textures;
            var buf = ctx.Buffers;

            // Upload g_MiniConst
            data.MiniConstBuffer.SetData(new[]
            {
                new SampleMiniConstants
                {
                    params0x = ctx.FrameState.frameIndex,
                    params0y = (ctx.FrameState.frameIndex & 1),
                }
            });

            var tlas = ctx.GpuScene?.AccelerationStructure;

            cmd.BeginSample("Rtxpt.BuildStablePlanes");
            {
                var ds = data.BuildDs;
                BindCommonRT(ds, ctx, data.MiniConstBuffer, tlas);

                ds.SetRWTexture("u_Throughput",                res.Throughput.NativePtr);
                ds.SetRWTexture("u_MotionVectors",             res.ScreenMotionVectors.NativePtr);
                ds.SetRWTexture("u_Depth",                     res.Depth.NativePtr);
                ds.SetRWTexture("u_StablePlanesHeader",        res.StablePlanesHeader.NativePtr);
                ds.SetRWTexture("u_StableRadiance",            res.StableRadiance.NativePtr);
                ds.SetRWTexture("u_SpecularHitT",              res.SpecularHitT.NativePtr);
                ds.SetRWTexture("u_ShaderDebugVizTextureBuffer", res.ShaderDebugViz.NativePtr);
                ds.SetRWStructuredBuffer("u_StablePlanesBuffer", buf.StablePlanesBuffer);

                data.BuildSP.Dispatch(cmd, ds, (uint)data.RenderRes.x, (uint)data.RenderRes.y);
            }
            cmd.EndSample("Rtxpt.BuildStablePlanes");
        }

        // ── Binding helpers ────────────────────────────────────────────────────

        private static void BindCommonRT(
            NativeRayTraceDescriptorSet ds,
            NativeRtxptPassContext ctx,
            GraphicsBuffer miniConstBuffer,
            RayTracingAccelerationStructure tlas)
        {
            ds.SetConstantBuffer("g_Const",    ctx.ConstantBuffer.GetNativeBufferPtr());
            ds.SetConstantBuffer("g_MiniConst", miniConstBuffer.GetNativeBufferPtr());
            ds.SetAccelerationStructure("SceneBVH", tlas);

            ctx.GpuScene.BindToShader(ds);

            var envCubePtr = ctx.BakedEnvCubePtr != IntPtr.Zero
                ? ctx.BakedEnvCubePtr
                : Texture2D.blackTexture.GetNativeTexturePtr();
            ds.SetTexture("t_EnvironmentMap", envCubePtr);

            var envLookupPtr = ctx.EnvLightLookupMapPtr != IntPtr.Zero
                ? ctx.EnvLightLookupMapPtr
                : Texture2D.blackTexture.GetNativeTexturePtr();
            ds.SetTexture("t_EnvLookupMap", envLookupPtr);

            ds.SetRWStructuredBuffer("u_FeedbackBuffer",
                ctx.Buffers.FeedbackBuffer.GetNativeBufferPtr(),
                ctx.Buffers.FeedbackBuffer.count,
                ctx.Buffers.FeedbackBuffer.stride);
        }
    }
}
