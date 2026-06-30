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
    /// Phase 2b: FillStablePlanes RT pass (PathTrace in original RTXPT).
    ///
    /// Realtime mode: dispatches FillStablePlanes to trace noisy diffuse paths
    /// from each StablePlane endpoint, accumulate radiance, and write DLSS-RR
    /// guide buffers + NEE-AT feedback.
    ///
    /// Reference mode: dispatches the Reference shader for accumulation.
    ///
    /// Must run AFTER LightingUpdateEnd.
    /// </summary>
    public class RtxptFillStablePlanesPass : ScriptableRenderPass, IDisposable
    {
        private readonly RayTracePipeline            _fillSP;
        private readonly NativeRayTraceDescriptorSet _fillDs;

        private readonly RayTracePipeline            _refSP;
        private readonly NativeRayTraceDescriptorSet _refDs;

        private          RtxptPassContext _ctx;
        private static readonly RootConstantsHint[] MiniConstRootConstantsHints =
        {
            new RootConstantsHint { Name = "g_MiniConst", Count = 16 }
        };

        /// <summary>Pipeline handles exposed for RtxptBuildTlasPass hit-table rebuilds.</summary>
        public RayTracePipeline FillPipeline => _fillSP;
        public RayTracePipeline RefPipeline  => _refSP;

        public RtxptFillStablePlanesPass(
            RayTraceShader fillStablePlanes,
            RayTraceShader reference,
            HitGroupShader[] fillHitGroups      = null,
            HitGroupShader[] referenceHitGroups = null)
        {
            _fillSP = fillHitGroups is { Length: > 0 }
                ? new RayTracePipeline(fillStablePlanes, fillHitGroups, MiniConstRootConstantsHints)
                : new RayTracePipeline(fillStablePlanes, MiniConstRootConstantsHints);
            _fillDs = new NativeRayTraceDescriptorSet(_fillSP);

            _refSP = referenceHitGroups is { Length: > 0 }
                ? new RayTracePipeline(reference, referenceHitGroups, MiniConstRootConstantsHints)
                : new RayTracePipeline(reference, MiniConstRootConstantsHints);
            _refDs = new NativeRayTraceDescriptorSet(_refSP);
        }

        public void Dispose()
        {
            _fillDs?.Dispose();
            _fillSP?.Dispose();
            _refDs?.Dispose();
            _refSP?.Dispose();
        }

        public void Setup(RtxptPassContext ctx) => _ctx = ctx;

        // ── Pass data ──────────────────────────────────────────────────────────

        private class PassData
        {
            internal RayTracePipeline            FillSP, RefSP;
            internal NativeRayTraceDescriptorSet FillDs, RefDs;
            internal RtxptPassContext      Ctx;
            internal int2                        RenderRes;
            internal bool                        IsRealtime;
        }

        // ── RenderGraph ────────────────────────────────────────────────────────

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            // PIX marker matches the original RTXPT (AdvancedSample PathTrace / RayGen_FILL, and the
            // REF pipeline in reference mode — RTXPT drives both through the single "PathTrace" marker).
            using var builder = renderGraph.AddUnsafePass<PassData>("PathTrace", out var passData);

            passData.FillSP             = _fillSP;
            passData.FillDs             = _fillDs;
            passData.RefSP              = _refSP;
            passData.RefDs              = _refDs;
            passData.Ctx                = _ctx;
            passData.RenderRes          = _ctx.RenderResolution;
            passData.IsRealtime         = _ctx.Setting.realtimeMode;

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

            if (data.IsRealtime)
            {
                cmd.BeginSample("PathTrace");
                {
                    var ds = data.FillDs;
                    BindCommonRT(ds, ctx, &miniConst, tlas);
                    BindLightBuffers(ds, ctx);

                    ds.SetRWTexture("u_ShaderDebugVizTextureBuffer", res.ShaderDebugViz.NativePtr);
                    ds.SetRWTexture("u_DebugOutputColor",            res.DebugOutputColor.NativePtr);
                    ds.SetRWTexture("u_StablePlanesHeader",          res.StablePlanesHeader.NativePtr);
                    ds.SetRWTexture("u_SpecularHitT",                res.SpecularHitT.NativePtr);
                    ds.SetRWStructuredBuffer("u_StablePlanesBuffer", buf.StablePlanesBufferPtr, buf.StablePlanesBuffer.count, buf.StablePlanesBuffer.stride);

                    data.FillSP.Dispatch(cmd, ds, (uint)data.RenderRes.x, (uint)data.RenderRes.y);
                }
                cmd.EndSample("PathTrace");
            }
            else
            {
                cmd.BeginSample("PathTrace");
                {
                    var ds = data.RefDs;
                    BindCommonRT(ds, ctx, &miniConst, tlas);
                    BindLightBuffers(ds, ctx);

                    ds.SetRWTexture("u_OutputColor",   res.OutputColor.NativePtr);
                    ds.SetRWTexture("u_Throughput",    res.Throughput.NativePtr);
                    ds.SetRWTexture("u_MotionVectors", res.ScreenMotionVectors.NativePtr);
                    ds.SetRWTexture("u_Depth",         res.Depth.NativePtr);
                    ds.SetRWTexture("u_SpecularHitT",  res.SpecularHitT.NativePtr);

                    data.RefSP.Dispatch(cmd, ds, (uint)data.RenderRes.x, (uint)data.RenderRes.y);
                }
                cmd.EndSample("PathTrace");
            }
        }

        // ── Binding helpers ────────────────────────────────────────────────────

        private static unsafe void BindCommonRT(
            NativeRayTraceDescriptorSet ds,
            RtxptPassContext ctx,
            SampleMiniConstants* miniConst,
            RayTracingAccelerationStructure tlas)
        {
            ds.SetConstantBuffer("g_Const",     ctx.ConstantBuffer);
            ds.SetRootConstants("g_MiniConst", miniConst);
            ds.SetAccelerationStructure("SceneBVH", tlas);

            ctx.GpuScene.BindToShader(ds);

            var envCubePtr = ctx.BakedEnvCubePtr;
            ds.SetTexture("t_EnvironmentMap", envCubePtr);

            var envLookupPtr = ctx.EnvLightLookupMapPtr;
            ds.SetTexture("t_EnvLookupMap", envLookupPtr);

            ds.SetRWStructuredBuffer("u_FeedbackBuffer",
                ctx.Buffers.FeedbackBufferPtr,
                ctx.Buffers.FeedbackBuffer.count,
                ctx.Buffers.FeedbackBuffer.stride);

            // ShaderDebug raw buffer (u125) + picked-pixel debug lines (u52, ENABLE_DEBUG_LINES_VIZ).
            // Both no-op when absent from the shader reflection or not allocated.
            if (ctx.Buffers.ShaderDebugBufferPtr != IntPtr.Zero)
                ds.SetRWBuffer("u_ShaderDebugBuffer", ctx.Buffers.ShaderDebugBufferPtr);
            if (ctx.Buffers.DebugLinesBufferPtr != IntPtr.Zero)
                ds.SetRWStructuredBuffer("u_DebugLinesBuffer", ctx.Buffers.DebugLinesBufferPtr,
                    RtxptBufferResources.MaxDebugLines, RtxptBufferResources.DebugLineStructSize);
        }

        private static void BindLightBuffers(NativeRayTraceDescriptorSet ds, RtxptPassContext ctx)
        {
            var buf = ctx.Buffers;
            if (buf == null) return;

            ds.SetStructuredBuffer("t_LightsCB", buf.LightControlBuffer, buf.LightControlBuffer.count, buf.LightControlBuffer.stride);
            ds.SetStructuredBuffer("t_Lights",   buf.LightBuffer,        buf.LightBuffer.count,        buf.LightBuffer.stride);

            ds.SetTypedBuffer("t_LightProxyCounters",       buf.LightProxyCountersPtr,   buf.LightProxyCounters.count,   (uint)Nri.DXGI_FORMAT.DXGI_FORMAT_R32_UINT);
            ds.SetTypedBuffer("t_LightProxyIndices",        buf.LightSamplingProxiesPtr, buf.LightSamplingProxies.count, (uint)Nri.DXGI_FORMAT.DXGI_FORMAT_R32_UINT);
            ds.SetTypedBuffer("t_LightLocalSamplingBuffer", buf.LocalSamplingBufferPtr,  buf.LocalSamplingBuffer.count,  (uint)Nri.DXGI_FORMAT.DXGI_FORMAT_R32_UINT);

            ds.SetStructuredBuffer("t_LightsEx", buf.LightExBuffer, buf.LightExBuffer.count, buf.LightExBuffer.stride);

            var tex = ctx.Textures;
            ds.SetRWTexture("u_LightFeedbackTotalWeight", tex.LightFeedbackTotalWeight.NativePtr);
            ds.SetRWTexture("u_LightFeedbackCandidates",  tex.LightFeedbackCandidates.NativePtr);
        }
    }
}
