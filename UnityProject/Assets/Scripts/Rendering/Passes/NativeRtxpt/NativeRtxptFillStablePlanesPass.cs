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
    public class NativeRtxptFillStablePlanesPass : ScriptableRenderPass, IDisposable
    {
        private readonly RayTracePipeline            _fillSP;
        private readonly NativeRayTraceDescriptorSet _fillDs;

        private readonly RayTracePipeline            _refSP;
        private readonly NativeRayTraceDescriptorSet _refDs;

        private          NativeRtxptPassContext _ctx;
        private readonly SampleMiniConstants[]  _miniConstArray = new SampleMiniConstants[1];
        private          GraphicsBuffer         _miniConstBuffer;
        private          IntPtr                 _miniConstBufferPtr;

        /// <summary>Pipeline handles exposed for NativeRtxptBuildTlasPass hit-table rebuilds.</summary>
        public RayTracePipeline FillPipeline => _fillSP;
        public RayTracePipeline RefPipeline  => _refSP;

        public NativeRtxptFillStablePlanesPass(
            RayTraceShader fillStablePlanes,
            RayTraceShader reference,
            HitGroupShader[] fillHitGroups      = null,
            HitGroupShader[] referenceHitGroups = null)
        {
            _fillSP = fillHitGroups is { Length: > 0 }
                ? new RayTracePipeline(fillStablePlanes, fillHitGroups)
                : new RayTracePipeline(fillStablePlanes);
            _fillDs = new NativeRayTraceDescriptorSet(_fillSP);

            _refSP = referenceHitGroups is { Length: > 0 }
                ? new RayTracePipeline(reference, referenceHitGroups)
                : new RayTracePipeline(reference);
            _refDs = new NativeRayTraceDescriptorSet(_refSP);

            _miniConstBuffer = new GraphicsBuffer(
                    GraphicsBuffer.Target.Constant, 1,
                    Marshal.SizeOf<SampleMiniConstants>())
                { name = "Rtxpt_MiniConst_Fill" };
            _miniConstBufferPtr = _miniConstBuffer.GetNativeBufferPtr();
        }

        public void Dispose()
        {
            _fillDs?.Dispose();
            _fillSP?.Dispose();
            _refDs?.Dispose();
            _refSP?.Dispose();
            _miniConstBuffer?.Dispose();
            _miniConstBuffer    = null;
            _miniConstBufferPtr = IntPtr.Zero;
        }

        public void Setup(NativeRtxptPassContext ctx) => _ctx = ctx;

        // ── Pass data ──────────────────────────────────────────────────────────

        private class PassData
        {
            internal RayTracePipeline            FillSP, RefSP;
            internal NativeRayTraceDescriptorSet FillDs, RefDs;
            internal NativeRtxptPassContext      Ctx;
            internal GraphicsBuffer              MiniConstBuffer;
            internal IntPtr                      MiniConstBufferPtr;
            internal int2                        RenderRes;
            internal bool                        IsRealtime;
        }

        // ── RenderGraph ────────────────────────────────────────────────────────

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxpt.FillStablePlanes", out var passData);

            passData.FillSP             = _fillSP;
            passData.FillDs             = _fillDs;
            passData.RefSP              = _refSP;
            passData.RefDs              = _refDs;
            passData.Ctx                = _ctx;
            passData.MiniConstBuffer    = _miniConstBuffer;
            passData.MiniConstBufferPtr = _miniConstBufferPtr;
            passData.RenderRes          = _ctx.RenderResolution;
            passData.IsRealtime         = _ctx.Setting.realtimeMode;

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

            // Upload g_MiniConst (same values as BuildStablePlanes, frame-consistent)
            data.MiniConstBuffer.SetData(new[]
            {
                new SampleMiniConstants
                {
                    params0x = ctx.FrameState.frameIndex,
                    params0y = (ctx.FrameState.frameIndex & 1),
                }
            });
            var miniConstBufferPtr = data.MiniConstBuffer.GetNativeBufferPtr();

            var tlas = ctx.GpuScene?.AccelerationStructure;

            if (data.IsRealtime)
            {
                cmd.BeginSample("Rtxpt.FillStablePlanes");
                {
                    var ds = data.FillDs;
                    BindCommonRT(ds, ctx, miniConstBufferPtr, tlas);
                    BindLightBuffers(ds, ctx);

                    ds.SetRWTexture("u_ShaderDebugVizTextureBuffer", res.ShaderDebugViz.NativePtr);
                    ds.SetRWTexture("u_DebugOutputColor",            res.DebugOutputColor.NativePtr);
                    ds.SetRWTexture("u_StablePlanesHeader",          res.StablePlanesHeader.NativePtr);
                    ds.SetRWTexture("u_SpecularHitT",                res.SpecularHitT.NativePtr);
                    ds.SetRWStructuredBuffer("u_StablePlanesBuffer", buf.StablePlanesBuffer);

                    data.FillSP.Dispatch(cmd, ds, (uint)data.RenderRes.x, (uint)data.RenderRes.y);
                }
                cmd.EndSample("Rtxpt.FillStablePlanes");
            }
            else
            {
                cmd.BeginSample("Rtxpt.Reference");
                {
                    var ds = data.RefDs;
                    BindCommonRT(ds, ctx, miniConstBufferPtr, tlas);
                    BindLightBuffers(ds, ctx);

                    ds.SetRWTexture("u_OutputColor",   res.OutputColor.NativePtr);
                    ds.SetRWTexture("u_Throughput",    res.Throughput.NativePtr);
                    ds.SetRWTexture("u_MotionVectors", res.ScreenMotionVectors.NativePtr);
                    ds.SetRWTexture("u_Depth",         res.Depth.NativePtr);
                    ds.SetRWTexture("u_SpecularHitT",  res.SpecularHitT.NativePtr);

                    data.RefSP.Dispatch(cmd, ds, (uint)data.RenderRes.x, (uint)data.RenderRes.y);
                }
                cmd.EndSample("Rtxpt.Reference");
            }
        }

        // ── Binding helpers ────────────────────────────────────────────────────

        private static void BindCommonRT(
            NativeRayTraceDescriptorSet ds,
            NativeRtxptPassContext ctx,
            IntPtr miniConstBufferPtr,
            RayTracingAccelerationStructure tlas)
        {
            ds.SetConstantBuffer("g_Const",     ctx.ConstantBufferPtr);
            ds.SetConstantBuffer("g_MiniConst", miniConstBufferPtr);
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
                ctx.Buffers.FeedbackBufferPtr,
                ctx.Buffers.FeedbackBuffer.count,
                ctx.Buffers.FeedbackBuffer.stride);
        }

        private static void BindLightBuffers(NativeRayTraceDescriptorSet ds, NativeRtxptPassContext ctx)
        {
            var buf = ctx.Buffers;
            if (buf == null) return;

            ds.SetStructuredBuffer("t_LightsCB", buf.LightControlBufferPtr, buf.LightControlBuffer.count, buf.LightControlBuffer.stride);
            ds.SetStructuredBuffer("t_Lights",   buf.LightBufferPtr,        buf.LightBuffer.count,        buf.LightBuffer.stride);

            ds.SetTypedBuffer("t_LightProxyCounters",       buf.LightProxyCountersPtr,   buf.LightProxyCounters.count,   (uint)Nri.DXGI_FORMAT.DXGI_FORMAT_R32_UINT);
            ds.SetTypedBuffer("t_LightProxyIndices",        buf.LightSamplingProxiesPtr, buf.LightSamplingProxies.count, (uint)Nri.DXGI_FORMAT.DXGI_FORMAT_R32_UINT);
            ds.SetTypedBuffer("t_LightLocalSamplingBuffer", buf.LocalSamplingBufferPtr,  buf.LocalSamplingBuffer.count,  (uint)Nri.DXGI_FORMAT.DXGI_FORMAT_R32_UINT);

            ds.SetStructuredBuffer("t_LightsEx", buf.LightExBuffer);

            var tex = ctx.Textures;
            ds.SetRWTexture("u_LightFeedbackTotalWeight", tex.LightFeedbackTotalWeight.NativePtr);
            ds.SetRWTexture("u_LightFeedbackCandidates",  tex.LightFeedbackCandidates.NativePtr);
        }
    }
}
