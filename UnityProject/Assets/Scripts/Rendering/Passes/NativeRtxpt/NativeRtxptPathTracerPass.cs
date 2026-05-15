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
    /// Phase 2: RTXPT main path-tracing pass.
    ///
    /// Realtime mode: dispatches <b>BuildStablePlanes</b> then <b>FillStablePlanes</b>.
    /// Reference mode: dispatches <b>Reference</b> (accumulation-friendly variant).
    ///
    /// Binding names follow BuildStablePlanes/FillStablePlanes/Reference reflection JSONs.
    /// </summary>
    public class NativeRtxptPathTracerPass : ScriptableRenderPass, IDisposable
    {
        private readonly RayTracePipeline            _buildSP;
        private readonly NativeRayTraceDescriptorSet _buildDs;

        private readonly RayTracePipeline            _fillSP;
        private readonly NativeRayTraceDescriptorSet _fillDs;

        private readonly RayTracePipeline            _refSP;
        private readonly NativeRayTraceDescriptorSet _refDs;

        private NativeRtxptPassContext _ctx;

        private readonly SampleMiniConstants[] _miniConstArray = new SampleMiniConstants[1];
        private          GraphicsBuffer         _miniConstBuffer;

        public NativeRtxptPathTracerPass(
            RayTraceShader buildStablePlanes,
            RayTraceShader fillStablePlanes,
            RayTraceShader reference)
        {
            _buildSP = new RayTracePipeline(buildStablePlanes);
            _buildDs = new NativeRayTraceDescriptorSet(_buildSP);

            _fillSP = new RayTracePipeline(fillStablePlanes);
            _fillDs = new NativeRayTraceDescriptorSet(_fillSP);

            _refSP = new RayTracePipeline(reference);
            _refDs = new NativeRayTraceDescriptorSet(_refSP);

            _miniConstBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Constant, 1,
                Marshal.SizeOf<SampleMiniConstants>())
            { name = "Rtxpt_MiniConst" };
        }

        public void Dispose()
        {
            _buildDs?.Dispose(); _buildSP?.Dispose();
            _fillDs?.Dispose();  _fillSP?.Dispose();
            _refDs?.Dispose();   _refSP?.Dispose();
            _miniConstBuffer?.Dispose();
            _miniConstBuffer = null;
        }

        public void Setup(NativeRtxptPassContext ctx)
        {
            _ctx = ctx;
        }

        // ── Pass data ──────────────────────────────────────────────────────────

        private class PassData
        {
            internal RayTracePipeline            BuildSP, FillSP, RefSP;
            internal NativeRayTraceDescriptorSet BuildDs, FillDs, RefDs;
            internal NativeRtxptPassContext      Ctx;
            internal GraphicsBuffer              MiniConstBuffer;
            internal int2                        RenderRes;
            internal bool                        IsRealtime;
        }

        // ── RenderGraph ────────────────────────────────────────────────────────

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxpt.PathTracer", out var passData);

            passData.BuildSP        = _buildSP;
            passData.BuildDs        = _buildDs;
            passData.FillSP         = _fillSP;
            passData.FillDs         = _fillDs;
            passData.RefSP          = _refSP;
            passData.RefDs          = _refDs;
            passData.Ctx            = _ctx;
            passData.MiniConstBuffer = _miniConstBuffer;
            passData.RenderRes      = _ctx.RenderResolution;
            passData.IsRealtime     = _ctx.Setting.realtimeMode;

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

            // ── Upload g_MiniConst ─────────────────────────────────────────────
            // params0x = sub-sample index (sampleIndex), params0y = pingPong flag
            var mini = new SampleMiniConstants
            {
                params0x = (uint)ctx.FrameState.frameIndex,
                params0y = (uint)(ctx.FrameState.frameIndex & 1),
            };
            data.MiniConstBuffer.SetData(new[] { mini });

            var tlas = ctx.NrdSampleResource?.WorldAS;

            if (data.IsRealtime)
            {
                // ── Phase 2a: BuildStablePlanes ────────────────────────────────
                cmd.BeginSample("Rtxpt.BuildStablePlanes");
                {
                    var ds = data.BuildDs;
                    BindCommonRT(ds, ctx, data.MiniConstBuffer, tlas);

                    // BuildStablePlanes-specific outputs
                    if (res.Throughput.IsCreated)       ds.SetRWTexture("u_Throughput",         res.Throughput.NativePtr);
                    if (res.ScreenMotionVectors.IsCreated) ds.SetRWTexture("u_MotionVectors",   res.ScreenMotionVectors.NativePtr);
                    if (res.Depth.IsCreated)            ds.SetRWTexture("u_Depth",              res.Depth.NativePtr);
                    if (res.StablePlanesHeader.IsCreated)  ds.SetRWTexture("u_StablePlanesHeader", res.StablePlanesHeader.NativePtr);
                    if (res.StableRadiance.IsCreated)   ds.SetRWTexture("u_StableRadiance",     res.StableRadiance.NativePtr);
                    if (res.SpecularHitT.IsCreated)     ds.SetRWTexture("u_SpecularHitT",       res.SpecularHitT.NativePtr);
                    if (buf.StablePlanesBuffer != null)
                        ds.SetRWStructuredBuffer("u_StablePlanesBuffer",
                            buf.StablePlanesBuffer.GetNativeBufferPtr(),
                            buf.StablePlanesBuffer.count, buf.StablePlanesBuffer.stride);

                    data.BuildSP.Dispatch(cmd, ds, (uint)data.RenderRes.x, (uint)data.RenderRes.y);
                }
                cmd.EndSample("Rtxpt.BuildStablePlanes");

                // ── Phase 2b: FillStablePlanes ─────────────────────────────────
                cmd.BeginSample("Rtxpt.FillStablePlanes");
                {
                    var ds = data.FillDs;
                    BindCommonRT(ds, ctx, data.MiniConstBuffer, tlas);
                    BindLightBuffers(ds, ctx);

                    if (res.StablePlanesHeader.IsCreated)  ds.SetRWTexture("u_StablePlanesHeader", res.StablePlanesHeader.NativePtr);
                    if (res.SpecularHitT.IsCreated)     ds.SetRWTexture("u_SpecularHitT",       res.SpecularHitT.NativePtr);
                    if (buf.StablePlanesBuffer != null)
                        ds.SetRWStructuredBuffer("u_StablePlanesBuffer",
                            buf.StablePlanesBuffer.GetNativeBufferPtr(),
                            buf.StablePlanesBuffer.count, buf.StablePlanesBuffer.stride);

                    data.FillSP.Dispatch(cmd, ds, (uint)data.RenderRes.x, (uint)data.RenderRes.y);
                }
                cmd.EndSample("Rtxpt.FillStablePlanes");
            }
            else
            {
                // ── Reference mode ─────────────────────────────────────────────
                cmd.BeginSample("Rtxpt.Reference");
                {
                    var ds = data.RefDs;
                    BindCommonRT(ds, ctx, data.MiniConstBuffer, tlas);
                    BindLightBuffers(ds, ctx);

                    if (res.OutputColor.IsCreated)      ds.SetRWTexture("u_OutputColor",        res.OutputColor.NativePtr);
                    if (res.Throughput.IsCreated)       ds.SetRWTexture("u_Throughput",         res.Throughput.NativePtr);
                    if (res.ScreenMotionVectors.IsCreated) ds.SetRWTexture("u_MotionVectors",   res.ScreenMotionVectors.NativePtr);
                    if (res.Depth.IsCreated)            ds.SetRWTexture("u_Depth",              res.Depth.NativePtr);
                    if (res.SpecularHitT.IsCreated)     ds.SetRWTexture("u_SpecularHitT",       res.SpecularHitT.NativePtr);

                    data.RefSP.Dispatch(cmd, ds, (uint)data.RenderRes.x, (uint)data.RenderRes.y);
                }
                cmd.EndSample("Rtxpt.Reference");
            }
        }

        // ── Binding helpers ────────────────────────────────────────────────────

        private static void BindCommonRT(
            NativeRayTraceDescriptorSet ds,
            NativeRtxptPassContext ctx,
            GraphicsBuffer miniConstBuffer,
            RayTracingAccelerationStructure tlas)
        {
            if (ctx.ConstantBuffer != null)
                ds.SetConstantBuffer("g_Const", ctx.ConstantBuffer.GetNativeBufferPtr());

            if (miniConstBuffer != null)
                ds.SetConstantBuffer("g_MiniConst", miniConstBuffer.GetNativeBufferPtr());

            if (tlas != null)
                ds.SetAccelerationStructure("SceneBVH", tlas);

            ctx.GpuScene?.BindToShader(ds);
        }

        private static void BindLightBuffers(NativeRayTraceDescriptorSet ds, NativeRtxptPassContext ctx)
        {
            var buf = ctx.Buffers;
            if (buf == null) return;

            if (buf.LightControlBuffer != null)
                ds.SetBuffer("t_LightsCB",
                    buf.LightControlBuffer.GetNativeBufferPtr());

            if (buf.LightBuffer != null)
                ds.SetStructuredBuffer("t_Lights",
                    buf.LightBuffer.GetNativeBufferPtr(),
                    buf.LightBuffer.count, buf.LightBuffer.stride);

            if (buf.LightProxyCounters != null)
                ds.SetBuffer("t_LightProxyCounters",
                    buf.LightProxyCounters.GetNativeBufferPtr());

            if (buf.LocalSamplingBuffer != null)
                ds.SetBuffer("t_LightLocalSamplingBuffer",
                    buf.LocalSamplingBuffer.GetNativeBufferPtr());
        }
    }
}
