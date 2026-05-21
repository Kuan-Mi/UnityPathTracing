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
        private          GraphicsBuffer        _miniConstBuffer;

        public NativeRtxptPathTracerPass(
            RayTraceShader buildStablePlanes,
            RayTraceShader fillStablePlanes,
            RayTraceShader reference,
            HitGroupShader[] buildHitGroups = null,
            HitGroupShader[] fillHitGroups = null,
            HitGroupShader[] referenceHitGroups = null)
        {
            _buildSP = buildHitGroups is { Length: > 0 }
                ? new RayTracePipeline(buildStablePlanes, buildHitGroups)
                : new RayTracePipeline(buildStablePlanes);
            _buildDs = new NativeRayTraceDescriptorSet(_buildSP);

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
                { name = "Rtxpt_MiniConst" };
        }

        public void Dispose()
        {
            _buildDs?.Dispose();
            _buildSP?.Dispose();
            _fillDs?.Dispose();
            _fillSP?.Dispose();
            _refDs?.Dispose();
            _refSP?.Dispose();
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

            passData.BuildSP         = _buildSP;
            passData.BuildDs         = _buildDs;
            passData.FillSP          = _fillSP;
            passData.FillDs          = _fillDs;
            passData.RefSP           = _refSP;
            passData.RefDs           = _refDs;
            passData.Ctx             = _ctx;
            passData.MiniConstBuffer = _miniConstBuffer;
            passData.RenderRes       = _ctx.RenderResolution;
            passData.IsRealtime      = _ctx.Setting.realtimeMode;

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
                params0x = ctx.FrameState.frameIndex,
                params0y = (ctx.FrameState.frameIndex & 1),
            };
            data.MiniConstBuffer.SetData(new[] { mini });

            var tlas = ctx.GpuScene?.AccelerationStructure;

            if (data.IsRealtime)
            {
                // ── Phase 2a: BuildStablePlanes ────────────────────────────────
                cmd.BeginSample("Rtxpt.BuildStablePlanes");
                {
                    var ds = data.BuildDs;
                    BindCommonRT(ds, ctx, data.MiniConstBuffer, tlas);

                    // BuildStablePlanes-specific outputs
                    ds.SetRWTexture("u_Throughput", res.Throughput.NativePtr);
                    ds.SetRWTexture("u_MotionVectors", res.ScreenMotionVectors.NativePtr);
                    ds.SetRWTexture("u_Depth", res.Depth.NativePtr);
                    ds.SetRWTexture("u_StablePlanesHeader", res.StablePlanesHeader.NativePtr);
                    ds.SetRWTexture("u_StableRadiance", res.StableRadiance.NativePtr);
                    ds.SetRWTexture("u_SpecularHitT", res.SpecularHitT.NativePtr);
                    
                    ds.SetRWStructuredBuffer("u_StablePlanesBuffer", buf.StablePlanesBuffer);

                    data.BuildSP.Dispatch(cmd, ds, (uint)data.RenderRes.x, (uint)data.RenderRes.y);
                }
                cmd.EndSample("Rtxpt.BuildStablePlanes");

                // ── Phase 2b: FillStablePlanes ─────────────────────────────────
                cmd.BeginSample("Rtxpt.FillStablePlanes");
                {
                    var ds = data.FillDs;
                    BindCommonRT(ds, ctx, data.MiniConstBuffer, tlas);
                    BindLightBuffers(ds, ctx);

                    ds.SetRWTexture("u_StablePlanesHeader", res.StablePlanesHeader.NativePtr);
                    ds.SetRWTexture("u_SpecularHitT", res.SpecularHitT.NativePtr);
                    ds.SetRWStructuredBuffer("u_StablePlanesBuffer", buf.StablePlanesBuffer);

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

                    ds.SetRWTexture("u_OutputColor", res.OutputColor.NativePtr);
                    ds.SetRWTexture("u_Throughput", res.Throughput.NativePtr);
                    ds.SetRWTexture("u_MotionVectors", res.ScreenMotionVectors.NativePtr);
                    ds.SetRWTexture("u_Depth", res.Depth.NativePtr);
                    ds.SetRWTexture("u_SpecularHitT", res.SpecularHitT.NativePtr);

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
            ds.SetConstantBuffer("g_Const", ctx.ConstantBuffer.GetNativeBufferPtr());
            ds.SetConstantBuffer("g_MiniConst", miniConstBuffer.GetNativeBufferPtr());
            ds.SetAccelerationStructure("SceneBVH", tlas);

            ctx.GpuScene.BindToShader(ds);

            // t_EnvironmentMap (t10): bind configured env map or fallback to black texture
            var envMap = ctx.Setting?.environmentMap != null
                ? ctx.Setting.environmentMap
                : Texture2D.blackTexture;
            ds.SetTexture("t_EnvironmentMap", envMap.GetNativeTexturePtr());

            // t_EnvLookupMap (t18): bind configured LUT or fallback to white texture
            var envLut = ctx.Setting?.environmentLookupMap != null
                ? ctx.Setting.environmentLookupMap
                : Texture2D.whiteTexture;
            ds.SetTexture("t_EnvLookupMap", envLut.GetNativeTexturePtr());

            // u_FeedbackBuffer (u51): debug stub buffer
            ds.SetRWStructuredBuffer("u_FeedbackBuffer",
                ctx.Buffers.FeedbackBuffer.GetNativeBufferPtr(),
                ctx.Buffers.FeedbackBuffer.count,
                ctx.Buffers.FeedbackBuffer.stride);
        }

        private static void BindLightBuffers(NativeRayTraceDescriptorSet ds, NativeRtxptPassContext ctx)
        {
            var buf = ctx.Buffers;
            if (buf == null) return;

            ds.SetBuffer("t_LightsCB", buf.LightControlBuffer.GetNativeBufferPtr());

            ds.SetStructuredBuffer("t_Lights", buf.LightBuffer.GetNativeBufferPtr(), buf.LightBuffer.count, buf.LightBuffer.stride);

            ds.SetBuffer("t_LightProxyCounters", buf.LightProxyCounters.GetNativeBufferPtr());

            ds.SetBuffer("t_LightLocalSamplingBuffer", buf.LocalSamplingBuffer.GetNativeBufferPtr());

            ds.SetStructuredBuffer("t_LightsEx", buf.LightExBuffer);
        }
    }
}