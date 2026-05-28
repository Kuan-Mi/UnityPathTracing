using System;
using NativeRender;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Phase 2c: LightingUpdateEnd — mirrors LightsBaker::UpdateFrameEnd() in original RTXPT.
    ///
    /// Runs between ExportVisibilityBuffer and FillStablePlanes.  By this point
    /// Depth and MotionVectors have been written by BuildStablePlanes / ExportVisibilityBuffer.
    ///
    /// Sub-passes (in order):
    ///   1. ProcessFeedbackHistoryP1a  — low-res temporal blend (blended resolution)
    ///   2. ProcessFeedbackHistoryP1b  — full-res temporal blend (render resolution)
    ///   3. ProcessFeedbackHistoryP2   — per-tile local sampling buffer construction
    ///   4. ProcessFeedbackHistoryP3   — history depth / confidence update (one group per tile)
    ///   5. ClearFeedbackHistory       — zero FeedbackTotalWeight + FeedbackCandidates
    ///                                   ready for the current frame's FillStablePlanes writes
    /// </summary>
    public class NativeRtxptLightingUpdateEndPass : ScriptableRenderPass, IDisposable
    {
        // ── 2D tile dispatch size (matches LLB_NUM_COMPUTE_THREADS_2D in NEEATBaker.hlsli) ─
        private const int Threads2D = 8;

        // ── DXGI format constants ─────────────────────────────────────────────
        private const uint DXGI_FORMAT_R32_FLOAT = 41u;
        private const uint DXGI_FORMAT_R32_UINT  = 42u;

        // ── Stride for LightControlBuffer ─────────────────────────────────────
        private static readonly int StrideCtrl = System.Runtime.InteropServices.Marshal.SizeOf<RtxptLightingControlData>();

        // ── GPU pipelines ──────────────────────────────────────────────────────
        private readonly NativeComputePipeline      _p1aCs;
        private readonly NativeComputeDescriptorSet _p1aDs;
        private readonly NativeComputePipeline      _p1bCs;
        private readonly NativeComputeDescriptorSet _p1bDs;
        private readonly NativeComputePipeline      _p2Cs;
        private readonly NativeComputeDescriptorSet _p2Ds;
        private readonly NativeComputePipeline      _p3Cs;
        private readonly NativeComputeDescriptorSet _p3Ds;
        private readonly NativeComputePipeline      _clearCs;
        private readonly NativeComputeDescriptorSet _clearDs;

        // ── Per-frame state ────────────────────────────────────────────────────
        private NativeRtxptPassContext _ctx;

        // ── Constructor ────────────────────────────────────────────────────────

        public NativeRtxptLightingUpdateEndPass(
            NativeComputeShader processFeedbackHistoryP1aCs,
            NativeComputeShader processFeedbackHistoryP1bCs,
            NativeComputeShader processFeedbackHistoryP2Cs,
            NativeComputeShader processFeedbackHistoryP3Cs,
            NativeComputeShader clearFeedbackHistoryCs)
        {
            _p1aCs   = new NativeComputePipeline(processFeedbackHistoryP1aCs);
            _p1aDs   = new NativeComputeDescriptorSet(_p1aCs);
            _p1bCs   = new NativeComputePipeline(processFeedbackHistoryP1bCs);
            _p1bDs   = new NativeComputeDescriptorSet(_p1bCs);
            _p2Cs    = new NativeComputePipeline(processFeedbackHistoryP2Cs);
            _p2Ds    = new NativeComputeDescriptorSet(_p2Cs);
            _p3Cs    = new NativeComputePipeline(processFeedbackHistoryP3Cs);
            _p3Ds    = new NativeComputeDescriptorSet(_p3Cs);
            _clearCs = new NativeComputePipeline(clearFeedbackHistoryCs);
            _clearDs = new NativeComputeDescriptorSet(_clearCs);
        }

        // ── IDisposable ────────────────────────────────────────────────────────

        public void Dispose()
        {
            _p1aDs?.Dispose();
            _p1aCs?.Dispose();
            _p1bDs?.Dispose();
            _p1bCs?.Dispose();
            _p2Ds?.Dispose();
            _p2Cs?.Dispose();
            _p3Ds?.Dispose();
            _p3Cs?.Dispose();
            _clearDs?.Dispose();
            _clearCs?.Dispose();
        }

        // ── Setup ──────────────────────────────────────────────────────────────

        public void Setup(NativeRtxptPassContext ctx) => _ctx = ctx;

        // ── Pass data ──────────────────────────────────────────────────────────

        private class PassData
        {
            internal NativeComputePipeline      P1aCs, P1bCs, P2Cs, P3Cs, ClearCs;
            internal NativeComputeDescriptorSet P1aDs, P1bDs, P2Ds, P3Ds, ClearDs;
            internal NativeRtxptPassContext     Ctx;
        }

        // ── RenderGraph ────────────────────────────────────────────────────────

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxpt.LightingUpdateEnd", out var pd);

            pd.P1aCs   = _p1aCs;
            pd.P1aDs   = _p1aDs;
            pd.P1bCs   = _p1bCs;
            pd.P1bDs   = _p1bDs;
            pd.P2Cs    = _p2Cs;
            pd.P2Ds    = _p2Ds;
            pd.P3Cs    = _p3Cs;
            pd.P3Ds    = _p3Ds;
            pd.ClearCs = _clearCs;
            pd.ClearDs = _clearDs;
            pd.Ctx     = _ctx;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData d, UnsafeGraphContext c) => ExecutePass(d, c));
        }

        // ── Execute ────────────────────────────────────────────────────────────

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var ctx = data.Ctx;
            var buf = ctx.Buffers;

            cmd.BeginSample(RenderPassMarkers.RtxptLightingUpdateEnd);

            // Shared pointers (all cached — immutable for lifetime of buffer objects)
            var pCtrl    = buf.LightControlBufferPtr;
            int cCtrl    = buf.LightControlBuffer.count;
            var pWeights = buf.LightWeightsBufferPtr;
            int cWeights = buf.LightWeightsBuffer.count;
            var pLocal   = buf.LocalSamplingBufferPtr;
            int cLocal   = buf.LocalSamplingBuffer.count;

            // Render resolution and derived resolutions
            int2 renderRes = ctx.RenderResolution;

            // Blended-feedback resolution = ceil(renderRes / NEEAT_TILE_SIZE=2)
            int blendW = (renderRes.x + LightingConfig.RTXPT_NEEAT_EARLY_FEEDBACK_TILE_SIZE - 1) / LightingConfig.RTXPT_NEEAT_EARLY_FEEDBACK_TILE_SIZE;
            int blendH = (renderRes.y + LightingConfig.RTXPT_NEEAT_EARLY_FEEDBACK_TILE_SIZE - 1) / LightingConfig.RTXPT_NEEAT_EARLY_FEEDBACK_TILE_SIZE;

            // Local sampling resolution = ceil(renderRes / SAMPLING_BUFFER_TILE_SIZE=8)
            int localW = (renderRes.x + LightingConfig.RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE - 1) / LightingConfig.RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE;
            int localH = (renderRes.y + LightingConfig.RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE - 1) / LightingConfig.RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE;

            // ----------------------------------------------------------------
            // 1. ProcessFeedbackHistoryP1a
            //    Aggregates blended early-feedback into scratch surfaces.
            //    Dispatch: ceil(blendW/8) x ceil(blendH/8)
            // ----------------------------------------------------------------
            {
                var ds = data.P1aDs;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWTexture("u_feedbackTotalWeight", ctx.FeedbackTotalWeightPtr);
                ds.SetRWTexture("u_feedbackCandidates", ctx.FeedbackCandidatesPtr);
                ds.SetRWTexture("u_feedbackTotalWeightScratch", ctx.FeedbackTotalWeightScratchPtr);
                ds.SetRWTexture("u_feedbackCandidatesScratch", ctx.FeedbackCandidatesScratchPtr);
                ds.SetRWTexture("u_feedbackTotalWeightBlended", ctx.FeedbackTotalWeightBlendedPtr);
                ds.SetRWTexture("u_feedbackCandidatesBlended", ctx.FeedbackCandidatesBlendedPtr);
                ds.SetRWTexture("u_ShaderDebugVizTextureBuffer", ctx.ShaderDebugVizPtr);
                ds.SetRWTypedBuffer("u_lightSamplingProxies", buf.LightSamplingProxiesPtr, buf.LightSamplingProxies.count, DXGI_FORMAT_R32_UINT);

                ds.SetRWTexture("u_historyDepth", ctx.NEEATHistoryDepthPtr);
                ds.SetTexture("t_depthBuffer", ctx.DepthPtr);
                ds.SetTexture("t_motionVectors", ctx.MotionVectorsPtr);
                uint gx = (uint)Math.Max(1, (blendW + Threads2D - 1) / Threads2D);
                uint gy = (uint)Math.Max(1, (blendH + Threads2D - 1) / Threads2D);
                cmd.BeginSample(RenderPassMarkers.RtxptProcessFeedbackHistoryP1a);
                data.P1aCs.Dispatch(cmd, ds, gx, gy, 1);
                cmd.EndSample(RenderPassMarkers.RtxptProcessFeedbackHistoryP1a);
            }

            // ----------------------------------------------------------------
            // 2. ProcessFeedbackHistoryP1b
            //    Full-res temporal blend from scratch into main surfaces.
            //    Dispatch: ceil(renderW/8) x ceil(renderH/8)
            // ----------------------------------------------------------------
            {
                var ds = data.P1bDs;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWTexture("u_feedbackTotalWeight", ctx.FeedbackTotalWeightPtr);
                ds.SetRWTexture("u_feedbackCandidates", ctx.FeedbackCandidatesPtr);
                ds.SetRWTexture("u_feedbackTotalWeightScratch", ctx.FeedbackTotalWeightScratchPtr);
                ds.SetRWTexture("u_feedbackCandidatesScratch", ctx.FeedbackCandidatesScratchPtr);
                ds.SetRWTexture("u_feedbackTotalWeightBlended", ctx.FeedbackTotalWeightBlendedPtr);
                ds.SetRWTexture("u_feedbackCandidatesBlended", ctx.FeedbackCandidatesBlendedPtr);
                ds.SetRWTexture("u_historyDepth", ctx.NEEATHistoryDepthPtr);
                ds.SetRWTexture("u_ShaderDebugVizTextureBuffer", ctx.ShaderDebugVizPtr);
                ds.SetRWTypedBuffer("u_historyRemapPastToCurrent", buf.HistoryRemapPastToCurrPtr, buf.HistoryRemapPastToCurrent.count, DXGI_FORMAT_R32_UINT);
                ds.SetRWTypedBuffer("u_lightSamplingProxies",      buf.LightSamplingProxiesPtr,   buf.LightSamplingProxies.count,   DXGI_FORMAT_R32_UINT);
                ds.SetRWTypedBuffer("u_localSamplingBuffer", pLocal, cLocal, DXGI_FORMAT_R32_UINT);


                ds.SetTexture("t_depthBuffer", ctx.DepthPtr);
                ds.SetTexture("t_motionVectors", ctx.MotionVectorsPtr);
                uint gx = (uint)Math.Max(1, (renderRes.x + Threads2D - 1) / Threads2D);
                uint gy = (uint)Math.Max(1, (renderRes.y + Threads2D - 1) / Threads2D);
                cmd.BeginSample(RenderPassMarkers.RtxptProcessFeedbackHistoryP1b);
                data.P1bCs.Dispatch(cmd, ds, gx, gy, 1);
                cmd.EndSample(RenderPassMarkers.RtxptProcessFeedbackHistoryP1b);
            }

            // ----------------------------------------------------------------
            // 3. ProcessFeedbackHistoryP2
            //    Builds per-tile local sampling buffer from blended feedback.
            //    Dispatch: ceil(localW/8) x ceil(localH/8)
            // ----------------------------------------------------------------
            {
                var ds = data.P2Ds;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWTypedBuffer("u_lightWeights", pWeights, cWeights, DXGI_FORMAT_R32_FLOAT);
                ds.SetRWTexture("u_feedbackTotalWeight", ctx.FeedbackTotalWeightPtr);
                ds.SetRWTexture("u_feedbackCandidates", ctx.FeedbackCandidatesPtr);
                ds.SetRWTexture("u_feedbackTotalWeightBlended", ctx.FeedbackTotalWeightBlendedPtr);
                ds.SetRWTexture("u_feedbackCandidatesBlended", ctx.FeedbackCandidatesBlendedPtr);
                ds.SetRWTexture("u_feedbackCandidatesScratch", ctx.FeedbackCandidatesScratchPtr);

                ds.SetRWTypedBuffer("u_localSamplingBuffer", pLocal, cLocal, DXGI_FORMAT_R32_UINT);
                uint gx = (uint)Math.Max(1, (localW + Threads2D - 1) / Threads2D);
                uint gy = (uint)Math.Max(1, (localH + Threads2D - 1) / Threads2D);
                cmd.BeginSample(RenderPassMarkers.RtxptProcessFeedbackHistoryP2);
                data.P2Cs.Dispatch(cmd, ds, gx, gy, 1);
                cmd.EndSample(RenderPassMarkers.RtxptProcessFeedbackHistoryP2);
            }

            // ----------------------------------------------------------------
            // 4. ProcessFeedbackHistoryP3
            //    History depth / confidence update, one thread-group per tile.
            //    numthreads = RTXPT_LIGHTING_LOCAL_PROXY_COUNT/2 = 64
            //    Dispatch: localW x localH x 1
            // ----------------------------------------------------------------
            {
                var ds = data.P3Ds;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWTexture("u_feedbackTotalWeight", ctx.FeedbackTotalWeightPtr);
                ds.SetRWTexture("u_feedbackCandidates", ctx.FeedbackCandidatesPtr);
                ds.SetRWTexture("u_historyDepth", ctx.NEEATHistoryDepthPtr);
                ds.SetRWTypedBuffer("u_localSamplingBuffer", pLocal, cLocal, DXGI_FORMAT_R32_UINT);
                cmd.BeginSample(RenderPassMarkers.RtxptProcessFeedbackHistoryP3);
                data.P3Cs.Dispatch(cmd, ds, (uint)Math.Max(1, localW), (uint)Math.Max(1, localH), 1);
                cmd.EndSample(RenderPassMarkers.RtxptProcessFeedbackHistoryP3);
            }

            // ----------------------------------------------------------------
            // 5. ClearFeedbackHistory
            //    Zeros FeedbackTotalWeight + FeedbackCandidates so the next
            //    FillStablePlanes can write fresh NEE samples.
            //    Dispatch: ceil(renderW/8) x ceil(renderH/8)
            // ----------------------------------------------------------------
            {
                var ds = data.ClearDs;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWTexture("u_feedbackTotalWeight", ctx.FeedbackTotalWeightPtr);
                ds.SetRWTexture("u_feedbackCandidates", ctx.FeedbackCandidatesPtr);
                ds.SetRWTexture("u_feedbackTotalWeightScratch", ctx.FeedbackTotalWeightScratchPtr);
                ds.SetRWTexture("u_feedbackCandidatesScratch", ctx.FeedbackCandidatesScratchPtr);
                ds.SetRWTexture("u_historyDepth", ctx.NEEATHistoryDepthPtr);
                ds.SetTexture("t_depthBuffer", ctx.DepthPtr);
                ds.SetRWTexture("u_ShaderDebugVizTextureBuffer", ctx.ShaderDebugVizPtr);


                uint gx = (uint)Math.Max(1, (renderRes.x + Threads2D - 1) / Threads2D);
                uint gy = (uint)Math.Max(1, (renderRes.y + Threads2D - 1) / Threads2D);
                cmd.BeginSample(RenderPassMarkers.RtxptClearFeedbackHistory);
                data.ClearCs.Dispatch(cmd, ds, gx, gy, 1);
                cmd.EndSample(RenderPassMarkers.RtxptClearFeedbackHistory);
            }

            cmd.EndSample(RenderPassMarkers.RtxptLightingUpdateEnd);
        }
    }
}
