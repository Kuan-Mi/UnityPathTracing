using System;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Pins the SLDenoiser render/present side to the frame currently being rendered, by
    /// forwarding this frame's Streamline FrameToken (minted on the main thread at the top of
    /// the frame by <see cref="SLDLRR.SLStreamlineFrameLoop"/>) to the render thread via
    /// <c>IssuePluginEventAndData</c>. The native begin event latches it (so the present-thread
    /// PCL markers + DLSS-G tagging use the same token as the main-thread Reflex sleep / sim
    /// markers) and emits the eRenderSubmitStart PCL marker at this point (render begin).
    ///
    /// This is needed for Reflex/PCL on EVERY adapter, not just Frame Generation — the SL proxy
    /// present path (and its PCL markers) runs even when FG is unavailable. (DLSS-RR-via-SL does
    /// not need this latch; it carries its token directly in its frame-data struct.)
    ///
    /// Must run BEFORE the RR pass, the FG-inputs pass and present — enqueue it at
    /// <see cref="RenderPassEvent.BeforeRendering"/> (earlier than those, which sit at
    /// BeforeRenderingPostProcessing). Mirror of <see cref="SLDlssgInputsPass"/>; replaces the
    /// legacy out-of-graph CommandBuffer path that used to execute at beginContextRendering.
    /// </summary>
    public class SLPclRenderSubmitStartPass : ScriptableRenderPass
    {
        private const int EventId = 0;

        private IntPtr _eventFunc;
        private IntPtr _frameToken;

        public void Setup(IntPtr eventFunc, IntPtr frameToken)
        {
            _eventFunc  = eventFunc;
            _frameToken = frameToken;
        }

        private class PassData
        {
            public IntPtr EventFunc;
            public IntPtr FrameToken;
        }

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            if (data.EventFunc == IntPtr.Zero || data.FrameToken == IntPtr.Zero) return;
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            natCmd.IssuePluginEventAndData(data.EventFunc, EventId, data.FrameToken);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            if (_eventFunc == IntPtr.Zero || _frameToken == IntPtr.Zero) return;

            using var builder = renderGraph.AddUnsafePass<PassData>("Streamline PCL Render Submit Start", out var passData);

            passData.EventFunc  = _eventFunc;
            passData.FrameToken = _frameToken;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }
    }
}
