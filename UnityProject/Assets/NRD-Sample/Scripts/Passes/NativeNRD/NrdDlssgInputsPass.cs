using System;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Pushes this frame's real DLSS-G inputs (native texture pointers + camera constants) to
    /// the Streamline probe via <c>IssuePluginEventAndData</c>, so the data is consumed on the
    /// render thread in command-stream order — no cross-thread mutex, and it reuses the frame
    /// token <c>BeginFrame</c> minted earlier this frame. Mirrors <see cref="DlssRRPass"/>.
    ///
    /// The <see cref="Setup"/> data pointer must remain valid until the render thread runs the
    /// event; the feature backs it with a persistent ring buffer.
    /// </summary>
    public class NrdDlssgInputsPass : ScriptableRenderPass
    {
        private const int EventId = 0; // StreamlineProbe ignores the id (single event kind)

        private IntPtr _eventFunc;
        private IntPtr _dataPtr;

        public void Setup(IntPtr eventFunc, IntPtr dataPtr)
        {
            _eventFunc = eventFunc;
            _dataPtr   = dataPtr;
        }

        private class PassData
        {
            public IntPtr EventFunc;
            public IntPtr DataPtr;
        }

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            if (data.EventFunc == IntPtr.Zero || data.DataPtr == IntPtr.Zero) return;
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            natCmd.IssuePluginEventAndData(data.EventFunc, EventId, data.DataPtr);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            if (_eventFunc == IntPtr.Zero || _dataPtr == IntPtr.Zero) return;

            using var builder = renderGraph.AddUnsafePass<PassData>("Streamline DLSS-G Inputs", out var passData);

            passData.EventFunc = _eventFunc;
            passData.DataPtr   = _dataPtr;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }
    }
}
