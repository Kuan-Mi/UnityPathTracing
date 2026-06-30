using System;
using PathTracing;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Pushes this frame's DLSS-G inputs (native depth/mvec pointers + camera constants) to
    /// the StreamlinePlugin plugin via <c>IssuePluginEventAndData</c>, on the render thread in
    /// command-stream order, reusing the frame token <c>SLStreamlineFrameLoop</c>'s begin tick minted
    /// earlier this frame. Mirror of <see cref="NrdDlssgInputsPass"/> (which targets the legacy
    /// StreamlineProbePlugin). The data pointer must stay valid until the render thread runs the
    /// event — back it with a persistent ring buffer in the feature.
    /// </summary>
    public class SLDlssgInputsPass : ScriptableRenderPass
    {
        private const int EventId = 0;

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

            using var builder = renderGraph.AddUnsafePass<PassData>("Streamline DLSS-G Inputs (SL)", out var passData);

            passData.EventFunc = _eventFunc;
            passData.DataPtr   = _dataPtr;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }
    }
}
