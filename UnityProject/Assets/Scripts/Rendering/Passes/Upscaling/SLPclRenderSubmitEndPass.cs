using System;
using PathTracing;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    public class SLPclRenderSubmitEndPass : ScriptableRenderPass
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

            using var builder = renderGraph.AddUnsafePass<PassData>("Streamline PCL Render Submit End", out var passData);

            passData.EventFunc  = _eventFunc;
            passData.FrameToken = _frameToken;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }
    }
}