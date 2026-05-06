using System;
using System.Runtime.InteropServices;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    public class NisPass : ScriptableRenderPass
    {
        private IntPtr _dataPtr;

        public void Setup(IntPtr dataPtr)
        {
            _dataPtr = dataPtr;
        }

        class PassData
        {
            internal IntPtr NISDataPtr;
        }

        [DllImport("Denoiser")]
        private static extern IntPtr GetRenderEventAndDataFunc();

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            natCmd.BeginSample(RenderPassMarkers.NisSharpening);
            natCmd.IssuePluginEventAndData(GetRenderEventAndDataFunc(), 4, data.NISDataPtr);
            natCmd.EndSample(RenderPassMarkers.NisSharpening);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NIS", out var passData);

            passData.NISDataPtr = _dataPtr;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }
    }
}
