using System;
using System.Runtime.InteropServices;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    public class DlssSRPass : ScriptableRenderPass
    {
        private IntPtr _dataPtr;

        public DlssSRPass()
        {
        }

        public void Setup(IntPtr dataPtr)
        {
            _dataPtr = dataPtr;
        }

        class PassData
        {
            internal IntPtr SRDataPtr;
        }

        [DllImport("Denoiser")]
        private static extern IntPtr GetRenderEventAndDataFunc();

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            natCmd.BeginSample(RenderPassMarkers.DlssUpscale);
            natCmd.IssuePluginEventAndData(GetRenderEventAndDataFunc(), 3, data.SRDataPtr);
            natCmd.EndSample(RenderPassMarkers.DlssUpscale);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("DLSS SR", out var passData);

            passData.SRDataPtr = _dataPtr;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }
    }
}
