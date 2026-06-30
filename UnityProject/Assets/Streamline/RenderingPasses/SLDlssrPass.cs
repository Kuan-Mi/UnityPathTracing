using System;
using System.Runtime.InteropServices;
using PathTracing;
using PathTracing.Profiling;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Issues the DLSS Super Resolution evaluate through Streamline (SL), the mirror of
    /// <see cref="DlssSRPass"/> which goes through the NRI <c>Denoiser</c> plugin, and the
    /// upscaling counterpart of <see cref="SLDlssrrPass"/>. Backed by the native
    /// <c>StreamlinePlugin</c> DLL; the data pointer is a packed
    /// <see cref="PathTracing.NativeInterop.Streamline.SLDlssr.GetInteropDataPtr"/> ring-buffer entry.
    /// </summary>
    public class SLDlssrPass : ScriptableRenderPass
    {
        private IntPtr _dataPtr;

        public SLDlssrPass()
        {
        }

        public void Setup(IntPtr dataPtr)
        {
            _dataPtr = dataPtr;
        }

        private class PassData
        {
            public IntPtr SRDataPtr;
        }

        [DllImport("StreamlinePlugin")]
        private static extern IntPtr GetSLDlssrRenderEventAndDataFunc();

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            var marker = RenderPassMarkers.DlssUpscale;
            natCmd.BeginSample(marker);
            // eventId is ignored by the native side (single feature); pass 0.
            natCmd.IssuePluginEventAndData(GetSLDlssrRenderEventAndDataFunc(), 0, data.SRDataPtr);
            natCmd.EndSample(marker);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("DLSS SR (SL)", out var passData);

            passData.SRDataPtr = _dataPtr;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }
    }
}