using System;
using System.Runtime.InteropServices;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Issues the DLSS Ray Reconstruction evaluate through Streamline (SL), the mirror of
    /// <see cref="DlssRRPass"/> which goes through the NRI <c>Denoiser</c> plugin. Backed by
    /// the native <c>SLDenoiser</c> DLL; the data pointer is a packed
    /// <see cref="SLDLRR.SLDlssrr.GetInteropDataPtr"/> ring-buffer entry.
    /// </summary>
    public class SLDlssrrPass : ScriptableRenderPass
    {
        private IntPtr   DataPtr;
        private Settings _settings;

        public SLDlssrrPass()
        {
        }

        public void Setup(IntPtr dataPtr, Settings settings)
        {
            this.DataPtr = dataPtr;
            _settings    = settings;
        }

        public class Settings
        {
            public bool tmpDisableRR;
        }

        private class PassData
        {
            public Settings Setting;
            public IntPtr   RRDataPtr;
        }

        [DllImport("SLDenoiser")]
        private static extern IntPtr GetSLDlssrrRenderEventAndDataFunc();

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            var marker = RenderPassMarkers.DlssDenoise;

            if (!data.Setting.tmpDisableRR)
            {
                natCmd.BeginSample(marker);
                // eventId is ignored by the native side (single feature); pass 0.
                natCmd.IssuePluginEventAndData(GetSLDlssrrRenderEventAndDataFunc(), 0, data.RRDataPtr);
                natCmd.EndSample(marker);
            }
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("DLSS RR (SL)", out var passData);

            passData.Setting   = _settings;
            passData.RRDataPtr = DataPtr;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}
