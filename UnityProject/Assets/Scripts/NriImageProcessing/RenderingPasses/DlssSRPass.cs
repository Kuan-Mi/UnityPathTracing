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
    /// DEPRECATED — issues the NRI/NGX DLSS Super Resolution evaluate through the
    /// <c>Denoiser</c> plugin. Cannot run concurrently with DLSS-G frame generation; use
    /// <see cref="SLDlssrPass"/> (Streamline) instead. Retained for offline A/B only.
    /// </summary>
    [Obsolete("Use SLDlssrPass (Streamline) — the NRI DLSS-SR path cannot run concurrently with DLSS-G frame generation.")]
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
            public IntPtr SRDataPtr;
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