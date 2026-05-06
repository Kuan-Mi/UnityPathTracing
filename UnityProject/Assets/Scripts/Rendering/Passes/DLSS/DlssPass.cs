using System;
using System.Runtime.InteropServices;
using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

using static PathTracing.ShaderIDs;

namespace PathTracing
{
    public class DlssPass : ScriptableRenderPass
    {
        private IntPtr _dataPtr;
        private Settings _settings;

        public DlssPass()
        {
        }

        public void Setup(IntPtr dataPtr, Settings settings)
        {
            _dataPtr  = dataPtr;
            _settings = settings;
        }

        public class Settings
        {
            internal bool disabled;
        }

        class PassData
        {
            internal Settings Setting;
            internal IntPtr    DataPtr;
        }

        [DllImport("Denoiser")]
        private static extern IntPtr GetRenderEventAndDataFunc();

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            if (data.Setting.disabled)
                return;

            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            natCmd.BeginSample(RenderPassMarkers.DlssUpscale);
            natCmd.IssuePluginEventAndData(GetRenderEventAndDataFunc(), 3, data.DataPtr);
            natCmd.EndSample(RenderPassMarkers.DlssUpscale);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("DLSS SR", out var passData);

            passData.Setting = _settings;
            passData.DataPtr = _dataPtr;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}
