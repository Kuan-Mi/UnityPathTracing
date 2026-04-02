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
    public class DlssRRPass : ScriptableRenderPass
    {

        private IntPtr DataPtr;
        private Settings _settings;

        public DlssRRPass()
        {
        }

        public void Setup(IntPtr DataPtr, Settings settings)
        {
            this.DataPtr = DataPtr;
            _settings = settings;
        }
 

        public class Resource
        {
        }

        public class Settings
        {
            internal bool tmpDisableRR;
        }

        class PassData
        {
            internal Settings Setting;
            internal IntPtr RRDataPtr;
        }

        [DllImport("RenderingPlugin")]
        private static extern IntPtr GetRenderEventAndDataFunc();

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            var dlssDenoiseMarker = RenderPassMarkers.DlssDenoise;

            // DLSS调用

            if (!data.Setting.tmpDisableRR)
            {
                natCmd.BeginSample(dlssDenoiseMarker);
                natCmd.IssuePluginEventAndData(GetRenderEventAndDataFunc(), 2, data.RRDataPtr);
                natCmd.EndSample(dlssDenoiseMarker);
            }
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("DLSS RR", out var passData);

            passData.Setting = _settings;
            passData.RRDataPtr = DataPtr;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}