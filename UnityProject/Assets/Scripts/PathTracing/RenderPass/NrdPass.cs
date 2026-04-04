using System;
using System.Runtime.InteropServices;
using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    public class NrdPass : ScriptableRenderPass
    {
        private IntPtr DataPtr;

        public void Setup(IntPtr DataPtr)
        {
            this.DataPtr = DataPtr;
        }

        class PassData
        {
            internal IntPtr DataPtr;
        }

        [DllImport("RenderingPlugin")]
        private static extern IntPtr GetRenderEventAndDataFunc();
        
        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            var nrdDenoiseMarker = RenderPassMarkers.NrdDenoise;

            natCmd.BeginSample(nrdDenoiseMarker);
            natCmd.IssuePluginEventAndData(GetRenderEventAndDataFunc(), 1, data.DataPtr);
            natCmd.EndSample(nrdDenoiseMarker);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("Nrd", out var passData);

            passData.DataPtr = DataPtr;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}