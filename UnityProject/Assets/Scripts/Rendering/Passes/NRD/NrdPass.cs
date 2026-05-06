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
        private IntPtr       DataPtr;
        private NamedMarker  _marker;

        public void Setup(IntPtr dataPtr, NamedMarker marker)
        {
            DataPtr  = dataPtr;
            _marker  = marker;
        }

        class PassData
        {
            internal IntPtr      DataPtr;
            internal NamedMarker Marker;
        }

        [DllImport("Denoiser")]
        private static extern IntPtr GetRenderEventAndDataFunc();
        
        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            natCmd.BeginSample(data.Marker);
            natCmd.IssuePluginEventAndData(GetRenderEventAndDataFunc(), 1, data.DataPtr);
            natCmd.EndSample(data.Marker);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("Nrd", out var passData);

            passData.DataPtr = DataPtr;
            passData.Marker  = _marker;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}