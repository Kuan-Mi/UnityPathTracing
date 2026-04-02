using System;
using System.Runtime.InteropServices;
using RTXDI;
using Unity.Mathematics;
using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    public class PrepareLightPass : ScriptableRenderPass
    {
        private PrepareLightResource _prepareLightResource;

        [DllImport("UnityRTXDI")]
        private static extern IntPtr GetRenderEventAndDataFunc();

        public void Setup(PrepareLightResource prepareLightResource)
        {
            _prepareLightResource = prepareLightResource;
        }

        class PassData
        {
            internal IntPtr DataPtr;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            var prepareLightMarker = RenderPassMarkers.PrepareLight;

            natCmd.BeginSample(prepareLightMarker);
            natCmd.IssuePluginEventAndData(GetRenderEventAndDataFunc(), 1, data.DataPtr);
            natCmd.EndSample(prepareLightMarker);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("Prepare Light", out var passData);

            passData.DataPtr = _prepareLightResource.GetInteropDataPtr();

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}