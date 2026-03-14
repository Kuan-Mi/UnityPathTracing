using System;
using System.Runtime.InteropServices;
using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace DefaultNamespace
{
    public class DemoRenderPass : ScriptableRenderPass
    {
        public DemoResource demoResource;
        
        [DllImport("UnityRTXDI")]
        private static extern IntPtr GetRenderEventAndDataFunc();


        class PassData
        {
            internal TextureHandle CameraTexture;
            // internal TextureHandle outputTexture;
            internal IntPtr DataPtr;
        }


        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var prepareLightMarker = new ProfilerMarker(ProfilerCategory.Render, "PrepareLight", MarkerFlags.SampleGPU);

            
            
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            
            natCmd.BeginSample(prepareLightMarker);
            natCmd.IssuePluginEventAndData(GetRenderEventAndDataFunc(), 1, data.DataPtr);
            natCmd.EndSample(prepareLightMarker);
            
            natCmd.SetRenderTarget(data.CameraTexture);
            
            // Blitter.BlitTexture (natCmd, data.outputTexture, new Vector4(1, 1, 0, 0),0,true);
            
        }


        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            
            using var builder = renderGraph.AddUnsafePass<PassData>("Demo Pass", out var passData);

            passData.DataPtr = demoResource.GetInteropDataPtr();

            var resourceData = frameData.Get<UniversalResourceData>();
            passData.CameraTexture = resourceData.activeColorTexture;
            // passData.outputTexture = renderGraph.ImportTexture(demoResource.GetOutputTexture());
            
            // builder.UseTexture(passData.outputTexture, AccessFlags.ReadWrite);
            builder.UseTexture(passData.CameraTexture, AccessFlags.Write);
            
            builder.AllowPassCulling(false);
            
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}