using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Profiling;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    public class GenerateMipsPass : ScriptableRenderPass
    {
        private readonly ComputeShader _genMipsCs;
        private Resource _resource;
        private Settings _settings;

        public GenerateMipsPass(ComputeShader cs)
        {
            _genMipsCs = cs;
        }

        public void Setup(Resource resource, Settings settings)
        {
            _resource = resource;
            _settings = settings;
        }

        public class Resource
        {
            // 需要生成 Mip 的 PDF 纹理
            internal RTHandle u_LocalLightPdfTexture;
        }

        public class Settings
        {
            // 可以放一些配置，比如是否开启调试
           public int mipCount;
           public int width;
           public int height;
        }

        class PassData
        {
            internal ComputeShader GenMipsCs;
            internal Resource Resource;
            internal Settings Settings;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var marker = new ProfilerMarker(ProfilerCategory.Render, "RtxdiGenerateMips", MarkerFlags.SampleGPU);

            natCmd.BeginSample(marker);

            var settings = data.Settings;
            var tex = data.Resource.u_LocalLightPdfTexture;
            int kernel = data.GenMipsCs.FindKernel("CSMain");

            
            natCmd.SetComputeTextureParam(data.GenMipsCs, kernel, "_SourceMip", tex, 0);
            
            
            // 逐级生成 Mip：从 Mip 0 到 Mip 1, 然后 Mip 1 到 Mip 2...
            for (int srcMip = 0; srcMip < settings.mipCount - 1; srcMip++)
            {
                int destMip = srcMip + 1;
                int destWidth = Mathf.Max(1, settings.width >> destMip);
                int destHeight = Mathf.Max(1, settings.height >> destMip);

                // 1. 设置源 Mip 层级和目标尺寸
                natCmd.SetComputeIntParam(data.GenMipsCs, "_SrcMipLevel", srcMip);
                natCmd.SetComputeVectorParam(data.GenMipsCs, "_TargetSize", new Vector4(destWidth, destHeight, 0, 0));

                // 2. 绑定目标 Mip (RWTexture2D 必须指定特定的 mipLevel)
                // UAV (RWTexture2D) 绑定特定 mip 是有效的，这会把该 mip 映射到 Shader 的 [0,0] 坐标系
                natCmd.SetComputeTextureParam(data.GenMipsCs, kernel, "_TargetMip", tex, destMip);

                int threadGroupsX = (destWidth + 7) / 8;
                int threadGroupsY = (destHeight + 7) / 8;

                natCmd.DispatchCompute(data.GenMipsCs, kernel, threadGroupsX, threadGroupsY, 1);
            }

            natCmd.EndSample(marker);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("RtxdiGenerateMips", out var passData);

            passData.GenMipsCs = _genMipsCs;
            passData.Resource = _resource;
            passData.Settings = _settings;

            builder.AllowPassCulling(false);

            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => 
            { 
                ExecutePass(data, context); 
            });
        }
    }
}