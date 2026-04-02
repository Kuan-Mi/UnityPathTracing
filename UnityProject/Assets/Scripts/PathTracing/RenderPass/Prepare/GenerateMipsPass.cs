using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Profiling;
using UnityEngine.Rendering.Universal;
using static PathTracing.ShaderIDs;

namespace PathTracing
{
    public class GenerateMipsPass : ScriptableRenderPass
    {
        private readonly ComputeShader _genMipsCs;
        private RtxdiPassContext _context;

        public GenerateMipsPass(ComputeShader cs)
        {
            _genMipsCs = cs;
        }

        public void Setup(RtxdiPassContext ctx)
        {
            _context = ctx;
        }

        class PassData
        {
            internal ComputeShader GenMipsCs;
            internal RtxdiPassContext Context;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var marker = RenderPassMarkers.GenerateMips;

            natCmd.BeginSample(marker);

            var tex = data.Context.LocalLightPdfTexture;
            int mipCount = tex.rt.mipmapCount;
            int width = tex.rt.width;
            int height = tex.rt.height;
            int kernel = data.GenMipsCs.FindKernel("CSMain");

            natCmd.SetComputeTextureParam(data.GenMipsCs, kernel, _SourceMipID, tex, 0);

            for (int srcMip = 0; srcMip < mipCount - 1; srcMip++)
            {
                int destMip = srcMip + 1;
                int destWidth = Mathf.Max(1, width >> destMip);
                int destHeight = Mathf.Max(1, height >> destMip);

                // 1. 设置源 Mip 层级和目标尺寸
                natCmd.SetComputeIntParam(data.GenMipsCs, _SrcMipLevelID, srcMip);
                natCmd.SetComputeVectorParam(data.GenMipsCs, _TargetSizeID, new Vector4(destWidth, destHeight, 0, 0));

                // 2. 绑定目标 Mip (RWTexture2D 必须指定特定的 mipLevel)
                // UAV (RWTexture2D) 绑定特定 mip 是有效的，这会把该 mip 映射到 Shader 的 [0,0] 坐标系
                natCmd.SetComputeTextureParam(data.GenMipsCs, kernel, _TargetMipID, tex, destMip);

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
            passData.Context = _context;

            builder.AllowPassCulling(false);

            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => 
            { 
                ExecutePass(data, context); 
            });
        }
    }
}