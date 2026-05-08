using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;
using static PathTracing.ShaderIDs;

namespace PathTracing
{
    /// <summary>
    /// Generates the full mip chain for <c>LocalLightPdfTexture</c> (and optionally
    /// <c>EnvironmentPdfTexture</c>) after PrepareLights has written mip 0.
    ///
    /// Mirrors FullSample's <c>GenerateMipsPass</c> (Source/RenderPasses/GenerateMipsPass.cpp)
    /// but is wired into the <see cref="NativeRtxdiPassContext"/> pipeline.
    ///
    /// The native <c>PreprocessEnvironmentMap.computeshader</c> uses an unbounded
    /// <c>RWTexture2D&lt;float&gt; u_IntegratedMips[]</c> array that requires per-mip UAV
    /// subresource binding — not exposed by <see cref="NativeComputeDescriptorSet"/> today.
    /// We therefore reuse Unity's managed <c>GenerateMips.compute</c> shader, which is
    /// identical in output and already used by <see cref="GenerateMipsPass"/>.
    /// </summary>
    public class NativeRtxdiPdfMipsPass : ScriptableRenderPass
    {
        private readonly ComputeShader _genMipsCs;
        private NativeRtxdiPassContext _context;

        public NativeRtxdiPdfMipsPass(ComputeShader genMipsCs)
        {
            _genMipsCs = genMipsCs;
        }

        public void Setup(NativeRtxdiPassContext ctx)
        {
            _context = ctx;
        }

        // -------------------------------------------------------------------------

        class PassData
        {
            internal ComputeShader         GenMipsCs;
            internal NativeRtxdiPassContext Context;
        }

        /// <summary>
        /// Dispatches the mip-generation compute shader for every mip level of <paramref name="tex"/>
        /// (mip 0 must already be written by the caller).
        /// </summary>
        static void GenerateMipsForTexture(CommandBuffer cmd, ComputeShader cs, RenderTexture tex)
        {
            int mipCount = tex.mipmapCount;
            if (mipCount <= 1) return;

            int kernel = cs.FindKernel("CSMain");
            int width  = tex.width;
            int height = tex.height;

            // Bind the full texture as the persistent SRV (_SourceMip always reads from mip 0
            // for the first pass, then we advance via _SrcMipLevel).
            cmd.SetComputeTextureParam(cs, kernel, _SourceMipID, tex, 0);

            for (int srcMip = 0; srcMip < mipCount - 1; srcMip++)
            {
                int destMip   = srcMip + 1;
                int destWidth  = Mathf.Max(1, width  >> destMip);
                int destHeight = Mathf.Max(1, height >> destMip);

                cmd.SetComputeIntParam(cs, _SrcMipLevelID, srcMip);
                cmd.SetComputeVectorParam(cs, _TargetSizeID, new Vector4(destWidth, destHeight, 0, 0));
                cmd.SetComputeTextureParam(cs, kernel, _TargetMipID, tex, destMip);

                int groupsX = (destWidth  + 7) / 8;
                int groupsY = (destHeight + 7) / 8;
                cmd.DispatchCompute(cs, kernel, groupsX, groupsY, 1);
            }
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var cs  = data.GenMipsCs;
            var res = data.Context.Resources;

            cmd.BeginSample(RenderPassMarkers.GenerateMips);

            // ---- Local Light PDF mip chain ----
            if (res?.LocalLightPdfTexture?.rt != null)
                GenerateMipsForTexture(cmd, cs, res.LocalLightPdfTexture.rt);

            // ---- Environment PDF mip chain ----
            // Only generate when the texture is larger than the stub 1×1 placeholder
            // (i.e. once a real environment map has been assigned).
            if (res?.EnvironmentPdfTexture?.rt != null
                && (res.EnvironmentPdfTexture.rt.width > 1 || res.EnvironmentPdfTexture.rt.height > 1))
            {
                GenerateMipsForTexture(cmd, cs, res.EnvironmentPdfTexture.rt);
            }

            cmd.EndSample(RenderPassMarkers.GenerateMips);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            if (_genMipsCs == null || _context?.Resources == null) return;

            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxdiPdfMips", out var passData);
            passData.GenMipsCs = _genMipsCs;
            passData.Context   = _context;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext ctx) => ExecutePass(data, ctx));
        }
    }
}
