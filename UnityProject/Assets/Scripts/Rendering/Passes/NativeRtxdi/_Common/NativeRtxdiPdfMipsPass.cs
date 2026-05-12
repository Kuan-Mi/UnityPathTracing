using System;
using System.Runtime.InteropServices;
using NativeRender;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Generates the full mip chain for <c>LocalLightPdfTexture</c> (and optionally
    /// <c>EnvironmentPdfTexture</c>) after PrepareLights has written mip 0.
    ///
    /// Uses the native <c>PreprocessEnvironmentMap.computeshader</c> which declares
    /// <c>RWTexture2D&lt;float&gt; u_IntegratedMips[]</c> — an unbounded UAV array.
    /// Each dispatch covers up to 5 mip levels using wave-ops hierarchical downsampling.
    /// All mips (0..N-1) are bound as UAV subresource views via <see cref="BindlessUAVTexture"/>.
    /// </summary>
    public class NativeRtxdiPdfMipsPass : ScriptableRenderPass, IDisposable
    {
        // DXGI_FORMAT_R32_FLOAT = 41
        private const uint DXGI_FORMAT_R32_FLOAT = 41;

        [StructLayout(LayoutKind.Sequential)]
        private struct PreprocessEnvironmentMapConstants
        {
            public uint sourceSizeX;
            public uint sourceSizeY;
            public uint sourceMipLevel;
            public uint numDestMipLevels;
        }

        private readonly NativeComputePipeline       _csLocalLight;   // INPUT_ENVIRONMENT_MAP=0
        private readonly NativeComputePipeline       _csEnvMap;       // INPUT_ENVIRONMENT_MAP=1
        private readonly NativeComputeDescriptorSet  _dsLocalLight;
        private readonly NativeComputeDescriptorSet  _dsEnvMap;
        private NativeRtxdiPassContext               _context;

        // Per-texture UAV arrays — lazily created and resized when mip count changes.
        private BindlessUAVTexture _localLightUav;
        private BindlessUAVTexture _envUav;
        private int                _localLightMipCount;
        private int                _envMipCount;

        public NativeRtxdiPdfMipsPass(
            NativeComputeShader preprocessLocalLightCs,
            NativeComputeShader preprocessEnvironmentMapCs)
        {
            var hints = new[] { new RootConstantsHint { Name = "g_Const", Count = 4 } };
            _csLocalLight  = new NativeComputePipeline(preprocessLocalLightCs,  hints);
            _csEnvMap      = new NativeComputePipeline(preprocessEnvironmentMapCs, hints);
            _dsLocalLight  = new NativeComputeDescriptorSet(_csLocalLight);
            _dsEnvMap      = new NativeComputeDescriptorSet(_csEnvMap);
        }

        public void Dispose()
        {
            _localLightUav?.Dispose();
            _envUav?.Dispose();
            _dsLocalLight?.Dispose();
            _dsEnvMap?.Dispose();
            _csLocalLight?.Dispose();
            _csEnvMap?.Dispose();
        }

        public void Setup(NativeRtxdiPassContext ctx)
        {
            _context = ctx;
        }

        // -------------------------------------------------------------------------

        class PassData
        {
            internal NativeRtxdiPdfMipsPass Pass;
            internal NativeRtxdiPassContext Context;
        }

        /// <summary>
        /// Ensures the <see cref="BindlessUAVTexture"/> has exactly <paramref name="mipCount"/> slots
        /// and fills each slot with the corresponding mip level of <paramref name="tex"/>.
        /// Returns the (possibly reallocated) UAV array.
        /// </summary>
        private static BindlessUAVTexture EnsureUAV(
            BindlessUAVTexture existing, ref int cachedMipCount,
            RenderTexture tex, int mipCount)
        {
            if (existing == null || cachedMipCount != mipCount)
            {
                existing?.Dispose();
                existing       = new BindlessUAVTexture(mipCount);
                cachedMipCount = mipCount;
            }
            for (int i = 0; i < mipCount; i++)
                existing.SetTexture(i, tex, i, DXGI_FORMAT_R32_FLOAT);
            return existing;
        }

        /// <summary>
        /// Dispatches the given pipeline/descriptor-set for every mip level of <paramref name="tex"/>.
        /// Mip 0 must already be written. Each call covers up to 5 mip levels.
        /// </summary>
        private unsafe void GenerateMipsNative(
            CommandBuffer cmd,
            NativeComputePipeline cs, NativeComputeDescriptorSet ds,
            RenderTexture tex,
            ref BindlessUAVTexture uav, ref int cachedMipCount)
        {
            int mipCount = tex.mipmapCount;
            if (mipCount <= 1) return;

            uav = EnsureUAV(uav, ref cachedMipCount, tex, mipCount);

            // Thread group size is 16×16 tiles (256 threads). Process mip 0 source.
            int srcWidth  = tex.width;
            int srcHeight = tex.height;

            // Each dispatch starts from sourceMipLevel and can write up to 5 smaller mips.
            for (int srcMip = 0; srcMip < mipCount - 1; srcMip += 5)
            {
                var consts = new PreprocessEnvironmentMapConstants
                {
                    sourceSizeX      = (uint)Mathf.Max(1, srcWidth  >> srcMip),
                    sourceSizeY      = (uint)Mathf.Max(1, srcHeight >> srcMip),
                    sourceMipLevel   = (uint)srcMip,
                    numDestMipLevels = (uint)mipCount,
                };

                ds.SetBindlessRWTexture("u_IntegratedMips", uav,
                    tex.GetNativeTexturePtr());
                ds.SetRootConstants("g_Const", &consts);

                // Groups cover the source mip level with 16×16 tiles.
                uint groupsX = (uint)Mathf.Max(1, ((int)consts.sourceSizeX + 15) / 16);
                uint groupsY = (uint)Mathf.Max(1, ((int)consts.sourceSizeY + 15) / 16);
                cs.Dispatch(cmd, ds, groupsX, groupsY, 1);
            }
        }

        private void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var res = data.Context.Resources;

            cmd.BeginSample(RenderPassMarkers.GenerateMips);

            // ---- Local Light PDF mip chain ----
            if (res?.LocalLightPdfTexture?.rt != null)
                GenerateMipsNative(cmd, _csLocalLight, _dsLocalLight,
                    res.LocalLightPdfTexture.rt, ref _localLightUav, ref _localLightMipCount);

            // // ---- Environment PDF mip chain ----
            // // Only generate when the texture is larger than the stub 1×1 placeholder.
            // if (res?.EnvironmentPdfTexture?.rt != null
            //     && (res.EnvironmentPdfTexture.rt.width > 1 || res.EnvironmentPdfTexture.rt.height > 1))
            // {
            //     GenerateMipsNative(cmd, _csEnvMap, _dsEnvMap,
            //         res.EnvironmentPdfTexture.rt, ref _envUav, ref _envMipCount);
            // }

            cmd.EndSample(RenderPassMarkers.GenerateMips);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            if (!_csLocalLight.IsValid || !_csEnvMap.IsValid || _context?.Resources == null) return;

            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxdiPdfMips", out var passData);
            passData.Pass    = this;
            passData.Context = _context;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext ctx) => data.Pass.ExecutePass(data, ctx));
        }
    }
}
