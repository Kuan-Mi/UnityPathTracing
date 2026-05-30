using System;
using System.Runtime.InteropServices;
using NativeRender;
using Nri;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Port of donut <c>BloomPass</c> (the bloom the original RTXPT uses), implemented as native
    /// compute so nothing interleaves Unity ops with the plugin's dispatches:
    ///
    ///   1. Downsample HDR → half → quarter (bilinear).
    ///   2. Separable Gaussian blur at quarter res (horizontal then vertical).
    ///   3. Composite the blurred bloom back into the HDR image in place:
    ///      result = lerp(original, bloom, intensity)  (matches donut's ConstantColor blend).
    ///
    /// Runs after DLSS-RR and before tone mapping, operating on the display-resolution HDR image.
    /// Gaussian weights and constants are taken verbatim from donut bloom_ps.hlsl / BloomPass.cpp.
    /// </summary>
    public class NativeRtxptBloomPass : ScriptableRenderPass, IDisposable
    {
        // Mirrors donut bloom_cb.h BloomConstants (32 bytes).
        [StructLayout(LayoutKind.Sequential)]
        private struct BloomConstants
        {
            public Vector2 pixstep;
            public float   argumentScale;
            public float   normalizationScale;
            public Vector3 padding;
            public float   numSamples;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct CompositeConstants
        {
            public float   blendFactor;
            public Vector3 padding;
        }

        private readonly NativeComputePipeline      _downsampleCs;
        private readonly NativeComputeDescriptorSet _downsample1Ds;
        private readonly NativeComputeDescriptorSet _downsample2Ds;
        private readonly NativeComputePipeline      _blurCs;
        private readonly NativeComputeDescriptorSet _blurHDs;
        private readonly NativeComputeDescriptorSet _blurVDs;
        private readonly NativeComputePipeline      _compositeCs;
        private readonly NativeComputeDescriptorSet _compositeDs;

        private readonly VolatileConstantBuffer _hBlurCb;
        private readonly VolatileConstantBuffer _vBlurCb;
        private readonly VolatileConstantBuffer _compositeCb;

        private NativeRtxptPassContext _ctx;
        private IntPtr                 _hdrPtr;

        public NativeRtxptBloomPass(NativeComputeShader downsampleCs, NativeComputeShader blurCs, NativeComputeShader compositeCs)
        {
            _downsampleCs  = new NativeComputePipeline(downsampleCs);
            _downsample1Ds = new NativeComputeDescriptorSet(_downsampleCs);
            _downsample2Ds = new NativeComputeDescriptorSet(_downsampleCs);
            _blurCs        = new NativeComputePipeline(blurCs);
            _blurHDs       = new NativeComputeDescriptorSet(_blurCs);
            _blurVDs       = new NativeComputeDescriptorSet(_blurCs);
            _compositeCs   = new NativeComputePipeline(compositeCs);
            _compositeDs   = new NativeComputeDescriptorSet(_compositeCs);

            _hBlurCb     = new VolatileConstantBuffer(Marshal.SizeOf<BloomConstants>());
            _vBlurCb     = new VolatileConstantBuffer(Marshal.SizeOf<BloomConstants>());
            _compositeCb = new VolatileConstantBuffer(Marshal.SizeOf<CompositeConstants>());
        }

        public void Dispose()
        {
            _downsample1Ds?.Dispose();
            _downsample2Ds?.Dispose();
            _downsampleCs?.Dispose();
            _blurHDs?.Dispose();
            _blurVDs?.Dispose();
            _blurCs?.Dispose();
            _compositeDs?.Dispose();
            _compositeCs?.Dispose();
            _hBlurCb?.Dispose();
            _vBlurCb?.Dispose();
            _compositeCb?.Dispose();
        }

        /// <summary>HDR image that is downsampled, blurred and composited back in place (e.g. DlssRrOutput).</summary>
        public void Setup(NativeRtxptPassContext ctx, NriTextureResource hdr)
        {
            _ctx    = ctx;
            _hdrPtr = hdr.NativePtr;
        }

        private class PassData
        {
            internal NativeComputePipeline      DownsampleCs, BlurCs, CompositeCs;
            internal NativeComputeDescriptorSet Downsample1Ds, Downsample2Ds, BlurHDs, BlurVDs, CompositeDs;
            internal VolatileConstantBuffer     HBlurCb, VBlurCb, CompositeCb;
            internal BloomConstants             HBlur, VBlur;
            internal CompositeConstants         Composite;
            internal IntPtr                     HdrPtr;
            internal IntPtr                     Down1Ptr, Down2Ptr, Pass1Ptr, Pass2Ptr;
            internal int2                        DisplayRes, HalfRes, QuarterRes;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            var s    = _ctx.Setting;
            var disp = _ctx.DisplayResolution;
            var half    = new int2((disp.x + 1) / 2, (disp.y + 1) / 2);
            var quarter = new int2((half.x + 1) / 2, (half.y + 1) / 2);

            // donut BloomPass::Render constant derivation.
            float sigma   = math.clamp(s.bloomRadius * 0.25f, 1f, 100f);
            float argScale = -1f / (2f * sigma * sigma);
            float normScale = 1f / (math.sqrt(2f * math.PI) * sigma);
            float numSamples = math.round(sigma * 4f);

            var hBlur = new BloomConstants
            {
                pixstep            = new Vector2(1f / quarter.x, 0f),
                argumentScale      = argScale,
                normalizationScale = normScale,
                padding            = Vector3.zero,
                numSamples         = numSamples,
            };
            var vBlur = hBlur;
            vBlur.pixstep = new Vector2(0f, 1f / quarter.y);

            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxpt.Bloom", out var pd);
            pd.DownsampleCs  = _downsampleCs;
            pd.Downsample1Ds = _downsample1Ds;
            pd.Downsample2Ds = _downsample2Ds;
            pd.BlurCs        = _blurCs;
            pd.BlurHDs       = _blurHDs;
            pd.BlurVDs       = _blurVDs;
            pd.CompositeCs   = _compositeCs;
            pd.CompositeDs   = _compositeDs;
            pd.HBlurCb       = _hBlurCb;
            pd.VBlurCb       = _vBlurCb;
            pd.CompositeCb   = _compositeCb;
            pd.HBlur         = hBlur;
            pd.VBlur         = vBlur;
            pd.Composite     = new CompositeConstants { blendFactor = s.bloomIntensity, padding = Vector3.zero };
            pd.HdrPtr        = _hdrPtr;
            pd.Down1Ptr      = _ctx.Textures.BloomDownscale1.NativePtr;
            pd.Down2Ptr      = _ctx.Textures.BloomDownscale2.NativePtr;
            pd.Pass1Ptr      = _ctx.Textures.BloomBlurPass1.NativePtr;
            pd.Pass2Ptr      = _ctx.Textures.BloomBlurPass2.NativePtr;
            pd.DisplayRes    = disp;
            pd.HalfRes       = half;
            pd.QuarterRes    = quarter;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }

        private static void Groups(int2 res, out uint gx, out uint gy)
        {
            gx = ((uint)res.x + 7u) / 8u;
            gy = ((uint)res.y + 7u) / 8u;
        }

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            cmd.BeginSample(RenderPassMarkers.Bloom);

            // 1. Downsample HDR → half.
            {
                var ds = data.Downsample1Ds;
                ds.SetTexture("t_Src", data.HdrPtr);
                ds.SetRWTexture("u_Dst", data.Down1Ptr);
                Groups(data.HalfRes, out var gx, out var gy);
                data.DownsampleCs.Dispatch(cmd, ds, gx, gy, 1);
            }

            // 2. Downsample half → quarter.
            {
                var ds = data.Downsample2Ds;
                ds.SetTexture("t_Src", data.Down1Ptr);
                ds.SetRWTexture("u_Dst", data.Down2Ptr);
                Groups(data.QuarterRes, out var gx, out var gy);
                data.DownsampleCs.Dispatch(cmd, ds, gx, gy, 1);
            }

            // 3a. Horizontal blur (quarter): downscale2 → pass1.
            {
                data.HBlurCb.UploadDirect(context.cmd, data.HBlur);
                var ds = data.BlurHDs;
                ds.SetConstantBuffer("c_Bloom", data.HBlurCb);
                ds.SetTexture("t_Src", data.Down2Ptr);
                ds.SetRWTexture("u_Dst", data.Pass1Ptr);
                Groups(data.QuarterRes, out var gx, out var gy);
                data.BlurCs.Dispatch(cmd, ds, gx, gy, 1);
            }

            // 3b. Vertical blur (quarter): pass1 → pass2.
            {
                data.VBlurCb.UploadDirect(context.cmd, data.VBlur);
                var ds = data.BlurVDs;
                ds.SetConstantBuffer("c_Bloom", data.VBlurCb);
                ds.SetTexture("t_Src", data.Pass1Ptr);
                ds.SetRWTexture("u_Dst", data.Pass2Ptr);
                Groups(data.QuarterRes, out var gx, out var gy);
                data.BlurCs.Dispatch(cmd, ds, gx, gy, 1);
            }

            // 4. Composite blurred bloom back into the HDR image (display res, in place).
            {
                data.CompositeCb.UploadDirect(context.cmd, data.Composite);
                var ds = data.CompositeDs;
                ds.SetConstantBuffer("c_Composite", data.CompositeCb);
                ds.SetTexture("t_Bloom", data.Pass2Ptr);
                ds.SetRWTexture("u_Output", data.HdrPtr);
                Groups(data.DisplayRes, out var gx, out var gy);
                data.CompositeCs.Dispatch(cmd, ds, gx, gy, 1);
            }

            cmd.EndSample(RenderPassMarkers.Bloom);
        }
    }
}
