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
    /// Faithful native replica of donut <c>BloomPass</c> (the bloom the original RTXPT uses), built
    /// entirely from the original shaders — each Unity asset is a thin verbatim #include wrapper:
    ///
    ///   1. downsample — BloomDownsample.rastershader (= donut fullscreen_vs + blit_ps.hlsl) renders the
    ///                   HDR into a half-res RT, then half → quarter. The single bilinear tap of blit_ps,
    ///                   driven at half-size, is the exact 2x2 box average CommonRenderPasses::BlitTexture
    ///                   does for the original's two downscales.
    ///   2. blur       — BloomBlur.rastershader (= donut fullscreen_vs + passes/bloom_ps.hlsl) runs the
    ///                   verbatim separable Gaussian twice at quarter res (horizontal then vertical).
    ///   3. composite  — BloomComposite.rastershader (= donut fullscreen_vs + blit_ps.hlsl) samples the
    ///                   blurred bloom and the constant-color blend hardware composites it back into the
    ///                   HDR image in place. The pipeline uses blendMode 4 (SrcBlend=ConstantColor,
    ///                   DestBlend=InvConstantColor) with blendFactor = bloom intensity, so
    ///                   result = lerp(hdr, bloom, intensity) — exactly donut's ConstantColor blend.
    ///
    /// Gaussian weights and constants are taken verbatim from donut bloom_ps.hlsl / BloomPass.cpp.
    /// Runs after DLSS-RR and before tone mapping, operating on the display-resolution HDR image.
    /// </summary>
    public class RtxptBloomPass : ScriptableRenderPass, IDisposable
    {
        private const uint kRGBA16F = (uint)DXGI_FORMAT.DXGI_FORMAT_R16G16B16A16_FLOAT;

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

        // ── Pipelines (built once) ──
        private readonly NativeRasterPipeline      _downsampleRaster;
        private readonly NativeRasterDescriptorSet _downsample1Ds;
        private readonly NativeRasterDescriptorSet _downsample2Ds;
        private readonly NativeRasterPipeline      _blurRaster;
        private readonly NativeRasterDescriptorSet _blurHDs;
        private readonly NativeRasterDescriptorSet _blurVDs;
        private readonly NativeRasterPipeline      _compositeRaster;
        private readonly NativeRasterDescriptorSet _compositeDs;

        private readonly VolatileConstantBuffer _hBlurCb;
        private readonly VolatileConstantBuffer _vBlurCb;

        // Single-RT format array shared by every draw (all bloom targets are RGBA16F).
        private readonly uint[] _colorFmt = { kRGBA16F };

        private RtxptPassContext _ctx;
        private IntPtr                 _hdrPtr;

        public RtxptBloomPass(NativeRasterShader downsampleShader,
                                    NativeRasterShader blurShader,
                                    NativeRasterShader compositeShader)
        {
            var opaque = NativeRenderPlugin.RasterPipelineStateDesc.FullscreenOpaque(kRGBA16F);

            // Composite: opaque fullscreen state but with the constant-color blend (donut's
            // ConstantColor/InvConstantColor); the blend constant is supplied per-draw.
            var constColor = NativeRenderPlugin.RasterPipelineStateDesc.FullscreenOpaque(kRGBA16F);
            constColor.blendMode = NativeRenderPlugin.RasterPipelineStateDesc.BlendModeConstantColor;

            _downsampleRaster = new NativeRasterPipeline(downsampleShader, opaque);
            _downsample1Ds    = new NativeRasterDescriptorSet(_downsampleRaster);
            _downsample2Ds    = new NativeRasterDescriptorSet(_downsampleRaster);
            _blurRaster       = new NativeRasterPipeline(blurShader, opaque);
            _blurHDs          = new NativeRasterDescriptorSet(_blurRaster);
            _blurVDs          = new NativeRasterDescriptorSet(_blurRaster);
            _compositeRaster  = new NativeRasterPipeline(compositeShader, constColor);
            _compositeDs      = new NativeRasterDescriptorSet(_compositeRaster);

            // Names mirror donut BloomPass.cpp debugName strings ("BloomConstantsH"/"BloomConstantsV").
            _hBlurCb = new VolatileConstantBuffer(Marshal.SizeOf<BloomConstants>(), "BloomConstantsH");
            _vBlurCb = new VolatileConstantBuffer(Marshal.SizeOf<BloomConstants>(), "BloomConstantsV");
        }

        public void Dispose()
        {
            _downsample1Ds?.Dispose();
            _downsample2Ds?.Dispose();
            _downsampleRaster?.Dispose();
            _blurHDs?.Dispose();
            _blurVDs?.Dispose();
            _blurRaster?.Dispose();
            _compositeDs?.Dispose();
            _compositeRaster?.Dispose();
            _hBlurCb?.Dispose();
            _vBlurCb?.Dispose();
        }

        /// <summary>HDR image that is downsampled, blurred and composited back in place (e.g. DlssRrOutput).</summary>
        public void Setup(RtxptPassContext ctx, NriTextureResource hdr)
        {
            _ctx    = ctx;
            _hdrPtr = hdr.NativePtr;
        }

        private class PassData
        {
            internal NativeRasterPipeline      DownsampleRaster, BlurRaster, CompositeRaster;
            internal NativeRasterDescriptorSet Downsample1Ds, Downsample2Ds, BlurHDs, BlurVDs, CompositeDs;
            internal VolatileConstantBuffer    HBlurCb, VBlurCb;
            internal BloomConstants            HBlur, VBlur;
            internal float                     BlendFactor;
            internal IntPtr                    HdrPtr;
            internal IntPtr                    Down1Ptr, Down2Ptr, Pass1Ptr, Pass2Ptr;
            internal int2                       DisplayRes, HalfRes, QuarterRes;
            internal uint[]                    ColorFmt;
            internal IntPtr[]                  ColorRes;   // scratch length-1, refilled per draw
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            var s    = _ctx.Setting;
            var disp = _ctx.DisplayResolution;
            // Intermediate sizes derive from the RENDER resolution (donut ceil(/2)), replicating the
            // original's quirk: Sample.cpp constructs BloomPass with the render-res *m_view, so the
            // first "Downscale" blit is a >2x reduction of the display-res HDR and the blur runs on
            // quarter-render-res buffers, while the final Apply composites at display res.
            var rend = _ctx.RenderResolution;
            var half    = new int2((rend.x + 1) / 2, (rend.y + 1) / 2);
            var quarter = new int2((half.x + 1) / 2, (half.y + 1) / 2);

            // donut BloomPass::Render constant derivation.
            float sigma      = math.clamp(s.bloomRadius * 0.25f, 1f, 100f);
            float argScale   = -1f / (2f * sigma * sigma);
            float normScale  = 1f / (math.sqrt(2f * math.PI) * sigma);
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

            using var builder = renderGraph.AddUnsafePass<PassData>("Bloom", out var pd);
            pd.DownsampleRaster = _downsampleRaster;
            pd.Downsample1Ds    = _downsample1Ds;
            pd.Downsample2Ds    = _downsample2Ds;
            pd.BlurRaster       = _blurRaster;
            pd.BlurHDs          = _blurHDs;
            pd.BlurVDs          = _blurVDs;
            pd.CompositeRaster  = _compositeRaster;
            pd.CompositeDs      = _compositeDs;
            pd.HBlurCb          = _hBlurCb;
            pd.VBlurCb          = _vBlurCb;
            pd.HBlur            = hBlur;
            pd.VBlur            = vBlur;
            pd.BlendFactor      = s.bloomIntensity;
            pd.HdrPtr           = _hdrPtr;
            pd.Down1Ptr         = _ctx.Textures.BloomDownscale1.NativePtr;
            pd.Down2Ptr         = _ctx.Textures.BloomDownscale2.NativePtr;
            pd.Pass1Ptr         = _ctx.Textures.BloomBlurPass1.NativePtr;
            pd.Pass2Ptr         = _ctx.Textures.BloomBlurPass2.NativePtr;
            pd.DisplayRes       = disp;
            pd.HalfRes          = half;
            pd.QuarterRes       = quarter;
            pd.ColorFmt         = _colorFmt;
            pd.ColorRes         = new IntPtr[1];

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            cmd.BeginSample(RenderPassMarkers.Bloom);

            // "Downscale" marker (donut BloomPass.cpp:189) wraps both half-scale blits.
            cmd.BeginSample(RenderPassMarkers.RtxptBloomDownscale);

            // 1. Downsample HDR → half (blit_ps bilinear, opaque).
            {
                var ds = data.Downsample1Ds;
                ds.SetTexture("tex", data.HdrPtr);
                data.ColorRes[0] = data.Down1Ptr;
                var draw = new RasterDrawDesc
                {
                    numRenderTargets = 1, colorResources = data.ColorRes, colorFormats = data.ColorFmt,
                    depthResource = IntPtr.Zero, viewport = new Rect(0, 0, data.HalfRes.x, data.HalfRes.y),
                    vertexCount = 4, instanceCount = 1,
                };
                data.DownsampleRaster.Draw(cmd, ds, in draw);
            }

            // 2. Downsample half → quarter.
            {
                var ds = data.Downsample2Ds;
                ds.SetTexture("tex", data.Down1Ptr);
                data.ColorRes[0] = data.Down2Ptr;
                var draw = new RasterDrawDesc
                {
                    numRenderTargets = 1, colorResources = data.ColorRes, colorFormats = data.ColorFmt,
                    depthResource = IntPtr.Zero, viewport = new Rect(0, 0, data.QuarterRes.x, data.QuarterRes.y),
                    vertexCount = 4, instanceCount = 1,
                };
                data.DownsampleRaster.Draw(cmd, ds, in draw);
            }

            cmd.EndSample(RenderPassMarkers.RtxptBloomDownscale);

            // "Blur" marker (donut BloomPass.cpp:220) wraps the H+V Gaussian draws.
            cmd.BeginSample(RenderPassMarkers.RtxptBloomBlur);

            // 3a. Horizontal blur (quarter): downscale2 → pass1 (verbatim bloom_ps).
            {
                data.HBlurCb.UploadDirect(context.cmd, data.HBlur);
                var ds = data.BlurHDs;
                ds.SetConstantBuffer("c_Bloom", data.HBlurCb);
                ds.SetTexture("t_Src", data.Down2Ptr);
                data.ColorRes[0] = data.Pass1Ptr;
                var draw = new RasterDrawDesc
                {
                    numRenderTargets = 1, colorResources = data.ColorRes, colorFormats = data.ColorFmt,
                    depthResource = IntPtr.Zero, viewport = new Rect(0, 0, data.QuarterRes.x, data.QuarterRes.y),
                    vertexCount = 4, instanceCount = 1,
                };
                data.BlurRaster.Draw(cmd, ds, in draw);
            }

            // 3b. Vertical blur (quarter): pass1 → pass2.
            {
                data.VBlurCb.UploadDirect(context.cmd, data.VBlur);
                var ds = data.BlurVDs;
                ds.SetConstantBuffer("c_Bloom", data.VBlurCb);
                ds.SetTexture("t_Src", data.Pass1Ptr);
                data.ColorRes[0] = data.Pass2Ptr;
                var draw = new RasterDrawDesc
                {
                    numRenderTargets = 1, colorResources = data.ColorRes, colorFormats = data.ColorFmt,
                    depthResource = IntPtr.Zero, viewport = new Rect(0, 0, data.QuarterRes.x, data.QuarterRes.y),
                    vertexCount = 4, instanceCount = 1,
                };
                data.BlurRaster.Draw(cmd, ds, in draw);
            }

            cmd.EndSample(RenderPassMarkers.RtxptBloomBlur);

            // "Apply" marker (donut BloomPass.cpp:260) wraps the composite draw.
            cmd.BeginSample(RenderPassMarkers.RtxptBloomApply);

            // 4. Composite blurred bloom back into the HDR image (display res, in place) via the
            //    constant-color blend: result = bloom*intensity + hdr*(1-intensity).
            {
                var ds = data.CompositeDs;
                ds.SetTexture("tex", data.Pass2Ptr);
                data.ColorRes[0] = data.HdrPtr;
                var draw = new RasterDrawDesc
                {
                    numRenderTargets = 1, colorResources = data.ColorRes, colorFormats = data.ColorFmt,
                    depthResource = IntPtr.Zero, viewport = new Rect(0, 0, data.DisplayRes.x, data.DisplayRes.y),
                    vertexCount = 4, instanceCount = 1, blendFactor = data.BlendFactor,
                };
                data.CompositeRaster.Draw(cmd, ds, in draw);
            }

            cmd.EndSample(RenderPassMarkers.RtxptBloomApply);

            cmd.EndSample(RenderPassMarkers.Bloom);
        }
    }
}
