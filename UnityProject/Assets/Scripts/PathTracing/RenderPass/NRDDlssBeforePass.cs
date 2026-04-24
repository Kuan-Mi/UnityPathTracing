using System;
using NativeRender;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Prepares DLSS-RR guide buffers using a NativeComputeShader.
    ///
    /// DlssBefore.computeshader — numthreads [16, 16, 1]
    ///
    /// -- SRV (3) --
    ///   gIn_Normal_Roughness              Texture2D  space0:t0
    ///   gIn_BaseColor_Metalness           Texture2D  space0:t1
    ///   gIn_Spec                          Texture2D  space0:t2
    ///
    /// -- UAV (5) --
    ///   gInOut_ViewZ                      RWTexture2D  space0:u0
    ///   gOut_DiffAlbedo                   RWTexture2D  space0:u1
    ///   gOut_SpecAlbedo                   RWTexture2D  space0:u2
    ///   gOut_SpecHitDistance              RWTexture2D  space0:u3
    ///   gOut_Normal_Roughness             RWTexture2D  space0:u4
    ///
    /// -- CBV (1) --
    ///   GlobalConstants                   ConstantBuffer  space4:b0
    /// </summary>
    public class NRDDlssBeforePass : ScriptableRenderPass, IDisposable
    {
        private readonly NativeComputePipeline      _cs;
        private readonly NativeComputeDescriptorSet _ds;
        private          Resource                   _resource;
        private          Settings                   _settings;

        public NRDDlssBeforePass(NativeComputeShader cs)
        {
            _cs = new NativeComputePipeline(cs);
            _ds = new NativeComputeDescriptorSet(_cs);
        }

        public void Dispose()
        {
            _ds?.Dispose();
            _cs?.Dispose();
        }

        public void Setup(Resource resource, Settings settings)
        {
            _resource = resource;
            _settings = settings;
        }

        // -------------------------------------------------------------------------
        // Resource / Settings
        // -------------------------------------------------------------------------

        public class Resource
        {
            internal IntPtr                  ConstantBuffer;
            internal PathTracingResourcePool Pool;
        }

        public class Settings
        {
            internal int rectGridW;
            internal int rectGridH;
        }

        // -------------------------------------------------------------------------
        // Pass data
        // -------------------------------------------------------------------------

        class PassData
        {
            internal NativeComputePipeline      Cs;
            internal NativeComputeDescriptorSet Ds;
            internal Resource                   Resource;
            internal Settings                   Settings;
            internal PathTracingResourcePool    Pool;
        }

        // -------------------------------------------------------------------------
        // Execution
        // -------------------------------------------------------------------------

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd  = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var cs   = data.Cs;
            var ds   = data.Ds;
            var res  = data.Resource;
            var pool = data.Pool;

            cmd.BeginSample(RenderPassMarkers.DlssBefore);

            // SRV inputs
            ds.SetTexture("gIn_Normal_Roughness", pool.GetPoint(RenderResourceType.NormalRoughness));
            ds.SetTexture("gIn_BaseColor_Metalness", pool.GetPoint(RenderResourceType.BaseColorMetalness));
            ds.SetTexture("gIn_Spec", pool.GetPoint(RenderResourceType.Unfiltered_Spec));

            // UAV outputs
            ds.SetRWTexture("gInOut_ViewZ", pool.GetPoint(RenderResourceType.Viewz));
            ds.SetRWTexture("gOut_DiffAlbedo", pool.GetPoint(RenderResourceType.RrGuideDiffAlbedo));
            ds.SetRWTexture("gOut_SpecAlbedo", pool.GetPoint(RenderResourceType.RrGuideSpecAlbedo));
            ds.SetRWTexture("gOut_SpecHitDistance", pool.GetPoint(RenderResourceType.RrGuideSpecHitDistance));
            ds.SetRWTexture("gOut_Normal_Roughness", pool.GetPoint(RenderResourceType.RrGuideNormalRoughness));

            // Constant buffer
            ds.SetConstantBuffer("GlobalConstants", res.ConstantBuffer);

            // Dispatch — numthreads [16, 16, 1]
            cs.Dispatch(cmd, ds, (uint)data.Settings.rectGridW, (uint)data.Settings.rectGridH, 1);

            cmd.EndSample(RenderPassMarkers.DlssBefore);
        }

        // -------------------------------------------------------------------------
        // RenderGraph
        // -------------------------------------------------------------------------

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NRDDlssBeforePass", out var passData);

            passData.Cs       = _cs;
            passData.Ds       = _ds;
            passData.Resource = _resource;
            passData.Settings = _settings;
            passData.Pool     = _resource.Pool;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }
    }
}