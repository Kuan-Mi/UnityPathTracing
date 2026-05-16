using System;
using NativeRender;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Native compute shader composition pass that dispatches Composition.computeshader.
    /// No TLAS required — binds only G-buffer textures and constant buffer.
    /// </summary>
    public class NRDCompositionPass : ScriptableRenderPass, IDisposable
    {
        private readonly NativeComputePipeline      _cs;
        private readonly NativeComputeDescriptorSet _ds;
        private          Resource                   _resource;
        private          Settings                   _settings;

        public NRDCompositionPass(NativeComputeShader cs)
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
        // Resource / Settings  (mirrors CompositionPass exactly)
        // -------------------------------------------------------------------------

        public class Resource
        {
            internal IntPtr ConstantBuffer;

            // RT textures sourced from the pool inside ExecutePass
            internal NativeNrdTextureResources Pool;
        }

        public class Settings
        {
            internal int  rectGridW;
            internal int  rectGridH;
            internal bool useRR;
        }

        // -------------------------------------------------------------------------
        // Pass data (RenderGraph)
        // -------------------------------------------------------------------------

        class PassData
        {
            internal NativeComputePipeline      Cs;
            internal NativeComputeDescriptorSet Ds;
            internal Resource                   Resource;
            internal Settings                   Settings;
            internal NativeNrdTextureResources    Pool;
        }

        // -------------------------------------------------------------------------
        // Execution
        // -------------------------------------------------------------------------

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var cs  = data.Cs;
            var ds  = data.Ds;
            var res = data.Resource;

            cmd.BeginSample(RenderPassMarkers.Composition);

            var pool = data.Pool;

            // SRV inputs
            ds.SetTexture("gIn_ViewZ", pool.ViewZ.NativePtr);
            ds.SetTexture("gIn_Normal_Roughness", pool.NormalRoughness.NativePtr);
            ds.SetTexture("gIn_BaseColor_Metalness", pool.BaseColorMetalness.NativePtr);
            ds.SetTexture("gIn_DirectLighting", pool.DirectLighting.NativePtr);
            ds.SetTexture("gIn_DirectEmission", pool.DirectEmission.NativePtr);
            ds.SetTexture("gIn_PsrThroughput", pool.PsrThroughput.NativePtr);

            bool useRR = data.Settings.useRR;

            ds.SetTexture("gIn_Shadow", useRR ? pool.Unfiltered_Translucency.NativePtr : pool.Shadow.NativePtr);
            ds.SetTexture("gIn_Diff",   useRR ? pool.Unfiltered_Diff.NativePtr          : pool.Diff.NativePtr);
            ds.SetTexture("gIn_Spec",   useRR ? pool.Unfiltered_Spec.NativePtr          : pool.Spec.NativePtr);

            // UAV outputs
            ds.SetRWTexture("gOut_ComposedDiff", pool.ComposedDiff.NativePtr);
            ds.SetRWTexture("gOut_ComposedSpec_ViewZ", pool.ComposedSpecViewZ.NativePtr);

            // Constant buffer
            ds.SetConstantBuffer("GlobalConstants", res.ConstantBuffer);

            // Dispatch — numthreads [16, 16, 1]
            cs.Dispatch(cmd, ds, (uint)data.Settings.rectGridW, (uint)data.Settings.rectGridH, 1);

            cmd.EndSample(RenderPassMarkers.Composition);
        }

        // -------------------------------------------------------------------------
        // RenderGraph
        // -------------------------------------------------------------------------

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NRDCompositionPass", out var passData);

            passData.Cs       = _cs;
            passData.Ds       = _ds;
            passData.Resource = _resource;
            passData.Settings = _settings;
            passData.Pool     = _resource.Pool;


            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}