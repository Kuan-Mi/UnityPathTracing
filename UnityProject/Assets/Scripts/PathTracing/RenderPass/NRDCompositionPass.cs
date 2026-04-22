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
        private readonly NativeComputePipeline _cs;
        private          Resource              _resource;
        private          Settings              _settings;

        public NRDCompositionPass(NativeComputeShader cs)
        {
            _cs = new NativeComputePipeline(cs);
        }

        public void Dispose()
        {
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
            internal GraphicsBuffer ConstantBuffer;

            // RT textures sourced from the pool inside ExecutePass
            internal PathTracingResourcePool Pool;
        }

        public class Settings
        {
            internal int rectGridW;
            internal int rectGridH;
        }

        // -------------------------------------------------------------------------
        // Pass data (RenderGraph)
        // -------------------------------------------------------------------------

        class PassData
        {
            internal NativeComputePipeline   Cs;
            internal Resource                Resource;
            internal Settings                Settings;
            internal PathTracingResourcePool Pool;
        }

        // -------------------------------------------------------------------------
        // Execution
        // -------------------------------------------------------------------------

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var cs  = data.Cs;
            var res = data.Resource;

            cmd.BeginSample(RenderPassMarkers.Composition);

            var pool = data.Pool;

            // SRV inputs
            cs.SetTexture("gIn_ViewZ", pool.GetRT(RenderResourceType.Viewz).rt);
            cs.SetTexture("gIn_Normal_Roughness", pool.GetRT(RenderResourceType.NormalRoughness).rt);
            cs.SetTexture("gIn_BaseColor_Metalness", pool.GetRT(RenderResourceType.BaseColorMetalness).rt);
            cs.SetTexture("gIn_DirectLighting", pool.GetRT(RenderResourceType.DirectLighting).rt);
            cs.SetTexture("gIn_DirectEmission", pool.GetRT(RenderResourceType.DirectEmission).rt);
            cs.SetTexture("gIn_PsrThroughput", pool.GetRT(RenderResourceType.PsrThroughput).rt);
            cs.SetTexture("gIn_Shadow", pool.GetRT(RenderResourceType.Shadow).rt);
            cs.SetTexture("gIn_Diff", pool.GetRT(RenderResourceType.Diff).rt);
            cs.SetTexture("gIn_Spec", pool.GetRT(RenderResourceType.Spec).rt);

            // UAV outputs
            cs.SetRWTexture("gOut_ComposedDiff", pool.GetRT(RenderResourceType.ComposedDiff).rt);
            cs.SetRWTexture("gOut_ComposedSpec_ViewZ", pool.GetRT(RenderResourceType.ComposedSpecViewZ).rt);

            // Constant buffer
            cs.SetConstantBuffer("GlobalConstants", res.ConstantBuffer);

            // Dispatch — numthreads [16, 16, 1]
            cs.Dispatch(cmd, (uint)data.Settings.rectGridW, (uint)data.Settings.rectGridH, 1);

            cmd.EndSample(RenderPassMarkers.Composition);
        }

        // -------------------------------------------------------------------------
        // RenderGraph
        // -------------------------------------------------------------------------

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NRDCompositionPass", out var passData);

            passData.Cs       = _cs;
            passData.Resource = _resource;
            passData.Settings = _settings;
            passData.Pool     = _resource.Pool;


            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}