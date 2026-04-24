using System;
using NativeRender;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Native-compute-shader Final pass that dispatches Final.computeshader,
    /// matching NRDSample.cpp "Final" section.
    ///
    /// Bindings:
    ///   gIn_PostAA    (t0) – TAA / upscaler output  (TaaHistory or TaaHistoryPrev)
    ///   gIn_PreAA     (t1) – Composed (pre-TAA tonemapped input)
    ///   gIn_Validation(t2) – Validation overlay
    ///   gOut_Final    (u0) – Final render target
    ///
    /// Dispatch: ceil(outputSize / 16).
    /// </summary>
    public class NRDFinalPass : ScriptableRenderPass, IDisposable
    {
        private readonly NativeComputePipeline      _cs;
        private readonly NativeComputeDescriptorSet _ds;
        private          Resource              _resource;
        private          Settings              _settings;

        public NRDFinalPass(NativeComputeShader cs)
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
            internal GraphicsBuffer ConstantBuffer;  // GlobalConstants CBV
            internal PathTracingResourcePool Pool;
            internal bool IsEven; // !(frameIndex & 1) — selects which TAA history is the output
        }

        public class Settings
        {
            internal int2 OutputResolution;
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
            internal bool                       IsEven;
        }

        // -------------------------------------------------------------------------
        // Execution
        // -------------------------------------------------------------------------

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd  = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var cs   = data.Cs;
            var ds   = data.Ds;
            var pool = data.Pool;
            var res  = data.Resource;

            cmd.BeginSample(RenderPassMarkers.Final);

            // TAA output is the history texture written this frame.
            // NRDTaaPass writes: isEven → TaaHistory, !isEven → TaaHistoryPrev
            var taaOutput = pool.GetPoint(data.IsEven ? RenderResourceType.TaaHistory : RenderResourceType.TaaHistoryPrev);

            ds.SetConstantBuffer("GlobalConstants", res.ConstantBuffer);

            ds.SetTexture("gIn_PostAA",     taaOutput);
            ds.SetTexture("gIn_PreAA",      pool.GetPoint(RenderResourceType.Composed));
            ds.SetTexture("gIn_Validation", pool.GetPoint(RenderResourceType.Validation));

            ds.SetRWTexture("gOut_Final", pool.GetPoint(RenderResourceType.Final));

            uint groupsX = ((uint)data.Settings.OutputResolution.x + 15u) / 16u;
            uint groupsY = ((uint)data.Settings.OutputResolution.y + 15u) / 16u;
            cs.Dispatch(cmd, ds, groupsX, groupsY, 1);

            cmd.EndSample(RenderPassMarkers.Final);
        }

        // -------------------------------------------------------------------------
        // RenderGraph
        // -------------------------------------------------------------------------

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NRDFinalPass", out var passData);

            passData.Cs       = _cs;
            passData.Ds       = _ds;
            passData.Resource = _resource;
            passData.Settings = _settings;
            passData.Pool     = _resource.Pool;
            passData.IsEven   = _resource.IsEven;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }
    }
}
