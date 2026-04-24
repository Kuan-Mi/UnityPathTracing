using System;
using System.Runtime.InteropServices;
using NativeRender;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Runs 5 ping-pong blur iterations over the SHARC gradient texture to produce
    /// a history-confidence value, matching NRDSample.cpp "History confidence - Blur".
    ///
    /// Ping-pong (matching NRDSample.cpp ConfidenceBlurPing/Pong descriptor sets):
    ///   i % 2 == 0 : gIn_Gradient = Gradient_Ping,  gOut_Gradient = Gradient_Pong
    ///   i % 2 == 1 : gIn_Gradient = Gradient_Pong,  gOut_Gradient = Gradient_Ping
    ///
    /// After 5 iterations (i = 0..4, last i = 4 → even) the final confidence
    /// value ends up in Gradient_Pong.
    ///
    /// Each iteration uses its own dedicated NativeComputeDescriptorSet (_ds[0..4]).
    /// </summary>
    public class NRDConfidenceBlurPass : ScriptableRenderPass, IDisposable
    {
        private const int IterationCount = 5; // must be odd per NRDSample

        private readonly NativeComputePipeline        _cs;
        private readonly NativeComputeDescriptorSet[] _ds; // one per iteration [0..4]
        private          Resource                     _resource;
        private          Settings                     _settings;

        [StructLayout(LayoutKind.Sequential)]
        private struct PushConstants
        {
            public int step;
        }

        public NRDConfidenceBlurPass(NativeComputeShader cs)
        {
            _cs = new NativeComputePipeline(cs);
            _ds = new NativeComputeDescriptorSet[IterationCount];
            for (int i = 0; i < IterationCount; i++)
                _ds[i] = new NativeComputeDescriptorSet(_cs);
        }

        public void Dispose()
        {
            if (_ds != null)
                foreach (var ds in _ds)
                    ds?.Dispose();
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
            internal GraphicsBuffer ConstantBuffer;   // GlobalConstants CBV
            internal PathTracingResourcePool Pool;
        }

        public class Settings
        {
            internal uint GroupsX; // sharcW / 16
            internal uint GroupsY; // sharcH / 16
        }

        // -------------------------------------------------------------------------
        // Pass data
        // -------------------------------------------------------------------------

        class PassData
        {
            internal NativeComputePipeline        Cs;
            internal NativeComputeDescriptorSet[] Ds;  // [0..4], one per iteration
            internal Resource                     Resource;
            internal Settings                     Settings;
            internal PathTracingResourcePool      Pool;
        }

        // -------------------------------------------------------------------------
        // Execution
        // -------------------------------------------------------------------------

        static unsafe void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd  = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var cs   = data.Cs;
            var res  = data.Resource;

            cmd.BeginSample(RenderPassMarkers.ConfidenceBlur);

            for (int i = 0; i < IterationCount; i++)
            {
                var ds = data.Ds[i];

                ds.SetConstantBuffer("GlobalConstants", res.ConstantBuffer);

                var push = new PushConstants { step = 1 + i };
                ds.SetRootConstants("g_PushConstants", &push);

                cs.Dispatch(cmd, ds, data.Settings.GroupsX, data.Settings.GroupsY, 1);
            }

            cmd.EndSample(RenderPassMarkers.ConfidenceBlur);
        }

        // -------------------------------------------------------------------------
        // RenderGraph
        // -------------------------------------------------------------------------

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NRDConfidenceBlurPass", out var passData);

            passData.Cs          = _cs;
            passData.Ds          = _ds;
            passData.Resource    = _resource;
            passData.Settings    = _settings;
            passData.Pool        = _resource.Pool;

            // Pre-bake gradient texture bindings for each iteration.
            // i%2==0: gIn=Gradient_Ping,  gOut=Gradient_Pong
            // i%2==1: gIn=Gradient_Pong,  gOut=Gradient_Ping
            var pool = _resource.Pool;
            for (int i = 0; i < IterationCount; i++)
            {
                bool isPing = (i % 2 == 0);
                _ds[i].SetTexture ("gIn_Gradient",  pool.GetRT(isPing ? RenderResourceType.Gradient_Ping : RenderResourceType.Gradient_Pong).rt);
                _ds[i].SetRWTexture("gOut_Gradient", pool.GetRT(isPing ? RenderResourceType.Gradient_Pong : RenderResourceType.Gradient_Ping).rt);
            }

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }
    }
}
