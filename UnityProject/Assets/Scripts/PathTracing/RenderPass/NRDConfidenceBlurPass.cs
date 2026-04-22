using System;
using System.Runtime.InteropServices;
using NativeRender;
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
    /// The "step" push-constant (value = 1 + i) is supplied via five pre-allocated
    /// 16-byte constant buffers (minimum D3D12 CBV size).
    /// </summary>
    public class NRDConfidenceBlurPass : ScriptableRenderPass, IDisposable
    {
        private const int IterationCount = 5; // must be odd per NRDSample

        private readonly NativeComputePipeline _cs;
        private          Resource              _resource;
        private          Settings              _settings;

        // Five tiny constant buffers, one per iteration, pre-filled with step = 1..5.
        // MinimumD3D12 CBV size = 256 bytes, but Unity's GraphicsBuffer manages alignment.
        // We store one int + 3 padding ints (16 bytes stride) to meet constant-buffer rules.
        private readonly GraphicsBuffer[] _stepBuffers = new GraphicsBuffer[IterationCount];

        [StructLayout(LayoutKind.Sequential)]
        private struct StepConstants
        {
            public int   step;
            public int   _pad0;
            public int   _pad1;
            public int   _pad2;
        }

        public NRDConfidenceBlurPass(NativeComputeShader cs)
        {
            _cs = new NativeComputePipeline(cs);

            var data = new StepConstants[1];
            for (int i = 0; i < IterationCount; i++)
            {
                _stepBuffers[i] = new GraphicsBuffer(GraphicsBuffer.Target.Constant, 1, Marshal.SizeOf<StepConstants>());
                data[0].step    = 1 + i;
                _stepBuffers[i].SetData(data);
            }
        }

        public void Dispose()
        {
            _cs?.Dispose();
            foreach (var buf in _stepBuffers)
                buf?.Release();
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
            internal NativeComputePipeline   Cs;
            internal GraphicsBuffer[]        StepBuffers;
            internal Resource                Resource;
            internal Settings                Settings;
            internal PathTracingResourcePool Pool;
        }

        // -------------------------------------------------------------------------
        // Execution
        // -------------------------------------------------------------------------

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd  = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var cs   = data.Cs;
            var pool = data.Pool;
            var res  = data.Resource;

            cmd.BeginSample(RenderPassMarkers.ConfidenceBlur);

            for (int i = 0; i < IterationCount; i++)
            {
                bool isPing = (i % 2 == 0);

                var inTex  = pool.GetRT(isPing ? RenderResourceType.Gradient_Ping : RenderResourceType.Gradient_Pong);
                var outTex = pool.GetRT(isPing ? RenderResourceType.Gradient_Pong : RenderResourceType.Gradient_Ping);

                cs.SetConstantBuffer("GlobalConstants",   res.ConstantBuffer);
                cs.SetConstantBuffer("g_PushConstants",   data.StepBuffers[i]);

                cs.SetTexture  ("gIn_Gradient",  inTex.rt);
                cs.SetRWTexture("gOut_Gradient", outTex.rt);

                cs.Dispatch(cmd, data.Settings.GroupsX, data.Settings.GroupsY, 1);
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
            passData.StepBuffers = _stepBuffers;
            passData.Resource    = _resource;
            passData.Settings    = _settings;
            passData.Pool        = _resource.Pool;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }
    }
}
