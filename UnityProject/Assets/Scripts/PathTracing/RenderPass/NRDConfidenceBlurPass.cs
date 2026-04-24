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

        private readonly NativeComputePipeline      _cs;
        private readonly NativeComputeDescriptorSet _dsPing; // i%2==0: gIn=Gradient_Ping,  gOut=Gradient_Pong
        private readonly NativeComputeDescriptorSet _dsPong; // i%2==1: gIn=Gradient_Pong,  gOut=Gradient_Ping
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
            _cs     = new NativeComputePipeline(cs);
            _dsPing = new NativeComputeDescriptorSet(_cs);
            _dsPong = new NativeComputeDescriptorSet(_cs);

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
            _dsPing?.Dispose();
            _dsPong?.Dispose();
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
            internal NativeComputePipeline      Cs;
            internal NativeComputeDescriptorSet DsPing;
            internal NativeComputeDescriptorSet DsPong;
            internal GraphicsBuffer[]           StepBuffers;
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
            var res  = data.Resource;

            cmd.BeginSample(RenderPassMarkers.ConfidenceBlur);

            for (int i = 0; i < IterationCount; i++)
            {
                // Select the pre-configured ping/pong descriptor set.
                // Gradient texture bindings were pre-baked in RecordRenderGraph.
                var ds = (i % 2 == 0) ? data.DsPing : data.DsPong;

                ds.SetConstantBuffer("GlobalConstants", res.ConstantBuffer);
                ds.SetConstantBuffer("g_PushConstants", data.StepBuffers[i]);

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
            passData.DsPing      = _dsPing;
            passData.DsPong      = _dsPong;
            passData.StepBuffers = _stepBuffers;
            passData.Resource    = _resource;
            passData.Settings    = _settings;
            passData.Pool        = _resource.Pool;

            // Pre-bake gradient texture bindings.
            // Ping (i%2==0): gIn=Gradient_Ping,  gOut=Gradient_Pong
            // Pong (i%2==1): gIn=Gradient_Pong,  gOut=Gradient_Ping
            var pool = _resource.Pool;
            _dsPing.SetTexture ("gIn_Gradient",  pool.GetRT(RenderResourceType.Gradient_Ping).rt);
            _dsPing.SetRWTexture("gOut_Gradient", pool.GetRT(RenderResourceType.Gradient_Pong).rt);

            _dsPong.SetTexture ("gIn_Gradient",  pool.GetRT(RenderResourceType.Gradient_Pong).rt);
            _dsPong.SetRWTexture("gOut_Gradient", pool.GetRT(RenderResourceType.Gradient_Ping).rt);

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }
    }
}
