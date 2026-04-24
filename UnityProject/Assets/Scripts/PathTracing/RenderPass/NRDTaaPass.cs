using System;
using NativeRender;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;
using static PathTracing.ShaderIDs;

namespace PathTracing
{
    /// <summary>
    /// Native compute shader TAA pass that dispatches Taa.computeshader.
    /// No TLAS required — binds only history/motion textures and constant buffer.
    /// </summary>
    public class NRDTaaPass : ScriptableRenderPass, IDisposable
    {
        private readonly NativeComputePipeline      _cs;
        private readonly NativeComputeDescriptorSet _dsPing; // isEven:  gIn_History=TaaHistoryPrev, gOut_Result=TaaHistory
        private readonly NativeComputeDescriptorSet _dsPong; // !isEven: gIn_History=TaaHistory,     gOut_Result=TaaHistoryPrev
        private          Resource              _resource;
        private          Settings              _settings;

        public NRDTaaPass(NativeComputeShader cs)
        {
            _cs     = new NativeComputePipeline(cs);
            _dsPing = new NativeComputeDescriptorSet(_cs);
            _dsPong = new NativeComputeDescriptorSet(_cs);
        }

        public void Dispose()
        {
            _dsPing?.Dispose();
            _dsPong?.Dispose();
            _cs?.Dispose();
        }

        public void Setup(Resource resource, Settings settings)
        {
            _resource = resource;
            _settings = settings;
        }

        // -------------------------------------------------------------------------
        // Resource / Settings  (mirrors TaaPass exactly)
        // -------------------------------------------------------------------------

        public class Resource
        {
            internal GraphicsBuffer ConstantBuffer;

            // RT textures sourced from the pool inside ExecutePass
            internal PathTracingResourcePool Pool;
            internal bool                    isEven;
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
            internal NativeComputePipeline      Cs;
            internal NativeComputeDescriptorSet DsPing;
            internal NativeComputeDescriptorSet DsPong;
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
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var cs  = data.Cs;
            var res = data.Resource;

            cmd.BeginSample(RenderPassMarkers.Taa);

            var pool = data.Pool;
            // Select the pre-configured ping/pong descriptor set.
            // History texture bindings were pre-baked in RecordRenderGraph.
            var ds = data.IsEven ? data.DsPing : data.DsPong;

            // Dynamic per-frame bindings (same regardless of ping/pong)
            ds.SetTexture("gIn_Mv",       pool.GetRT(RenderResourceType.MV).rt);
            ds.SetTexture("gIn_Composed", pool.GetRT(RenderResourceType.Composed).rt);
            ds.SetConstantBuffer("GlobalConstants", res.ConstantBuffer);

            cs.Dispatch(cmd, ds, (uint)data.Settings.rectGridW, (uint)data.Settings.rectGridH, 1);

            cmd.EndSample(RenderPassMarkers.Taa);
        }

        // -------------------------------------------------------------------------
        // RenderGraph
        // -------------------------------------------------------------------------

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NRDTaaPass", out var passData);

            passData.Cs       = _cs;
            passData.DsPing   = _dsPing;
            passData.DsPong   = _dsPong;
            passData.Resource = _resource;
            passData.Settings = _settings;
            passData.Pool     = _resource.Pool;
            passData.IsEven   = _resource.isEven;

            // Pre-bake history texture bindings.
            // Ping (isEven):  gIn_History = TaaHistoryPrev, gOut_Result = TaaHistory
            // Pong (!isEven): gIn_History = TaaHistory,     gOut_Result = TaaHistoryPrev
            var pool = _resource.Pool;
            _dsPing.SetTexture ("gIn_History",  pool.GetRT(RenderResourceType.TaaHistoryPrev).rt);
            _dsPing.SetRWTexture("gOut_Result", pool.GetRT(RenderResourceType.TaaHistory).rt);

            _dsPong.SetTexture ("gIn_History",  pool.GetRT(RenderResourceType.TaaHistory).rt);
            _dsPong.SetRWTexture("gOut_Result", pool.GetRT(RenderResourceType.TaaHistoryPrev).rt);
 

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}
