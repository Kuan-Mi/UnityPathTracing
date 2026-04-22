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
        private readonly NativeComputePipeline _cs;
        private          Resource              _resource;
        private          Settings              _settings;

        public NRDTaaPass(NativeComputeShader cs)
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
            internal NativeComputePipeline   Cs;
            internal Resource                Resource;
            internal Settings                Settings;
            internal PathTracingResourcePool Pool;
            internal bool                    IsEven;
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

            var pool   = data.Pool;
            var isEven = data.IsEven;

            // SRV inputs
            cs.SetTexture("gIn_Mv",       pool.GetRT(RenderResourceType.MV).rt);
            cs.SetTexture("gIn_Composed", pool.GetRT(RenderResourceType.Composed).rt);
            cs.SetTexture("gIn_History",  pool.GetRT(isEven ? RenderResourceType.TaaHistoryPrev : RenderResourceType.TaaHistory).rt);

            // UAV output
            cs.SetRWTexture("gOut_Result", pool.GetRT(isEven ? RenderResourceType.TaaHistory : RenderResourceType.TaaHistoryPrev).rt);

            // Constant buffer
            cs.SetConstantBuffer("GlobalConstants", res.ConstantBuffer);

            // Dispatch — numthreads [16, 16, 1]
            cs.Dispatch(cmd, (uint)data.Settings.rectGridW, (uint)data.Settings.rectGridH, 1);

            cmd.EndSample(RenderPassMarkers.Taa);
        }

        // -------------------------------------------------------------------------
        // RenderGraph
        // -------------------------------------------------------------------------

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NRDTaaPass", out var passData);

            passData.Cs       = _cs;
            passData.Resource = _resource;
            passData.Settings = _settings;
            passData.Pool     = _resource.Pool;
            passData.IsEven   = _resource.isEven;
 

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}
