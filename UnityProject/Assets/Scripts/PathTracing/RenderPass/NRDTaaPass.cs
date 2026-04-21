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

            internal RTHandle Mv;
            internal RTHandle Composed;
            internal RTHandle taaSrc;
            internal RTHandle taaDst;

            internal RTHandle Output;
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
            internal NativeComputePipeline Cs;
            internal Resource              Resource;
            internal Settings              Settings;

            internal TextureHandle OutputTexture;
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

            // SRV inputs
            cs.SetTexture("gIn_Mv",       res.Mv.rt);
            cs.SetTexture("gIn_Composed", res.Composed.rt);
            cs.SetTexture("gIn_History",  res.taaSrc.rt);

            // UAV output
            cs.SetRWTexture("gOut_Result", res.taaDst.rt);

            // Debug output (matches original TaaPass gOut_Debug → Output)
            cs.SetRWTexture("gOut_Debug", data.OutputTexture);

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

            passData.OutputTexture = renderGraph.ImportTexture(_resource.Output);

            builder.UseTexture(passData.OutputTexture, AccessFlags.ReadWrite);

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}
