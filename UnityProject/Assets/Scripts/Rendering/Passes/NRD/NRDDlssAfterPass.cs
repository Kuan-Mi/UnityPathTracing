using System;
using NativeRender;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Post-processes the DLSS output (tone-map / color corrections).
    /// Runs after both DLSS-SR and DLSS-RR.
    ///
    /// DlssAfter.computeshader — numthreads [16, 16, 1]
    ///
    /// -- UAV (1) --
    ///   gOut_Image    RWTexture2D  space0:u0   (DlssOutput, read-modify-write)
    ///
    /// -- CBV (1) --
    ///   GlobalConstants  ConstantBuffer  space4:b0
    /// </summary>
    public class NRDDlssAfterPass : ScriptableRenderPass, IDisposable
    {
        private readonly NativeComputePipeline      _cs;
        private readonly NativeComputeDescriptorSet _ds;
        private          Resource                   _resource;
        private          Settings                   _settings;

        public NRDDlssAfterPass(NativeComputeShader cs)
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

        public class Resource
        {
            internal IntPtr                  ConstantBuffer;
            internal PathTracingResourcePool Pool;
        }

        public class Settings
        {
            internal int outputGridW;
            internal int outputGridH;
        }

        class PassData
        {
            internal NativeComputePipeline      Cs;
            internal NativeComputeDescriptorSet Ds;
            internal Resource                   Resource;
            internal Settings                   Settings;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var cs  = data.Cs;
            var ds  = data.Ds;
            var res = data.Resource;

            cmd.BeginSample(RenderPassMarkers.DlssAfter);

            ds.SetRWTexture("gOut_Image", res.Pool.GetPoint(RenderResourceType.DlssOutput));
            ds.SetConstantBuffer("GlobalConstants", res.ConstantBuffer);

            cs.Dispatch(cmd, ds, (uint)data.Settings.outputGridW, (uint)data.Settings.outputGridH, 1);

            cmd.EndSample(RenderPassMarkers.DlssAfter);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NRDDlssAfterPass", out var passData);

            passData.Cs       = _cs;
            passData.Ds       = _ds;
            passData.Resource = _resource;
            passData.Settings = _settings;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }
    }
}
