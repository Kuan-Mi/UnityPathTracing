
using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;
using static PathTracing.ShaderIDs;

namespace PathTracing
{
    public class TaaPass: ScriptableRenderPass
    {
        private readonly ComputeShader _taaCs;
        private Resource _resource;
        private Settings _settings;


        public TaaPass(ComputeShader taaCs)
        {
            _taaCs = taaCs;
        }

        public void Setup(Resource sharcResource, Settings sharcSettings)
        {
            _resource = sharcResource;
            _settings = sharcSettings;
        }

        public class Resource
        {
            internal GraphicsBuffer ConstantBuffer;

            internal RTHandle Mv;
            internal RTHandle Composed;
            internal RTHandle taaSrc;
            internal RTHandle taaDst;
        }

        public class Settings
        {
            internal int rectGridW;
            internal int rectGridH;
        }

        class PassData
        {
            internal ComputeShader TaaCs;
            internal Resource Resource;
            internal Settings Settings;

            internal TextureHandle OutputTexture;
            
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            var taaMarker = new ProfilerMarker(ProfilerCategory.Render, "TAA", MarkerFlags.SampleGPU);
            
            natCmd.BeginSample(taaMarker);

            natCmd.SetComputeConstantBufferParam(data.TaaCs, paramsID, data.Resource.ConstantBuffer, 0, data.Resource.ConstantBuffer.stride);
            natCmd.SetComputeTextureParam(data.TaaCs, 0, gIn_MvID, data.Resource.Mv);
            natCmd.SetComputeTextureParam(data.TaaCs, 0, gIn_ComposedID, data.Resource.Composed);
            natCmd.SetComputeTextureParam(data.TaaCs, 0, gIn_HistoryID, data.Resource.taaSrc);
            natCmd.SetComputeTextureParam(data.TaaCs, 0, gOut_ResultID, data.Resource.taaDst);
            natCmd.SetComputeTextureParam(data.TaaCs, 0, gOut_DebugID, data.OutputTexture);
            natCmd.DispatchCompute(data.TaaCs, 0, (int)data.Settings.rectGridW, (int)data.Settings.rectGridH, 1);
            natCmd.EndSample(taaMarker);
        }


        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("Taa Pass", out var passData);

            passData.TaaCs = _taaCs;

            passData.Resource = _resource;
            passData.Settings = _settings;
            
            
            var ptContextItem = frameData.Get<PTContextItem>();

            passData.OutputTexture = ptContextItem.OutputTexture; 

            builder.UseTexture(passData.OutputTexture,  AccessFlags.ReadWrite); 

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}