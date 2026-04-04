using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;
using static PathTracing.ShaderIDs;

namespace PathTracing
{
    public class AccumulatePass: ScriptableRenderPass
    {
        private readonly ComputeShader _accumulateCs;
        private Resource _resource;
        private Settings _settings;


        public AccumulatePass(ComputeShader accumulateCs)
        {
            _accumulateCs = accumulateCs;
        }

        public void Setup(Resource sharcResource, Settings sharcSettings)
        {
            _resource = sharcResource;
            _settings = sharcSettings;
        }

        public class Resource
        {
            internal RTHandle noise;
            internal RTHandle accumulation;
        }

        public class Settings
        {
            internal int rectGridW;
            internal int rectGridH;
            internal int convergenceStep;
        }

        class PassData
        {
            internal ComputeShader AccCs;
            internal Resource Resource;
            internal Settings Settings;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            var compositionMarker = RenderPassMarkers.Acc;

            // 合成
            {
                natCmd.BeginSample(compositionMarker);
                 natCmd.SetComputeTextureParam(data.AccCs, 0, gIn_noiseID, data.Resource.noise);
                natCmd.SetComputeTextureParam(data.AccCs, 0, gIn_AccumulatedID, data.Resource.accumulation);
                natCmd.SetComputeIntParam(data.AccCs, g_ConvergenceStepID, data.Settings.convergenceStep);

                natCmd.DispatchCompute(data.AccCs, 0, (int)data.Settings.rectGridW, (int)data.Settings.rectGridH, 1);

                natCmd.EndSample(compositionMarker);
            }
        }


        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("Acc", out var passData);

            passData.AccCs = _accumulateCs;

            passData.Resource = _resource;
            passData.Settings = _settings;
            
            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}