using Unity.Mathematics;
using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;
using static PathTracing.ShaderIDs;

namespace PathTracing
{
    public class AutoExposurePass: ScriptableRenderPass
    {
        private readonly ComputeShader _aeCs;
        
        private Resource _sharcResource;
        private Settings _sharcSettings;
        
        
        public AutoExposurePass( ComputeShader aeCs)
        {
            _aeCs = aeCs;
        }

        public void Setup( Resource sharcResource, Settings sharcSettings)
        {
            _sharcResource = sharcResource;
            _sharcSettings = sharcSettings;
        }

        public class Resource
        {
            internal GraphicsBuffer AeHistogramBuffer;
            internal GraphicsBuffer AeExposureBuffer;
            
            internal RTHandle Composed;
        }

        public class Settings
        {
            internal bool AeEnabled;
            internal float AeEVMin;
            internal float AeEVMax;
            internal float AeLowPercent;
            internal float AeHighPercent;
            internal float AeSpeedUp;
            internal float AeSpeedDown;
            internal float AeDeltaTime;
            internal float AeExposureCompensation;
            internal float AeMinExposure;
            internal float AeMaxExposure;
            internal uint AeTexWidth;
            internal uint AeTexHeight;
            internal float ManualExposure;
        }
        
        class SharcPassData
        {
            internal ComputeShader AeCs;
            internal Resource Resource;
            internal Settings Settings;
        }

        static void ExecutePass(SharcPassData data, UnsafeGraphContext context)
        {
            
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            
            var aeMarker = RenderPassMarkers.AutoExposure;
            
            
            // ── Auto-exposure: histogram build + reduce (after transparent, before TAA) ──
            // if (data.AeEnabled && data.AeCs != null && data.AeHistogramBuffer != null && data.AeExposureBuffer != null)
            {
                natCmd.BeginSample(aeMarker);

                int kernelClear  = data.AeCs.FindKernel("ClearHistogram");
                int kernelBuild  = data.AeCs.FindKernel("BuildHistogram");
                int kernelReduce = data.AeCs.FindKernel("ReduceHistogram");

                // -- Kernel 0: Clear --
                natCmd.SetComputeBufferParam(data.AeCs, kernelClear, _AE_HistogramBufferID, data.Resource.AeHistogramBuffer);
                natCmd.DispatchCompute(data.AeCs, kernelClear, 1, 1, 1);

                // -- Kernel 1: Build --
                natCmd.SetComputeTextureParam(data.AeCs, kernelBuild, _AE_ComposedTextureID, data.Resource.Composed);
                natCmd.SetComputeBufferParam(data.AeCs, kernelBuild, _AE_HistogramBufferID, data.Resource.AeHistogramBuffer);
                natCmd.SetComputeIntParam(data.AeCs, _AE_TexWidthID,  (int)data.Settings.AeTexWidth);
                natCmd.SetComputeIntParam(data.AeCs, _AE_TexHeightID, (int)data.Settings.AeTexHeight);
                natCmd.SetComputeFloatParam(data.AeCs, _AE_EVMinID, data.Settings.AeEVMin);
                natCmd.SetComputeFloatParam(data.AeCs, _AE_EVMaxID, data.Settings.AeEVMax);
                uint buildX = (data.Settings.AeTexWidth  + 15u) / 16u;
                uint buildY = (data.Settings.AeTexHeight + 15u) / 16u;
                natCmd.DispatchCompute(data.AeCs, kernelBuild, (int)buildX, (int)buildY, 1);

                // -- Kernel 2: Reduce --
                natCmd.SetComputeBufferParam(data.AeCs, kernelReduce, _AE_HistogramBufferID, data.Resource.AeHistogramBuffer);
                natCmd.SetComputeBufferParam(data.AeCs, kernelReduce, _AE_ExposureBufferID,  data.Resource.AeExposureBuffer);
                natCmd.SetComputeFloatParam(data.AeCs, _AE_EVMinID,                data.Settings.AeEVMin);
                natCmd.SetComputeFloatParam(data.AeCs, _AE_EVMaxID,                data.Settings.AeEVMax);
                natCmd.SetComputeFloatParam(data.AeCs, _AE_LowPercentID,           data.Settings.AeLowPercent);
                natCmd.SetComputeFloatParam(data.AeCs, _AE_HighPercentID,          data.Settings.AeHighPercent);
                natCmd.SetComputeFloatParam(data.AeCs, _AE_SpeedUpID,              data.Settings.AeSpeedUp);
                natCmd.SetComputeFloatParam(data.AeCs, _AE_SpeedDownID,            data.Settings.AeSpeedDown);
                natCmd.SetComputeFloatParam(data.AeCs, _AE_DeltaTimeID,            data.Settings.AeDeltaTime);
                natCmd.SetComputeFloatParam(data.AeCs, _AE_ExposureCompensationID, data.Settings.AeExposureCompensation);
                natCmd.SetComputeFloatParam(data.AeCs, _AE_MinExposureID,          data.Settings.AeMinExposure);
                natCmd.SetComputeFloatParam(data.AeCs, _AE_MaxExposureID,          data.Settings.AeMaxExposure);
                natCmd.DispatchCompute(data.AeCs, kernelReduce, 1, 1, 1);

                natCmd.EndSample(aeMarker);
            }
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<SharcPassData>("Auto Exposure", out var passData);
            
            passData.AeCs = _aeCs;

            passData.Resource = _sharcResource;
            passData.Settings = _sharcSettings;
            
            builder.AllowPassCulling(false);
            builder.SetRenderFunc((SharcPassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}