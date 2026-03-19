using mini;
using Unity.Mathematics;
using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;
using static PathTracing.ShaderIDs;
using static PathTracing.PathTracingUtils;

namespace PathTracing
{
    public class CompositionPass : ScriptableRenderPass
    {
        private readonly ComputeShader _compositionCs;
        private Resource _resource;
        private Settings _settings;


        public CompositionPass(ComputeShader compositionCs)
        {
            _compositionCs = compositionCs;
        }

        public void Setup(Resource sharcResource, Settings sharcSettings)
        {
            _resource = sharcResource;
            _settings = sharcSettings;
        }

        public class Resource
        {
            internal GraphicsBuffer ConstantBuffer;

            internal RTHandle ViewZ;
            internal RTHandle NormalRoughness;
            internal RTHandle BaseColorMetalness;
            internal RTHandle PsrThroughput;

            internal RTHandle Shadow;
            internal RTHandle Diff;
            internal RTHandle Spec;
        }

        public class Settings
        {
            internal int rectGridW;
            internal int rectGridH;
        }

        class PassData
        {
            internal ComputeShader CompositionCs;
            internal Resource Resource;
            internal Settings Settings;

            internal TextureHandle DirectLighting;
            internal TextureHandle DirectEmission;
            internal TextureHandle ComposedDiff;
            internal TextureHandle ComposedSpecViewZ;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            var compositionMarker = new ProfilerMarker(ProfilerCategory.Render, "Composition", MarkerFlags.SampleGPU);

            // 合成
            {
                natCmd.BeginSample(compositionMarker);
                natCmd.SetComputeConstantBufferParam(data.CompositionCs, paramsID, data.Resource.ConstantBuffer, 0, data.Resource.ConstantBuffer.stride);
                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_ViewZID, data.Resource.ViewZ);
                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_Normal_RoughnessID, data.Resource.NormalRoughness);
                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_BaseColor_MetalnessID, data.Resource.BaseColorMetalness);
                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_DirectLightingID, data.DirectLighting);
                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_DirectEmissionID, data.DirectEmission);

                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_ShadowID, data.Resource.Shadow);
                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_DiffID, data.Resource.Diff);
                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_SpecID, data.Resource.Spec);

                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_PsrThroughputID, data.Resource.PsrThroughput);
                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gOut_ComposedDiffID, data.ComposedDiff);
                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gOut_ComposedSpec_ViewZID, data.ComposedSpecViewZ);

                natCmd.DispatchCompute(data.CompositionCs, 0, (int)data.Settings.rectGridW, (int)data.Settings.rectGridH, 1);

                natCmd.EndSample(compositionMarker);
            }
        }


        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("Composition", out var passData);

            passData.CompositionCs = _compositionCs;

            passData.Resource = _resource;
            passData.Settings = _settings;
            
            
            var ptContextItem = frameData.Get<PTContextItem>();

            passData.DirectLighting = ptContextItem.DirectLighting;
            passData.DirectEmission = ptContextItem.DirectEmission;
            passData.ComposedDiff = ptContextItem.ComposedDiff;
            passData.ComposedSpecViewZ = ptContextItem.ComposedSpecViewZ;

            builder.UseTexture(passData.DirectLighting,  AccessFlags.ReadWrite);
            builder.UseTexture(passData.DirectEmission,  AccessFlags.ReadWrite);
            builder.UseTexture(passData.ComposedDiff,  AccessFlags.ReadWrite);
            builder.UseTexture(passData.ComposedSpecViewZ,  AccessFlags.ReadWrite);

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}