using System;
using System.Runtime.InteropServices;
using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using PathTracing;
using PathTracing.Profiling;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;
using static PathTracing.ShaderIDs;

namespace PathTracing
{
    public class DlssBeforePass : ScriptableRenderPass
    {
        private readonly ComputeShader DlssBeforeCs;

        private Resource _resource;
        private Settings _settings;

        public DlssBeforePass(ComputeShader dlssBeforeCs)
        {
            DlssBeforeCs = dlssBeforeCs;
        }

        public void Setup(Resource resource, Settings settings)
        {
            _resource = resource;
            _settings = settings;
        }


        public class Resource
        {
            public GraphicsBuffer ConstantBuffer;

            public RTHandle NormalRoughness;
            public RTHandle BaseColorMetalness;
            public RTHandle Spec;

            public RTHandle ViewZ;
            public RTHandle RRGuide_DiffAlbedo;
            public RTHandle RRGuide_SpecAlbedo;
            public RTHandle RRGuide_SpecHitDistance;
            public RTHandle RRGuide_Normal_Roughness;
        }

        public class Settings
        {
            public int  rectGridW;
            public int  rectGridH;
            public bool tmpDisableRR;
        }

        class PassData
        {
            public ComputeShader DlssBeforeCs;
            public Resource      Resource;
            public Settings      Setting;
        }

        [DllImport("Denoiser")]
        private static extern IntPtr GetRenderEventAndDataFunc();

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            var dlssBeforeMarker = RenderPassMarkers.DlssBefore;

            // dlss Before
            natCmd.BeginSample(dlssBeforeMarker);
            natCmd.SetComputeConstantBufferParam(data.DlssBeforeCs, paramsID, data.Resource.ConstantBuffer, 0, data.Resource.ConstantBuffer.stride);

            natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gIn_Normal_Roughness", data.Resource.NormalRoughness);
            natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gIn_BaseColor_Metalness", data.Resource.BaseColorMetalness);
            natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gIn_Spec", data.Resource.Spec);

            natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gInOut_ViewZ", data.Resource.ViewZ);
            natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gOut_DiffAlbedo", data.Resource.RRGuide_DiffAlbedo);
            natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gOut_SpecAlbedo", data.Resource.RRGuide_SpecAlbedo);
            natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gOut_SpecHitDistance", data.Resource.RRGuide_SpecHitDistance);
            natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gOut_Normal_Roughness", data.Resource.RRGuide_Normal_Roughness);


            natCmd.DispatchCompute(data.DlssBeforeCs, 0, (int)data.Setting.rectGridW, (int)data.Setting.rectGridH, 1);
            natCmd.EndSample(dlssBeforeMarker);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("DLSS RR Before", out var passData);

            passData.DlssBeforeCs = DlssBeforeCs;
            passData.Resource     = _resource;
            passData.Setting      = _settings;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}