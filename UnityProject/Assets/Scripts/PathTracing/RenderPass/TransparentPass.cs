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
    public class TransparentPass : ScriptableRenderPass
    {
        private readonly RayTracingShader _transparentTs;
        private Resource _resource;
        private Settings _settings;


        public TransparentPass(RayTracingShader transparentCs)
        {
            _transparentTs = transparentCs;
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
            internal RTHandle NormalRoughness;

            internal GraphicsBuffer HashEntriesBuffer;
            internal GraphicsBuffer AccumulationBuffer;
            internal GraphicsBuffer ResolvedBuffer;

            internal GraphicsBuffer SpotLightBuffer;
            internal GraphicsBuffer AreaLightBuffer;
            internal GraphicsBuffer PointLightBuffer;

            internal GraphicsBuffer AeExposureBuffer;
        }

        public class Settings
        {
            internal int2 m_RenderResolution;        
            internal int convergenceStep;

        }

        class PassData
        {
            internal RayTracingShader TransparentTs;
            internal Resource Resource;
            internal Settings Settings;

            internal TextureHandle ComposedDiff;
            internal TextureHandle ComposedSpecViewZ;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);


            var transparentTracingMarker = RenderPassMarkers.TransparentTracing;
            natCmd.BeginSample(transparentTracingMarker);

            natCmd.SetRayTracingShaderPass(data.TransparentTs, "Test2");
            natCmd.SetRayTracingConstantBufferParam(data.TransparentTs, paramsID, data.Resource.ConstantBuffer, 0, data.Resource.ConstantBuffer.stride);

            natCmd.SetRayTracingBufferParam(data.TransparentTs, g_HashEntriesID, data.Resource.HashEntriesBuffer);
            natCmd.SetRayTracingBufferParam(data.TransparentTs, g_AccumulationBufferID, data.Resource.AccumulationBuffer);
            natCmd.SetRayTracingBufferParam(data.TransparentTs, g_ResolvedBufferID, data.Resource.ResolvedBuffer);


            natCmd.SetRayTracingTextureParam(data.TransparentTs, gIn_ComposedDiffID, data.ComposedDiff);
            natCmd.SetRayTracingTextureParam(data.TransparentTs, gIn_ComposedSpec_ViewZID, data.ComposedSpecViewZ);
            natCmd.SetRayTracingTextureParam(data.TransparentTs, g_Normal_RoughnessID, data.Resource.NormalRoughness);
            natCmd.SetRayTracingTextureParam(data.TransparentTs, gOut_ComposedID, data.Resource.Composed);
            natCmd.SetRayTracingTextureParam(data.TransparentTs, GInOutMv, data.Resource.Mv);

            natCmd.SetRayTracingBufferParam(data.TransparentTs, gIn_SpotLightsID, data.Resource.SpotLightBuffer);
            natCmd.SetRayTracingBufferParam(data.TransparentTs, gIn_AreaLightsID, data.Resource.AreaLightBuffer);
            natCmd.SetRayTracingBufferParam(data.TransparentTs, gIn_PointLightsID, data.Resource.PointLightBuffer);
            natCmd.SetRayTracingBufferParam(data.TransparentTs, _AE_ExposureBufferID, data.Resource.AeExposureBuffer);
            natCmd.SetRayTracingIntParam(data.TransparentTs, g_ConvergenceStepID, data.Settings.convergenceStep);

            
            
            natCmd.DispatchRays(data.TransparentTs, "MainRayGenShader", (uint)data.Settings.m_RenderResolution.x, (uint)data.Settings.m_RenderResolution.y, 1);
            natCmd.EndSample(transparentTracingMarker);
        }


        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("Transparent", out var passData);

            passData.TransparentTs = _transparentTs;

            passData.Resource = _resource;
            passData.Settings = _settings;

            var ptContextItem = frameData.Get<PTContextItem>();

            passData.ComposedDiff = ptContextItem.ComposedDiff;
            passData.ComposedSpecViewZ = ptContextItem.ComposedSpecViewZ;

            builder.UseTexture(passData.ComposedDiff, AccessFlags.ReadWrite);
            builder.UseTexture(passData.ComposedSpecViewZ, AccessFlags.ReadWrite);

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}