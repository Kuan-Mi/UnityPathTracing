using System;
using mini;
using NativeRender;
using Unity.Mathematics;
using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;
using static PathTracing.ShaderIDs;

namespace PathTracing
{
    public class NativeOpaquePass : ScriptableRenderPass, IDisposable
    {
        private readonly RayTracePipeline _opaqueTs;
        private          Resource       _resource;
        private          Settings       _settings;


        public NativeOpaquePass(RayTraceShader opaqueTs)
        {
            _opaqueTs = new RayTracePipeline(opaqueTs);
        }

        public void Dispose()
        {
            _opaqueTs?.Dispose();
        }

        public void Setup(Resource sharcResource, Settings sharcSettings)
        {
            _resource = sharcResource;
            _settings = sharcSettings;
        }

        public class Resource
        {
            internal GraphicsBuffer ConstantBuffer;

            internal GraphicsBuffer HashEntriesBuffer;
            internal GraphicsBuffer AccumulationBuffer;
            internal GraphicsBuffer ResolvedBuffer;

            internal GraphicsBuffer SpotLightBuffer;
            internal GraphicsBuffer AreaLightBuffer;
            internal GraphicsBuffer PointLightBuffer;

            internal GraphicsBuffer ScramblingRanking;
            internal GraphicsBuffer Sobol;


            internal RTHandle Mv;
            internal RTHandle ViewZ;
            internal RTHandle NormalRoughness;
            internal RTHandle BaseColorMetalness;
            internal RTHandle GeoNormal;
            internal RTHandle DirectLighting;


            internal RTHandle Penumbra;
            internal RTHandle Diff;
            internal RTHandle Spec;


            internal RTHandle PrevViewZ;
            internal RTHandle PrevNormalRoughness;
            internal RTHandle PrevBaseColorMetalness;
            internal RTHandle PrevGeoNormal;

            internal RTHandle PsrThroughput;

            internal RTHandle Output;
            internal RTHandle DirectEmission;
            internal RTHandle ComposedDiff;
            internal RTHandle ComposedSpecViewZ;
            
            // internal TextureHandle OutputTexture;
        }

        public class Settings
        {
            internal int2 m_RenderResolution;
            internal float resolutionScale;     
            internal int convergenceStep;
        }

        class PassData
        {
            internal RayTracePipeline OpaqueTs;
            internal Resource       Resource;
            internal Settings       Settings;

        }

        void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            
            var opaqueTracingMarker = RenderPassMarkers.OpaqueTracing;

            natCmd.BeginSample(opaqueTracingMarker);

            
            _gpuScene.BuildAccelerationStructure(natCmd);
            
            data.OpaqueTs.SetAccelerationStructure("gWorldTlas", _gpuScene.AccelerationStructure);
            
            
            
            var resource = data.Resource;
            var settings = data.Settings;


            data.OpaqueTs.SetConstantBuffer("GlobalConstants", resource.ConstantBuffer);

            data.OpaqueTs.SetBuffer("gIn_ScramblingRanking", resource.ScramblingRanking);
            data.OpaqueTs.SetBuffer("gIn_Sobol", resource.Sobol);

            data.OpaqueTs.SetRWStructuredBuffer("gInOut_SharcHashEntriesBuffer", resource.HashEntriesBuffer);
            data.OpaqueTs.SetRWStructuredBuffer("gInOut_SharcAccumulated",      resource.AccumulationBuffer);
            data.OpaqueTs.SetRWStructuredBuffer("gInOut_SharcResolved",         resource.ResolvedBuffer);


            data.OpaqueTs.SetRWTexture("g_Output",              resource.Output.rt);

            data.OpaqueTs.SetRWTexture("gOut_Mv",               resource.Mv.rt);
            data.OpaqueTs.SetRWTexture("gOut_ViewZ",            resource.ViewZ.rt);
            data.OpaqueTs.SetRWTexture("gOut_Normal_Roughness", resource.NormalRoughness.rt);
            data.OpaqueTs.SetRWTexture("gOut_BaseColor_Metalness", resource.BaseColorMetalness.rt);

            data.OpaqueTs.SetRWTexture("gOut_DirectLighting",   resource.DirectLighting.rt);
            data.OpaqueTs.SetRWTexture("gOut_DirectEmission",   resource.DirectEmission.rt);
            data.OpaqueTs.SetRWTexture("gOut_PsrThroughput",    resource.PsrThroughput.rt);

            data.OpaqueTs.SetRWTexture("gOut_ShadowData",       resource.Penumbra.rt);
            data.OpaqueTs.SetRWTexture("gOut_Diff",             resource.Diff.rt);
            data.OpaqueTs.SetRWTexture("gOut_Spec",             resource.Spec.rt);

            data.OpaqueTs.SetTexture("gIn_PrevComposedDiff",         resource.ComposedDiff.rt);
            data.OpaqueTs.SetTexture("gIn_PrevComposedSpec_PrevViewZ", resource.ComposedSpecViewZ.rt);

            data.OpaqueTs.SetTexture("gIn_PrevViewZ",            resource.PrevViewZ.rt);
            data.OpaqueTs.SetTexture("gIn_PrevNormalRoughness",  resource.PrevNormalRoughness.rt);
            data.OpaqueTs.SetTexture("gIn_PrevBaseColorMetalness", resource.PrevBaseColorMetalness.rt);

            data.OpaqueTs.SetRWTexture("gOut_GeoNormal",         resource.GeoNormal.rt);
            data.OpaqueTs.SetTexture("gIn_PrevGeoNormal",        resource.PrevGeoNormal.rt);

            data.OpaqueTs.SetStructuredBuffer("gIn_SpotLights",  resource.SpotLightBuffer);
            data.OpaqueTs.SetStructuredBuffer("gIn_AreaLights",  resource.AreaLightBuffer);
            data.OpaqueTs.SetStructuredBuffer("gIn_PointLights", resource.PointLightBuffer);


            uint rectWmod = (uint)(settings.m_RenderResolution.x * settings.resolutionScale + 0.5f);
            uint rectHmod = (uint)(settings.m_RenderResolution.y * settings.resolutionScale + 0.5f);

            _gpuScene.BindToShader(data.OpaqueTs);
            data.OpaqueTs.Dispatch(natCmd, rectWmod, rectHmod);

            natCmd.EndSample(opaqueTracingMarker);
        }
        private GPUScene _gpuScene;
        public void SetGPUScene(GPUScene scene)
        {
            _gpuScene = scene;
        }
        private TextureHandle CreateTex(TextureDesc textureDesc, RenderGraph renderGraph, string name, GraphicsFormat format)
        {
            textureDesc.format = format;
            textureDesc.name = name;
            return renderGraph.CreateTexture(textureDesc);
        }


        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("Opaque", out var passData);

            passData.OpaqueTs = _opaqueTs;

            passData.Resource = _resource;
            passData.Settings = _settings;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}