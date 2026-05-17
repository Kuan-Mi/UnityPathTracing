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
        private readonly RayTracePipeline            _opaqueTs;
        private readonly NativeRayTraceDescriptorSet _ds;
        private          Resource                    _resource;
        private          Settings                    _settings;


        public NativeOpaquePass(RayTraceShader opaqueTs)
        {
            _opaqueTs = new RayTracePipeline(opaqueTs);
            _ds       = new NativeRayTraceDescriptorSet(_opaqueTs);
        }

        public void Dispose()
        {
            _ds?.Dispose();
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
            internal RayTracePipeline            OpaqueTs;
            internal NativeRayTraceDescriptorSet Ds;
            internal Resource                    Resource;
            internal Settings                    Settings;

        }

        void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            
            var opaqueTracingMarker = RenderPassMarkers.OpaqueTracing;

            natCmd.BeginSample(opaqueTracingMarker);

            
            _gpuScene.BuildAccelerationStructure(natCmd);

            var ds       = data.Ds;
            var resource = data.Resource;
            var settings = data.Settings;

            ds.SetAccelerationStructure("gWorldTlas", _gpuScene.AccelerationStructure);

            ds.SetConstantBuffer("GlobalConstants", resource.ConstantBuffer.GetNativeBufferPtr());

            ds.SetBuffer("gIn_ScramblingRanking", resource.ScramblingRanking.GetNativeBufferPtr());
            ds.SetBuffer("gIn_Sobol",             resource.Sobol.GetNativeBufferPtr());

            ds.SetRWStructuredBuffer("gInOut_SharcHashEntriesBuffer", resource.HashEntriesBuffer.GetNativeBufferPtr(), resource.HashEntriesBuffer.count, resource.HashEntriesBuffer.stride);
            ds.SetRWStructuredBuffer("gInOut_SharcAccumulated",       resource.AccumulationBuffer.GetNativeBufferPtr(), resource.AccumulationBuffer.count, resource.AccumulationBuffer.stride);
            ds.SetRWStructuredBuffer("gInOut_SharcResolved",          resource.ResolvedBuffer.GetNativeBufferPtr(), resource.ResolvedBuffer.count, resource.ResolvedBuffer.stride);

            ds.SetRWTexture("g_Output",               resource.Output.rt.GetNativeTexturePtr());

            ds.SetRWTexture("gOut_Mv",                resource.Mv.rt.GetNativeTexturePtr());
            ds.SetRWTexture("gOut_ViewZ",             resource.ViewZ.rt.GetNativeTexturePtr());
            ds.SetRWTexture("gOut_Normal_Roughness",  resource.NormalRoughness.rt.GetNativeTexturePtr());
            ds.SetRWTexture("gOut_BaseColor_Metalness", resource.BaseColorMetalness.rt.GetNativeTexturePtr());

            ds.SetRWTexture("gOut_DirectLighting",    resource.DirectLighting.rt.GetNativeTexturePtr());
            ds.SetRWTexture("gOut_DirectEmission",    resource.DirectEmission.rt.GetNativeTexturePtr());
            ds.SetRWTexture("gOut_PsrThroughput",     resource.PsrThroughput.rt.GetNativeTexturePtr());

            ds.SetRWTexture("gOut_ShadowData",        resource.Penumbra.rt.GetNativeTexturePtr());
            ds.SetRWTexture("gOut_Diff",              resource.Diff.rt.GetNativeTexturePtr());
            ds.SetRWTexture("gOut_Spec",              resource.Spec.rt.GetNativeTexturePtr());

            ds.SetTexture("gIn_PrevComposedDiff",          resource.ComposedDiff.rt.GetNativeTexturePtr());
            ds.SetTexture("gIn_PrevComposedSpec_PrevViewZ", resource.ComposedSpecViewZ.rt.GetNativeTexturePtr());

            ds.SetTexture("gIn_PrevViewZ",             resource.PrevViewZ.rt.GetNativeTexturePtr());
            ds.SetTexture("gIn_PrevNormalRoughness",   resource.PrevNormalRoughness.rt.GetNativeTexturePtr());
            ds.SetTexture("gIn_PrevBaseColorMetalness", resource.PrevBaseColorMetalness.rt.GetNativeTexturePtr());

            ds.SetRWTexture("gOut_GeoNormal",          resource.GeoNormal.rt.GetNativeTexturePtr());
            ds.SetTexture("gIn_PrevGeoNormal",         resource.PrevGeoNormal.rt.GetNativeTexturePtr());

            ds.SetStructuredBuffer("gIn_SpotLights",   resource.SpotLightBuffer.GetNativeBufferPtr(), resource.SpotLightBuffer.count, resource.SpotLightBuffer.stride);
            ds.SetStructuredBuffer("gIn_AreaLights",   resource.AreaLightBuffer.GetNativeBufferPtr(), resource.AreaLightBuffer.count, resource.AreaLightBuffer.stride);
            ds.SetStructuredBuffer("gIn_PointLights",  resource.PointLightBuffer.GetNativeBufferPtr(), resource.PointLightBuffer.count, resource.PointLightBuffer.stride);

            uint rectWmod = (uint)(settings.m_RenderResolution.x * settings.resolutionScale + 0.5f);
            uint rectHmod = (uint)(settings.m_RenderResolution.y * settings.resolutionScale + 0.5f);

            _gpuScene.BindToShader(ds);
            data.OpaqueTs.Dispatch(natCmd, ds, rectWmod, rectHmod);

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
            passData.Ds       = _ds;

            passData.Resource = _resource;
            passData.Settings = _settings;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}