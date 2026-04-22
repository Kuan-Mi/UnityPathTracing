using System;
using NativeRender;
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
    /// <summary>
    /// Native compute shader opaque pass that dispatches TraceOpaque.computeshader.
    /// Scene TLAS / instance-data / SHARC UAVs are sourced from an <see cref="NRDSampleResource"/>
    /// and bound directly inside ExecutePass (not delegated to NRDSampleResource).
    /// </summary>
    public class NRDOpaquePass : ScriptableRenderPass, IDisposable
    {
        private readonly NativeComputePipeline _cs;
        private          Resource              _resource;
        private          Settings              _settings;
        private          NRDSampleResource     _nrdResource;

        public NRDOpaquePass(NativeComputeShader cs)
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

        public void SetNRDSampleResource(NRDSampleResource nrdResource)
        {
            _nrdResource = nrdResource;
        }

        // -------------------------------------------------------------------------
        // Resource / Settings descriptors
        // -------------------------------------------------------------------------

        public class Resource
        {
            internal GraphicsBuffer ConstantBuffer;

            // SHARC (passed in from PathTracingFeature, same as NativeOpaquePass)
            internal GraphicsBuffer HashEntriesBuffer;
            internal GraphicsBuffer AccumulationBuffer;
            internal GraphicsBuffer ResolvedBuffer;

            // Stochastic sampling textures
            internal Texture2D ScramblingRanking;
            internal Texture2D Sobol;

            // RT textures sourced from the pool inside ExecutePass
            internal PathTracingResourcePool Pool;
        }

        public class Settings
        {
            internal int2  m_RenderResolution;
            internal float resolutionScale;
        }

        // -------------------------------------------------------------------------
        // Pass data (RenderGraph)
        // -------------------------------------------------------------------------

        class PassData
        {
            internal NativeComputePipeline  Cs;
            internal NRDSampleResource      NrdResource;
            internal Resource               Resource;
            internal Settings               Settings;
            internal PathTracingResourcePool Pool;
        }

        // -------------------------------------------------------------------------
        // Execution
        // -------------------------------------------------------------------------

        

// numthreads  [16, 16, 1]
//
// -- SRV (8) --
//   gIn_InstanceData                  Buffer<mixed4>                        space4:t2
//   gIn_PrimitiveData                 Buffer<mixed4>                        space4:t3
//   gIn_MorphPrimitivePositionsPrev   Buffer<mixed4>                        space4:t4
//   gIn_Textures                      Texture2D<float4>                     space1:t0
//   gIn_PrevComposedDiff              Texture2D<float4>                     space0:t0
//   gIn_PrevComposedSpec_PrevViewZ    Texture2D<float4>                     space0:t1
//   gIn_ScramblingRanking             Texture2D<uint4>                      space0:t2
//   gIn_Sobol                         Texture2D<uint4>                      space0:t3
//
// -- UAV (13) --
//   gInOut_SharcHashEntriesBuffer     RWBuffer<mixed4>                      space2:u0
//   gInOut_SharcResolved              RWBuffer<mixed4>                      space2:u2
//   gOut_Mv                           RWTexture2D<float4>                   space0:u0
//   gOut_ViewZ                        RWTexture2D<float4>                   space0:u1
//   gOut_Normal_Roughness             RWTexture2D<float4>                   space0:u2
//   gOut_BaseColor_Metalness          RWTexture2D<float4>                   space0:u3
//   gOut_DirectLighting               RWTexture2D<float4>                   space0:u4
//   gOut_DirectEmission               RWTexture2D<float4>                   space0:u5
//   gOut_PsrThroughput                RWTexture2D<float4>                   space0:u6
//   gOut_ShadowData                   RWTexture2D<float4>                   space0:u7
//   gOut_Shadow_Translucency          RWTexture2D<float4>                   space0:u8
//   gOut_Diff                         RWTexture2D<float4>                   space0:u9
//   gOut_Spec                         RWTexture2D<float4>                   space0:u10
//
// -- CBV (1) --
//   GlobalConstants                   ConstantBuffer                        space4:b0
//
// -- Sampler (2) --
//   gLinearMipmapLinearSampler        SamplerState                          space4:s0
//   gNearestClamp                     SamplerState                          space4:s4
//
// -- TLAS (2) --
//   gWorldTlas                        RaytracingAccelerationStructure       space4:t0
//   gLightTlas                        RaytracingAccelerationStructure       space4:t1

        

        void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            var opaqueTracingMarker = RenderPassMarkers.OpaqueTracing;
            cmd.BeginSample(opaqueTracingMarker);

            var cs       = data.Cs;
            var res      = data.Resource;
            var settings = data.Settings;
            var nrd      = data.NrdResource;

            // 1. Scene TLAS
            cs.SetAccelerationStructure("gWorldTlas", nrd.WorldAS);
            cs.SetAccelerationStructure("gLightTlas", nrd.LightAS);

            // 3. Scene structured buffers
            cs.SetStructuredBuffer("gIn_InstanceData", nrd.InstanceDataBuf);
            cs.SetStructuredBuffer("gIn_PrimitiveData", nrd.PrimitiveDataBuf);
            cs.SetStructuredBuffer("gIn_MorphPrimitivePositionsPrev", nrd.MorphPrimitivePositionsPrevBuf);

            // 4. SHARC UAVs
            cs.SetRWBuffer("gInOut_SharcHashEntriesBuffer", res.HashEntriesBuffer);
            cs.SetRWBuffer("gInOut_SharcAccumulated", res.AccumulationBuffer);
            cs.SetRWBuffer("gInOut_SharcResolved", res.ResolvedBuffer);

            // 5. Bindless material textures
            cs.SetBindlessTexture("gIn_Textures", nrd.Textures);

            // 6. Constant buffer
            cs.SetConstantBuffer("GlobalConstants", res.ConstantBuffer);

            var pool = data.Pool;
            
            // SRV
            cs.SetTexture("gIn_PrevComposedDiff",           pool.GetRT(RenderResourceType.ComposedDiff).rt);
            cs.SetTexture("gIn_PrevComposedSpec_PrevViewZ", pool.GetRT(RenderResourceType.ComposedSpecViewZ).rt);
            cs.SetTexture("gIn_ScramblingRanking", res.ScramblingRanking);
            cs.SetTexture("gIn_Sobol", res.Sobol);


            // UAV
            cs.SetRWTexture("gOut_Mv",                  pool.GetRT(RenderResourceType.MV).rt);
            cs.SetRWTexture("gOut_ViewZ",               pool.GetRT(RenderResourceType.Viewz).rt);
            cs.SetRWTexture("gOut_Normal_Roughness",    pool.GetRT(RenderResourceType.NormalRoughness).rt);
            cs.SetRWTexture("gOut_BaseColor_Metalness", pool.GetRT(RenderResourceType.BaseColorMetalness).rt);
            cs.SetRWTexture("gOut_DirectLighting",      pool.GetRT(RenderResourceType.DirectLighting).rt);
            cs.SetRWTexture("gOut_DirectEmission",      pool.GetRT(RenderResourceType.DirectEmission).rt);
            cs.SetRWTexture("gOut_PsrThroughput",       pool.GetRT(RenderResourceType.PsrThroughput).rt);
            cs.SetRWTexture("gOut_ShadowData",          pool.GetRT(RenderResourceType.Unfiltered_Penumbra).rt);
            cs.SetRWTexture("gOut_Shadow_Translucency", pool.GetRT(RenderResourceType.Unfiltered_Translucency).rt);
            cs.SetRWTexture("gOut_Diff",                pool.GetRT(RenderResourceType.Unfiltered_Diff).rt);
            cs.SetRWTexture("gOut_Spec",                pool.GetRT(RenderResourceType.Unfiltered_Spec).rt);


            // 10. Dispatch — numthreads [16, 16, 1]
            uint w       = (uint)(settings.m_RenderResolution.x * settings.resolutionScale + 0.5f);
            uint h       = (uint)(settings.m_RenderResolution.y * settings.resolutionScale + 0.5f);
            uint groupsX = (w + 15u) / 16u;
            uint groupsY = (h + 15u) / 16u;

            cs.Dispatch(cmd, groupsX, groupsY, 1);

            cmd.EndSample(opaqueTracingMarker);
        }

        // -------------------------------------------------------------------------
        // RenderGraph
        // -------------------------------------------------------------------------

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NRDOpaquePass", out var passData);

            passData.Cs          = _cs;
            passData.NrdResource = _nrdResource;
            passData.Resource    = _resource;
            passData.Settings    = _settings;
            passData.Pool        = _resource.Pool;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }
    }
}