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
        private readonly NativeComputePipeline      _cs;
        private readonly NativeComputeDescriptorSet _ds;
        private          Resource                   _resource;
        private          Settings                   _settings;
        private          NRDSampleResource          _nrdResource;

        public NRDOpaquePass(NativeComputeShader cs)
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
            internal NativeComputePipeline      Cs;
            internal NativeComputeDescriptorSet Ds;
            internal NRDSampleResource          NrdResource;
            internal Resource                   Resource;
            internal Settings                   Settings;
            internal PathTracingResourcePool    Pool;
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
            var ds       = data.Ds;
            var res      = data.Resource;
            var settings = data.Settings;
            var nrd      = data.NrdResource;

            // 1. Scene TLAS
            ds.SetAccelerationStructure("gWorldTlas", nrd.WorldAS);
            ds.SetAccelerationStructure("gLightTlas", nrd.LightAS);

            // 3. Scene structured buffers
            ds.SetStructuredBuffer("gIn_InstanceData", nrd.InstanceDataBuf);
            ds.SetStructuredBuffer("gIn_PrimitiveData", nrd.PrimitiveDataBuf);
            ds.SetStructuredBuffer("gIn_MorphPrimitivePositionsPrev", nrd.MorphPrimitivePositionsPrevBuf);

            // 4. SHARC UAVs
            ds.SetRWStructuredBuffer("gInOut_SharcHashEntriesBuffer", res.HashEntriesBuffer.GetNativeBufferPtr(),res.HashEntriesBuffer.count,res.HashEntriesBuffer.stride);
            ds.SetRWStructuredBuffer("gInOut_SharcAccumulated",       res.AccumulationBuffer.GetNativeBufferPtr(),res.AccumulationBuffer.count,res.AccumulationBuffer.stride);
            ds.SetRWStructuredBuffer("gInOut_SharcResolved",          res.ResolvedBuffer.GetNativeBufferPtr(),res.ResolvedBuffer.count,res.ResolvedBuffer.stride);

            // 5. Bindless material textures
            ds.SetBindlessTexture("gIn_Textures", nrd.Textures);

            // 6. Constant buffer
            ds.SetConstantBuffer("GlobalConstants", res.ConstantBuffer);

            var pool = data.Pool;

            // SRV
            ds.SetTexture("gIn_PrevComposedDiff", pool.GetPoint(RenderResourceType.ComposedDiff));
            ds.SetTexture("gIn_PrevComposedSpec_PrevViewZ", pool.GetPoint(RenderResourceType.ComposedSpecViewZ));
            ds.SetTexture("gIn_ScramblingRanking", res.ScramblingRanking.GetNativeTexturePtr());
            ds.SetTexture("gIn_Sobol", res.Sobol.GetNativeTexturePtr());


            // UAV
            ds.SetRWTexture("gOut_Mv", pool.GetPoint(RenderResourceType.MV));
            ds.SetRWTexture("gOut_ViewZ", pool.GetPoint(RenderResourceType.Viewz));
            ds.SetRWTexture("gOut_Normal_Roughness", pool.GetPoint(RenderResourceType.NormalRoughness));
            ds.SetRWTexture("gOut_BaseColor_Metalness", pool.GetPoint(RenderResourceType.BaseColorMetalness));
            ds.SetRWTexture("gOut_DirectLighting", pool.GetPoint(RenderResourceType.DirectLighting));
            ds.SetRWTexture("gOut_DirectEmission", pool.GetPoint(RenderResourceType.DirectEmission));
            ds.SetRWTexture("gOut_PsrThroughput", pool.GetPoint(RenderResourceType.PsrThroughput));
            ds.SetRWTexture("gOut_ShadowData", pool.GetPoint(RenderResourceType.Unfiltered_Penumbra));
            ds.SetRWTexture("gOut_Shadow_Translucency", pool.GetPoint(RenderResourceType.Unfiltered_Translucency));
            ds.SetRWTexture("gOut_Diff", pool.GetPoint(RenderResourceType.Unfiltered_Diff));
            ds.SetRWTexture("gOut_Spec", pool.GetPoint(RenderResourceType.Unfiltered_Spec));


            // 10. Dispatch — numthreads [16, 16, 1]
            uint w       = (uint)(settings.m_RenderResolution.x * settings.resolutionScale + 0.5f);
            uint h       = (uint)(settings.m_RenderResolution.y * settings.resolutionScale + 0.5f);
            uint groupsX = (w + 15u) / 16u;
            uint groupsY = (h + 15u) / 16u;

            cs.Dispatch(cmd, ds, groupsX, groupsY, 1);

            cmd.EndSample(opaqueTracingMarker);
        }

        // -------------------------------------------------------------------------
        // RenderGraph
        // -------------------------------------------------------------------------

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NRDOpaquePass", out var passData);

            passData.Cs          = _cs;
            passData.Ds          = _ds;
            passData.NrdResource = _nrdResource;
            passData.Resource    = _resource;
            passData.Settings    = _settings;
            passData.Pool        = _resource.Pool;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }
    }
}