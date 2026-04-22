using System;
using NativeRender;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;
using static PathTracing.ShaderIDs;

namespace PathTracing
{
    /// <summary>
    /// Native-compute-shader SHARC pass.
    ///
    /// Dispatches SharcResolve then SharcUpdate using NativeComputeShader, following
    /// NRDSample.cpp ping-pong pattern:
    ///   isEven = !(frameIndex &amp; 1)
    ///   Even: gIn_PrevGradient = StoredPing,  gOut_CurrGradient = StoredPong
    ///   Odd:  gIn_PrevGradient = StoredPong,  gOut_CurrGradient = StoredPing
    ///   Always: gOut_Gradient = GradientPing
    /// </summary>
    public class NRDSharcPass : ScriptableRenderPass, IDisposable
    {
        private readonly NativeComputePipeline _resolve;
        private readonly NativeComputePipeline _update;
        private          Resource              _resource;
        private          Settings              _settings;
        private          NRDSampleResource     _nrdResource;

        public NRDSharcPass(NativeComputeShader resolve, NativeComputeShader update)
        {
            _resolve = new NativeComputePipeline(resolve);
            _update  = new NativeComputePipeline(update);
        }

        public void Dispose()
        {
            _resolve?.Dispose();
            _update?.Dispose();
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
        // Resource / Settings
        // -------------------------------------------------------------------------

        public class Resource
        {
            internal GraphicsBuffer ConstantBuffer;
            internal GraphicsBuffer HashEntriesBuffer;
            internal GraphicsBuffer AccumulationBuffer;
            internal GraphicsBuffer ResolvedBuffer;

            // Gradient textures sourced from the pool inside ExecutePass
            internal PathTracingResourcePool Pool;
        }

        public class Settings
        {
            internal int2  RenderResolution;
            internal float sharcDownscale;
            internal bool  isEven; // !(frameIndex & 1), matching NRDSample.cpp
        }

        // -------------------------------------------------------------------------
        // Pass data (RenderGraph)
        // -------------------------------------------------------------------------

        class PassData
        {
            internal NativeComputePipeline Resolve;
            internal NativeComputePipeline Update;
            internal NRDSampleResource     NrdResource;
            internal Resource              Resource;
            internal Settings              Settings;
            internal PathTracingResourcePool Pool;
 
        }

        // -------------------------------------------------------------------------
        // Execution
        // -------------------------------------------------------------------------

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            var resolve   = data.Resolve;
            var update    = data.Update;
            var res       = data.Resource;
            var settings  = data.Settings;
            var nrd       = data.NrdResource;

            // ── SharcResolve [numthreads(256,1,1)] ────────────────────────────────
            cmd.BeginSample(RenderPassMarkers.SharcResolve);

            resolve.SetConstantBuffer("GlobalConstants", res.ConstantBuffer);
            resolve.SetRWBuffer("gInOut_SharcHashEntriesBuffer", res.HashEntriesBuffer);
            resolve.SetRWBuffer("gInOut_SharcAccumulated",       res.AccumulationBuffer);
            resolve.SetRWBuffer("gInOut_SharcResolved",          res.ResolvedBuffer);

            uint resolveGroups = (uint)((PathTracingFeature.Capacity + 255) / 256);
            resolve.Dispatch(cmd, resolveGroups, 1, 1);

            cmd.EndSample(RenderPassMarkers.SharcResolve);

            // ── SharcUpdate [numthreads(16,16,1)] ─────────────────────────────────
            cmd.BeginSample(RenderPassMarkers.SharcUpdate);

            // 1. Acceleration structures
            update.SetAccelerationStructure("gWorldTlas", nrd.WorldAS);
            update.SetAccelerationStructure("gLightTlas", nrd.LightAS);

            // 3. Scene structured buffers
            update.SetStructuredBuffer("gIn_InstanceData",  nrd.InstanceDataBuf);
            update.SetStructuredBuffer("gIn_PrimitiveData", nrd.PrimitiveDataBuf);

            // 4. SHARC UAVs
            update.SetRWBuffer("gInOut_SharcHashEntriesBuffer", res.HashEntriesBuffer);
            update.SetRWBuffer("gInOut_SharcAccumulated",       res.AccumulationBuffer);
            update.SetRWBuffer("gInOut_SharcResolved",          res.ResolvedBuffer);

            // 5. Bindless material textures
            update.SetBindlessTexture("gIn_Textures", nrd.Textures);

            // 6. Constant buffer
            update.SetConstantBuffer("GlobalConstants", res.ConstantBuffer);

            // 7. Gradient ping-pong (NRDSample.cpp line 4062+4400)
            //    isEven = !(frameIndex & 1)
            //    Even: PrevGradient = StoredPing (SRV), CurrGradient = StoredPong (UAV)
            //    Odd:  PrevGradient = StoredPong (SRV), CurrGradient = StoredPing (UAV)
            var pool         = data.Pool;
            var prevGradient = settings.isEven ? pool.GetRT(RenderResourceType.Gradient_StoredPing).rt : pool.GetRT(RenderResourceType.Gradient_StoredPong).rt;
            var currGradient = settings.isEven ? pool.GetRT(RenderResourceType.Gradient_StoredPong).rt : pool.GetRT(RenderResourceType.Gradient_StoredPing).rt;

            update.SetTexture("gIn_PrevGradient",   prevGradient);
            update.SetRWTexture("gOut_CurrGradient", currGradient);
            update.SetRWTexture("gOut_Gradient",     pool.GetRT(RenderResourceType.Gradient_Ping).rt);

            // 8. Dispatch at SHARC resolution
            //    sharcDims = 16 * ceil(renderRes / sharcDownscale / 16)
            uint sharcW = (uint)(settings.RenderResolution.x / settings.sharcDownscale);
            uint sharcH = (uint)(settings.RenderResolution.y / settings.sharcDownscale);
            uint groupsX = (sharcW + 15u) / 16u;
            uint groupsY = (sharcH + 15u) / 16u;

            update.Dispatch(cmd, groupsX, groupsY, 1);

            cmd.EndSample(RenderPassMarkers.SharcUpdate);
        }

        // -------------------------------------------------------------------------
        // RenderGraph
        // -------------------------------------------------------------------------

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NRDSharcPass", out var passData);

            passData.Resolve     = _resolve;
            passData.Update      = _update;
            passData.NrdResource = _nrdResource;
            passData.Resource    = _resource;
            passData.Settings    = _settings;

            passData.Pool = _resource.Pool; 

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }
    }
}
