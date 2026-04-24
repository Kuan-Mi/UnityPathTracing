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
        private readonly NativeComputePipeline       _resolve;
        private readonly NativeComputePipeline       _update;
        private readonly NativeComputeDescriptorSet  _resolveDs;
        private readonly NativeComputeDescriptorSet  _updateDsPing; // isEven: StoredPing→in, StoredPong→out
        private readonly NativeComputeDescriptorSet  _updateDsPong; // !isEven: StoredPong→in, StoredPing→out
        private          Resource              _resource;
        private          Settings              _settings;
        private          NRDSampleResource     _nrdResource;

        public NRDSharcPass(NativeComputeShader resolve, NativeComputeShader update)
        {
            _resolve      = new NativeComputePipeline(resolve);
            _update       = new NativeComputePipeline(update);
            _resolveDs    = new NativeComputeDescriptorSet(_resolve);
            _updateDsPing = new NativeComputeDescriptorSet(_update);
            _updateDsPong = new NativeComputeDescriptorSet(_update);
        }

        public void Dispose()
        {
            _resolveDs?.Dispose();
            _updateDsPing?.Dispose();
            _updateDsPong?.Dispose();
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
            internal IntPtr ConstantBuffer;

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
            internal NativeComputePipeline      Resolve;
            internal NativeComputePipeline      Update;
            internal NativeComputeDescriptorSet ResolveDs;
            internal NativeComputeDescriptorSet UpdateDsPing;
            internal NativeComputeDescriptorSet UpdateDsPong;
            internal NRDSampleResource          NrdResource;
            internal Resource                   Resource;
            internal Settings                   Settings;
            internal PathTracingResourcePool    Pool;
        }

        // -------------------------------------------------------------------------
        // Execution
        // -------------------------------------------------------------------------

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            var resolve   = data.Resolve;
            var update    = data.Update;
            var resolveDs = data.ResolveDs;
            var res       = data.Resource;
            var settings  = data.Settings;
            var nrd       = data.NrdResource;

            // ── SharcUpdate [numthreads(16,16,1)] ─────────────────────────────────
            // Select the pre-configured ping or pong descriptor set.
            // Gradient texture bindings were pre-baked in RecordRenderGraph.
            cmd.BeginSample(RenderPassMarkers.SharcUpdate);

            var updateDs = settings.isEven ? data.UpdateDsPing : data.UpdateDsPong;

            // Dynamic per-frame bindings (same regardless of ping/pong)
            updateDs.SetAccelerationStructure("gWorldTlas", nrd.WorldAS);
            updateDs.SetAccelerationStructure("gLightTlas", nrd.LightAS);
            
            updateDs.SetStructuredBuffer("gIn_InstanceData",  nrd.InstanceDataBufPtr, nrd.InstanceDataBuf.count, nrd.InstanceDataBuf.stride);
            updateDs.SetStructuredBuffer("gIn_PrimitiveData", nrd.PrimitiveDataBufPtr, nrd.PrimitiveDataBuf.count, nrd.PrimitiveDataBuf.stride);
            updateDs.SetRWStructuredBuffer("gInOut_SharcHashEntriesBuffer", nrd.HashEntriesBufferPtr, nrd.HashEntriesBuffer.count, nrd.HashEntriesBuffer.stride);
            updateDs.SetRWStructuredBuffer("gInOut_SharcAccumulated",       nrd.AccumulationBufferPtr, nrd.AccumulationBuffer.count, nrd.AccumulationBuffer.stride);
            updateDs.SetRWStructuredBuffer("gInOut_SharcResolved",          nrd.ResolvedBufferPtr, nrd.ResolvedBuffer.count, nrd.ResolvedBuffer.stride);
            updateDs.SetBindlessTexture("gIn_Textures", nrd.Textures);
            updateDs.SetConstantBuffer("GlobalConstants", res.ConstantBuffer);

            uint sharcW  = (uint)(settings.RenderResolution.x / settings.sharcDownscale);
            uint sharcH  = (uint)(settings.RenderResolution.y / settings.sharcDownscale);
            uint groupsX = (sharcW + 15u) / 16u;
            uint groupsY = (sharcH + 15u) / 16u;

            update.Dispatch(cmd, updateDs, groupsX, groupsY, 1);

            cmd.EndSample(RenderPassMarkers.SharcUpdate);


            // ── SharcResolve [numthreads(256,1,1)] ────────────────────────────────
            cmd.BeginSample(RenderPassMarkers.SharcResolve);

            resolveDs.SetConstantBuffer("GlobalConstants", res.ConstantBuffer);
            resolveDs.SetRWStructuredBuffer("gInOut_SharcHashEntriesBuffer",  nrd.HashEntriesBufferPtr, nrd.HashEntriesBuffer.count, nrd.HashEntriesBuffer.stride);
            resolveDs.SetRWStructuredBuffer("gInOut_SharcAccumulated",        nrd.AccumulationBufferPtr, nrd.AccumulationBuffer.count, nrd.AccumulationBuffer.stride);
            resolveDs.SetRWStructuredBuffer("gInOut_SharcResolved",           nrd.ResolvedBufferPtr, nrd.ResolvedBuffer.count, nrd.ResolvedBuffer.stride);

            uint resolveGroups = (uint)((NRDSampleResource.SharcCapacity + 255) / 256);
            resolve.Dispatch(cmd, resolveDs, resolveGroups, 1, 1);

            cmd.EndSample(RenderPassMarkers.SharcResolve);
        }

        // -------------------------------------------------------------------------
        // RenderGraph
        // -------------------------------------------------------------------------

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NRDSharcPass", out var passData);

            passData.Resolve      = _resolve;
            passData.Update       = _update;
            passData.ResolveDs    = _resolveDs;
            passData.UpdateDsPing = _updateDsPing;
            passData.UpdateDsPong = _updateDsPong;
            passData.NrdResource  = _nrdResource;
            passData.Resource     = _resource;
            passData.Settings     = _settings;

            passData.Pool = _resource.Pool;

            // Pre-bake gradient texture bindings into the Ping/Pong descriptor sets.
            // Even frame (Ping): StoredPing → in,  StoredPong → out
            // Odd  frame (Pong): StoredPong → in,  StoredPing → out
            // gOut_Gradient always writes to Gradient_Ping.
            var pool = _resource.Pool;
            _updateDsPing.SetTexture ("gIn_PrevGradient",   pool.GetPoint(RenderResourceType.Gradient_StoredPing));
            _updateDsPing.SetRWTexture("gOut_CurrGradient", pool.GetPoint(RenderResourceType.Gradient_StoredPong));
            _updateDsPing.SetRWTexture("gOut_Gradient",     pool.GetPoint(RenderResourceType.Gradient_Ping));

            _updateDsPong.SetTexture ("gIn_PrevGradient",   pool.GetPoint(RenderResourceType.Gradient_StoredPong));
            _updateDsPong.SetRWTexture("gOut_CurrGradient", pool.GetPoint(RenderResourceType.Gradient_StoredPing));
            _updateDsPong.SetRWTexture("gOut_Gradient",     pool.GetPoint(RenderResourceType.Gradient_Ping));

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }
    }
}
