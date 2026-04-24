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
    /// Native compute shader transparent pass that dispatches TraceTransparent.computeshader.
    /// Scene TLAS / instance-data are sourced from an <see cref="NRDSampleResource"/>.
    /// </summary>
    public class NRDTransparentPass : ScriptableRenderPass, IDisposable
    {
        private readonly NativeComputePipeline      _cs;
        private readonly NativeComputeDescriptorSet _ds;
        private          Resource              _resource;
        private          Settings              _settings;
        private          NRDSampleResource     _nrdResource;

        public NRDTransparentPass(NativeComputeShader cs)
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
        // Resource / Settings
        // -------------------------------------------------------------------------

        public class Resource
        {
            internal GraphicsBuffer ConstantBuffer;

            internal GraphicsBuffer HashEntriesBuffer;
            internal GraphicsBuffer ResolvedBuffer;

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

            internal TextureHandle ComposedDiff;
            internal TextureHandle ComposedSpecViewZ;
        }

        // -------------------------------------------------------------------------
        // Execution
        // -------------------------------------------------------------------------

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            cmd.BeginSample(RenderPassMarkers.TransparentTracing);

            var cs   = data.Cs;
            var ds   = data.Ds;
            var res  = data.Resource;
            var nrd  = data.NrdResource;

            // 1. Acceleration structures
            ds.SetAccelerationStructure("gWorldTlas", nrd.WorldAS);

            // 3. Scene structured buffers
            ds.SetStructuredBuffer("gIn_InstanceData",                nrd.InstanceDataBuf);
            ds.SetStructuredBuffer("gIn_PrimitiveData",               nrd.PrimitiveDataBuf);
            ds.SetStructuredBuffer("gIn_MorphPrimitivePositionsPrev", nrd.MorphPrimitivePositionsPrevBuf);

            // 4. Bindless material textures
            ds.SetBindlessTexture("gIn_Textures", nrd.Textures);

            // 5. SHARC UAVs (only hash entries + resolved; no accumulation in transparent shader)
            ds.SetRWStructuredBuffer("gInOut_SharcHashEntriesBuffer", res.HashEntriesBuffer);
            ds.SetRWStructuredBuffer("gInOut_SharcResolved",          res.ResolvedBuffer);

            var pool = data.Pool;

            // 6. SRV inputs
            ds.SetTexture("gIn_ComposedDiff",       data.ComposedDiff);
            ds.SetTexture("gIn_ComposedSpec_ViewZ", data.ComposedSpecViewZ);

            // 7. UAV outputs
            ds.SetRWTexture("gOut_Composed",         pool.GetRT(RenderResourceType.Composed).rt);
            ds.SetRWTexture("gInOut_Mv",             pool.GetRT(RenderResourceType.MV).rt);
            ds.SetRWTexture("gOut_Normal_Roughness", pool.GetRT(RenderResourceType.NormalRoughness).rt);

            // 8. Constant buffer
            ds.SetConstantBuffer("GlobalConstants", res.ConstantBuffer);

            // 9. Dispatch — numthreads [16, 16, 1]
            uint w       = (uint)(data.Settings.m_RenderResolution.x * data.Settings.resolutionScale + 0.5f);
            uint h       = (uint)(data.Settings.m_RenderResolution.y * data.Settings.resolutionScale + 0.5f);
            uint groupsX = (w + 15u) / 16u;
            uint groupsY = (h + 15u) / 16u;

            cs.Dispatch(cmd, ds, groupsX, groupsY, 1);

            cmd.EndSample(RenderPassMarkers.TransparentTracing);
        }

        // -------------------------------------------------------------------------
        // RenderGraph
        // -------------------------------------------------------------------------

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NRDTransparentPass", out var passData);

            passData.Cs          = _cs;
            passData.Ds          = _ds;
            passData.NrdResource = _nrdResource;
            passData.Resource    = _resource;
            passData.Settings    = _settings;
            passData.Pool        = _resource.Pool;

            passData.ComposedDiff      = renderGraph.ImportTexture(_resource.Pool.GetRT(RenderResourceType.ComposedDiff));
            passData.ComposedSpecViewZ = renderGraph.ImportTexture(_resource.Pool.GetRT(RenderResourceType.ComposedSpecViewZ));

            builder.UseTexture(passData.ComposedDiff,      AccessFlags.ReadWrite);
            builder.UseTexture(passData.ComposedSpecViewZ, AccessFlags.ReadWrite);

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}
