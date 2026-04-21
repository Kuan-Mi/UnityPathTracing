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
        private readonly NativeComputePipeline _cs;
        private          Resource              _resource;
        private          Settings              _settings;
        private          NRDSampleResource     _nrdResource;

        public NRDTransparentPass(NativeComputeShader cs)
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
        // Resource / Settings
        // -------------------------------------------------------------------------

        public class Resource
        {
            internal GraphicsBuffer ConstantBuffer;

            internal GraphicsBuffer HashEntriesBuffer;
            internal GraphicsBuffer ResolvedBuffer;

            internal RTHandle ComposedDiff;
            internal RTHandle ComposedSpecViewZ;

            internal RTHandle Composed;
            internal RTHandle Mv;
            internal RTHandle NormalRoughness;
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
            internal NativeComputePipeline Cs;
            internal NRDSampleResource     NrdResource;
            internal Resource              Resource;
            internal Settings              Settings;

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
            var res  = data.Resource;
            var nrd  = data.NrdResource;

            // 1. Build TLAS
            nrd.BuildAccelerationStructures(cmd);

            // 2. Acceleration structures
            cs.SetAccelerationStructure("gWorldTlas", nrd.WorldAS);

            // 3. Scene structured buffers
            cs.SetStructuredBuffer("gIn_InstanceData",                nrd.InstanceDataBuf);
            cs.SetStructuredBuffer("gIn_PrimitiveData",               nrd.PrimitiveDataBuf);
            cs.SetStructuredBuffer("gIn_MorphPrimitivePositionsPrev", nrd.MorphPrimitivePositionsPrevBuf);

            // 4. Bindless material textures
            cs.SetBindlessTexture("gIn_Textures", nrd.Textures);

            // 5. SHARC UAVs (only hash entries + resolved; no accumulation in transparent shader)
            cs.SetRWBuffer("gInOut_SharcHashEntriesBuffer", res.HashEntriesBuffer);
            cs.SetRWBuffer("gInOut_SharcResolved",          res.ResolvedBuffer);

            // 6. SRV inputs
            cs.SetTexture("gIn_ComposedDiff",       data.ComposedDiff);
            cs.SetTexture("gIn_ComposedSpec_ViewZ", data.ComposedSpecViewZ);

            // 7. UAV outputs
            cs.SetRWTexture("gOut_Composed",      res.Composed.rt);
            cs.SetRWTexture("gInOut_Mv",          res.Mv.rt);
            cs.SetRWTexture("gOut_Normal_Roughness", res.NormalRoughness.rt);

            // 8. Constant buffer
            cs.SetConstantBuffer("GlobalConstants", res.ConstantBuffer);

            // 9. Dispatch — numthreads [16, 16, 1]
            uint w       = (uint)(data.Settings.m_RenderResolution.x * data.Settings.resolutionScale + 0.5f);
            uint h       = (uint)(data.Settings.m_RenderResolution.y * data.Settings.resolutionScale + 0.5f);
            uint groupsX = (w + 15u) / 16u;
            uint groupsY = (h + 15u) / 16u;

            cs.Dispatch(cmd, groupsX, groupsY, 1);

            cmd.EndSample(RenderPassMarkers.TransparentTracing);
        }

        // -------------------------------------------------------------------------
        // RenderGraph
        // -------------------------------------------------------------------------

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NRDTransparentPass", out var passData);

            passData.Cs          = _cs;
            passData.NrdResource = _nrdResource;
            passData.Resource    = _resource;
            passData.Settings    = _settings;

            passData.ComposedDiff      = renderGraph.ImportTexture(_resource.ComposedDiff);
            passData.ComposedSpecViewZ = renderGraph.ImportTexture(_resource.ComposedSpecViewZ);

            builder.UseTexture(passData.ComposedDiff,      AccessFlags.ReadWrite);
            builder.UseTexture(passData.ComposedSpecViewZ, AccessFlags.ReadWrite);

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}
