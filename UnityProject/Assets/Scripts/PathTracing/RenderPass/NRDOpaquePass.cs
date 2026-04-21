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
    public class NRDOpaquePass : ScriptableRenderPass
    {
        private readonly NativeComputeShader _cs;
        private          Resource            _resource;
        private          Settings            _settings;
        private          NRDSampleResource   _nrdResource;

        public NRDOpaquePass(NativeComputeShader cs)
        {
            _cs = cs;
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

            // Output G-buffer / denoising inputs (UAVs)
            internal RTHandle Mv;
            internal RTHandle ViewZ;
            internal RTHandle NormalRoughness;
            internal RTHandle BaseColorMetalness;
            internal RTHandle DirectLighting;
            internal RTHandle DirectEmission;
            internal RTHandle PsrThroughput;
            internal RTHandle ShadowData;
            internal RTHandle ShadowTranslucency;
            internal RTHandle Diff;
            internal RTHandle Spec;

            // History SRVs
            internal RTHandle PrevComposedDiff;
            internal RTHandle PrevComposedSpecViewZ;
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
            internal NativeComputeShader Cs;
            internal NRDSampleResource   NrdResource;
            internal Resource            Resource;
            internal Settings            Settings;

            internal TextureHandle DirectEmission;
            internal TextureHandle PrevComposedDiff;
            internal TextureHandle PrevComposedSpecViewZ;
        }

        // -------------------------------------------------------------------------
        // Execution
        // -------------------------------------------------------------------------

        private static readonly ProfilerMarker s_OpaqueTracingMarker =
            new ProfilerMarker(ProfilerCategory.Render, "NRDOpaqueTracing",
                MarkerFlags.SampleGPU);

        void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            cmd.BeginSample(s_OpaqueTracingMarker);

            var cs       = data.Cs;
            var res      = data.Resource;
            var settings = data.Settings;
            var nrd      = data.NrdResource;

            // 1. Build TLAS (must happen before binding)
            nrd.BuildAccelerationStructures(cmd);

            // 2. Scene TLAS
            cs.SetAccelerationStructure("gWorldTlas", nrd.WorldAS);
            cs.SetAccelerationStructure("gLightTlas", nrd.LightAS);

            // 3. Scene structured buffers
            cs.SetStructuredBuffer("gIn_InstanceData",                nrd.InstanceDataBuf);
            cs.SetStructuredBuffer("gIn_PrimitiveData",               nrd.PrimitiveDataBuf);
            cs.SetStructuredBuffer("gIn_MorphPrimitivePositionsPrev", nrd.MorphPrimitivePositionsPrevBuf);

            // 4. SHARC UAVs
            cs.SetRWBuffer("gInOut_SharcHashEntriesBuffer", res.HashEntriesBuffer);
            cs.SetRWBuffer("gInOut_SharcAccumulated",       res.AccumulationBuffer);
            cs.SetRWBuffer("gInOut_SharcResolved",          res.ResolvedBuffer);

            // 5. Bindless material textures
            cs.SetBindlessTexture("gIn_Textures", nrd.Textures);

            // 6. Constant buffer
            cs.SetConstantBuffer("GlobalConstants", res.ConstantBuffer);

            // 7. Stochastic sampling (Texture2D<uint4> in shader)
            cs.SetTexture("gIn_ScramblingRanking", res.ScramblingRanking);
            cs.SetTexture("gIn_Sobol",             res.Sobol);

            // 8. UAV outputs
            cs.SetRWTexture("gOut_Mv",                  res.Mv.rt);
            cs.SetRWTexture("gOut_ViewZ",               res.ViewZ.rt);
            cs.SetRWTexture("gOut_Normal_Roughness",    res.NormalRoughness.rt);
            cs.SetRWTexture("gOut_BaseColor_Metalness", res.BaseColorMetalness.rt);
            cs.SetRWTexture("gOut_DirectLighting",      res.DirectLighting.rt);
            cs.SetRWTexture("gOut_DirectEmission",      data.DirectEmission);
            cs.SetRWTexture("gOut_PsrThroughput",       res.PsrThroughput.rt);
            cs.SetRWTexture("gOut_ShadowData",          res.ShadowData.rt);
            cs.SetRWTexture("gOut_Shadow_Translucency", res.ShadowTranslucency.rt);
            cs.SetRWTexture("gOut_Diff",                res.Diff.rt);
            cs.SetRWTexture("gOut_Spec",                res.Spec.rt);

            // 9. History SRVs
            cs.SetTexture("gIn_PrevComposedDiff",            data.PrevComposedDiff);
            cs.SetTexture("gIn_PrevComposedSpec_PrevViewZ",  data.PrevComposedSpecViewZ);

            // 10. Dispatch — numthreads [16, 16, 1]
            uint w = (uint)(settings.m_RenderResolution.x * settings.resolutionScale + 0.5f);
            uint h = (uint)(settings.m_RenderResolution.y * settings.resolutionScale + 0.5f);
            uint groupsX = (w + 15u) / 16u;
            uint groupsY = (h + 15u) / 16u;

            cs.Dispatch(cmd, groupsX, groupsY, 1);

            cmd.EndSample(s_OpaqueTracingMarker);
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

            passData.DirectEmission        = renderGraph.ImportTexture(_resource.DirectEmission);
            passData.PrevComposedDiff      = renderGraph.ImportTexture(_resource.PrevComposedDiff);
            passData.PrevComposedSpecViewZ = renderGraph.ImportTexture(_resource.PrevComposedSpecViewZ);

            builder.UseTexture(passData.DirectEmission,        AccessFlags.ReadWrite);
            builder.UseTexture(passData.PrevComposedDiff,      AccessFlags.Read);
            builder.UseTexture(passData.PrevComposedSpecViewZ, AccessFlags.Read);

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }
    }
}
