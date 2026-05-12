using System;
using System.Runtime.InteropServices;
using NativeRender;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Dispatches <c>CompositingPass.computeshader</c> to combine denoised diffuse/specular
    /// illumination with GBuffer material data and produce the final composited image in
    /// <c>RenderResourceType.DirectLighting</c>.
    ///
    /// Mirrors <c>SceneRenderer::Compositing()</c> in
    /// <c>RTXDI/Samples/FullSample/Source/App/SceneRenderer.cpp</c>.
    ///
    /// CompositingPass.hlsl register layout:
    ///   b0  : CompositingConstants
    ///   u0  : u_Output          → DirectLighting
    ///   u1  : u_MotionVectors   → RtxdiMotionVectors
    ///   t0  : t_GBufferDepth    → RtxdiDeviceDepth
    ///   t1  : t_GBufferNormals  → RtxdiNormals
    ///   t2  : t_GBufferDiffuseAlbedo  → RtxdiDiffuseAlbedo
    ///   t3  : t_GBufferSpecularRough  → RtxdiSpecularRough
    ///   t4  : t_GBufferEmissive       → RtxdiEmissive
    ///   t5  : t_Diffuse               → RtxdiDiffuseLighting   (noisy)
    ///   t6  : t_Specular              → RtxdiSpecularLighting  (noisy)
    ///   t7  : t_DenoisedDiffuse       → RtxdiDenoisedDiffuseLighting
    ///   t8  : t_DenoisedSpecular      → RtxdiDenoisedSpecularLighting
    ///   t9  : t_PSRDiffuseAlbedo      → RtxdiNormals (dummy; PSR disabled)
    ///   t10 : t_PSRSpecularF0         → RtxdiNormals (dummy; PSR disabled)
    ///   s0  : s_EnvironmentSampler    (not used when enableEnvironmentMap=0)
    /// </summary>
    public class NativeRtxdiCompositingPass : ScriptableRenderPass, IDisposable
    {
        private readonly NativeComputePipeline      _cs;
        private readonly NativeComputeDescriptorSet _ds;
        private          Resource                   _resource;
        private          Settings                   _settings;

        private const uint GroupSize = 8u;

        public NativeRtxdiCompositingPass(NativeComputeShader cs)
        {
            _cs = new NativeComputePipeline(cs);
            _ds = new NativeComputeDescriptorSet(_cs);
        }

        public void Dispose()
        {
            _ds?.Dispose();
            _cs?.Dispose();
        }

        // -------------------------------------------------------------------------
        // Resource / Settings
        // -------------------------------------------------------------------------

        public class Resource
        {
            internal IntPtr                      ConstantBuffer;
            internal IntPtr                      CurrentDepth;
            internal NativeRtxdiTextureResources Pool;
            internal NativeRtxdiGPUScene         GpuScene;
        }

        public class Settings
        {
            internal int renderW;
            internal int renderH;
        }

        public void Setup(Resource resource, Settings settings)
        {
            _resource = resource;
            _settings = settings;
        }

        // -------------------------------------------------------------------------
        // Pass data
        // -------------------------------------------------------------------------

        class PassData
        {
            internal NativeComputePipeline      Cs;
            internal NativeComputeDescriptorSet Ds;
            internal Resource                   Resource;
            internal Settings                   Settings;
        }

        // -------------------------------------------------------------------------
        // Execution
        // -------------------------------------------------------------------------

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd  = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var cs   = data.Cs;
            var ds   = data.Ds;
            var pool = data.Resource.Pool;

            cmd.BeginSample(RenderPassMarkers.Composition);

            // CBV
            ds.SetConstantBuffer("g_Const", data.Resource.ConstantBuffer);

            // UAV outputs
            ds.SetRWTexture("u_Output",        pool.HdrColor.NativePtr);
            ds.SetRWTexture("u_MotionVectors",  pool.MotionVectors.NativePtr);

            // SRV inputs – GBuffer
            ds.SetTexture("t_GBufferDepth", data.Resource.CurrentDepth);
            ds.SetTexture("t_GBufferNormals",        pool.GBufferNormals.NativePtr);
            ds.SetTexture("t_GBufferDiffuseAlbedo",  pool.GBufferDiffuseAlbedo.NativePtr);
            ds.SetTexture("t_GBufferSpecularRough",  pool.GBufferSpecularRough.NativePtr);
            ds.SetTexture("t_GBufferEmissive",       pool.GBufferEmissive.NativePtr);

            // SRV inputs – lighting (noisy)
            ds.SetTexture("t_Diffuse",  pool.DiffuseLighting.NativePtr);
            ds.SetTexture("t_Specular", pool.SpecularLighting.NativePtr);

            // SRV inputs – denoised
            ds.SetTexture("t_DenoisedDiffuse",  pool.DenoisedDiffuseLighting.NativePtr);
            ds.SetTexture("t_DenoisedSpecular", pool.DenoisedSpecularLighting.NativePtr);

            // SRV inputs – PSR (not used; bind normals as dummy to avoid null hazard)
            ds.SetTexture("t_PSRDiffuseAlbedo", pool.GBufferNormals.NativePtr);
            ds.SetTexture("t_PSRSpecularF0",    pool.GBufferNormals.NativePtr);

            // Bindless scene textures/buffers (space1/space2)
            data.Resource.GpuScene?.BindToShader(ds);

            uint gx = ((uint)data.Settings.renderW + GroupSize - 1u) / GroupSize;
            uint gy = ((uint)data.Settings.renderH + GroupSize - 1u) / GroupSize;
            cs.Dispatch(cmd, ds, gx, gy, 1);

            cmd.EndSample(RenderPassMarkers.Composition);
        }

        // -------------------------------------------------------------------------
        // RenderGraph
        // -------------------------------------------------------------------------

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxdiCompositingPass", out var passData);

            passData.Cs       = _cs;
            passData.Ds       = _ds;
            passData.Resource = _resource;
            passData.Settings = _settings;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }
    }
}
