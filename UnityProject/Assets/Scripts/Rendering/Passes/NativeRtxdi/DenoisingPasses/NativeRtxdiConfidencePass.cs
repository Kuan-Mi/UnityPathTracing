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
    /// Converts filtered gradients into NRD-compatible confidence values with one-frame
    /// temporal blending.  Matches FullSample ConfidencePass.cpp / ConfidencePass.hlsl.
    ///
    /// Default parameters mirror FullSample LightingSettings:
    ///   darknessBias          = exp2(-12)  ≈ 0.000244
    ///   sensitivity           = 8.0
    ///   confidenceHistoryLength = 0.75  → blendFactor = 1 / (0.75 + 1) ≈ 0.571
    ///
    /// inputBufferIndex = 0: after 4 FilterGradients passes the output sits in slice 0
    /// of the gradient Texture2DArray (even number of passes, each pass flips the slice).
    /// </summary>
    public class NativeRtxdiConfidencePass : ScriptableRenderPass, IDisposable
    {
        // FullSample defaults
        private const float GradientLogDarknessBias  = -12f;
        private const float GradientSensitivity      =  8f;
        private const float ConfidenceHistoryLength  =  0.75f;
        // After 4 FilterGradients passes final data is in slice 0.
        private const int   FilteredGradientSlice    =  0;

        [StructLayout(LayoutKind.Sequential)]
        private struct ConfidenceConstants
        {
            public uint  viewportSizeX;       // offset  0
            public uint  viewportSizeY;       // offset  4
            public float invGradTexSizeX;     // offset  8
            public float invGradTexSizeY;     // offset 12
            public float darknessBias;        // offset 16
            public float sensitivity;         // offset 20
            public uint  checkerboard;        // offset 24
            public int   inputBufferIndex;    // offset 28
            public float blendFactor;         // offset 32
        }

        private readonly NativeComputePipeline      _cs;
        private readonly NativeComputeDescriptorSet _ds;
        private readonly GraphicsBuffer             _cbuffer;

        private uint _groupsX;
        private uint _groupsY;

        public NativeRtxdiConfidencePass(NativeComputeShader cs)
        {
            _cs      = new NativeComputePipeline(cs);
            _ds      = new NativeComputeDescriptorSet(_cs);
            _cbuffer = new GraphicsBuffer(GraphicsBuffer.Target.Constant, 1, Marshal.SizeOf<ConfidenceConstants>());
        }

        public void Dispose()
        {
            _ds?.Dispose();
            _cbuffer?.Release();
            _cs?.Dispose();
        }

        /// <param name="ctx">Frame context providing motion vectors and confidence ping-pong ptrs.</param>
        /// <param name="gradientArrayPtr">Native ptr to the Texture2DArray (2 slices) written by FilterGradientsPass.</param>
        /// <param name="gradDims">Gradient texture dimensions = ceil(renderRes / RTXDI_GRAD_FACTOR).</param>
        public void Setup(NativeRtxdiPassContext ctx, IntPtr gradientArrayPtr, int2 gradDims)
        {
            int renderW = ctx.RenderResolution.x;
            int renderH = ctx.RenderResolution.y;
            _groupsX = (uint)((renderW + 7) / 8);
            _groupsY = (uint)((renderH + 7) / 8);

            float blendFactor = 1f / (ConfidenceHistoryLength + 1f);

            var tmp = new ConfidenceConstants[]
            {
                new ConfidenceConstants
                {
                    viewportSizeX    = (uint)renderW,
                    viewportSizeY    = (uint)renderH,
                    invGradTexSizeX  = 1f / gradDims.x,
                    invGradTexSizeY  = 1f / gradDims.y,
                    darknessBias     = math.exp2(GradientLogDarknessBias),
                    sensitivity      = GradientSensitivity,
                    checkerboard     = 0,
                    inputBufferIndex = FilteredGradientSlice,
                    blendFactor      = blendFactor,
                }
            };
            _cbuffer.SetData(tmp);

            // Pre-bake descriptor set bindings (all ptrs known at Setup time).
            _ds.SetConstantBuffer("g_Const", _cbuffer.GetNativeBufferPtr());
            _ds.SetTexture("t_Gradients",            gradientArrayPtr);
            _ds.SetTexture("t_MotionVectors",         ctx.MotionVectorsPtr);
            _ds.SetTexture("t_PrevDiffuseConfidence", ctx.PrevDiffuseConfidencePtr);
            _ds.SetTexture("t_PrevSpecularConfidence", ctx.PrevSpecularConfidencePtr);
            _ds.SetRWTexture("u_DiffuseConfidence",   ctx.DiffuseConfidencePtr);
            _ds.SetRWTexture("u_SpecularConfidence",  ctx.SpecularConfidencePtr);
        }

        // ── Render graph ─────────────────────────────────────────────────────────

        class PassData
        {
            internal NativeComputePipeline      Cs;
            internal NativeComputeDescriptorSet Ds;
            internal uint                       GroupsX;
            internal uint                       GroupsY;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            cmd.BeginSample("NativeRtxdiConfidencePass");
            data.Cs.Dispatch(cmd, data.Ds, data.GroupsX, data.GroupsY, 1);
            cmd.EndSample("NativeRtxdiConfidencePass");
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxdiConfidencePass", out var passData);

            passData.Cs      = _cs;
            passData.Ds      = _ds;
            passData.GroupsX = _groupsX;
            passData.GroupsY = _groupsY;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext ctx) => ExecutePass(data, ctx));
        }
    }
}
