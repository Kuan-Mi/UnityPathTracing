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
    /// A-trous spatial filter over the gradient Texture2DArray (2 slices).
    /// Runs 4 passes with doubling step sizes (1, 2, 4, 8 pixels), ping-ponging
    /// between array slices 0 and 1 internally via <c>passIndex</c>.
    ///
    /// Matches FullSample FilterGradientsPass.cpp / FilterGradientsPass.hlsl.
    /// RTXDI_GRAD_FACTOR = 3 is hardcoded in ShaderParameters.h; gradient texture
    /// dimensions = ceil(renderRes / 3).
    ///
    /// After 4 passes (passIndex 0-3), final output is written to slice 0:
    ///   pass 0: reads slice 0, writes slice 1
    ///   pass 1: reads slice 1, writes slice 0
    ///   pass 2: reads slice 0, writes slice 1
    ///   pass 3: reads slice 1, writes slice 0
    /// </summary>
    public class NativeRtxdiFilterGradientsPass : ScriptableRenderPass, IDisposable
    {
        private const int NumPasses = 4;

        [StructLayout(LayoutKind.Sequential)]
        private struct FilterGradientsConstants
        {
            public uint viewportSizeX;
            public uint viewportSizeY;
            public int  passIndex;
            public uint checkerboard;
        }

        private readonly NativeComputePipeline        _cs;
        private readonly NativeComputeDescriptorSet[] _ds;
        private readonly GraphicsBuffer[]             _cbuffers;
        private          IntPtr                       _gradientArrayPtr;
        private          uint                         _groupsX;
        private          uint                         _groupsY;

        public NativeRtxdiFilterGradientsPass(NativeComputeShader cs)
        {
            _cs       = new NativeComputePipeline(cs);
            _ds       = new NativeComputeDescriptorSet[NumPasses];
            _cbuffers = new GraphicsBuffer[NumPasses];
            for (int i = 0; i < NumPasses; i++)
            {
                _ds[i]       = new NativeComputeDescriptorSet(_cs);
                _cbuffers[i] = new GraphicsBuffer(GraphicsBuffer.Target.Constant, 1, Marshal.SizeOf<FilterGradientsConstants>());
            }
        }

        public void Dispose()
        {
            for (int i = 0; i < NumPasses; i++)
            {
                _ds[i]?.Dispose();
                _cbuffers[i]?.Release();
            }
            _cs?.Dispose();
        }

        /// <param name="gradientArrayPtr">Native ptr to the Texture2DArray (2 slices, RGBA16_SFloat).</param>
        /// <param name="gradDims">Dimensions of the gradient texture = ceil(renderRes / RTXDI_GRAD_FACTOR).</param>
        public void Setup(IntPtr gradientArrayPtr, int2 gradDims)
        {
            _gradientArrayPtr = gradientArrayPtr;
            _groupsX          = (uint)((gradDims.x + 7) / 8);
            _groupsY          = (uint)((gradDims.y + 7) / 8);

            var tmp = new FilterGradientsConstants[1];
            for (int i = 0; i < NumPasses; i++)
            {
                tmp[0] = new FilterGradientsConstants
                {
                    viewportSizeX = (uint)gradDims.x,
                    viewportSizeY = (uint)gradDims.y,
                    passIndex     = i,
                    checkerboard  = 0,
                };
                _cbuffers[i].SetData(tmp);
            }
        }

        // ── Render graph ─────────────────────────────────────────────────────────

        class PassData
        {
            internal NativeComputePipeline        Cs;
            internal NativeComputeDescriptorSet[] Ds;
            internal uint                         GroupsX;
            internal uint                         GroupsY;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            cmd.BeginSample("NativeRtxdiFilterGradients");
            for (int i = 0; i < NumPasses; i++)
                data.Cs.Dispatch(cmd, data.Ds[i], data.GroupsX, data.GroupsY, 1);
            cmd.EndSample("NativeRtxdiFilterGradients");
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxdiFilterGradients", out var passData);

            passData.Cs      = _cs;
            passData.Ds      = _ds;
            passData.GroupsX = _groupsX;
            passData.GroupsY = _groupsY;

            // Pre-bake bindings: all passes share the same Texture2DArray;
            // slice selection is encoded in g_Const.passIndex inside the shader.
            for (int i = 0; i < NumPasses; i++)
            {
                _ds[i].SetConstantBuffer("g_Const", _cbuffers[i].GetNativeBufferPtr());
                _ds[i].SetRWTexture("u_Gradients", _gradientArrayPtr);
            }

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext ctx) => ExecutePass(data, ctx));
        }
    }
}
