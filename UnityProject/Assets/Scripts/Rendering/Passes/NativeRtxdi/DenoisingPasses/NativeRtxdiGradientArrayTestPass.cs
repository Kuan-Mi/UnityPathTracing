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
    /// Minimal test pass that writes known colors into both slices of a Texture2DArray
    /// allocated via <see cref="Nri.NriTextureArrayResource"/>.
    ///
    /// Slice 0 → red  (1, 0, 0, 1)
    /// Slice 1 → green (0, 1, 0, 1)
    ///
    /// Assign <c>GradientArrayTest.computeshader</c> to the matching field on
    /// <see cref="NativeRtxdiFeature"/> to enable the pass.
    /// </summary>
    public class NativeRtxdiGradientArrayTestPass : ScriptableRenderPass, IDisposable
    {
        [StructLayout(LayoutKind.Sequential)]
        private struct TestConstants
        {
            public uint width;
            public uint height;
        }

        private readonly NativeComputePipeline  _cs;
        private readonly NativeComputeDescriptorSet _ds;
        private readonly GraphicsBuffer         _cbuffer;

        private IntPtr _gradientArrayPtr;
        private uint   _groupsX;
        private uint   _groupsY;

        public NativeRtxdiGradientArrayTestPass(NativeComputeShader cs)
        {
            _cs      = new NativeComputePipeline(cs);
            _ds      = new NativeComputeDescriptorSet(_cs);
            _cbuffer = new GraphicsBuffer(GraphicsBuffer.Target.Constant, 1, Marshal.SizeOf<TestConstants>());
        }

        public void Dispose()
        {
            _ds?.Dispose();
            _cbuffer?.Release();
            _cs?.Dispose();
        }

        /// <param name="gradientArrayPtr">NRI-wrapped native ptr from <c>NriTextureArrayResource.NriPtr</c>.</param>
        /// <param name="dims">Texture dimensions (width × height).</param>
        public void Setup(IntPtr gradientArrayPtr, int2 dims)
        {
            _gradientArrayPtr = gradientArrayPtr;
            _groupsX          = (uint)((dims.x + 7) / 8);
            _groupsY          = (uint)((dims.y + 7) / 8);

            var constants = new TestConstants[] { new() { width = (uint)dims.x, height = (uint)dims.y } };
            _cbuffer.SetData(constants);

            _ds.SetConstantBuffer("g_Const", _cbuffer.GetNativeBufferPtr());
            _ds.SetRWTexture("u_Output", _gradientArrayPtr);
        }

        // ── Render graph ──────────────────────────────────────────────────────

        private class PassData
        {
            internal NativeComputePipeline     Cs;
            internal NativeComputeDescriptorSet Ds;
            internal uint GroupsX;
            internal uint GroupsY;
        }

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            cmd.BeginSample("NativeRtxdiGradientArrayTest");
            data.Cs.Dispatch(cmd, data.Ds, data.GroupsX, data.GroupsY, 1);
            cmd.EndSample("NativeRtxdiGradientArrayTest");
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            if (_gradientArrayPtr == IntPtr.Zero) return;

            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxdiGradientArrayTest", out var passData);

            passData.Cs      = _cs;
            passData.Ds      = _ds;
            passData.GroupsX = _groupsX;
            passData.GroupsY = _groupsY;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext ctx) => ExecutePass(data, ctx));
        }
    }
}
