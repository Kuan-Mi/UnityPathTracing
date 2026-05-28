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
    /// Phase 4: Bilateral specular hit-distance filter (×2 ping-pong).
    ///
    /// Shader: DenoisingGuidesBaker_DenoiseSpecHitT.computeshader  numthreads [8,8,1]
    ///
    /// Bindings (from shader source DenoisingGuidesBaker.hlsl):
    ///   b1  g_denoisingConstants  (DenoisingGuidesBakerConstants)
    ///   u6  u_Depth
    ///   u7  u_SpecularHitT
    ///   u8  u_ScratchFloat1
    ///
    /// Pass 0  (Ping=1): reads SpecularHitT, writes ScratchFloat1
    /// Pass 1  (Ping=0): reads ScratchFloat1, writes SpecularHitT
    /// </summary>
    public class NativeRtxptDenoisingGuidesBakePass : ScriptableRenderPass, IDisposable
    {
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        private struct DenoisingGuidesBakerConstants
        {
            public uint RenderResX,  RenderResY;
            public uint DisplayResX, DisplayResY;
            public int  DebugView;
            public uint Ping;
            public uint _pad1,  _pad2;
            public uint _pad3x, _pad3y, _pad3z, _pad3w;
            public uint _pad4x, _pad4y, _pad4z, _pad4w;
        }

        private readonly NativeComputePipeline      _cs;
        private readonly NativeComputeDescriptorSet _pingDs;
        private readonly NativeComputeDescriptorSet _pongDs;
        private          NativeRtxptPassContext     _ctx;

        private readonly DenoisingGuidesBakerConstants[] _pingCbData = new DenoisingGuidesBakerConstants[1];
        private readonly DenoisingGuidesBakerConstants[] _pongCbData = new DenoisingGuidesBakerConstants[1];
        private          GraphicsBuffer                  _pingCb;
        private          GraphicsBuffer                  _pongCb;
        private          IntPtr                          _pingCbPtr;
        private          IntPtr                          _pongCbPtr;

        public NativeRtxptDenoisingGuidesBakePass(NativeComputeShader shader)
        {
            _cs = new NativeComputePipeline(shader);
            _pingDs = new NativeComputeDescriptorSet(_cs);
            _pongDs = new NativeComputeDescriptorSet(_cs);
            _pingCb = new GraphicsBuffer(GraphicsBuffer.Target.Constant, 1,
                    Marshal.SizeOf<DenoisingGuidesBakerConstants>())
                { name = "Rtxpt_DenoisingGuidesBakerConstants_Ping" };
            _pingCbPtr = _pingCb.GetNativeBufferPtr();
            _pongCb = new GraphicsBuffer(GraphicsBuffer.Target.Constant, 1,
                    Marshal.SizeOf<DenoisingGuidesBakerConstants>())
                { name = "Rtxpt_DenoisingGuidesBakerConstants_Pong" };
            _pongCbPtr = _pongCb.GetNativeBufferPtr();
        }

        public void Dispose()
        {
            _pingDs?.Dispose();
            _pongDs?.Dispose();
            _cs?.Dispose();
            _pingCb?.Dispose();
            _pingCb    = null;
            _pingCbPtr = IntPtr.Zero;
            _pongCb?.Dispose();
            _pongCb    = null;
            _pongCbPtr = IntPtr.Zero;
        }

        public void Setup(NativeRtxptPassContext ctx) => _ctx = ctx;

        // ── Pass data ─────────────────────────────────────────────────────────

        private class PassData
        {
            internal NativeComputePipeline           Cs;
            internal NativeComputeDescriptorSet      PingDs;
            internal NativeComputeDescriptorSet      PongDs;
            internal NativeRtxptPassContext          Ctx;
            internal GraphicsBuffer                  PingCb;
            internal GraphicsBuffer                  PongCb;
            internal IntPtr                          PingCbPtr;
            internal IntPtr                          PongCbPtr;
            internal DenoisingGuidesBakerConstants[] PingCbData;
            internal DenoisingGuidesBakerConstants[] PongCbData;
        }

        // ── RenderGraph ───────────────────────────────────────────────────────

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxpt.DenoiseSpecHitT", out var passData);
            passData.Cs         = _cs;
            passData.PingDs     = _pingDs;
            passData.PongDs     = _pongDs;
            passData.Ctx        = _ctx;
            passData.PingCb     = _pingCb;
            passData.PingCbPtr  = _pingCbPtr;
            passData.PongCb     = _pongCb;
            passData.PongCbPtr  = _pongCbPtr;
            passData.PingCbData = _pingCbData;
            passData.PongCbData = _pongCbData;
            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData d, UnsafeGraphContext c) => ExecutePass(d, c));
        }

        // ── Execute ───────────────────────────────────────────────────────────

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var ctx = data.Ctx;
            var res = ctx.Textures;

            uint gx = ((uint)ctx.RenderResolution.x + 7u) / 8u;
            uint gy = ((uint)ctx.RenderResolution.y + 7u) / 8u;

            cmd.BeginSample("Rtxpt.DenoiseSpecHitT[Ping]");
            DispatchDenoiseSpecHitT(cmd, data, data.PingDs, data.PingCb, data.PingCbPtr, data.PingCbData, 1u, gx, gy, res, ctx);
            cmd.EndSample("Rtxpt.DenoiseSpecHitT[Ping]");

            cmd.BeginSample("Rtxpt.DenoiseSpecHitT[Pong]");
            DispatchDenoiseSpecHitT(cmd, data, data.PongDs, data.PongCb, data.PongCbPtr, data.PongCbData, 0u, gx, gy, res, ctx);
            cmd.EndSample("Rtxpt.DenoiseSpecHitT[Pong]");
        }

        private static void DispatchDenoiseSpecHitT(
            CommandBuffer cmd,
            PassData data,
            NativeComputeDescriptorSet ds,
            GraphicsBuffer cb,
            IntPtr cbPtr,
            DenoisingGuidesBakerConstants[] cbData,
            uint ping,
            uint gx,
            uint gy,
            NativeRtxptTextureResources res,
            NativeRtxptPassContext ctx)
        {
            cbData[0] = new DenoisingGuidesBakerConstants
            {
                RenderResX  = (uint)ctx.RenderResolution.x,
                RenderResY  = (uint)ctx.RenderResolution.y,
                DisplayResX = (uint)ctx.DisplayResolution.x,
                DisplayResY = (uint)ctx.DisplayResolution.y,
                DebugView   = 0,
                Ping        = ping,
            };
            cb.SetData(cbData);
            cbPtr = cb.GetNativeBufferPtr();
            ds.SetConstantBuffer("g_denoisingConstants", cbPtr);

            ds.SetRWTexture("u_Depth", res.Depth.NativePtr);
            ds.SetRWTexture("u_SpecularHitT", res.SpecularHitT.NativePtr);
            ds.SetRWTexture("u_ScratchFloat1", res.ScratchFloat1.NativePtr);

            data.Cs.Dispatch(cmd, ds, gx, gy, 1);
        }
    }
}
