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
    public class RtxptDenoisingGuidesBakePass : ScriptableRenderPass, IDisposable
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
        private          RtxptPassContext     _ctx;

        public RtxptDenoisingGuidesBakePass(NativeComputeShader shader)
        {
            // Root-constant hints (g_denoisingConstants) live on the .computeshader asset (the importer).
            _cs = new NativeComputePipeline(shader);
            _pingDs = new NativeComputeDescriptorSet(_cs);
            _pongDs = new NativeComputeDescriptorSet(_cs);
        }

        public void Dispose()
        {
            _pingDs?.Dispose();
            _pongDs?.Dispose();
            _cs?.Dispose();
        }

        public void Setup(RtxptPassContext ctx) => _ctx = ctx;

        // ── Pass data ─────────────────────────────────────────────────────────

        private class PassData
        {
            internal NativeComputePipeline           Cs;
            internal NativeComputeDescriptorSet      PingDs;
            internal NativeComputeDescriptorSet      PongDs;
            internal RtxptPassContext          Ctx;
        }

        // ── RenderGraph ───────────────────────────────────────────────────────

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("Denoising Guides Bake", out var passData);
            passData.Cs         = _cs;
            passData.PingDs     = _pingDs;
            passData.PongDs     = _pongDs;
            passData.Ctx        = _ctx;
            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData d, UnsafeGraphContext c) => ExecutePass(d, c));
        }

        // ── Execute ───────────────────────────────────────────────────────────

        private static unsafe void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var ctx = data.Ctx;
            var res = ctx.Textures;

            uint gx = ((uint)ctx.RenderResolution.x + 7u) / 8u;
            uint gy = ((uint)ctx.RenderResolution.y + 7u) / 8u;

            // Two dispatches (ping then pong) both labelled "DenoiseSpecHitT" to mirror the
            // original RTXPT DenoisingGuidesBaker, which emits the inner marker twice by that
            // name under the "Denoising Guides Bake" group — so PIX captures line up.
            cmd.BeginSample("DenoiseSpecHitT");
            DispatchDenoiseSpecHitT(cmd, data, data.PingDs, 1u, gx, gy, res, ctx);
            cmd.EndSample("DenoiseSpecHitT");

            cmd.BeginSample("DenoiseSpecHitT");
            DispatchDenoiseSpecHitT(cmd, data, data.PongDs, 0u, gx, gy, res, ctx);
            cmd.EndSample("DenoiseSpecHitT");
        }

        private static unsafe void DispatchDenoiseSpecHitT(
            CommandBuffer cmd,
            PassData data,
            NativeComputeDescriptorSet ds,
            uint ping,
            uint gx,
            uint gy,
            RtxptTextureResources res,
            RtxptPassContext ctx)
        {
            var cb = new DenoisingGuidesBakerConstants
            {
                RenderResX  = (uint)ctx.RenderResolution.x,
                RenderResY  = (uint)ctx.RenderResolution.y,
                DisplayResX = (uint)ctx.DisplayResolution.x,
                DisplayResY = (uint)ctx.DisplayResolution.y,
                DebugView   = 0,
                Ping        = ping,
            };
            ds.SetRootConstants("g_denoisingConstants", &cb);

            ds.SetRWTexture("u_Depth", res.Depth.NativePtr);
            ds.SetRWTexture("u_SpecularHitT", res.SpecularHitT.NativePtr);
            ds.SetRWTexture("u_ScratchFloat1", res.ScratchFloat1.NativePtr);

            data.Cs.Dispatch(cmd, ds, gx, gy, 1);
        }
    }
}
