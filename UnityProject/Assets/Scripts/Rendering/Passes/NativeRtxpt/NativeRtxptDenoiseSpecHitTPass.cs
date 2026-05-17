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
    /// Pass 0  (Ping=0): reads SpecularHitT → writes ScratchFloat1
    /// Pass 1  (Ping=1): reads ScratchFloat1 → writes SpecularHitT
    /// </summary>
    public class NativeRtxptDenoiseSpecHitTPass : ScriptableRenderPass, IDisposable
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
        private readonly NativeComputeDescriptorSet _ds;
        private          NativeRtxptPassContext     _ctx;

        private readonly DenoisingGuidesBakerConstants[] _cbData = new DenoisingGuidesBakerConstants[1];
        private          GraphicsBuffer                  _cb;

        public NativeRtxptDenoiseSpecHitTPass(NativeComputeShader shader)
        {
            _cs = new NativeComputePipeline(shader);
            _ds = new NativeComputeDescriptorSet(_cs);
            _cb = new GraphicsBuffer(GraphicsBuffer.Target.Constant, 1,
                    Marshal.SizeOf<DenoisingGuidesBakerConstants>())
                { name = "Rtxpt_DenoisingGuidesBakerConstants" };
        }

        public void Dispose()
        {
            _ds?.Dispose();
            _cs?.Dispose();
            _cb?.Dispose();
            _cb = null;
        }

        public void Setup(NativeRtxptPassContext ctx) => _ctx = ctx;

        // ── Pass data ─────────────────────────────────────────────────────────

        private class PassData
        {
            internal NativeComputePipeline           Cs;
            internal NativeComputeDescriptorSet      Ds;
            internal NativeRtxptPassContext          Ctx;
            internal GraphicsBuffer                  Cb;
            internal DenoisingGuidesBakerConstants[] CbData;
        }

        // ── RenderGraph ───────────────────────────────────────────────────────

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxpt.DenoiseSpecHitT", out var passData);
            passData.Cs     = _cs;
            passData.Ds     = _ds;
            passData.Ctx    = _ctx;
            passData.Cb     = _cb;
            passData.CbData = _cbData;
            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData d, UnsafeGraphContext c) => ExecutePass(d, c));
        }

        // ── Execute ───────────────────────────────────────────────────────────

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var ctx = data.Ctx;
            var ds  = data.Ds;
            var res = ctx.Textures;

            uint gx = ((uint)ctx.RenderResolution.x + 7u) / 8u;
            uint gy = ((uint)ctx.RenderResolution.y + 7u) / 8u;

            // Two ping-pong passes
            for (uint ping = 0; ping <= 1; ping++)
            {
                cmd.BeginSample(ping == 0 ? "Rtxpt.DenoiseSpecHitT[0]" : "Rtxpt.DenoiseSpecHitT[1]");

                data.CbData[0] = new DenoisingGuidesBakerConstants
                {
                    RenderResX  = (uint)ctx.RenderResolution.x,
                    RenderResY  = (uint)ctx.RenderResolution.y,
                    DisplayResX = (uint)ctx.DisplayResolution.x,
                    DisplayResY = (uint)ctx.DisplayResolution.y,
                    DebugView   = 0,
                    Ping        = ping,
                };
                data.Cb.SetData(data.CbData);
                ds.SetConstantBuffer("g_denoisingConstants", data.Cb.GetNativeBufferPtr());

                ds.SetRWTexture("u_Depth", res.Depth.NativePtr);
                ds.SetRWTexture("u_SpecularHitT", res.SpecularHitT.NativePtr);
                ds.SetRWTexture("u_ScratchFloat1", res.ScratchFloat1.NativePtr);

                data.Cs.Dispatch(cmd, ds, gx, gy, 1);

                cmd.EndSample(ping == 0 ? "Rtxpt.DenoiseSpecHitT[0]" : "Rtxpt.DenoiseSpecHitT[1]");
            }
        }
    }
}