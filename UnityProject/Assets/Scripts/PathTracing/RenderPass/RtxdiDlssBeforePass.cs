using System;
using System.Runtime.InteropServices;
using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;
using static PathTracing.ShaderIDs;

namespace PathTracing
{
    public class RxtdiDlssBeforePass : ScriptableRenderPass
    {
        private readonly ComputeShader DlssBeforeCs;

        private RtxdiPassContext _context;
        private RTHandle _rrDiffAlbedo;
        private RTHandle _rrSpecAlbedo;
        private RTHandle _rrSpecHitDist;
        private RTHandle _rrNormalRoughness;
        private int _rectGridW;
        private int _rectGridH;

        public RxtdiDlssBeforePass(ComputeShader dlssBeforeCs)
        {
            DlssBeforeCs = dlssBeforeCs;
        }

        public void Setup(RtxdiPassContext ctx, RTHandle rrDiffAlbedo, RTHandle rrSpecAlbedo, RTHandle rrSpecHitDist, RTHandle rrNormalRoughness, int rectGridW, int rectGridH)
        {
            _context = ctx;
            _rrDiffAlbedo = rrDiffAlbedo;
            _rrSpecAlbedo = rrSpecAlbedo;
            _rrSpecHitDist = rrSpecHitDist;
            _rrNormalRoughness = rrNormalRoughness;
            _rectGridW = rectGridW;
            _rectGridH = rectGridH;
        }

        class PassData
        {
            internal ComputeShader DlssBeforeCs;
            internal RtxdiPassContext Context;
            internal RTHandle RRDiffAlbedo;
            internal RTHandle RRSpecAlbedo;
            internal RTHandle RRSpecHitDist;
            internal RTHandle RRNormalRoughness;
            internal int RectGridW;
            internal int RectGridH;
        }

        [DllImport("RenderingPlugin")]
        private static extern IntPtr GetRenderEventAndDataFunc();

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            var dlssBeforeMarker = RenderPassMarkers.DlssBefore;

            // dlss Before
            natCmd.BeginSample(dlssBeforeMarker);
            var ctx = data.Context;
            natCmd.SetComputeConstantBufferParam(data.DlssBeforeCs, paramsID, ctx.ConstantBuffer, 0, ctx.ConstantBuffer.stride);

            natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, gIn_ViewDepthID, ctx.ViewDepth);
            natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, gIn_DiffuseAlbedoID, ctx.DiffuseAlbedo);
            natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, gIn_SpecularRoughID, ctx.SpecularRough);
            natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, gIn_NormalsID, ctx.Normals);

            natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, gOut_DiffAlbedoID, data.RRDiffAlbedo);
            natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, gOut_SpecAlbedoID, data.RRSpecAlbedo);
            natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, gOut_SpecHitDistanceID, data.RRSpecHitDist);
            natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, g_Normal_RoughnessID, data.RRNormalRoughness);

            natCmd.DispatchCompute(data.DlssBeforeCs, 0, data.RectGridW, data.RectGridH, 1);
            natCmd.EndSample(dlssBeforeMarker);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("DLSS RR Before", out var passData);

            passData.DlssBeforeCs = DlssBeforeCs;
            passData.Context = _context;
            passData.RRDiffAlbedo = _rrDiffAlbedo;
            passData.RRSpecAlbedo = _rrSpecAlbedo;
            passData.RRSpecHitDist = _rrSpecHitDist;
            passData.RRNormalRoughness = _rrNormalRoughness;
            passData.RectGridW = _rectGridW;
            passData.RectGridH = _rectGridH;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}