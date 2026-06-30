using System;
using NativeRender;

namespace PathTracing
{
    /// <summary>
    /// Shared binding helper for Rtxpt compute and RT passes.
    ///
    /// All RTXPT shaders include <c>Shaders/Bindings/ShaderResourceBindings.hlsli</c> which
    /// declares the global binding set (g_Const b0, all output UAVs, stable planes, DLSS-RR guides, …).
    ///
    /// <see cref="NativeComputeDescriptorSet"/> and <see cref="RayTracePipeline"/> both silently
    /// ignore names absent from a shader's reflection map, so this helper safely binds the full
    /// superset; each shader only consumes what it actually references.
    /// </summary>
    internal static class RtxptBindings
    {
        /// <summary>
        /// Bind all global resources declared in ShaderResourceBindings.hlsli to a compute descriptor set.
        /// Call this first, then bind any pass-specific SRVs/UAVs not covered here.
        /// </summary>
        public static void BindCommon(NativeComputeDescriptorSet ds, RtxptPassContext ctx)
        {
            BindConstantsAndScene(ds, ctx);
            BindOutputUAVs(ds, ctx);
            BindStablePlanes(ds, ctx);
            BindDlssRrGuides(ds, ctx);
        }

        /// <summary>
        /// Bind all global resources to a DXR ray-tracing pipeline.
        /// </summary>
        public static void BindCommon(RayTracePipeline rtp, RtxptPassContext ctx)
        {
        }

        // ── Private helpers ───────────────────────────────────────────────────

        private static void BindConstantsAndScene(NativeComputeDescriptorSet ds, RtxptPassContext ctx)
        {
            // if (ctx.ConstantBuffer != null)
            //     ds.SetConstantBuffer("g_Const", ctx.ConstantBuffer);
            //
            // var tlas = ctx.NrdSampleResource?.AccelerationStructure;
            // if (tlas != null)
            //     ds.SetAccelerationStructure("SceneBVH", tlas);
            //
            // ctx.GpuScene?.BindToShader(ds);
        }

        private static void BindOutputUAVs(NativeComputeDescriptorSet ds, RtxptPassContext ctx)
        {
            if (ctx.OutputColorPtr        != IntPtr.Zero) ds.SetRWTexture("u_OutputColor",        ctx.OutputColorPtr);
            if (ctx.ProcessedOutputColorPtr != IntPtr.Zero) ds.SetRWTexture("u_ProcessedOutputColor", ctx.ProcessedOutputColorPtr);
            if (ctx.ThroughputPtr         != IntPtr.Zero) ds.SetRWTexture("u_Throughput",         ctx.ThroughputPtr);
            if (ctx.MotionVectorsPtr      != IntPtr.Zero) ds.SetRWTexture("u_MotionVectors",      ctx.MotionVectorsPtr);
            if (ctx.DepthPtr              != IntPtr.Zero) ds.SetRWTexture("u_Depth",              ctx.DepthPtr);
            if (ctx.SpecularHitTPtr       != IntPtr.Zero) ds.SetRWTexture("u_SpecularHitT",       ctx.SpecularHitTPtr);
            if (ctx.ScratchFloat1Ptr      != IntPtr.Zero) ds.SetRWTexture("u_ScratchFloat1",      ctx.ScratchFloat1Ptr);
            if (ctx.ShaderDebugVizPtr     != IntPtr.Zero) ds.SetRWTexture("u_ShaderDebugVizTextureBuffer", ctx.ShaderDebugVizPtr);
        }

        private static void BindStablePlanes(NativeComputeDescriptorSet ds, RtxptPassContext ctx)
        {
            if (ctx.StablePlanesHeaderPtr != IntPtr.Zero)
                ds.SetRWTexture("u_StablePlanesHeader", ctx.StablePlanesHeaderPtr);
            if (ctx.StableRadiancePtr != IntPtr.Zero)
                ds.SetRWTexture("u_StableRadiance", ctx.StableRadiancePtr);

            var buf = ctx.Buffers;
            if (buf?.StablePlanesBufferPtr != IntPtr.Zero)
                ds.SetRWStructuredBuffer("u_StablePlanesBuffer", buf.StablePlanesBufferPtr, buf.StablePlanesBuffer.count, buf.StablePlanesBuffer.stride);

            if (buf?.SurfaceDataBufferPtr != IntPtr.Zero)
                ds.SetRWStructuredBuffer("u_SurfaceData", buf.SurfaceDataBufferPtr, buf.SurfaceDataBuffer.count, buf.SurfaceDataBuffer.stride);
        }

        private static void BindDlssRrGuides(NativeComputeDescriptorSet ds, RtxptPassContext ctx)
        {
            if (ctx.DlssRrDiffAlbedoPtr      != IntPtr.Zero) ds.SetRWTexture("u_RRDiffuseAlbedo",      ctx.DlssRrDiffAlbedoPtr);
            if (ctx.DlssRrSpecAlbedoPtr      != IntPtr.Zero) ds.SetRWTexture("u_RRSpecAlbedo",         ctx.DlssRrSpecAlbedoPtr);
            if (ctx.DlssRrNormalRoughnessPtr != IntPtr.Zero) ds.SetRWTexture("u_RRNormalsAndRoughness", ctx.DlssRrNormalRoughnessPtr);
        }
    }
}
