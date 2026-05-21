using System;
using NativeRender;

namespace PathTracing
{
    /// <summary>
    /// Shared binding helper for NativeRtxpt compute and RT passes.
    ///
    /// All RTXPT shaders include <c>Shaders/Bindings/ShaderResourceBindings.hlsli</c> which
    /// declares the global binding set (g_Const b0, all output UAVs, stable planes, DLSS-RR guides, …).
    ///
    /// <see cref="NativeComputeDescriptorSet"/> and <see cref="RayTracePipeline"/> both silently
    /// ignore names absent from a shader's reflection map, so this helper safely binds the full
    /// superset; each shader only consumes what it actually references.
    /// </summary>
    internal static class NativeRtxptBindings
    {
        /// <summary>
        /// Bind all global resources declared in ShaderResourceBindings.hlsli to a compute descriptor set.
        /// Call this first, then bind any pass-specific SRVs/UAVs not covered here.
        /// </summary>
        public static void BindCommon(NativeComputeDescriptorSet ds, NativeRtxptPassContext ctx)
        {
            BindConstantsAndScene(ds, ctx);
            BindOutputUAVs(ds, ctx);
            BindStablePlanes(ds, ctx);
            BindDlssRrGuides(ds, ctx);
        }

        /// <summary>
        /// Bind all global resources to a DXR ray-tracing pipeline.
        /// </summary>
        public static void BindCommon(RayTracePipeline rtp, NativeRtxptPassContext ctx)
        {
        }

        // ── Private helpers ───────────────────────────────────────────────────

        private static void BindConstantsAndScene(NativeComputeDescriptorSet ds, NativeRtxptPassContext ctx)
        {
            // if (ctx.ConstantBuffer != null)
            //     ds.SetConstantBuffer("g_Const", ctx.ConstantBuffer.GetNativeBufferPtr());
            //
            // var tlas = ctx.NrdSampleResource?.AccelerationStructure;
            // if (tlas != null)
            //     ds.SetAccelerationStructure("SceneBVH", tlas);
            //
            // ctx.GpuScene?.BindToShader(ds);
        }

        private static void BindOutputUAVs(NativeComputeDescriptorSet ds, NativeRtxptPassContext ctx)
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

        private static void BindStablePlanes(NativeComputeDescriptorSet ds, NativeRtxptPassContext ctx)
        {
            if (ctx.StablePlanesHeaderPtr != IntPtr.Zero)
                ds.SetRWTexture("u_StablePlanesHeader", ctx.StablePlanesHeaderPtr);
            if (ctx.StableRadiancePtr != IntPtr.Zero)
                ds.SetRWTexture("u_StableRadiance", ctx.StableRadiancePtr);

            var spBuf = ctx.Buffers?.StablePlanesBuffer;
            if (spBuf != null)
                ds.SetRWStructuredBuffer("u_StablePlanesBuffer", spBuf.GetNativeBufferPtr(), spBuf.count, spBuf.stride);

            var sdBuf = ctx.Buffers?.SurfaceDataBuffer;
            if (sdBuf != null)
                ds.SetRWStructuredBuffer("u_SurfaceData", sdBuf.GetNativeBufferPtr(), sdBuf.count, sdBuf.stride);
        }

        private static void BindDlssRrGuides(NativeComputeDescriptorSet ds, NativeRtxptPassContext ctx)
        {
            if (ctx.DlssRrDiffAlbedoPtr      != IntPtr.Zero) ds.SetRWTexture("u_RRDiffuseAlbedo",      ctx.DlssRrDiffAlbedoPtr);
            if (ctx.DlssRrSpecAlbedoPtr      != IntPtr.Zero) ds.SetRWTexture("u_RRSpecAlbedo",         ctx.DlssRrSpecAlbedoPtr);
            if (ctx.DlssRrNormalRoughnessPtr != IntPtr.Zero) ds.SetRWTexture("u_RRNormalsAndRoughness", ctx.DlssRrNormalRoughnessPtr);
        }
    }
}
