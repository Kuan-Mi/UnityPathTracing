using NativeRender;

namespace PathTracing
{
    /// <summary>
    /// Shared binding helper for NativeRtxdi DI/GI compute passes.
    ///
    /// Every DI/GI pass shipped under <c>Assets/RTXDI/Shaders/LightingPasses/</c> includes
    /// <c>RtxdiApplicationBridge/RAB_Buffers.hlsli</c>, which declares a fixed common binding
    /// set (g_Const, g_PerPassConstants, GBuffer SRVs, light/RIS resources, screen UAVs, ...).
    /// <see cref="NativeComputeDescriptorSet"/> silently drops names absent from a shader's
    /// reflection map, so this helper safely binds the full superset; each individual shader
    /// only consumes what it actually references.
    /// </summary>
    internal static class NativeRtxdiBindings
    {
        /// <summary>
        /// Bind every resource declared in RAB_Buffers.hlsli that has a value in the context.
        /// Caller still must bind pass-specific UAVs (e.g. u_DiffuseLighting / u_GIReservoirs) afterwards if any.
        /// </summary>
        public static void BindRabCommon(NativeComputeDescriptorSet ds, NativeRtxdiPassContext ctx)
        {
            // ---------- Constant buffers ----------
            // Most lighting passes use ResamplingConstants for g_Const; prep passes that bind
            // GlobalConstants instead pass their own buffer via ctx.ConstantBuffer.
            if (ctx.ResamplingConstantBuffer != null)
                ds.SetConstantBuffer("g_Const", ctx.ResamplingConstantBuffer.GetNativeBufferPtr());

            if (ctx.PerPassConstantBuffer != null)
                ds.SetConstantBuffer("g_PerPassConstants", ctx.PerPassConstantBuffer.GetNativeBufferPtr());

            // ---------- TLAS ----------
            var tlas = ctx.RtxdiGpuScene?.AccelerationStructure;
            if (tlas != null)
                ds.SetAccelerationStructure("SceneBVH", tlas);
            if (tlas != null)
                ds.SetAccelerationStructure("PrevSceneBVH", tlas);

            // ---------- Scene buffers + bindless arrays ----------
            ctx.RtxdiGpuScene?.BindToShader(ds);

            // ---------- GBuffer SRVs (current frame, t0..t4) ----------
            if (ctx.ViewDepthPtr != System.IntPtr.Zero) ds.SetTexture("t_GBufferDepth", ctx.ViewDepthPtr);
            if (ctx.NormalsPtr != System.IntPtr.Zero) ds.SetTexture("t_GBufferNormals", ctx.NormalsPtr);
            if (ctx.GeoNormalsPtr != System.IntPtr.Zero) ds.SetTexture("t_GBufferGeoNormals", ctx.GeoNormalsPtr);
            if (ctx.DiffuseAlbedoPtr != System.IntPtr.Zero) ds.SetTexture("t_GBufferDiffuseAlbedo", ctx.DiffuseAlbedoPtr);
            if (ctx.SpecularRoughPtr != System.IntPtr.Zero) ds.SetTexture("t_GBufferSpecularRough", ctx.SpecularRoughPtr);

            // ---------- GBuffer SRVs (previous frame, t5..t9) ----------
            if (ctx.PrevViewDepthPtr != System.IntPtr.Zero) ds.SetTexture("t_PrevGBufferDepth", ctx.PrevViewDepthPtr);
            if (ctx.PrevNormalsPtr != System.IntPtr.Zero) ds.SetTexture("t_PrevGBufferNormals", ctx.PrevNormalsPtr);
            if (ctx.PrevGeoNormalsPtr != System.IntPtr.Zero) ds.SetTexture("t_PrevGBufferGeoNormals", ctx.PrevGeoNormalsPtr);
            if (ctx.PrevDiffuseAlbedoPtr != System.IntPtr.Zero) ds.SetTexture("t_PrevGBufferDiffuseAlbedo", ctx.PrevDiffuseAlbedoPtr);
            if (ctx.PrevSpecularRoughPtr != System.IntPtr.Zero) ds.SetTexture("t_PrevGBufferSpecularRough", ctx.PrevSpecularRoughPtr);

            // ---------- Per-frame aux SRVs (t10..t12) ----------
            if (ctx.PrevRestirLuminancePtr != System.IntPtr.Zero) ds.SetTexture("t_PrevRestirLuminance", ctx.PrevRestirLuminancePtr);
            if (ctx.MotionVectorsPtr != System.IntPtr.Zero) ds.SetTexture("t_MotionVectors", ctx.MotionVectorsPtr);
            if (ctx.DenoiserNormalRoughnessPtr != System.IntPtr.Zero) ds.SetTexture("t_DenoiserNormalRoughness", ctx.DenoiserNormalRoughnessPtr);

            // ---------- Light / RTXDI SRVs (t20..t25) ----------
            var rtx = ctx.Resources;
            if (rtx?.LightDataBuffer != null)
                ds.SetStructuredBuffer("t_LightDataBuffer",
                    rtx.LightDataBuffer.GetNativeBufferPtr(),
                    rtx.LightDataBuffer.count, rtx.LightDataBuffer.stride);

            if (rtx?.NeighborOffsetsBuffer != null)
                // Buffer<float2> requires a typed SRV: DXGI_FORMAT_R32G32_FLOAT
                ds.SetTypedBuffer("t_NeighborOffsets", rtx.NeighborOffsetsBuffer.GetNativeBufferPtr(), rtx.NeighborOffsetsBuffer.count, (uint)Nri.DXGI_FORMAT.DXGI_FORMAT_R32G32_FLOAT);

            if (rtx?.LightIndexMappingBuffer != null)
                ds.SetStructuredBuffer("t_LightIndexMappingBuffer",
                    rtx.LightIndexMappingBuffer.GetNativeBufferPtr(),
                    rtx.LightIndexMappingBuffer.count, rtx.LightIndexMappingBuffer.stride);

            // EnvironmentPdfTexture: prefer the IntPtr explicitly set by the feature; fall back
            // to the RTHandle owned by NativeRtxdiResources.
            if (ctx.EnvironmentPdfTexturePtr != System.IntPtr.Zero)
                ds.SetTexture("t_EnvironmentPdfTexture", ctx.EnvironmentPdfTexturePtr);
            else if (rtx?.EnvironmentPdfTexture?.rt != null)
                ds.SetTexture("t_EnvironmentPdfTexture", rtx.EnvironmentPdfTexture.rt.GetNativeTexturePtr());

            if (ctx.LocalLightPdfTexturePtr != System.IntPtr.Zero)
                ds.SetTexture("t_LocalLightPdfTexture", ctx.LocalLightPdfTexturePtr);
            else if (rtx?.LocalLightPdfTexture?.rt != null)
                ds.SetTexture("t_LocalLightPdfTexture", rtx.LocalLightPdfTexture.rt.GetNativeTexturePtr());

            if (rtx?.GeometryInstanceToLight != null)
                ds.SetStructuredBuffer("t_GeometryInstanceToLight",
                    rtx.GeometryInstanceToLight.GetNativeBufferPtr(),
                    rtx.GeometryInstanceToLight.count, rtx.GeometryInstanceToLight.stride);

            // ---------- Reservoir / RIS UAVs (u0, u10, u11, u6) ----------
            if (rtx?.LightReservoirBuffer != null)
                ds.SetRWStructuredBuffer("u_LightReservoirs",
                    rtx.LightReservoirBuffer.GetNativeBufferPtr(),
                    rtx.LightReservoirBuffer.count, rtx.LightReservoirBuffer.stride);
            if (rtx?.RisBuffer != null)
                // RWBuffer<uint2> requires a typed UAV: DXGI_FORMAT_R32G32_UINT
                ds.SetRWTypedBuffer("u_RisBuffer", rtx.RisBuffer.GetNativeBufferPtr(), rtx.RisBuffer.count, (uint)Nri.DXGI_FORMAT.DXGI_FORMAT_R32G32_UINT);
            if (rtx?.RisLightDataBuffer != null)
                // RWBuffer<uint4> requires a typed UAV: DXGI_FORMAT_R32G32B32A32_UINT
                ds.SetRWTypedBuffer("u_RisLightDataBuffer", rtx.RisLightDataBuffer.GetNativeBufferPtr(), rtx.RisLightDataBuffer.count, (uint)Nri.DXGI_FORMAT.DXGI_FORMAT_R32G32B32A32_UINT);
            if (rtx?.GIReservoirBuffer != null)
                ds.SetRWStructuredBuffer("u_GIReservoirs",
                    rtx.GIReservoirBuffer.GetNativeBufferPtr(),
                    rtx.GIReservoirBuffer.count, rtx.GIReservoirBuffer.stride);
            if (rtx?.SecondaryGBuffer != null)
                ds.SetRWStructuredBuffer("u_SecondaryGBuffer",
                    rtx.SecondaryGBuffer.GetNativeBufferPtr(),
                    rtx.SecondaryGBuffer.count, rtx.SecondaryGBuffer.stride);

            if (ctx.RayCountBuffer != null)
                ds.SetRWBuffer("u_RayCountBuffer", ctx.RayCountBuffer.GetNativeBufferPtr());

            // ---------- Screen UAVs (u1..u5, u17) ----------
            if (ctx.DiffuseLightingPtr != System.IntPtr.Zero) ds.SetRWTexture("u_DiffuseLighting", ctx.DiffuseLightingPtr);
            if (ctx.SpecularLightingPtr != System.IntPtr.Zero) ds.SetRWTexture("u_SpecularLighting", ctx.SpecularLightingPtr);
            if (ctx.TemporalSamplePositionsPtr != System.IntPtr.Zero) ds.SetRWTexture("u_TemporalSamplePositions", ctx.TemporalSamplePositionsPtr);
            if (ctx.GradientsPtr != System.IntPtr.Zero) ds.SetRWTexture("u_Gradients", ctx.GradientsPtr);
            if (ctx.RestirLuminancePtr != System.IntPtr.Zero) ds.SetRWTexture("u_RestirLuminance", ctx.RestirLuminancePtr);
            if (ctx.DirectLightingRawPtr != System.IntPtr.Zero) ds.SetRWTexture("u_DirectLightingRaw", ctx.DirectLightingRawPtr);
            if (ctx.IndirectLightingRawPtr != System.IntPtr.Zero) ds.SetRWTexture("u_IndirectLightingRaw", ctx.IndirectLightingRawPtr);
        }
    }
}