namespace PathTracing
{
    /// <summary>
    /// Unified enum covering all render textures managed by PathTracingResourcePool.
    ///
    /// Groups:
    ///   NRD I/O   – standard NRD denoiser inputs/outputs (passed to the C++ denoiser via NRI pointers)
    ///   NRI-interop – non-NRD textures that still require a native NRI pointer (DLSS/RR)
    ///   RTHandle-only – cross-frame buffers that only need a Unity RTHandle (TAA history, prev GBuffer, …)
    /// </summary>
    public enum RenderResourceType
    {
        // ── NRD standard non-noisy inputs ──────────────────────────────────────
        Viewz,
        MV,
        NormalRoughness,
        PsrThroughput,
        BaseColorMetalness,
        DirectLighting,
        DirectEmission,
        Shadow,
        Diff,
        Spec,

        GeoNormal,

        // ── NRD standard noisy inputs ───────────────────────────────────────────
        Unfiltered_Penumbra,
        Unfiltered_Diff,
        Unfiltered_Spec,
        Unfiltered_Translucency,

        Validation,
        Composed,

        Gradient_StoredPing, // gIn_PrevGradient (even) / gOut_CurrGradient (odd)
        Gradient_StoredPong, // gOut_CurrGradient (even) / gIn_PrevGradient (odd)
        Gradient_Ping, // gOut_Gradient (always)
        Gradient_Pong, // reserved for downstream ConfidenceBlur

        // History
        ComposedDiff,
        ComposedSpecViewZ,
        TaaHistory,
        TaaHistoryPrev,

        DlssOutput,
        PreFinal,
        
        Final,
        
        // RR guides
        RrGuideDiffAlbedo,
        RrGuideSpecAlbedo,
        RrGuideSpecHitDistance,
        RrGuideNormalRoughness,


        // Previous-frame GBuffer for RTXDI temporal reuse
        PrevViewZ,
        PrevNormalRoughness,
        PrevBaseColorMetalness,
        PrevGeoNormal,


        // rtxdi
        RtxdiViewDepth,
        RtxdiDiffuseAlbedo,
        RtxdiSpecularRough,
        RtxdiNormals,
        RtxdiGeoNormals,
        RtxdiEmissive,
        RtxdiMotionVectors,
        RtxdiPrevViewDepth,
        RtxdiPrevDiffuseAlbedo,
        RtxdiPrevSpecularRough,
        RtxdiPrevNormals,
        RtxdiPrevGeoNormals,

        // RTXDI lighting output UAVs (screen-sized)
        RtxdiDiffuseLighting,       // u_DiffuseLighting  RGBA16_SFloat
        RtxdiSpecularLighting,      // u_SpecularLighting RGBA16_SFloat
        RtxdiTemporalSamplePos,     // u_TemporalSamplePositions RG16_SInt
        RtxdiRestirLuminance,       // u_RestirLuminance  RG16_SFloat (ping)
        RtxdiPrevRestirLuminance,   // t_PrevRestirLuminance RG16_SFloat (pong)
        RtxdiDirectLightingRaw,       // u_DirectLightingRaw      RGBA16_SFloat
        RtxdiIndirectLightingRaw,    // u_IndirectLightingRaw    RGBA16_SFloat
        RtxdiDenoiserNormalRoughness, // t_DenoiserNormalRoughness RGBA8_UNorm
    }
}