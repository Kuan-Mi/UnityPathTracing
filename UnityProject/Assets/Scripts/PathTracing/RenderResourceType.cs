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
        MV,
        NormalRoughness,
        Viewz,
        BasecolorMetalness,
        GeoNormal,

        // ── NRD standard noisy inputs ───────────────────────────────────────────
        DiffRadianceHitdist,
        SpecRadianceHitdist,
        Penumbra,

        // ── NRD standard outputs ────────────────────────────────────────────────
        OutDiffRadianceHitdist,
        OutSpecRadianceHitdist,
        OutShadowTranslucency,
        Validation,

        // ── NRI-interop resources (DLSS / composition) ──────────────────────────
        Composed,
        DirectLighting,
        DlssOutput,
        RrGuideDiffAlbedo,
        RrGuideSpecAlbedo,
        RrGuideSpecHitDistance,
        RrGuideNormalRoughness,

        // ── Cross-frame RTHandle-only resources ─────────────────────────────────
        TaaHistory,
        TaaHistoryPrev,
        PsrThroughput,

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
    }
}