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
        IN_MV,
        IN_NORMAL_ROUGHNESS,
        IN_VIEWZ,
        IN_BASECOLOR_METALNESS,
        IN_DIFF_CONFIDENCE,
        IN_SPEC_CONFIDENCE,
        IN_DISOCCLUSION_THRESHOLD_MIX,

        // ── NRD standard noisy inputs ───────────────────────────────────────────
        IN_DIFF_RADIANCE_HITDIST,
        IN_SPEC_RADIANCE_HITDIST,
        IN_DIFF_HITDIST,
        IN_SPEC_HITDIST,
        IN_DIFF_DIRECTION_HITDIST,
        IN_DIFF_SH0,
        IN_DIFF_SH1,
        IN_SPEC_SH0,
        IN_SPEC_SH1,
        IN_PENUMBRA,
        IN_TRANSLUCENCY,
        IN_SIGNAL,

        // ── NRD standard outputs ────────────────────────────────────────────────
        OUT_DIFF_RADIANCE_HITDIST,
        OUT_SPEC_RADIANCE_HITDIST,
        OUT_DIFF_SH0,
        OUT_DIFF_SH1,
        OUT_SPEC_SH0,
        OUT_SPEC_SH1,
        OUT_DIFF_HITDIST,
        OUT_SPEC_HITDIST,
        OUT_DIFF_DIRECTION_HITDIST,
        OUT_SHADOW_TRANSLUCENCY,
        OUT_SIGNAL,
        OUT_VALIDATION,

        // ── NRI-interop resources (DLSS / composition) ──────────────────────────
        Composed,
        DlssOutput,
        RRGuide_DiffAlbedo,
        RRGuide_SpecAlbedo,
        RRGuide_SpecHitDistance,
        RRGuide_Normal_Roughness,

        // ── Cross-frame RTHandle-only resources ─────────────────────────────────
        TaaHistory,
        TaaHistoryPrev,
        PsrThroughput,

        // Previous-frame GBuffer for RTXDI temporal reuse
        Prev_ViewZ,
        Prev_NormalRoughness,
        Prev_BaseColorMetalness,
    }
}
