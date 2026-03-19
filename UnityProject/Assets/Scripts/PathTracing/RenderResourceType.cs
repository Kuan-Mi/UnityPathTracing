namespace PathTracing
{
    /// <summary>
    /// Resources managed by PathTracingResourcePool that are NOT part of the NRD denoiser API.
    /// Split into two semantic groups based on lifetime/interop requirements.
    /// </summary>
    public enum RenderResourceType
    {
        // ──────────────────────────────────────────────────────────────────────────
        // Resources that require an NRI pointer (native D3D12 interop via DLSS/RR)
        // ──────────────────────────────────────────────────────────────────────────
        Composed,
        DlssOutput,
        RRGuide_DiffAlbedo,
        RRGuide_SpecAlbedo,
        RRGuide_SpecHitDistance,
        RRGuide_Normal_Roughness,

        // ──────────────────────────────────────────────────────────────────────────
        // Cross-frame RTHandle-only resources (no NRI pointer needed)
        // ──────────────────────────────────────────────────────────────────────────
        TaaHistory,
        TaaHistoryPrev,
        PsrThroughput,

        // Previous-frame GBuffer for RTXDI temporal reuse
        Prev_ViewZ,
        Prev_NormalRoughness,
        Prev_BaseColorMetalness,
    }
}
