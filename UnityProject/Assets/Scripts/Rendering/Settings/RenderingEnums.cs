using UnityEngine;

namespace PathTracing
{
    public enum ShowMode
    {
        None,
        BaseColor,
        Metalness,
        Normal,
        Roughness,
        NoiseShadow,
        Shadow,
        Diffuse,
        Specular,
        DenoisedDiffuse,
        DenoisedSpecular,
        DirectLight,
        Emissive,
        Out,
        ComposedDiff,
        ComposedSpec,
        Composed,
        Taa,
        Final,
        DLSS_DiffuseAlbedo,
        DLSS_SpecularAlbedo,
        DLSS_SpecularHitDistance,
        DLSS_NormalRoughness,
        DLSS_Output,
        ViewZ,
        Gradient,

        // ── Rtxdi native GBuffer debug views ──────────────────────────────────
        Rtxdi_ViewDepth, // R32_SFloat    – linear view-space depth (greyscale)
        Rtxdi_DiffuseAlbedo, // R32_UINT pack R11G11B10_UFLOAT – diffuse albedo
        Rtxdi_SpecularF0, // R32_UINT pack R8G8B8A8_Gamma_UFLOAT – specular F0 (RGB)
        Rtxdi_Roughness, // R32_UINT pack R8G8B8A8_Gamma_UFLOAT – roughness (A)
        Rtxdi_Normal, // R32_UINT oct32 – shading normal as colour
        Rtxdi_GeoNormal, // R32_UINT oct32 – geometry normal as colour
        Rtxdi_DiffuseLighting, // RTXDI diffuse lighting output
        Rtxdi_SpecularLighting, // RTXDI specular lighting output
        Rtxdi_LocalLightPdf, // LocalLightPdfTexture mip slice (log-scale heat map)
        Rtxdi_EnvironmentPdf, // EnvironmentPdfTexture mip slice (log-scale heat map)
        Rtxdi_DirectLightingRaw, // EnvironmentPdfTexture mip slice (log-scale heat map)
    }

    public enum UpscalerMode : byte // Scaling factor       // Min jitter phases (or just use unclamped Halton2D)
    {
        NATIVE, // 1.0x                 8
        ULTRA_QUALITY, // 1.3x                 14
        QUALITY, // 1.5x                 18
        BALANCED, // 1.7x                 23
        PERFORMANCE, // 2.0x                 32
        ULTRA_PERFORMANCE // 3.0x                 72
    }

    public enum DenoiserType
    {
        DENOISER_REBLUR    = 0,
        DENOISER_RELAX     = 1,
        DENOISER_REFERENCE = 2,
    }

    public enum ForceMaterial
    {
        Null = 0,
        Gypsum = 1,
        Cobalt = 2,
    }

    public enum RtxDiDenoiserType
    {
        DENOISER_MODE_OFF    = 0,
        DENOISER_MODE_REBLUR = 1,
        DENOISER_MODE_RELAX  = 2,
    }

    public enum RESOLUTION
    {
        RESOLUTION_FULL               = 0,
        RESOLUTION_FULL_PROBABILISTIC = 1,
        RESOLUTION_HALF               = 2,
    }

    public enum DirectLightingMode
    {
        None,
        Brdf,
        ReStir
    }

    public enum IndirectLightingMode
    {
        None,
        Brdf,
        ReGirGI,
        ReStirGI,
        ReStirPT
    };

    public enum OnScreen
    {
        // - HDR,
        SHOW_FINAL,
        SHOW_DENOISED_DIFFUSE,
        SHOW_DENOISED_SPECULAR,

        // - LDR,
        SHOW_AMBIENT_OCCLUSION,
        SHOW_SPECULAR_OCCLUSION,
        SHOW_SHADOW,
        SHOW_BASE_COLOR,
        SHOW_NORMAL,
        SHOW_ROUGHNESS,
        SHOW_METALNESS,
        SHOW_MATERIAL_ID,
        SHOW_PSR_THROUGHPUT,
        SHOW_WORLD_UNITS,
        SHOW_INSTANCE_INDEX,
        SHOW_UV,
        SHOW_CURVATURE,
        SHOW_MIP_PRIMARY,
        SHOW_MIP_SPECULAR,
    }

    /// <summary>
    /// Debug / display mode used exclusively by <see cref="NativeRtxdiFeature"/>.
    /// Keeps NativeRtxdi concerns separate from the shared <see cref="ShowMode"/> enum.
    /// </summary>
    public enum NativeRtxdiShowMode
    {
        // ── Main output ────────────────────────────────────────────────────
        /// <summary>Final composited image (DlssOutput when SR is on, HdrColor otherwise).</summary>
        Final,

        // ── Intermediate lighting buffers ──────────────────────────────────
        /// <summary>Raw HDR composited lighting written by CompositingPass (= original HdrColor).</summary>
        HdrColor,
        LdrColor,

        /// <summary>DLSS-SR upscaled output (display resolution).</summary>
        DlssOutput,

        // ── Denoiser inputs / outputs ──────────────────────────────────────
        DiffuseLighting,
        SpecularLighting,
        DenoisedDiffuse,
        DenoisedSpecular,
        DirectLightingRaw,
        IndirectLightingRaw,

        // ── NRD validation overlay ─────────────────────────────────────────
        NrdValidation,

        // ── GBuffer ────────────────────────────────────────────────────────
        ViewDepth, // R32_SFloat linear depth
        DiffuseAlbedo, // R32_UINT  R11G11B10_UFLOAT → albedo
        SpecularF0, // R32_UINT  R8G8B8A8_Gamma   → specular F0
        Roughness, // R32_UINT  R8G8B8A8_Gamma   → roughness (alpha)
        Normal, // R32_UINT  oct32             → shading normal
        GeoNormal, // R32_UINT  oct32             → geometry normal

        // ── Light PDF debug ────────────────────────────────────────────────
        LocalLightPdf,
        EnvironmentPdf,

        // ── Gradient Texture2DArray debug ─────────────────────────────────
        /// <summary>Slice 0 of the gradient Texture2DArray (FilterGradientsPass input/output).</summary>
        GradientArraySlice0,
        /// <summary>Slice 1 of the gradient Texture2DArray.</summary>
        GradientArraySlice1,
    }

    /// <summary>
    /// Debug / display modes for <see cref="NativeRtxptFeature"/>'s
    /// <see cref="NativeRtxptOutputBlitPass"/>.
    /// </summary>
    public enum NativeRtxptShowMode
    {
        // ── Final outputs ──────────────────────────────────────────────────
        /// <summary>DLSS-RR denoised + upscaled output (realtime mode).</summary>
        DlssRrOutput,
        /// <summary>Post-accumulation tone-mapped output (reference mode).</summary>
        ProcessedOutput,
        /// <summary>Raw PT output color before denoising.</summary>
        OutputColor,

        // ── GBuffer ────────────────────────────────────────────────────────
        BaseColor,
        RoughnessMetal,
        SpecNormal,

        // ── Depth / motion ─────────────────────────────────────────────────
        Depth,
        MotionVectors,

        // ── Stable planes ─────────────────────────────────────────────────
        SpecularHitT,
        StableRadiance,

        // ── DLSS-RR guide buffers ──────────────────────────────────────────
        DiffuseAlbedo,
        SpecularAlbedo,
        NormalRoughness,
        SpecMotionVectors,

        // ── Debug ──────────────────────────────────────────────────────────
        ShaderDebugViz,
    }

    /// <summary>
    /// Debug / display modes for <see cref="NativeNrdFeature"/>'s dedicated
    /// <see cref="NativeNrdOutputBlitPass"/>.  Only NRD-pipeline buffers are listed here;
    /// RTXDI-specific GBuffer and PDF views are intentionally absent.
    /// </summary>
    public enum NativeNrdShowMode
    {
        // ── Main output ────────────────────────────────────────────────────
        /// <summary>Tone-mapped final image written by NRDFinalPass.</summary>
        Final,

        // ── GBuffer ────────────────────────────────────────────────────────
        BaseColor,
        Metalness,
        Normal,
        Roughness,
        ViewZ,

        // ── Denoiser inputs ────────────────────────────────────────────────
        NoiseDiffuse,
        NoiseSpecular,
        NoiseShadow,

        // ── Denoiser outputs ───────────────────────────────────────────────
        DenoisedDiffuse,
        DenoisedSpecular,
        DenoisedShadow,

        // ── Intermediate lighting ──────────────────────────────────────────
        DirectLight,
        Emissive,
        ComposedDiff,
        ComposedSpec,
        Composed,

        // ── TAA output ─────────────────────────────────────────────────────
        Taa,

        // ── DLSS/RR guide buffers ──────────────────────────────────────────
        DLSS_DiffuseAlbedo,
        DLSS_SpecularAlbedo,
        DLSS_SpecularHitDistance,
        DLSS_NormalRoughness,
        DLSS_Output,

        // ── SHARC confidence gradient ──────────────────────────────────────
        Gradient,
    }
}
