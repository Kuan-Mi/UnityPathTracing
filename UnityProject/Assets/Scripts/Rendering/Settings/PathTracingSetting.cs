using Rtxdi.DI;
using Rtxdi.GI;
using Rtxdi.PT;
using Rtxdi.ReGIR;
using UnityEngine;
using UnityEngine.Serialization;

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
        Rtxdi_ViewDepth,        // R32_SFloat    – linear view-space depth (greyscale)
        Rtxdi_DiffuseAlbedo,    // R32_UINT pack R11G11B10_UFLOAT – diffuse albedo
        Rtxdi_SpecularF0,       // R32_UINT pack R8G8B8A8_Gamma_UFLOAT – specular F0 (RGB)
        Rtxdi_Roughness,        // R32_UINT pack R8G8B8A8_Gamma_UFLOAT – roughness (A)
        Rtxdi_Normal,           // R32_UINT oct32 – shading normal as colour
        Rtxdi_GeoNormal,        // R32_UINT oct32 – geometry normal as colour
        Rtxdi_DiffuseLighting,  // RTXDI diffuse lighting output
        Rtxdi_SpecularLighting, // RTXDI specular lighting output
        Rtxdi_LocalLightPdf,   // LocalLightPdfTexture mip slice (log-scale heat map)
        Rtxdi_EnvironmentPdf,  // EnvironmentPdfTexture mip slice (log-scale heat map)
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
        ReStirGI
    };


    [System.Serializable]
    public class RtxdiSetting
    {
        [FoldoutHeader("Base Settings")]
        [Range(-10f, 10f)]
        public float exposureEv = 0.0f;

        public float exposure => Mathf.Pow(2, exposureEv);

        public bool         cameraJitter = true;
        public ShowMode     showMode     = ShowMode.Final;
        public bool         showMv;
        /// <summary>Mip level to visualise when showMode is Rtxdi_LocalLightPdf or Rtxdi_EnvironmentPdf.</summary>
        [Range(0, 15)]
        public int          pdfMipLevel      = 0;
        /// <summary>Exposure in stops for the PDF heat-map. +1 = 2× brighter (more sensitive), -1 = 2× darker.</summary>
        [Range(-10f, 10f)]
        public float        pdfExposureStops = 0f;
        public UpscalerMode upscalerMode = UpscalerMode.NATIVE;

        public bool tmpDisableRR;
        public bool skipRightEyeInVR = true;
        public bool enableDenoiser   = true;

        [FoldoutHeader("RTXDI")]
        public UnityRtxdiFeature.RenderSettings lightingSettings = UnityRtxdiFeature.RenderSettings.Default();

        public DirectLightingMode   directLightingMode   = DirectLightingMode.ReStir;
        public IndirectLightingMode indirectLightingMode = IndirectLightingMode.ReStirGI;

        public bool enableGIFinalShading = true;
        public bool enableDIFinalShading = true;

        public bool enableEnv        = true;
        public bool useRasterGBuffer = true;

        public ReGIRDynamicParameters regirDynamicParams = ReGIRDynamicParameters.Default();

        [FoldoutHeader("Compute OR Raster")]
        public bool useComputeForGis = true;

        public bool useComputeForTemporalResampling     = true;
        public bool useComputeForSpatialResampling      = true;
        public bool useComputeForShadeSamples           = true;
        public bool useComputeForShadeSecondarySurfaces = true;
        public bool useComputeForGITemporalResampling   = true;
        public bool useComputeForGISpatialResampling    = true;
        public bool useComputeForGIFinalShading         = true;

        [FoldoutHeader("ReSTIR DI")]
        public ReSTIRDI_ResamplingMode diResamplingMode = ReSTIRDI_ResamplingMode.TemporalAndSpatial;

        public RTXDI_DIInitialSamplingParameters    initialSamplingParams    = ReSTIRDIDefaults.GetDefaultInitialSamplingParams();
        public RTXDI_DITemporalResamplingParameters temporalResamplingParams = ReSTIRDIDefaults.GetDefaultTemporalResamplingParams();
        public RTXDI_DISpatialResamplingParameters  spatialResamplingParams  = ReSTIRDIDefaults.GetDefaultSpatialResamplingParams();
        public RTXDI_ShadingParameters              shadingParams            = ReSTIRDIDefaults.GetDefaultShadingParams();

        [FoldoutHeader("ReSTIR GI")]
        public ReSTIRGI_ResamplingMode giResamplingMode = ReSTIRGI_ResamplingMode.TemporalAndSpatial;

        public RTXDI_GITemporalResamplingParameters giTemporalResamplingParams = ReSTIRGIDefaults.GetDefaultTemporalResamplingParams();
        public RTXDI_GISpatialResamplingParameters  giSpatialResamplingParams  = ReSTIRGIDefaults.GetDefaultSpatialResamplingParams();
        public RTXDI_GIFinalShadingParameters       giFinalShadingParams       = ReSTIRGIDefaults.GetDefaultFinalShadingParams();
        public BRDFPathTracing_Parameters           brdfptParams               = BRDFPathTracing_Parameters.Default();
    }
    [System.Serializable]
    public class NativeRtxdiSetting
    {
        [FoldoutHeader("Base Settings")]
        [Range(-10f, 10f)]
        public float exposureEv = 0.0f;

        public float exposure => Mathf.Pow(2, exposureEv);

        public bool         cameraJitter = true;
        public ShowMode     showMode     = ShowMode.Final;
        public bool         showMv;
        /// <summary>Mip level to visualise when showMode is Rtxdi_LocalLightPdf or Rtxdi_EnvironmentPdf.</summary>
        [Range(0, 15)]
        public int          pdfMipLevel      = 0;
        /// <summary>Exposure in stops for the PDF heat-map. +1 = 2× brighter (more sensitive), -1 = 2× darker.</summary>
        [Range(-10f, 10f)]
        public float        pdfExposureStops = 0f;
        public UpscalerMode upscalerMode = UpscalerMode.NATIVE;

        public bool skipRightEyeInVR = true;
        public bool enableDenoiser   = true;

        [FoldoutHeader("RTXDI")]
        public UnityRtxdiFeature.RenderSettings lightingSettings = UnityRtxdiFeature.RenderSettings.Default();

        public DirectLightingMode   directLightingMode   = DirectLightingMode.ReStir;
        public IndirectLightingMode indirectLightingMode = IndirectLightingMode.ReStirGI;

        public bool enableGIFinalShading = true;
        public bool enableDIFinalShading = true;

        public bool enableEnv        = true;
        public bool useRasterGBuffer = true;

        public ReGIRDynamicParameters regirDynamicParams = ReGIRDynamicParameters.Default();

        [FoldoutHeader("ReSTIR DI")]
        public ReSTIRDI_ResamplingMode diResamplingMode = ReSTIRDI_ResamplingMode.TemporalAndSpatial;

        public RTXDI_DIInitialSamplingParameters    initialSamplingParams    = ReSTIRDIDefaults.GetDefaultInitialSamplingParams();
        public RTXDI_DITemporalResamplingParameters temporalResamplingParams = ReSTIRDIDefaults.GetDefaultTemporalResamplingParams();
        public RTXDI_DISpatialResamplingParameters  spatialResamplingParams  = ReSTIRDIDefaults.GetDefaultSpatialResamplingParams();
        public RTXDI_ShadingParameters              shadingParams            = ReSTIRDIDefaults.GetDefaultShadingParams();

        [FoldoutHeader("ReSTIR GI")]
        public ReSTIRGI_ResamplingMode giResamplingMode = ReSTIRGI_ResamplingMode.TemporalAndSpatial;

        public RTXDI_GITemporalResamplingParameters giTemporalResamplingParams = ReSTIRGIDefaults.GetDefaultTemporalResamplingParams();
        public RTXDI_GISpatialResamplingParameters  giSpatialResamplingParams  = ReSTIRGIDefaults.GetDefaultSpatialResamplingParams();
        public RTXDI_GIFinalShadingParameters       giFinalShadingParams       = ReSTIRGIDefaults.GetDefaultFinalShadingParams();
        public BRDFPathTracing_Parameters           brdfptParams               = BRDFPathTracing_Parameters.Default();

        [FoldoutHeader("ReSTIR PT")]
        public bool                    enableReSTIRPT   = false;
        public ReSTIRPT_ResamplingMode ptResamplingMode = ReSTIRPT_ResamplingMode.TemporalAndSpatial;
    }

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

    [System.Serializable]
    public class PathTracingSetting
    {
        [FoldoutHeader("显示模式")]
        [Range(-10f, 10f)]
        public float exposureEv = 0.0f;

        public bool     cameraJitter = true;
        public ShowMode showMode     = ShowMode.Final;
        public bool     showMv;
        public bool     showValidation;
        public bool     skipRightEyeInVR = true;

        [FoldoutHeader("Base Settings")]
        [Range(0.001f, 10f)]
        public float sunAngularDiameter = 0.533f;

        public float exposure => Mathf.Pow(2, exposureEv);

        public UpscalerMode upscalerMode = UpscalerMode.NATIVE;

        public float mipBias = -0.5f;

        public RESOLUTION   tracingMode = RESOLUTION.RESOLUTION_FULL_PROBABILISTIC;
        public DenoiserType denoiser    = DenoiserType.DENOISER_REBLUR;

        public float emissionIntensity = 1.0f;

        public bool psr                  = true;
        public bool emission             = true;
        public bool usePrevFrame         = true;
        public bool TAA                  = true;
        public bool indirectDiffuse      = true;
        public bool indirectSpecular     = true;
        public bool importanceSampling   = false;
        public bool SHARC                = true;
        public bool specularLobeTrimming = true;
        public bool boost                = false;

        [Range(0.0f, 10.0f)]
        public float boostFactor = 0.6667f;

        public bool SR           = false;
        public bool RR           = true;
        public bool tmpDisableRR = false;

        [Range(0.5f, 1.0f)]
        public float resolutionScale = 1.0f;

        [FoldoutHeader("NRD Common Settings")]
        [Range(0.1f, 10000.0f)]
        public float denoisingRange = 1000;

        [Range(0.0f, 1.0f)]
        public float splitScreen;

        public bool isBaseColorMetalnessAvailable = true;

        [FoldoutHeader("NRD Sigma Settings")]
        [Range(0.0f, 1.0f)]
        public float planeDistanceSensitivity = 0.02f;

        [Range(0, 7)]
        public uint maxStabilizedFrameNum = 5;

        [FoldoutHeader("景深")]
        [Range(0, 100f)]
        public float dofAperture;

        [Range(0.1f, 10f)]
        public float dofFocalDistance = 5;


        [FoldoutHeader("自动曝光")]
        public bool enableAutoExposure = false;

        [Tooltip("Histogram EV range lower bound (log2 luminance)")]
        [Range(-16f, 0f)]
        public float aeEVMin = -10f;

        [Tooltip("Histogram EV range upper bound (log2 luminance)")]
        [Range(0f, 16f)]
        public float aeEVMax = 10f;

        [Tooltip("Fraction of darkest pixels to ignore when computing average EV")]
        [Range(0f, 0.5f)]
        public float aeLowPercent = 0.05f;

        [Tooltip("Fraction of brightest pixels to keep when computing average EV")]
        [Range(0.5f, 1f)]
        public float aeHighPercent = 0.95f;

        [Tooltip("Adaptation speed (EV/s) when scene becomes brighter")]
        [Range(0.01f, 10f)]
        public float aeAdaptationSpeedUp = 2.0f;

        [Tooltip("Adaptation speed (EV/s) when scene becomes darker")]
        [Range(0.01f, 10f)]
        public float aeAdaptationSpeedDown = 1.0f;

        [Tooltip("Artistic EV offset added to computed target exposure")]
        [Range(-5f, 5f)]
        public float aeExposureCompensation = 0f;

        [Tooltip("Minimum allowed output exposure multiplier")]
        [Range(0.001f, 1f)]
        public float aeMinExposure = 0.01f;

        [Tooltip("Maximum allowed output exposure multiplier")]
        [Range(1f, 1000f)]
        public float aeMaxExposure = 100f;

        // [FoldoutHeader("TAA")]
        // [Range(0f, 1f)]
        // public float taa = 1.0f;

        [FoldoutHeader("采样")]
        [Range(1, 4)]
        public uint rpp = 1;

        [Range(1, 4)]
        public uint bounceNum = 1;

        [FoldoutHeader("SHARC")]
        [Range(1, 8)]
        public float sharcDownscale = 4;

        [Range(10, 100)]
        public float sharcSceneScale = 45;

        public bool sharcDebug = false;

        [FoldoutHeader("次表面散射")]
        [Tooltip("SSS 阴影阈值：皮肤在该 NoL 值以下时开始渐入散射（默认 -0.2，允许背透光）")]
        [Range(-1.0f, 0.1f)]
        public float sssMinThreshold = -0.2f;

        [Tooltip("SSS 采样每个 BSDF 的样本数量（默认 4，过高会显著增加渲染时间）")]
        [Range(0, 16)]
        public int sssTransmissionBsdfSampleCount = 4;

        [Tooltip("SSS 采样每个 BSDF 散射样本数量（默认 4，过高会显著增加渲染时间）")]
        [Range(0, 16)]
        public int sssTransmissionPerBsdfScatteringSampleCount = 4;

        [Tooltip("Burley 散射尺度参数（以 SSS_METERS_UNIT=0.01m 为单位，0.4 ≈ 4mm 散射半径）")]
        [Range(0.0f, 100.0f)]
        public float sssScale = 0.4f;

        [Tooltip("Burley 散射各向异性参数（-1 完全向后散射，0 各向同性，1 完全向前散射，默认 0）")]
        [Range(-1.0f, 1.0f)]
        public float sssAnisotropy = 0.0f;

        [Tooltip("Burley 采样最大盘半径（世界单位/m，默认 0.004 = 4mm）")]
        [Range(0.0001f, 0.1f)]
        public float sssMaxSampleRadius = 0.004f;


        [FoldoutHeader("参考路径追踪")]
        public bool useReferencePathTracing;

        [Range(0, 16)]
        public int referenceBounceNum = 4;

        [Range(0.0f, 1.0f)]
        public float split;

        public bool     accumulateReference = true;
        public bool     accumulate          = false;
        public OnScreen gOnScreen;
    }

    /// <summary>
    /// Direct C# port of the C++ <c>Settings</c> struct in NRDSample.cpp.
    /// All fields match the original defaults.  Use this with
    /// <c>CameraFrameState.GetNrdConstants(renderingData, NrdSampleSetting)</c>
    /// which faithfully follows the C++ <c>UpdateConstantBuffer</c> logic.
    /// </summary>
    [System.Serializable]
    public class NrdSampleSetting
    {
        public bool showValidation = false;
        public bool showMV = false;
        public bool mergeBlas      = false;
        // ── Animation / timing (not used by shader, kept for completeness) ──
        // public double motionStartTime        = 0.0;
        // public float  emulateMotionSpeed     = 1.0f;
        // public float  animatedObjectScale    = 1.0f;
        // public float  animationProgress      = 0.0f;
        // public float  animationSpeed         = 0.0f;
        // public int    animatedObjectNum      = 5;
        // public uint   activeAnimation        = 0;
        // public int    motionMode             = 0;
        // public bool   animatedObjects        = false;
        // public bool   animateScene           = false;
        // public bool   animateSun             = false;
        // public bool   nineBrothers           = false;
        // public bool   blink                  = false;
        // public bool   pauseAnimation         = true;
        // public bool   linearMotion           = true;
        // public bool   emissiveObjects        = false;
        // public bool   windowAlignment        = true;

        // ── Camera ───────────────────────────────────────────────────────
        // public float maxFps          = 60.0f;
        // public float camFov          = 90.0f;        // horizontal FOV in degrees
        // public bool  limitFps        = false;
        // public bool  ortho           = false;
        public bool  cameraJitter    = true;
        // public int   mvType          = 1;            // 0 = MV_2D, 1 = MV_25D

        // ── Sun ──────────────────────────────────────────────────────────
        public float sunAngularDiameter = 0.533f;   // degrees

        // ── Rendering ────────────────────────────────────────────────────
        public float exposure               = 80.0f;
        public float roughnessOverride      = 0.0f;
        public float metalnessOverride      = 0.0f;
        public float emissionIntensityLights = 1.0f;
        public float emissionIntensityCubes  = 1.0f;
        public float debug                  = 0.0f;
        public float meterToUnitsMultiplier = 1.0f;
        
        [Range(0.0f, 1.0f)]
        public float separator              = 0.0f;
        public float hitDistScale           = 3.0f;
        public float resolutionScale        = 1.0f;
        // public float sharpness              = 0.15f;

        [Range(1f, 120f)]
        public uint         maxAccumulatedFrameNum     = 31;
        [Range(1f, 20f)]
        public uint         maxFastAccumulatedFrameNum = 7;
        public OnScreen     onScreen             = 0;
        public int          forcedMaterial       = 0;
        public DenoiserType denoiser             = 0;  // 0 = DENOISER_REBLUR
        public int          rpp                  = 1;
        public int          bounceNum            = 1;
        public RESOLUTION   tracingMode          = RESOLUTION.RESOLUTION_HALF;  // RESOLUTION_HALF
        public bool         SHARC                = true;
        public bool         PSR                  = false;
        public bool         normalMap            = true;
        public bool         TAA                  = true;
        public bool         emission             = true;
        public bool         importanceSampling   = true;
        public bool         specularLobeTrimming = true;
        public bool         adaptiveAccumulation = true;
        public bool         usePrevFrame         = true;
        public bool         boost                = false;
        public bool         SR                   = false;
        public bool         RR                   = false;
        public bool         tmpDisableRR         = false;
        public UpscalerMode upscalerMode         = UpscalerMode.NATIVE;
        public bool         confidence           = true;
        public ShowMode     showMode;
        public bool         update;
        public bool         updateTick;
        public float        denoisingRange = 1000f;

        [Range(0.0f, 1.0f)]
        public float nisSharpness = 0.2f;

        public bool skipRightEyeInVR;
    }
}