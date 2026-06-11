using UnityEngine;

namespace PathTracing
{
    /// <summary>
    /// Mirrors SampleUIData::NEEType / LightingControlData::ImportanceSamplingType.
    /// Numeric values are part of the shader ABI.
    /// </summary>
    public enum NativeRtxptNeeType
    {
        Uniform = 0,
        Power   = 1,
        NEEAT   = 2,
    }

    /// <summary>
    /// Mirrors RTXPT ToneMapping_cb.h ToneMapperOperator. Numeric values are the shader ABI.
    /// </summary>
    public enum NativeRtxptToneMapOperator
    {
        Linear           = 0,
        Reinhard         = 1,
        ReinhardModified = 2,
        HejiHableAlu     = 3,
        HableUc2         = 4,
        Aces             = 5,
    }

    /// <summary>
    /// Inspector-facing settings for <see cref="NativeRtxptFeature"/>.
    /// Mirrors <c>SampleUIData</c> from RTXPT/Rtxpt/SampleUI.h, adapted for Unity.
    /// Denoising is handled exclusively by DLSS Ray Reconstruction (DLSS-RR).
    /// </summary>
    [System.Serializable]
    public class NativeRtxptSetting
    {
        // ── Mode ─────────────────────────────────────────────────────────────
        /// <summary>false = Realtime (stable planes + DLSS-RR), true = Reference (accumulation).</summary>
        public bool realtimeMode = true;

        // ── AA / Upscaler ─────────────────────────────────────────────────────
        // realtimeAA: 0=None, 1=TAA, 2=DLSS-SR, 3=DLSS-RR
        public int          realtimeAA   = 3;
        // C++ default: SampleUIData::DLSSModeDefault = eBalanced.
        public UpscalerMode upscalerMode = UpscalerMode.BALANCED;

        // ── Camera ────────────────────────────────────────────────────────────
        public bool  cameraJitter        = true;
        public float cameraAperture      = 0.0f;
        public float cameraFocalDistance = 10000.0f;
        public float cameraMoveSpeed     = 1.0f;

        // ── Path tracer quality ───────────────────────────────────────────────
        /// <summary>Samples per pixel per frame (realtime). SampleUIData::RealtimeSamplesPerPixel.</summary>
        [Range(1, 8)]
        public int realtimeSamplesPerPixel = 1;

        /// <summary>Max total bounces. SampleUIData::BounceCount (C++ default 20).</summary>
        [Range(1, 48)]
        public int bounceCount = 20;

        /// <summary>Max diffuse-only bounces. SampleUIData::DiffuseBounceCount (C++ default 2).</summary>
        [Range(1, 8)]
        public int diffuseBounceCount = 2;

        /// <summary>Texture LOD bias applied during tracing. SampleUIData::TexLODBias (C++ default -1.0).</summary>
        [Range(-4f, 4f)]
        public float texLODBias = -1.0f;

        // ── Reference mode accumulation ───────────────────────────────────────
        /// <summary>Reference mode: target sample count. SampleUIData::AccumulationTarget.</summary>
        [Range(1, 65536)]
        public int accumulationTarget = 4096;

        public bool accumulationPreWarmRealtimeCaches = true;
        public bool accumulationAA                   = true;

        // ── NEE / Lighting ────────────────────────────────────────────────────
        public bool  useNEE              = true;
        // C++ default: CommandLineOptions::NEEType = 2 (NEE-AT).
        public NativeRtxptNeeType neeType = NativeRtxptNeeType.NEEAT;
        [Range(1, 16)]
        public int   neeCandidateSamples = 5;
        // C++ default: SampleUIData::NEEFullSamples = 1 (each full sample casts a shadow ray).
        [Range(1, 4)]
        public int   neeFullSamples      = 1;
        /// <summary>0=Full MIS always; 1=Full in ref / approx in realtime; 2=Approx always.</summary>
        public int   neeMisType          = 1;
        public float neeatGlobalTemporalFeedbackWeight = 0.75f;
        public float neeatLocalToGlobalSampleRatio     = 0.65f;
        public float neeatDistantVsLocalImportance     = 1.0f;
        // Advanced NEE-AT baker knobs (mirror LightsBaker.cpp m_advSetting_*).
        public float neeatScreenSpaceVsWorldSpaceThreshold = 0.3f;
        public float neeatDepthDisocclusionThreshold       = 1.5f;
        public float neeatReservoirHistoryDropoff          = 0.005f;
        public bool  neeatEnableMotionReprojection         = true;
        // Original RTXPT enables both boosts by default (LightsBaker.h:245-249):
        //   IntensityDelta = m_importanceBoost_IntensityDelta(true) ? 64.0 : 0
        //   FrustumMul     = m_importanceBoost_Frustum(true)        ? 8.0  : 0
        //   FrustumFadeDistance = 5.0 (ungated)
        public float neeatImportanceBoostIntensityDelta    = 64.0f;
        public float neeatImportanceBoostFrustumMul        = 8.0f;
        public float neeatImportanceBoostFrustumFadeDistance = 5.0f;
        public float neeatSceneAverageContentsDistance     = 10.0f;
        /// <summary>
        /// Sample the baked env cube for distant-light radiance (LightsBaker.h:241
        /// m_advSetting_SampleBakedEnvironment). Compile-time macro
        /// NEE_AT_SAMPLE_BAKED_ENVIRONMENT — applied via the shader-macro sync (reimport).
        /// </summary>
        public bool  neeatSampleBakedEnvironment           = true;

        public bool useReSTIRDI = false;
        public bool useReSTIRGI = false;

        // ── Material / shading features ───────────────────────────────────────
        public bool normalMap            = true;
        public bool specularLobeTrimming = true;
        public bool importanceSampling   = true;
        public bool enableLDSamplerForBSDF = true;
        public bool indirectDiffuse      = true;
        public bool indirectSpecular     = true;
        public bool enableRussianRoulette = true;
        public bool emission             = true;
        [Range(0, 2)]
        public int  nestedDielectricsQuality = 1; // 0=off, 1=fast, 2=quality
        public bool usePrevFrame         = true;
        public bool useFp16Types         = true;

        // ── Ray-tracing pipeline (compile-time macros, applied via shader-macro sync) ──
        // C++ defaults to the NVAPI HitObject extension (SampleUI.h:221-225); the Unity plugin has
        // no NVAPI integration, so the SM 6.9 DX HitObject path is used instead (USE_NVAPI_* stay 0).
        /// <summary>USE_DX_HIT_OBJECT_EXTENSION — DX1.x HitObject API (SER) on SM 6.9.</summary>
        public bool dxHitObjectExtension  = true;
        /// <summary>USE_DX_MAYBE_REORDER_THREADS — MaybeReorderThread() with the DX HitObject path.</summary>
        public bool dxMaybeReorderThreads = true;

        // ── PSR / SHARC / Stable Planes ───────────────────────────────────────
        public bool  allowPrimarySurfaceReplacement          = true;
        public bool  stablePlanesSuppressPrimaryIndirectSpecular = true;
        [Range(0f, 1f)]
        public float stablePlanesSuppressPrimaryIndirectSpecularK = 0.6f;
        [Range(1, 3)]
        public int   stablePlanesActiveCount     = PathTracerConfig.cStablePlaneCount;
        [Range(1, 9)]
        public int   stablePlanesMaxVertexDepth  = 9;
        [Range(0f, 1f)]
        public float stablePlanesSplitStopThreshold      = 0.95f;
        [Range(0f, 1f)]
        public float stablePlanesAntiAliasingFallthrough = 0.6f;

        // ── Firefly filter ────────────────────────────────────────────────────
        public bool  realtimeFireflyFilterEnabled       = true;
        [Range(0f, 2f)]
        public float realtimeFireflyFilterThreshold     = 0.10f;
        public bool  referenceFireflyFilterEnabled      = true;
        [Range(0f, 20f)]
        public float referenceFireflyFilterThreshold    = 5.0f;

        // ── Denoiser / DLSS-RR ────────────────────────────────────────────────
        [Range(0f, 32f)]
        public float denoiserRadianceClampK  = 8.0f;
        [Range(0f, 8192f)]
        public float dlssrrBrightnessClampK  = 4096.0f;
        public float dlssrrMicroJitter       = 0.1f;
        public bool  disableReSTIRsWithDLSSRR = true;
        public bool  tmpDisableDlssRR        = false;

        // ── Bloom ─────────────────────────────────────────────────────────────
        public bool  enableBloom     = true;
        [Range(0f, 32f)]
        public float bloomRadius     = 8.0f;
        [Range(0f, 0.1f)]
        public float bloomIntensity  = 0.004f;

        // ── Environment map ───────────────────────────────────────────────────
        [Range(0, 5)]
        public int       environmentMapDiffuseSampleMIPLevel = 2;
        public bool      environmentMapEnabled   = true;
        public float     environmentMapIntensity = 1.0f;
        public Color     environmentMapTint      = Color.white;
        /// <summary>Equirectangular HDR environment map. Bound as t_EnvironmentMap (t10).</summary>
        public Texture environmentMap          = null;

        // ── Tone mapping ──────────────────────────────────────────────────────
        // Mirrors RTXPT ToneMappingParameters (ToneMapper/ToneMappingPasses.h).
        public bool enableToneMapping = true;

        /// <summary>Tone-map operator. Matches the RTXPT reference capture (HableUc2).</summary>
        public NativeRtxptToneMapOperator toneMapOperator = NativeRtxptToneMapOperator.HableUc2;

        /// <summary>
        /// Enable histogram-free auto-exposure (geometric-mean luminance). Off by default, matching
        /// RTXPT's C++ default (scene JSON can override it there) and the reference capture, which
        /// uses photographic manual exposure (filmSpeed/fNumber/shutter) with exposureCompensation.
        /// </summary>
        public bool autoExposure = false;

        /// <summary>
        /// Exposure compensation in stops (EV). Set to 1.5 to reproduce the RTXPT reference
        /// capture's manual color-transform scale k = 2^1.5 ≈ 2.828 (with neutral film/aperture:
        /// 2^EC · (filmSpeed/100)/(shutter·fNumber²) = 2^1.5 · 1). RTXPT's Sample.cpp default is 2.0.
        /// </summary>
        [Range(-8f, 8f)]
        public float exposureCompensation = 1.5f;

        /// <summary>Manual exposure value (EV) used to derive shutter/aperture when autoExposure is off.</summary>
        public float exposureValue = 0.0f;

        /// <summary>Auto-exposure clamp range, in EV.</summary>
        public float exposureValueMin = -16.0f;
        public float exposureValueMax = 16.0f;

        // Photographic controls (manual exposure path).
        public float filmSpeed = 100.0f;
        public float fNumber   = 1.0f;
        public float shutter   = 1.0f;

        // Color grading / operator parameters.
        public bool  toneMapWhiteBalance   = false;
        public float toneMapWhitePoint     = 6500.0f;
        public float toneMapWhiteMaxLuminance = 1.0f;
        public float toneMapWhiteScale     = 5.1f;
        public bool  toneMapClamped        = true;

        // ── Debug ─────────────────────────────────────────────────────────────
        public bool showValidation    = false;
        public bool enableShaderDebug = true;
        public bool dbgFreezeRealtimeNoiseSeed   = false;
        public bool dbgDiscardNonNEELighting     = false;
        public bool dbgDiscardNEELighting        = false;
        public bool dbgDisablePostProcessFilters = false;
        /// <summary>RTXPT_DISABLE_SER_TERMINATION_HINT (SampleUI.h:299). Compile-time macro.</summary>
        public bool dbgDisableSERTerminationHint = false;

        /// <summary>Which buffer to display in the output blit pass.</summary>
        public NativeRtxptShowMode showMode = NativeRtxptShowMode.DlssRrOutput;

        /// <summary>
        /// Selects which surface attribute to visualise via the path tracer's
        /// built-in debug-viz channel (ShaderDebugViz texture).
        /// Mirrors HLSL <c>DebugViewType</c>.
        /// When non-Disabled the output blit pass will show the debug texture.
        /// </summary>
        public RtxptDebugViewType debugViewType = RtxptDebugViewType.Disabled;

        /// <summary>
        /// Which stable-plane index to inspect when <see cref="debugViewType"/>
        /// is a StablePlane_* mode. -1 means all planes combined.
        /// </summary>
        [Range(-1, 2)]
        public int debugViewStablePlaneIndex = -1;

        // ── Misc ──────────────────────────────────────────────────────────────
        public bool skipRightEyeInVR = true;
    }
}
