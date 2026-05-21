using UnityEngine;

namespace PathTracing
{
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
        public UpscalerMode upscalerMode = UpscalerMode.QUALITY;

        // ── Camera ────────────────────────────────────────────────────────────
        public bool  cameraJitter        = true;
        public float cameraAperture      = 0.0f;
        public float cameraFocalDistance = 10000.0f;
        public float cameraMoveSpeed     = 1.0f;

        // ── Path tracer quality ───────────────────────────────────────────────
        /// <summary>Samples per pixel per frame (realtime). SampleUIData::RealtimeSamplesPerPixel.</summary>
        [Range(1, 8)]
        public int realtimeSamplesPerPixel = 1;

        /// <summary>Max total bounces. SampleUIData::BounceCount.</summary>
        [Range(1, 48)]
        public int bounceCount = 24;

        /// <summary>Max diffuse-only bounces. SampleUIData::DiffuseBounceCount.</summary>
        [Range(1, 8)]
        public int diffuseBounceCount = 3;

        /// <summary>Texture LOD bias applied during tracing. SampleUIData::TexLODBias.</summary>
        [Range(-4f, 4f)]
        public float texLODBias = -1.5f;

        // ── Reference mode accumulation ───────────────────────────────────────
        /// <summary>Reference mode: target sample count. SampleUIData::AccumulationTarget.</summary>
        [Range(1, 65536)]
        public int accumulationTarget = 4096;

        public bool accumulationPreWarmRealtimeCaches = true;
        public bool accumulationAA                   = true;

        // ── NEE / Lighting ────────────────────────────────────────────────────
        public bool  useNEE              = true;
        /// <summary>0=Uniform, 1=Power, 2=NEE-AT. SampleUIData::NEEType.</summary>
        public int   neeType             = 1;
        [Range(1, 16)]
        public int   neeCandidateSamples = 5;
        [Range(1, 4)]
        public int   neeFullSamples      = 2;
        /// <summary>0=Full MIS always; 1=Full in ref / approx in realtime; 2=Approx always.</summary>
        public int   neeMisType          = 1;
        public float neeatGlobalTemporalFeedbackWeight = 0.75f;
        public float neeatLocalToGlobalSampleRatio     = 0.65f;
        public float neeatDistantVsLocalImportance     = 1.0f;

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

        // ── PSR / SHARC / Stable Planes ───────────────────────────────────────
        public bool  allowPrimarySurfaceReplacement          = true;
        public bool  stablePlanesSuppressPrimaryIndirectSpecular = true;
        [Range(0f, 1f)]
        public float stablePlanesSuppressPrimaryIndirectSpecularK = 0.6f;
        [Range(1, 3)]
        public int   stablePlanesActiveCount     = 3; // cStablePlaneCount
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
        public Texture2D environmentMap          = null;
        /// <summary>Precomputed environment CDF LUT for importance sampling. Bound as t_EnvLookupMap (t18).</summary>
        public Texture2D environmentLookupMap    = null;

        // ── Tone mapping ──────────────────────────────────────────────────────
        public bool enableToneMapping = true;

        // ── Debug ─────────────────────────────────────────────────────────────
        public bool showValidation    = false;
        public bool enableShaderDebug = true;
        public bool dbgFreezeRealtimeNoiseSeed   = false;
        public bool dbgDiscardNonNEELighting     = false;
        public bool dbgDiscardNEELighting        = false;
        public bool dbgDisablePostProcessFilters = false;

        /// <summary>Which buffer to display in the output blit pass.</summary>
        public NativeRtxptShowMode showMode = NativeRtxptShowMode.DlssRrOutput;

        // ── Misc ──────────────────────────────────────────────────────────────
        public bool skipRightEyeInVR = true;
    }
}
