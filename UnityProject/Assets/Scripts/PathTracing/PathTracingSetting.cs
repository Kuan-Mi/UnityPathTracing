using Rtxdi.DI;
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
        DENOISER_REBLUR = 0,
        DENOISER_RELAX = 1,
        DENOISER_REFERENCE = 2,
    }

    public enum RESOLUTION
    {
        RESOLUTION_FULL = 0,
        RESOLUTION_FULL_PROBABILISTIC = 1,
        RESOLUTION_HALF = 2,
    }


    [System.Serializable]
    public class PathTracingSetting
    {
        [FoldoutHeader("显示模式")]
        public ShowMode showMode = ShowMode.Final;

        public bool showMv;
        public bool showValidation;

        [FoldoutHeader("Base Settings")]
        [Range(0.001f, 10f)]
        public float sunAngularDiameter = 0.533f;

        [Range(0.1f, 100f)]
        public float exposure = 1.0f;

        public UpscalerMode upscalerMode = UpscalerMode.NATIVE;

        public float mipBias = -0.5f;

        public RESOLUTION tracingMode = RESOLUTION.RESOLUTION_FULL_PROBABILISTIC;
        public DenoiserType denoiser = DenoiserType.DENOISER_REBLUR;

        public float emissionIntensity = 1.0f;

        public bool cameraJitter = true;
        public bool psr = true;
        public bool emission = true;
        public bool usePrevFrame = true;
        public bool TAA = true;
        public bool indirectDiffuse = true;
        public bool indirectSpecular = true;
        public bool importanceSampling = false;
        public bool SHARC = true;
        public bool specularLobeTrimming = true;
        public bool boost = false;

        [Range(0.0f, 10.0f)]
        public float boostFactor = 0.6667f;

        public bool SR = false;
        public bool RR = true;
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


        [FoldoutHeader("自动曝光 (Histogram Auto Exposure)")]
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

        [FoldoutHeader("SSS (次表面散射)")]
        [Tooltip("皮肤散射颜色，暖橙红色为典型皮肤值")]
        [ColorUsage(false, true)]
        public Color sssScatteringColor = new Color(1.0f, 0.3f, 0.1f);

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

        [FoldoutHeader("RTXDI")]
        public bool enableRtxdi;

        [Range(0, 16)]
        public uint localLightSamples;

        [Range(0, 16)]
        public uint spatialSamples;

        [Range(0, 16)]
        public uint brdfSamples;

        public bool enableSpatialResampling => resamplingMode is ReSTIRDI_ResamplingMode.Spatial or ReSTIRDI_ResamplingMode.TemporalAndSpatial;
        public bool enableTemporalResampling => resamplingMode is ReSTIRDI_ResamplingMode.Temporal or ReSTIRDI_ResamplingMode.TemporalAndSpatial;

        public ReSTIRDI_ResamplingMode resamplingMode;
        public bool gShowLight;

        [FoldoutHeader("参考路径追踪")]
        public bool useReferencePathTracing;

        [Range(0, 16)]
        public int referenceBounceNum = 4;

        [Range(0.0f, 1.0f)]
        public float split;

        public bool accumulateReference = true;
        public bool accumulate = false;
    }
}