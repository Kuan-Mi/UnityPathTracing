using Rtxdi.DI;
using Rtxdi.GI;
using Rtxdi.PT;
using Rtxdi.ReGIR;
using UnityEngine;

namespace PathTracing
{
    [System.Serializable]
    public class NativeRtxdiSetting
    {
        [FoldoutHeader("Base Settings")]
        public Texture2D environmentMap = null; // Equirectangular env map for RTXDI importance sampling

        [Range(0f, 360f)]
        public float environmentRotation = 0f; // Horizontal rotation in degrees

        public float               environmentScale = 1; // Radiance multiplier
        public bool                cameraJitter     = true;
        public NativeRtxdiShowMode showMode         = NativeRtxdiShowMode.Final;
        public bool                showMv;
        public bool                showValidation;

        /// <summary>Mip level to visualise when showMode is LocalLightPdf or EnvironmentPdf.</summary>
        [Range(0, 15)]
        public int pdfMipLevel = 0;

        /// <summary>Exposure in stops for the PDF heat-map. +1 = 2× brighter (more sensitive), -1 = 2× darker.</summary>
        [Range(-10f, 10f)]
        public float pdfExposureStops = 0f;

        public UpscalerMode upscalerMode = UpscalerMode.NATIVE;

        public bool SR = false;

        public bool              skipRightEyeInVR = true;
        public RtxDiDenoiserType denoiserMode     = RtxDiDenoiserType.DENOISER_MODE_OFF;
        public bool              enableGradients  = true;

        [FoldoutHeader("RTXDI")]
        public UnityRtxdiFeature.RenderSettings lightingSettings = UnityRtxdiFeature.RenderSettings.Default();

        public DirectLightingMode   directLightingMode   = DirectLightingMode.ReStir;
        public IndirectLightingMode indirectLightingMode = IndirectLightingMode.ReStirGI;

        public bool                   enableEnv          = true;
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
        public ReSTIRPT_ResamplingMode ptResamplingMode = ReSTIRPT_ResamplingMode.TemporalAndSpatial;

        public RTXDI_PTTemporalResamplingParameters ptTemporalResamplingParams = ReSTIRPTDefaults.GetDefaultTemporalResamplingParams();
        public RTXDI_PTSpatialResamplingParameters  ptSpatialResamplingParams  = ReSTIRPTDefaults.GetDefaultSpatialResamplingParams();

        [FoldoutHeader("Tone Mapping")]
        public bool enableToneMapping = true;

        public NativeToneMappingPass.ToneMappingParameters toneMappingParams;
    }
}
