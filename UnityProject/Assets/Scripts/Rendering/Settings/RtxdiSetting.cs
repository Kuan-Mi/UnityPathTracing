using Rtxdi.DI;
using Rtxdi.GI;
using Rtxdi.PT;
using Rtxdi.ReGIR;
using UnityEngine;

namespace PathTracing
{
    [System.Serializable]
    public class RtxdiSetting
    {
        [FoldoutHeader("Base Settings")]
        [Range(-10f, 10f)]
        public float exposureEv = 0.0f;

        public float exposure => Mathf.Pow(2, exposureEv);

        public bool     cameraJitter = true;
        public ShowMode showMode     = ShowMode.Final;
        public bool     showMv;

        /// <summary>Mip level to visualise when showMode is Rtxdi_LocalLightPdf or Rtxdi_EnvironmentPdf.</summary>
        [Range(0, 15)]
        public int pdfMipLevel = 0;

        /// <summary>Exposure in stops for the PDF heat-map. +1 = 2× brighter (more sensitive), -1 = 2× darker.</summary>
        [Range(-10f, 10f)]
        public float pdfExposureStops = 0f;

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
}
