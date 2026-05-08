using mini;
using Rtxdi;
using Rtxdi.DI;
using Rtxdi.GI;
using Rtxdi.ReGIR;
using RTXDI;
using Unity.Mathematics;

namespace PathTracing
{
    /// <summary>
    /// Shared helper for building <see cref="ResamplingConstants"/> from RTXDI runtime state.
    /// Used by both <see cref="UnityRtxdiFeature"/> and <see cref="NativeRtxdiFeature"/>.
    /// </summary>
    internal static class RtxdiConstantsBuilder
    {
        public static ResamplingConstants Build(
            NativeRtxdiSetting              setting,
            UnityRtxdiFeature.RenderSettings localSettings,
            ImportanceSamplingContext isContext,
            CameraFrameState          frameState,
            RTXDI_LightBufferParameters lightBufferParams,
            uint2                     localLightPdfTextureSize,
            bool enableIndirect,
            bool enableAdditiveBlend,
            bool enableEmissiveSurfaces,
            bool enableAccumulation,
            bool enableReSTIRGI)
        {
            var restirDIContext = isContext.GetReSTIRDIContext();
            var restirGIContext = isContext.GetReSTIRGIContext();

            isContext.SetLightBufferParams(lightBufferParams);
            restirDIContext.SetFrameIndex(frameState.frameIndex);
            restirDIContext.SetResamplingMode(setting.diResamplingMode);
            restirDIContext.SetInitialSamplingParameters(setting.initialSamplingParams);
            restirDIContext.SetTemporalResamplingParameters(setting.temporalResamplingParams);
            restirDIContext.SetSpatialResamplingParameters(setting.spatialResamplingParams);
            restirDIContext.SetShadingParameters(setting.shadingParams);

            restirGIContext.SetFrameIndex(frameState.frameIndex);
            restirGIContext.SetResamplingMode(setting.giResamplingMode);
            restirGIContext.SetTemporalResamplingParameters(setting.giTemporalResamplingParams);
            restirGIContext.SetSpatialResamplingParameters(setting.giSpatialResamplingParams);
            restirGIContext.SetFinalShadingParameters(setting.giFinalShadingParams);

            var regirContext = isContext.GetReGIRContext();
            setting.regirDynamicParams.center = frameState.camPos;
            regirContext.SetDynamicParameters(setting.regirDynamicParams);

            var constants = new ResamplingConstants
            {
                frameIndex             = restirDIContext.GetFrameIndex(),
                denoiserMode           = localSettings.denoiserMode,
                enableBrdfIndirect     = enableIndirect      ? 1u : 0u,
                enableBrdfAdditiveBlend = enableAdditiveBlend ? 1u : 0u,
                enableAccumulation     = enableAccumulation  ? 1u : 0u,
            };

            FillResamplingConstants(ref constants, localSettings, isContext, localLightPdfTextureSize);
            FillBRDFPTConstants(ref constants.brdfPT, setting);

            constants.brdfPT.enableIndirectEmissiveSurfaces = enableEmissiveSurfaces ? 1u : 0u;
            constants.brdfPT.enableReSTIRGI                 = enableReSTIRGI         ? 1u : 0u;

            return constants;
        }

        private static void FillResamplingConstants(
            ref ResamplingConstants  constants,
            UnityRtxdiFeature.RenderSettings localSettings,
            ImportanceSamplingContext isContext,
            uint2                     localLightPdfTextureSize)
        {
            constants.enablePreviousTLAS  = localSettings.denoiserMode;
            constants.denoiserMode        = localSettings.denoiserMode;
            constants.visualizeRegirCells = localSettings.visualizeRegirCells ? 1u : 0u;

            constants.lightBufferParams                      = isContext.GetLightBufferParameters();
            constants.localLightsRISBufferSegmentParams      = isContext.GetLocalLightRISBufferSegmentParams();
            constants.environmentLightRISBufferSegmentParams = isContext.GetEnvironmentLightRISBufferSegmentParams();
            constants.runtimeParams                          = isContext.GetReSTIRDIContext().GetRuntimeParams();

            FillReSTIRDIConstants(ref constants.restirDI, isContext.GetReSTIRDIContext(), isContext.GetLightBufferParameters());
            FillReGIRConstants(ref constants.regir,    isContext.GetReGIRContext());
            FillReSTIRGIConstants(ref constants.restirGI, isContext.GetReSTIRGIContext());

            constants.localLightPdfTextureSize = localLightPdfTextureSize;
        }

        private static void FillBRDFPTConstants(ref BRDFPathTracing_Parameters constants, RtxdiSetting setting)
        {
            constants = setting.brdfptParams;
            constants.materialOverrideParams.roughnessOverride     = -1.0f;
            constants.materialOverrideParams.metalnessOverride     = -1.0f;
            constants.secondarySurfaceReSTIRDIParams.initialSamplingParams.environmentMapImportanceSampling = 0;
            if (constants.secondarySurfaceReSTIRDIParams.initialSamplingParams.environmentMapImportanceSampling == 0)
                constants.secondarySurfaceReSTIRDIParams.initialSamplingParams.numEnvironmentSamples = 0;
        }
        private static void FillBRDFPTConstants(ref BRDFPathTracing_Parameters constants, NativeRtxdiSetting setting)
        {
            constants = setting.brdfptParams;
            constants.materialOverrideParams.roughnessOverride     = -1.0f;
            constants.materialOverrideParams.metalnessOverride     = -1.0f;
            constants.secondarySurfaceReSTIRDIParams.initialSamplingParams.environmentMapImportanceSampling = 0;
            if (constants.secondarySurfaceReSTIRDIParams.initialSamplingParams.environmentMapImportanceSampling == 0)
                constants.secondarySurfaceReSTIRDIParams.initialSamplingParams.numEnvironmentSamples = 0;
        }

        private static void FillReSTIRDIConstants(
            ref RTXDI_Parameters        rparams,
            ReSTIRDIContext             restirDIContext,
            RTXDI_LightBufferParameters lightBufferParameters)
        {
            rparams.reservoirBufferParams = restirDIContext.GetReservoirBufferParameters();
            rparams.bufferIndices         = restirDIContext.GetBufferIndices();
            rparams.initialSamplingParams = restirDIContext.GetInitialSamplingParameters();

            rparams.initialSamplingParams.environmentMapImportanceSampling = lightBufferParameters.environmentLightParams.lightPresent;
            if (rparams.initialSamplingParams.environmentMapImportanceSampling == 0)
                rparams.initialSamplingParams.numEnvironmentSamples = 0;
            rparams.temporalResamplingParams        = restirDIContext.GetTemporalResamplingParameters();
            rparams.boilingFilterParams             = restirDIContext.GetBoilingFilterParameters();
            rparams.spatialResamplingParams         = restirDIContext.GetSpatialResamplingParameters();
            rparams.spatioTemporalResamplingParams  = restirDIContext.GetSpatioTemporalResamplingParameters();
            rparams.shadingParams                  = restirDIContext.GetShadingParameters();
        }

        private static void FillReSTIRGIConstants(ref RTXDI_GIParameters constants, ReSTIRGIContext restirGIContext)
        {
            constants.reservoirBufferParams         = restirGIContext.GetReservoirBufferParameters();
            constants.bufferIndices                 = restirGIContext.GetBufferIndices();
            constants.temporalResamplingParams      = restirGIContext.GetTemporalResamplingParameters();
            constants.boilingFilterParams           = restirGIContext.GetBoilingFilterParameters();
            constants.spatialResamplingParams       = restirGIContext.GetSpatialResamplingParameters();
            constants.spatioTemporalResamplingParams = restirGIContext.GetSpatioTemporalResamplingParameters();
            constants.finalShadingParams            = restirGIContext.GetFinalShadingParameters();
        }

        private static void FillReGIRConstants(ref ReGIR_Parameters reGirParams, ReGIRContext regirContext)
        {
            var staticParams  = regirContext.GetReGIRStaticParameters();
            var dynamicParams = regirContext.GetReGIRDynamicParameters();
            var onionParams   = regirContext.GetReGIROnionCalculatedParameters();

            reGirParams.gridParams.cellsX = staticParams.gridParameters.GridSize.x;
            reGirParams.gridParams.cellsY = staticParams.gridParameters.GridSize.y;
            reGirParams.gridParams.cellsZ = staticParams.gridParameters.GridSize.z;

            reGirParams.commonParams.numRegirBuildSamples = dynamicParams.regirNumBuildSamples;
            reGirParams.commonParams.risBufferOffset      = regirContext.GetReGIRCellOffset();
            reGirParams.commonParams.lightsPerCell        = staticParams.LightsPerCell;
            reGirParams.commonParams.centerX              = dynamicParams.center.x;
            reGirParams.commonParams.centerY              = dynamicParams.center.y;
            reGirParams.commonParams.centerZ              = dynamicParams.center.z;
            reGirParams.commonParams.cellSize = (staticParams.Mode == ReGIRMode.Onion)
                ? dynamicParams.regirCellSize * 0.5f
                : dynamicParams.regirCellSize;
            reGirParams.commonParams.localLightSamplingFallbackMode = (uint)dynamicParams.fallbackSamplingMode;
            reGirParams.commonParams.localLightPresamplingMode      = (uint)dynamicParams.presamplingMode;
            reGirParams.commonParams.samplingJitter                 = math.max(0.0f, dynamicParams.regirSamplingJitter * 2.0f);
            reGirParams.onionParams.cubicRootFactor                 = onionParams.regirOnionCubicRootFactor;
            reGirParams.onionParams.linearFactor                    = onionParams.regirOnionLinearFactor;
            reGirParams.onionParams.numLayerGroups                  = (uint)onionParams.regirOnionLayers.Count;

            for (int group = 0; group < onionParams.regirOnionLayers.Count; group++)
            {
                var layer = onionParams.regirOnionLayers[group];
                layer.innerRadius *= reGirParams.commonParams.cellSize;
                layer.outerRadius *= reGirParams.commonParams.cellSize;
                reGirParams.onionParams.SetLayer(group, layer);
            }

            for (int n = 0; n < onionParams.regirOnionRings.Count; n++)
                reGirParams.onionParams.SetRing(n, onionParams.regirOnionRings[n]);

            reGirParams.onionParams.cubicRootFactor = regirContext.GetReGIROnionCalculatedParameters().regirOnionCubicRootFactor;
        }
    }
}
