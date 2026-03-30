#pragma max_recursion_depth 1

#include "../RtxdiApplicationBridge/RtxdiApplicationBridge.hlsl"
#include <Assets/Shaders/RTXDI/DI/TemporalResampling.hlsl>

#ifdef USE_RAY_QUERY
[numthreads(RTXDI_SCREEN_SPACE_GROUP_SIZE, RTXDI_SCREEN_SPACE_GROUP_SIZE, 1)]
void main(uint2 GlobalIndex : SV_DispatchThreadID, uint LocalIndex : SV_GroupIndex)
#else
[shader("raygeneration")]
void MainRayGenShader()
#endif
{
    #ifndef USE_RAY_QUERY
    uint2 GlobalIndex = DispatchRaysIndex().xy;
    uint LocalIndex = 0;
    #endif

    const RTXDI_RuntimeParameters params = g_Const.runtimeParams;

    uint2 pixelPosition = RTXDI_ReservoirPosToPixelPos(GlobalIndex, params.activeCheckerboardField);

    RAB_RandomSamplerState rng = RAB_InitRandomSampler(pixelPosition, 2);

    RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);

    bool usePermutationSampling = false;
    if (g_Const.restirDI.temporalResamplingParams.enablePermutationSampling)
    {
        // Permutation sampling makes more noise on thin, high-detail objects.
        usePermutationSampling = !IsComplexSurface(pixelPosition, surface);
    }

    // usePermutationSampling = true;
    RTXDI_DIReservoir temporalResult = RTXDI_EmptyDIReservoir();
    int2 temporalSamplePixelPos = -1;

    if (RAB_IsSurfaceValid(surface))
    {
        RTXDI_DIReservoir curSample = RTXDI_LoadDIReservoir(g_Const.restirDI.reservoirBufferParams,
                                                            GlobalIndex, g_Const.restirDI.bufferIndices.initialSamplingOutputBufferIndex);

        float3 motionVector = t_MotionVectors[pixelPosition].xyz;

        RTXDI_DITemporalResamplingParameters tparams;
        tparams.screenSpaceMotion = motionVector;
        tparams.sourceBufferIndex = g_Const.restirDI.bufferIndices.temporalResamplingInputBufferIndex;
        tparams.maxHistoryLength = g_Const.restirDI.temporalResamplingParams.maxHistoryLength;
        tparams.biasCorrectionMode = g_Const.restirDI.temporalResamplingParams.temporalBiasCorrection;
        tparams.depthThreshold = g_Const.restirDI.temporalResamplingParams.temporalDepthThreshold;
        tparams.normalThreshold = g_Const.restirDI.temporalResamplingParams.temporalNormalThreshold;
        tparams.enableVisibilityShortcut = g_Const.restirDI.temporalResamplingParams.discardInvisibleSamples;
        tparams.enablePermutationSampling = usePermutationSampling;
        tparams.uniformRandomNumber = g_Const.restirDI.temporalResamplingParams.uniformRandomNumber;

        RAB_LightSample selectedLightSample = (RAB_LightSample)0;

        temporalResult = RTXDI_DITemporalResampling(pixelPosition, surface, curSample,
                                                    rng, params, g_Const.restirDI.reservoirBufferParams, tparams, temporalSamplePixelPos, selectedLightSample);
        //
        // float3 finalColor = ShadeSurfaceWithLightSample(selectedLightSample, surface) * RTXDI_GetDIReservoirInvPdf(temporalResult);
        //
        // RAB_LightInfo lightInfo = RAB_LoadLightInfo(RTXDI_GetDIReservoirLightIndex(temporalResult), false);
        // RAB_LightSample lightSample = RAB_SamplePolymorphicLight(lightInfo, surface, RTXDI_GetDIReservoirSampleUV(temporalResult));
        // float3 finalColor2 = ShadeSurfaceWithLightSample(lightSample, surface) * RTXDI_GetDIReservoirInvPdf(temporalResult);
        //
        // gOut_DirectLighting[pixelPosition] = finalColor2;
        //  
    }

    #ifdef RTXDI_ENABLE_BOILING_FILTER
    if (g_Const.restirDI.temporalResamplingParams.enableBoilingFilter)
    {
        RTXDI_BoilingFilter(LocalIndex, g_Const.restirDI.temporalResamplingParams.boilingFilterStrength, temporalResult);
    }
    #endif
    // u_TemporalSamplePositions[pixelPos] = temporalSamplePixelPos;

    RTXDI_StoreDIReservoir(temporalResult, g_Const.restirDI.reservoirBufferParams, GlobalIndex, g_Const.restirDI.bufferIndices.temporalResamplingOutputBufferIndex);
}
