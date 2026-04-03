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

    RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(pixelPosition, g_Const.runtimeParams.frameIndex, RTXDI_DI_TEMPORAL_RESAMPLING_RANDOM_SEED);

    RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);

    bool usePermutationSampling = false;
    if (g_Const.restirDI.temporalResamplingParams.enablePermutationSampling)
    {
        // Permutation sampling makes more noise on thin, high-detail objects.
        usePermutationSampling = !IsComplexSurface(pixelPosition, surface);
    }

    RTXDI_DITemporalResamplingParameters tParams = g_Const.restirDI.temporalResamplingParams;
    tParams.enablePermutationSampling = usePermutationSampling;
    
    // usePermutationSampling = true;
    RTXDI_DIReservoir temporalResult = RTXDI_EmptyDIReservoir();
    int2 temporalSamplePixelPos = -1;

    if (RAB_IsSurfaceValid(surface))
    {
        RTXDI_DIReservoir curSample = RTXDI_LoadDIReservoir(g_Const.restirDI.reservoirBufferParams,
                                                            GlobalIndex, g_Const.restirDI.bufferIndices.initialSamplingOutputBufferIndex);

        float3 motionVector = t_MotionVectors[pixelPosition].xyz;

        
        uint sourceBufferIndex = g_Const.restirDI.bufferIndices.temporalResamplingInputBufferIndex;
        
        
        RAB_LightSample selectedLightSample = (RAB_LightSample)0;
        
        temporalResult = RTXDI_DITemporalResampling(pixelPosition, surface, curSample,
            rng, params, g_Const.restirDI.reservoirBufferParams, motionVector, sourceBufferIndex, tParams, temporalSamplePixelPos, selectedLightSample);
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
