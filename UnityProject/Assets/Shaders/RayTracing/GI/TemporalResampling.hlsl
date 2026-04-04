#pragma max_recursion_depth 1

#include "../RtxdiApplicationBridge/RtxdiApplicationBridge.hlsl"

#ifdef USE_RAY_QUERY
#define RTXDI_ENABLE_BOILING_FILTER
#define RTXDI_BOILING_FILTER_GROUP_SIZE RTXDI_SCREEN_SPACE_GROUP_SIZE
#endif

#include <Assets/Shaders/RTXDI/GI/BoilingFilter.hlsl>
#include <Assets/Shaders/RTXDI/GI/TemporalResampling.hlsl>

#ifdef USE_RAY_QUERY
[numthreads(RTXDI_SCREEN_SPACE_GROUP_SIZE, RTXDI_SCREEN_SPACE_GROUP_SIZE, 1)]
void main(uint2 GlobalIndex : SV_DispatchThreadID, uint2 LocalIndex : SV_GroupThreadID)
#else
[shader("raygeneration")]
void MainRayGenShader()
#endif
{
    #ifndef USE_RAY_QUERY
    uint2 GlobalIndex = DispatchRaysIndex().xy;
    uint2 LocalIndex = 0;
    #endif

    const RTXDI_RuntimeParameters params = g_Const.runtimeParams;

    uint2 pixelPosition = RTXDI_ReservoirPosToPixelPos(GlobalIndex, params.activeCheckerboardField);

    RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(GlobalIndex, g_Const.runtimeParams.frameIndex, RTXDI_GI_TEMPORAL_RESAMPLING_RANDOM_SEED);

    const RAB_Surface primarySurface = RAB_GetGBufferSurface(pixelPosition, false);

    const uint2 reservoirPosition = RTXDI_PixelPosToReservoirPos(pixelPosition, params.activeCheckerboardField);
    RTXDI_GIReservoir reservoir = RTXDI_LoadGIReservoir(g_Const.restirGI.reservoirBufferParams, reservoirPosition,
        g_Const.restirGI.bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex);

    float3 motionVector = t_MotionVectors[pixelPosition].xyz;
    // motionVector = convertMotionVectorToPixelSpace(g_Const.view, g_Const.prevView, pixelPosition, motionVector);

    if (RAB_IsSurfaceValid(primarySurface))
    {
        RTXDI_GITemporalResamplingParameters tParams = g_Const.restirGI.temporalResamplingParams;

        // Age threshold should vary.
        // This is to avoid to die a bunch of GI reservoirs at once at a disoccluded area.
        tParams.maxReservoirAge = g_Const.restirGI.temporalResamplingParams.maxReservoirAge * (0.5 + RTXDI_GetNextRandom(rng) * 0.5);

        reservoir = RTXDI_GITemporalResampling(pixelPosition, primarySurface, motionVector, g_Const.restirGI.bufferIndices.temporalResamplingInputBufferIndex, reservoir, rng, g_Const.runtimeParams, g_Const.restirGI.reservoirBufferParams, tParams);
   
    }

#ifdef RTXDI_ENABLE_BOILING_FILTER
    if (g_Const.restirGI.boilingFilterParams.enableBoilingFilter)
    {
        RTXDI_GIBoilingFilter(LocalIndex, g_Const.restirGI.boilingFilterParams.boilingFilterStrength, reservoir);
    }
#endif

    RTXDI_StoreGIReservoir(reservoir, g_Const.restirGI.reservoirBufferParams, reservoirPosition,
        g_Const.restirGI.bufferIndices.temporalResamplingOutputBufferIndex);
}
