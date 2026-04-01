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

    RAB_RandomSamplerState rng = RAB_InitRandomSampler(pixelPosition, 7);

    const RAB_Surface primarySurface = RAB_GetGBufferSurface(pixelPosition, false);

    const uint2 reservoirPosition = RTXDI_PixelPosToReservoirPos(pixelPosition, params.activeCheckerboardField);
    RTXDI_GIReservoir reservoir = RTXDI_LoadGIReservoir(g_Const.restirGI.reservoirBufferParams, reservoirPosition,
        g_Const.restirGI.bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex);

    float3 motionVector = t_MotionVectors[pixelPosition].xyz;
    // motionVector = convertMotionVectorToPixelSpace(g_Const.view, g_Const.prevView, pixelPosition, motionVector);

    if (RAB_IsSurfaceValid(primarySurface))
    {
        RTXDI_GITemporalResamplingParameters tParams;

        tParams.screenSpaceMotion = motionVector;
        tParams.sourceBufferIndex = g_Const.restirGI.bufferIndices.temporalResamplingInputBufferIndex;
        tParams.maxHistoryLength = g_Const.restirGI.temporalResamplingParams.maxHistoryLength;
        tParams.biasCorrectionMode = g_Const.restirGI.temporalResamplingParams.temporalBiasCorrectionMode;
        tParams.depthThreshold = g_Const.restirGI.temporalResamplingParams.depthThreshold;
        tParams.normalThreshold = g_Const.restirGI.temporalResamplingParams.normalThreshold;
        tParams.enablePermutationSampling = g_Const.restirGI.temporalResamplingParams.enablePermutationSampling;
        tParams.enableFallbackSampling = g_Const.restirGI.temporalResamplingParams.enableFallbackSampling;
        tParams.uniformRandomNumber = g_Const.restirGI.temporalResamplingParams.uniformRandomNumber;
        tParams.maxReservoirAge = g_Const.restirGI.temporalResamplingParams.maxReservoirAge * (0.5 + RAB_GetNextRandom(rng) * 0.5);

        reservoir = RTXDI_GITemporalResampling(pixelPosition, primarySurface, reservoir, rng,
            params, g_Const.restirGI.reservoirBufferParams, tParams);
    }

#ifdef RTXDI_ENABLE_BOILING_FILTER
    if (g_Const.restirGI.temporalResamplingParams.enableBoilingFilter)
    {
        RTXDI_GIBoilingFilter(LocalIndex, g_Const.restirGI.temporalResamplingParams.boilingFilterStrength, reservoir);
    }
#endif

    RTXDI_StoreGIReservoir(reservoir, g_Const.restirGI.reservoirBufferParams, reservoirPosition,
        g_Const.restirGI.bufferIndices.temporalResamplingOutputBufferIndex);
}
