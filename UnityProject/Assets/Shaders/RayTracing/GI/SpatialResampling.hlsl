#pragma max_recursion_depth 1

#include "../RtxdiApplicationBridge/RtxdiApplicationBridge.hlsl"

#include <Assets/Shaders/RTXDI/GI/SpatialResampling.hlsl>

#ifdef USE_RAY_QUERY
[numthreads(RTXDI_SCREEN_SPACE_GROUP_SIZE, RTXDI_SCREEN_SPACE_GROUP_SIZE, 1)]
void main(uint2 GlobalIndex : SV_DispatchThreadID)
#else
[shader("raygeneration")]
void MainRayGenShader()
#endif
{
    #ifndef USE_RAY_QUERY
    uint2 GlobalIndex = DispatchRaysIndex().xy;
    #endif
    uint2 pixelPosition = RTXDI_ReservoirPosToPixelPos(GlobalIndex, g_Const.runtimeParams.activeCheckerboardField);

    if (any(pixelPosition > int2(gRectSize)))
        return;

    RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(GlobalIndex, g_Const.runtimeParams.frameIndex, RTXDI_GI_SPATIAL_RESAMPLING_RANDOM_SEED);

    const RAB_Surface primarySurface = RAB_GetGBufferSurface(pixelPosition, false);

    const uint2 reservoirPosition = RTXDI_PixelPosToReservoirPos(pixelPosition, g_Const.runtimeParams.activeCheckerboardField);
    RTXDI_GIReservoir reservoir = RTXDI_LoadGIReservoir(g_Const.restirGI.reservoirBufferParams, reservoirPosition,
        g_Const.restirGI.bufferIndices.spatialResamplingInputBufferIndex);

    if (RAB_IsSurfaceValid(primarySurface))
    {
        uint sourceBufferIndex = g_Const.restirGI.bufferIndices.spatialResamplingInputBufferIndex;
        RTXDI_GISpatialResamplingParameters sparams = g_Const.restirGI.spatialResamplingParams;
        reservoir = RTXDI_GISpatialResampling(pixelPosition, primarySurface, sourceBufferIndex, reservoir, rng, g_Const.runtimeParams, g_Const.restirGI.reservoirBufferParams, sparams);
    }

    RTXDI_StoreGIReservoir(reservoir, g_Const.restirGI.reservoirBufferParams, reservoirPosition,
        g_Const.restirGI.bufferIndices.spatialResamplingOutputBufferIndex);
}
