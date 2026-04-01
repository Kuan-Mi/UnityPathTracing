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

    const RTXDI_RuntimeParameters params = g_Const.runtimeParams;

    uint2 pixelPosition = RTXDI_ReservoirPosToPixelPos(GlobalIndex, params.activeCheckerboardField);

    if (any(pixelPosition > int2(gRectSize)))
        return;

    RAB_RandomSamplerState rng = RAB_InitRandomSampler(pixelPosition, 8);

    const RAB_Surface primarySurface = RAB_GetGBufferSurface(pixelPosition, false);

    const uint2 reservoirPosition = RTXDI_PixelPosToReservoirPos(pixelPosition, params.activeCheckerboardField);
    RTXDI_GIReservoir reservoir = RTXDI_LoadGIReservoir(g_Const.restirGI.reservoirBufferParams, reservoirPosition,
        g_Const.restirGI.bufferIndices.spatialResamplingInputBufferIndex);

    if (RAB_IsSurfaceValid(primarySurface))
    {
        RTXDI_GISpatialResamplingParameters sparams;

        sparams.sourceBufferIndex = g_Const.restirGI.bufferIndices.spatialResamplingInputBufferIndex;
        sparams.biasCorrectionMode = g_Const.restirGI.spatialResamplingParams.spatialBiasCorrectionMode;
        sparams.depthThreshold = g_Const.restirGI.spatialResamplingParams.spatialDepthThreshold;
        sparams.normalThreshold = g_Const.restirGI.spatialResamplingParams.spatialNormalThreshold;
        sparams.numSamples = g_Const.restirGI.spatialResamplingParams.numSpatialSamples;
        sparams.samplingRadius = g_Const.restirGI.spatialResamplingParams.spatialSamplingRadius;

        reservoir = RTXDI_GISpatialResampling(pixelPosition, primarySurface, reservoir, rng,
            params, g_Const.restirGI.reservoirBufferParams, sparams);
    }

    RTXDI_StoreGIReservoir(reservoir, g_Const.restirGI.reservoirBufferParams, reservoirPosition,
        g_Const.restirGI.bufferIndices.spatialResamplingOutputBufferIndex);
}
