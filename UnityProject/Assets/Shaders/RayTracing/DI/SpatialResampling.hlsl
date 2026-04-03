#pragma max_recursion_depth 1

RWTexture2D<int2> u_TemporalSamplePositions;

#include "../RtxdiApplicationBridge/RtxdiApplicationBridge.hlsl"
#include <Assets/Shaders/RTXDI/DI/SpatialResampling.hlsl>

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

    RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(pixelPosition, g_Const.runtimeParams.frameIndex, RTXDI_DI_SPATIAL_RESAMPLING_RANDOM_SEED);

    RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);

    RTXDI_DIReservoir spatialResult = RTXDI_EmptyDIReservoir();

    
    if (RAB_IsSurfaceValid(surface))
    {
        RTXDI_DIReservoir centerSample = RTXDI_LoadDIReservoir(g_Const.restirDI.reservoirBufferParams,
            GlobalIndex, g_Const.restirDI.bufferIndices.spatialResamplingInputBufferIndex);

        uint sourceBufferIndex = g_Const.restirDI.bufferIndices.spatialResamplingInputBufferIndex;
        
        RAB_LightSample lightSample = (RAB_LightSample)0;
        spatialResult = RTXDI_DISpatialResampling(pixelPosition, surface, centerSample, 
             rng, params, g_Const.restirDI.reservoirBufferParams, sourceBufferIndex, g_Const.restirDI.spatialResamplingParams, lightSample);
        
    }

    RTXDI_StoreDIReservoir(spatialResult, g_Const.restirDI.reservoirBufferParams, GlobalIndex, g_Const.restirDI.bufferIndices.spatialResamplingOutputBufferIndex);
}
