#include "Assets/Shaders/Include/Shared.hlsl"
#include "Assets/Shaders/Include/RayTracingShared.hlsl"

#include "Assets/Shaders/NRD/NRD.hlsli"

#pragma max_recursion_depth 1

#include "Assets/Shaders/Rtxdi/RtxdiParameters.h"
#include "Assets/Shaders/Rtxdi/DI/ReSTIRDIParameters.h"
#include "Assets/Shaders/donut/packing.hlsli"
#include "Assets/Shaders/donut/brdf.hlsli"

#include "ResamplingConstants.hlsl"

#include "../RtxdiApplicationBridge/RtxdiApplicationBridge.hlsl"
#include <Assets/Shaders/RTXDI/DI/TemporalResampling.hlsl>

[shader("raygeneration")]
void MainRayGenShader()
{
    uint2 GlobalIndex = DispatchRaysIndex().xy;

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

    RTXDI_DIReservoir temporalResult = RTXDI_EmptyDIReservoir();
    int2 temporalSamplePixelPos = -1;

    if (RAB_IsSurfaceValid(surface))
    {
        RTXDI_DIReservoir curSample = RTXDI_LoadDIReservoir(g_Const.restirDI.reservoirBufferParams,
                                                            GlobalIndex, g_Const.restirDI.bufferIndices.initialSamplingOutputBufferIndex);


        float3 motionVector = gOut_Mv[pixelPosition].xyz;

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

        temporalResult = RTXDI_DITemporalResampling(pixelPosition, surface, curSample, rng, params, g_Const.restirDI.reservoirBufferParams, tparams, temporalSamplePixelPos, selectedLightSample);
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
