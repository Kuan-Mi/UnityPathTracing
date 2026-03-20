#include "Assets/Shaders/Include/Shared.hlsl"
#include "Assets/Shaders/Include/RayTracingShared.hlsl"

#include "Assets/Shaders/NRD/NRD.hlsli"

#pragma max_recursion_depth 1
Texture2D<float4> gOut_Mv;
Texture2D<float> gOut_ViewZ;
Texture2D<float4> gOut_Normal_Roughness;
Texture2D<float4> gOut_BaseColor_Metalness;
Texture2D<uint> gOut_GeoNormal;


// RTXDI：上一帧 GBuffer
Texture2D<float> gIn_PrevViewZ;
Texture2D<float4> gIn_PrevNormalRoughness;
Texture2D<float4> gIn_PrevBaseColorMetalness;
Texture2D<uint>   gIn_PrevGeoNormal;

RWTexture2D<float3> gOut_DirectLighting;

// RWTexture2D<int2> u_TemporalSamplePositions;

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
    uint2 pixelPos = DispatchRaysIndex().xy;
    
    const RTXDI_RuntimeParameters params = g_Const.runtimeParams;
    
    RAB_RandomSamplerState rng = RAB_InitRandomSampler(pixelPos, 2);

    RAB_Surface surface = RAB_GetGBufferSurface(pixelPos, false);

    
    bool usePermutationSampling = false;
    if (g_Const.restirDI.temporalResamplingParams.enablePermutationSampling)
    {
        // Permutation sampling makes more noise on thin, high-detail objects.
        usePermutationSampling = !IsComplexSurface(pixelPos, surface);
    }
    
    RTXDI_DIReservoir temporalResult = RTXDI_EmptyDIReservoir();
    int2 temporalSamplePixelPos = -1;
    
    if (RAB_IsSurfaceValid(surface))
    {
        RTXDI_DIReservoir curSample = RTXDI_LoadDIReservoir(g_Const.restirDI.reservoirBufferParams,
        pixelPos, g_Const.restirDI.bufferIndices.initialSamplingOutputBufferIndex);

        
        float3 motionVector = gOut_Mv[pixelPos].xyz;

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
        
        temporalResult = RTXDI_DITemporalResampling(pixelPos, surface, curSample,
                rng, params, g_Const.restirDI.reservoirBufferParams, tparams, temporalSamplePixelPos, selectedLightSample);
  
    }
    
    // u_TemporalSamplePositions[pixelPos] = temporalSamplePixelPos;
    
    RTXDI_StoreDIReservoir(temporalResult, g_Const.restirDI.reservoirBufferParams, pixelPos, g_Const.restirDI.bufferIndices.temporalResamplingOutputBufferIndex);
}