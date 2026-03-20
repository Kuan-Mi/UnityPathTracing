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


#include "Assets/Shaders/Rtxdi/RtxdiParameters.h"
#include "Assets/Shaders/donut/packing.hlsli"
#include "Assets/Shaders/donut/brdf.hlsli"

#include "ResamplingConstants.hlsl"

#include "../RtxdiApplicationBridge/RtxdiApplicationBridge.hlsl"
#include "Assets/Shaders/RTXDI/DI/InitialSampling.hlsl"
#include <Assets/Shaders/RTXDI/DI/SpatioTemporalResampling.hlsl>

[shader("raygeneration")]
void MainRayGenShader()
{
    uint2 pixelPos = DispatchRaysIndex().xy;
    
    // Test RTXDI

    const RTXDI_LightBufferParameters lightBufferParams = g_Const.lightBufferParams;

    RAB_Surface primarySurface =  RAB_GetGBufferSurface(pixelPos,false);

    RTXDI_DIReservoir reservoir = RTXDI_EmptyDIReservoir();

    RAB_RandomSamplerState rng = RAB_InitRandomSampler(pixelPos, 1);

    RTXDI_SampleParameters sampleParams = RTXDI_InitSampleParameters(
        g_Const.numInitialSamples, // local light samples 
        // 局部光源采样数
        0, // infinite light samples
        // 无限光源采样数
        0, // environment map samples
        // 环境贴图采样数
        g_Const.numInitialBRDFSamples,
        g_Const.brdfCutoff,
        0.001f);

    // Generate the initial sample
    RAB_LightSample lightSample = RAB_EmptyLightSample();
    RTXDI_DIReservoir localReservoir = RTXDI_SampleLocalLights(rng, rng, primarySurface, sampleParams, ReSTIRDI_LocalLightSamplingMode_UNIFORM, lightBufferParams.localLightBufferRegion, lightSample);
    RTXDI_CombineDIReservoirs(reservoir, localReservoir, 0.5, localReservoir.targetPdf);


    // Resample BRDF samples.
    RAB_LightSample brdfSample = RAB_EmptyLightSample();
    RTXDI_DIReservoir brdfReservoir = RTXDI_SampleBrdf(rng, primarySurface, sampleParams, lightBufferParams, brdfSample);
    bool selectBrdf = RTXDI_CombineDIReservoirs(reservoir, brdfReservoir, RAB_GetNextRandom(rng), brdfReservoir.targetPdf);
    if (selectBrdf)
    {
        lightSample = brdfSample;
    }

    RTXDI_FinalizeResampling(reservoir, 1.0, 1.0);
    reservoir.M = 1;

    // BRDF was generated with a trace so no need to trace visibility again
    // BRDF 是通过追踪生成的，因此无需再次追踪可见性
    if (RTXDI_IsValidDIReservoir(reservoir) && !selectBrdf)
    // if (RTXDI_IsValidDIReservoir(reservoir))
    {
        // See if the initial sample is visible from the surface
        // 查看初始样本对于表面是否可见
        if (!RAB_GetConservativeVisibility(primarySurface, lightSample))
        {
            // If not visible, discard the sample (but keep the M)
            // 如果不可见，则丢弃样本（但保留 M 值）
            RTXDI_StoreVisibilityInDIReservoir(reservoir, 0, true);
        }
    }

    RTXDI_StoreDIReservoir(reservoir, g_Const.restirDI.reservoirBufferParams, pixelPos, g_Const.restirDI.bufferIndices.initialSamplingOutputBufferIndex);
    // RTXDI_StoreDIReservoir(reservoir, g_Const.restirDIReservoirBufferParams, pixelPos, g_Const.outputBufferIndex);


    
}