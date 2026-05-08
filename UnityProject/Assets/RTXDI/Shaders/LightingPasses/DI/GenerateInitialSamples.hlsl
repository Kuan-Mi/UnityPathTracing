/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma pack_matrix(row_major)

#include "../RtxdiApplicationBridge/RtxdiApplicationBridge.hlsli"

#include <Rtxdi/DI/InitialSampling.hlsli>
#include <Rtxdi/DI/ReservoirStorage.hlsli>

bool DEBUG_StreamLocalLightAtUVIntoReservoir(
    inout RTXDI_RandomSamplerState rng,
    RTXDI_InitialSamplingMisData misData,
    RAB_Surface surface,
    float brdfCutoff,
    float localLightMisWeight,
    uint lightIndex,
    float2 uv,
    float invSourcePdf,
    RAB_LightInfo lightInfo,
    inout RTXDI_DIReservoir state,
    inout RAB_LightSample o_selectedSample,float2 pixelPosition)
{
    RAB_LightSample candidateSample = RAB_SamplePolymorphicLight(lightInfo, surface, uv);
    float blendedSourcePdf = RTXDI_LightBrdfMisWeight(surface, candidateSample, 1.0 / invSourcePdf,
        misData.localLightMisWeight, false, misData.brdfMisWeight, brdfCutoff);
    float targetPdf = RAB_GetLightSampleTargetPdfForSurface(candidateSample, surface);
    float risRnd = RTXDI_GetNextRandom(rng);

    if (blendedSourcePdf == 0)
    {
        return false;
    }
    bool selected = RTXDI_StreamSample(state, lightIndex, uv, risRnd, targetPdf, 1.0 / blendedSourcePdf);

    float3 debugColor = selected;
        u_DiffuseLighting[pixelPosition] = float4(debugColor, 0);
    if (selected) {
        o_selectedSample = candidateSample;
 

        u_DiffuseLighting[pixelPosition] = float4(o_selectedSample.radiance, 0);
    }
        // u_DiffuseLighting[pixelPosition] = float4(o_selectedSample.radiance, 0);
    return true;
}

void DEBUG_RandomlySelectLightUniformly(
    float rnd,
    RTXDI_LightBufferRegion region,
    out RAB_LightInfo lightInfo,
    out uint lightIndex,
    out float invSourcePdf,float2 pixelPosition)
{
    invSourcePdf = float(region.numLights);
    lightIndex = region.firstLightIndex + min(uint(floor(rnd * region.numLights)), region.numLights - 1);
    lightInfo = RAB_LoadLightInfo(lightIndex, false);
}

#if RTXDI_ENABLE_PRESAMPLING
void DEBUG_RandomlySelectLocalLightFromRISTile(
    float rnd,
    const RTXDI_RISTileInfo risTileInfo,
    out RAB_LightInfo lightInfo,
    out uint lightIndex,
    out float invSourcePdf)
{
    uint2 risTileData;
    uint risBufferPtr;
    RTXDI_RandomlySelectLightDataFromRISTile(rnd, risTileInfo, risTileData, risBufferPtr);
    RTXDI_UnpackLocalLightFromRISLightData(risTileData, risBufferPtr, lightInfo, lightIndex, invSourcePdf);
}
#endif

void DEBUG_SelectNextLocalLight(
    RTXDI_LocalLightSelectionContext ctx,
    float rnd,
    out RAB_LightInfo lightInfo,
    out uint lightIndex,
    out float invSourcePdf,float2 pixelPosition)
{
    switch (ctx.mode)
    {
#if RTXDI_ENABLE_PRESAMPLING
    case RTXDI_LocalLightContextSamplingMode_RIS:
        DEBUG_RandomlySelectLocalLightFromRISTile(rnd, ctx.risTileInfo, lightInfo, lightIndex, invSourcePdf);
        break;
#endif
    default:
    case RTXDI_LocalLightContextSamplingMode_UNIFORM:
        DEBUG_RandomlySelectLightUniformly(rnd, ctx.lightBufferRegion, lightInfo, lightIndex, invSourcePdf,pixelPosition);
        break;
    }
}

RTXDI_DIReservoir DEBUG_SampleLocalLightsInternal(
    inout RTXDI_RandomSamplerState rng,
    inout RTXDI_RandomSamplerState coherentRng,
    RAB_Surface surface,
    RTXDI_DIInitialSamplingParameters sampleParams,
    RTXDI_InitialSamplingMisData misData,
    ReSTIRDI_LocalLightSamplingMode localLightSamplingMode,
    RTXDI_LightBufferRegion localLightBufferRegion,
#if RTXDI_ENABLE_PRESAMPLING
    RTXDI_RISBufferSegmentParameters localLightRISBufferSegmentParams,
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
    ReGIR_Parameters regirParams,
#endif
#endif
    out RAB_LightSample o_selectedSample,float2 pixelPosition)
{
    RTXDI_DIReservoir state = RTXDI_EmptyDIReservoir();

    RTXDI_LocalLightSelectionContext lightSelectionContext = RTXDI_InitializeLocalLightSelectionContext(coherentRng, localLightSamplingMode, localLightBufferRegion
#if RTXDI_ENABLE_PRESAMPLING
        , localLightRISBufferSegmentParams
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
        , regirParams
        , surface
#endif
#endif
    );

    for (uint i = 0; i < sampleParams.numLocalLightSamples; i++)
    {
        uint lightIndex;
        RAB_LightInfo lightInfo;
        float invSourcePdf;

        float rnd = RTXDI_GetNextRandom(rng);
#if RTXDI_STRATIFY_LOCAL_SAMPLING
        rnd = (rnd + i) / sampleParams.numLocalLightSamples;
#endif

        DEBUG_SelectNextLocalLight(lightSelectionContext, rnd, lightInfo, lightIndex, invSourcePdf,pixelPosition);
        // RTXDI_SelectNextLocalLight(lightSelectionContext, rnd, lightInfo, lightIndex, invSourcePdf);
        float2 uv = RTXDI_RandomlySelectLocalLightUV(rng);

        float3 debugColor = (lightIndex - localLightBufferRegion.firstLightIndex) / (float)localLightBufferRegion.numLights;
        // debugColor = rnd;
        debugColor = lightInfo.logRadiance/1000.0f;

        RAB_LightSample lightSample = RAB_SamplePolymorphicLight(lightInfo, surface, float2(0,0));
        debugColor = lightSample.radiance ;       
        u_DiffuseLighting[pixelPosition] = float4(debugColor, 0);

        bool zeroPdf = DEBUG_StreamLocalLightAtUVIntoReservoir(rng, misData, surface, sampleParams.brdfCutoff, misData.localLightMisWeight, lightIndex, uv, invSourcePdf, lightInfo, state, o_selectedSample
        ,pixelPosition);


        // u_DiffuseLighting[pixelPosition] = float4(o_selectedSample.radiance, 0);

        if (zeroPdf)
            continue;


    }

    RTXDI_FinalizeResampling(state, 1.0, misData.numMisSamples);
    state.M = 1;

    return state;
}

RTXDI_DIReservoir DEBUG_SampleLocalLights(
    inout RTXDI_RandomSamplerState rng,
    inout RTXDI_RandomSamplerState coherentRng,
    RAB_Surface surface,
    RTXDI_DIInitialSamplingParameters sampleParams,
    RTXDI_InitialSamplingMisData misData,
    ReSTIRDI_LocalLightSamplingMode localLightSamplingMode,
    RTXDI_LightBufferRegion localLightBufferRegion,
#if RTXDI_ENABLE_PRESAMPLING
    RTXDI_RISBufferSegmentParameters localLightRISBufferSegmentParams,
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
    ReGIR_Parameters regirParams,
#endif
#endif
    out RAB_LightSample o_selectedSample,float2 pixelPosition)
{
    o_selectedSample = RAB_EmptyLightSample();

    if (localLightBufferRegion.numLights == 0)
        return RTXDI_EmptyDIReservoir();

    if (sampleParams.numLocalLightSamples == 0)
        return RTXDI_EmptyDIReservoir();

    return DEBUG_SampleLocalLightsInternal(rng, coherentRng, surface, sampleParams, misData, localLightSamplingMode, localLightBufferRegion,
#if RTXDI_ENABLE_PRESAMPLING
        localLightRISBufferSegmentParams,
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
        regirParams,
#endif
#endif
        o_selectedSample,pixelPosition);
}

RTXDI_DIReservoir DEBUG_SampleLightsForSurface(
    inout RTXDI_RandomSamplerState rng,
    inout RTXDI_RandomSamplerState coherentRng,
    RAB_Surface surface,
    RTXDI_DIInitialSamplingParameters sampleParams,
    RTXDI_LightBufferParameters lightBufferParams,
#if RTXDI_ENABLE_PRESAMPLING
    RTXDI_RISBufferSegmentParameters localLightRISBufferSegmentParams,
    RTXDI_RISBufferSegmentParameters environmentLightRISBufferSegmentParams,
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
    ReGIR_Parameters regirParams,
#endif
#endif
    out RAB_LightSample o_lightSample,float2 pixelPosition)
{
    o_lightSample = RAB_EmptyLightSample();

    RTXDI_DIReservoir localReservoir;
    RAB_LightSample localSample = RAB_EmptyLightSample();

    RTXDI_InitialSamplingMisData misData = RTXDI_ComputeInitialSamplingMisData(sampleParams);

    localReservoir = DEBUG_SampleLocalLights(rng, coherentRng, surface,
        sampleParams, misData, sampleParams.localLightSamplingMode, lightBufferParams.localLightBufferRegion,
#if RTXDI_ENABLE_PRESAMPLING
        localLightRISBufferSegmentParams,
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
        regirParams,
#endif
#endif
        localSample,pixelPosition);

    RAB_LightSample infiniteSample = RAB_EmptyLightSample();
    RTXDI_DIReservoir infiniteReservoir = RTXDI_SampleInfiniteLights(rng, surface,
        sampleParams.numInfiniteLightSamples, lightBufferParams.infiniteLightBufferRegion, infiniteSample);

#if RTXDI_ENABLE_PRESAMPLING
    RAB_LightSample environmentSample = RAB_EmptyLightSample();
    RTXDI_DIReservoir environmentReservoir = RTXDI_SampleEnvironmentMap(rng, coherentRng, surface,
        sampleParams, misData, lightBufferParams.environmentLightParams, environmentLightRISBufferSegmentParams, environmentSample);
#endif // RTXDI_ENABLE_PRESAMPLING

    RAB_LightSample brdfSample = RAB_EmptyLightSample();
    RTXDI_DIReservoir brdfReservoir = RTXDI_SampleBrdf(rng, surface, sampleParams.numBrdfSamples, sampleParams.brdfCutoff, sampleParams.brdfRayMinT, misData, coherentRng, lightBufferParams, brdfSample);

    RTXDI_DIReservoir state = RTXDI_EmptyDIReservoir();
    RTXDI_CombineDIReservoirs(state, localReservoir, 0.5, localReservoir.targetPdf);
    bool selectInfinite = RTXDI_CombineDIReservoirs(state, infiniteReservoir, RTXDI_GetNextRandom(rng), infiniteReservoir.targetPdf);
#if RTXDI_ENABLE_PRESAMPLING
    bool selectEnvironment = RTXDI_CombineDIReservoirs(state, environmentReservoir, RTXDI_GetNextRandom(rng), environmentReservoir.targetPdf);
#endif // RTXDI_ENABLE_PRESAMPLING
    bool selectBrdf = RTXDI_CombineDIReservoirs(state, brdfReservoir, RTXDI_GetNextRandom(rng), brdfReservoir.targetPdf);

    RTXDI_FinalizeResampling(state, 1.0, 1.0);
    state.M = 1;

    if (selectBrdf)
        o_lightSample = brdfSample;
    else
#if RTXDI_ENABLE_PRESAMPLING
    if (selectEnvironment)
        o_lightSample = environmentSample;
    else
#endif // RTXDI_ENABLE_PRESAMPLING
    if (selectInfinite)
        o_lightSample = infiniteSample;
    else
        o_lightSample = localSample;

    if (sampleParams.enableInitialVisibility && RTXDI_IsValidDIReservoir(state))
    {
        if (!RAB_GetConservativeVisibility(surface, o_lightSample))
        {
            RTXDI_StoreVisibilityInDIReservoir(state, 0, true);
        }
    }

    return state;
}

#if USE_RAY_QUERY
[numthreads(RTXDI_SCREEN_SPACE_GROUP_SIZE, RTXDI_SCREEN_SPACE_GROUP_SIZE, 1)]
void main(uint2 GlobalIndex : SV_DispatchThreadID)
#else
[shader("raygeneration")]
void RayGen()
#endif
{
#if !USE_RAY_QUERY
    uint2 GlobalIndex = DispatchRaysIndex().xy;
#endif

    const RTXDI_RuntimeParameters params = g_Const.runtimeParams;

    uint2 pixelPosition = RTXDI_ReservoirPosToPixelPos(GlobalIndex, params.activeCheckerboardField);

    RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(pixelPosition, g_Const.runtimeParams.frameIndex, RTXDI_DI_GENERATE_INITIAL_SAMPLES_RANDOM_SEED);
    RTXDI_RandomSamplerState tileRng = RTXDI_InitRandomSampler(pixelPosition / RTXDI_TILE_SIZE_IN_PIXELS, g_Const.runtimeParams.frameIndex, RTXDI_DI_GENERATE_INITIAL_SAMPLES_RANDOM_SEED);

    RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);

    // DEBUG: 强制只采 1 个 local light，禁用其他采样策略
    RTXDI_DIInitialSamplingParameters debugSampleParams = g_Const.restirDI.initialSamplingParams;
    // debugSampleParams.numLocalLightSamples = 1;

    RAB_LightSample lightSample;
    RTXDI_DIReservoir reservoir = DEBUG_SampleLightsForSurface(rng, tileRng, surface,
        // g_Const.restirDI.initialSamplingParams, g_Const.lightBufferParams,
        debugSampleParams, g_Const.lightBufferParams,
#ifdef RTXDI_ENABLE_PRESAMPLING
        g_Const.localLightsRISBufferSegmentParams, g_Const.environmentLightRISBufferSegmentParams,
#if RTXDI_REGIR_MODE != RTXDI_REGIR_MODE_DISABLED
        g_Const.regir,
#endif
#endif
        lightSample,pixelPosition);

    RTXDI_StoreDIReservoir(reservoir, g_Const.restirDI.reservoirBufferParams, GlobalIndex, g_Const.restirDI.bufferIndices.initialSamplingOutputBufferIndex);
    
    // debug 
    // u_DiffuseLighting[pixelPosition] = float4(1,0,0, 0);

    // float3 debugColor = g_Const.restirDI.initialSamplingParams.numLocalLightSamples / 2.0f;
    // u_DiffuseLighting[pixelPosition] = float4(debugColor, 0);

// if (g_Const.restirDI.initialSamplingParams.numLocalLightSamples == 1) 
//     u_DiffuseLighting[pixelPosition] = float4(0, 1, 0, 0); // 绿色表示确认为1
// else 
//     u_DiffuseLighting[pixelPosition] = float4(1, 0, 0, 0); // 红色表示不是1


    // u_DiffuseLighting[pixelPosition] = float4(lightSample.radiance, 0);
    // u_DiffuseLighting[pixelPosition] = float4(surface.material.diffuseAlbedo, 0);
}