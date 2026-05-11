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

    // ---------------------------------------------------------------------
    // Inlined RTXDI_SampleLightsForSurface for debugging / analysis.
    // Original: External/Rtxdi/Include/Rtxdi/DI/InitialSampling.hlsli
    // ---------------------------------------------------------------------
    RTXDI_DIInitialSamplingParameters sampleParams = g_Const.restirDI.initialSamplingParams;
    RTXDI_LightBufferParameters lightBufferParams = g_Const.lightBufferParams;

    RAB_LightSample lightSample = RAB_EmptyLightSample();

    // --- MIS data ---
    RTXDI_InitialSamplingMisData misData = RTXDI_ComputeInitialSamplingMisData(sampleParams);

    // --- Local lights (INLINED for debugging, includes ReGIR path) ---
    RAB_LightSample localSample = RAB_EmptyLightSample();
    RTXDI_DIReservoir localReservoir = RTXDI_EmptyDIReservoir();

    // Debug-friendly flags so we can observe which path each pixel took.
    int  dbg_regirCellIndex   = -1;   // -1 means pixel is outside the ReGIR grid (uses fallback)
    uint dbg_localSampleMode  = 0;    // 0 = UNIFORM, 1 = POWER_RIS, 2 = REGIR_RIS (cell hit)

    if (lightBufferParams.localLightBufferRegion.numLights != 0 &&
        sampleParams.numLocalLightSamples != 0)
    {
        // ----- Build RTXDI_LocalLightSelectionContext (inlined) -----
        RTXDI_LocalLightSelectionContext lightSelectionContext;
        lightSelectionContext.lightBufferRegion = lightBufferParams.localLightBufferRegion;

#ifdef RTXDI_ENABLE_PRESAMPLING
        if (sampleParams.localLightSamplingMode == ReSTIRDI_LocalLightSamplingMode_REGIR_RIS)
        {
#if RTXDI_REGIR_MODE != RTXDI_REGIR_MODE_DISABLED
            // ----- Inlined RTXDI_CalculateReGIRCellIndex -----
            float3 cellJitter = float3(
                RTXDI_GetNextRandom(tileRng),
                RTXDI_GetNextRandom(tileRng),
                RTXDI_GetNextRandom(tileRng)) - 0.5;

            float3 samplingPos  = RAB_GetSurfaceWorldPos(surface);
            float  jitterScale  = RTXDI_ReGIR_GetJitterScale(g_Const.regir, samplingPos);
            samplingPos        += cellJitter * jitterScale;

            int cellIndex = RTXDI_ReGIR_WorldPosToCellIndex(g_Const.regir, samplingPos);
            dbg_regirCellIndex = cellIndex;

            if (cellIndex >= 0)
            {
                // ReGIR cell hit -> use the per-cell pre-sampled RIS tile
                RTXDI_RISTileInfo tileInfo;
                tileInfo.risTileOffset = uint(cellIndex) * g_Const.regir.commonParams.lightsPerCell
                                       + g_Const.regir.commonParams.risBufferOffset;
                tileInfo.risTileSize   = g_Const.regir.commonParams.lightsPerCell;

                lightSelectionContext.mode        = RTXDI_LocalLightContextSamplingMode_RIS;
                lightSelectionContext.risTileInfo = tileInfo;
                dbg_localSampleMode = 2; // REGIR_RIS
            }
            else if (g_Const.regir.commonParams.localLightSamplingFallbackMode
                     == ReSTIRDI_LocalLightSamplingMode_POWER_RIS)
            {
                // Fallback: global Power-RIS buffer
                lightSelectionContext.mode = RTXDI_LocalLightContextSamplingMode_RIS;
                lightSelectionContext.risTileInfo = RTXDI_RandomlySelectRISTile(
                    tileRng, g_Const.localLightsRISBufferSegmentParams);
                dbg_localSampleMode = 1; // POWER_RIS (fallback)
            }
            else
            {
                // Fallback: uniform sampling over the whole local light buffer
                lightSelectionContext.mode = RTXDI_LocalLightContextSamplingMode_UNIFORM;
                dbg_localSampleMode = 0; // UNIFORM (fallback)
            }
#else
            // REGIR_RIS requested but RTXDI_REGIR_MODE is disabled -> uniform
            lightSelectionContext.mode = RTXDI_LocalLightContextSamplingMode_UNIFORM;
            dbg_localSampleMode = 0;
#endif // RTXDI_REGIR_MODE != RTXDI_REGIR_MODE_DISABLED
        }
        else if (sampleParams.localLightSamplingMode == ReSTIRDI_LocalLightSamplingMode_POWER_RIS)
        {
            lightSelectionContext.mode = RTXDI_LocalLightContextSamplingMode_RIS;
            lightSelectionContext.risTileInfo = RTXDI_RandomlySelectRISTile(
                tileRng, g_Const.localLightsRISBufferSegmentParams);
            dbg_localSampleMode = 1; // POWER_RIS
        }
        else
#endif // RTXDI_ENABLE_PRESAMPLING
        {
            lightSelectionContext.mode = RTXDI_LocalLightContextSamplingMode_UNIFORM;
            dbg_localSampleMode = 0; // UNIFORM
        }

        // ----- Sampling loop (inlined from RTXDI_SampleLocalLightsInternal) -----
        for (uint i = 0; i < sampleParams.numLocalLightSamples; i++)
        {
            uint           lightIndex;
            RAB_LightInfo  lightInfo;
            float          invSourcePdf;

            float rnd = RTXDI_GetNextRandom(rng);
#if RTXDI_STRATIFY_LOCAL_SAMPLING
            rnd = (rnd + i) / sampleParams.numLocalLightSamples;
#endif
            RTXDI_SelectNextLocalLight(lightSelectionContext, rnd,
                                       lightInfo, lightIndex, invSourcePdf);

            float2 uv = RTXDI_RandomlySelectLocalLightUV(rng);
            RTXDI_StreamLocalLightAtUVIntoReservoir(
                rng, misData, surface, sampleParams.brdfCutoff,
                misData.localLightMisWeight, lightIndex, uv, invSourcePdf,
                lightInfo, localReservoir, localSample);
        }

        RTXDI_FinalizeResampling(localReservoir, 1.0, misData.numMisSamples);
        localReservoir.M = 1;
    }

    // --- Infinite lights ---
    RAB_LightSample infiniteSample = RAB_EmptyLightSample();
    RTXDI_DIReservoir infiniteReservoir = RTXDI_SampleInfiniteLights(
        rng, surface,
        sampleParams.numInfiniteLightSamples,
        lightBufferParams.infiniteLightBufferRegion,
        infiniteSample);

#ifdef RTXDI_ENABLE_PRESAMPLING
    // --- Environment map ---
    RAB_LightSample environmentSample = RAB_EmptyLightSample();
    RTXDI_DIReservoir environmentReservoir = RTXDI_SampleEnvironmentMap(
        rng, tileRng, surface,
        sampleParams, misData,
        lightBufferParams.environmentLightParams,
        g_Const.environmentLightRISBufferSegmentParams,
        environmentSample);
#endif // RTXDI_ENABLE_PRESAMPLING

    // --- BRDF samples ---
    RAB_LightSample brdfSample = RAB_EmptyLightSample();
    RTXDI_DIReservoir brdfReservoir = RTXDI_SampleBrdf(
        rng, surface,
        sampleParams.numBrdfSamples, sampleParams.brdfCutoff, sampleParams.brdfRayMinT,
        misData, tileRng, lightBufferParams, brdfSample);

    // --- Combine reservoirs ---
    RTXDI_DIReservoir reservoir = RTXDI_EmptyDIReservoir();
    RTXDI_CombineDIReservoirs(reservoir, localReservoir, 0.5, localReservoir.targetPdf);
    bool selectInfinite = RTXDI_CombineDIReservoirs(reservoir, infiniteReservoir, RTXDI_GetNextRandom(rng), infiniteReservoir.targetPdf);
#ifdef RTXDI_ENABLE_PRESAMPLING
    bool selectEnvironment = RTXDI_CombineDIReservoirs(reservoir, environmentReservoir, RTXDI_GetNextRandom(rng), environmentReservoir.targetPdf);
#endif // RTXDI_ENABLE_PRESAMPLING
    bool selectBrdf = RTXDI_CombineDIReservoirs(reservoir, brdfReservoir, RTXDI_GetNextRandom(rng), brdfReservoir.targetPdf);

    RTXDI_FinalizeResampling(reservoir, 1.0, 1.0);
    reservoir.M = 1;

    // --- Pick the light sample that matches the selected reservoir ---
    if (selectBrdf)
        lightSample = brdfSample;
    else
#ifdef RTXDI_ENABLE_PRESAMPLING
    if (selectEnvironment)
        lightSample = environmentSample;
    else
#endif // RTXDI_ENABLE_PRESAMPLING
    if (selectInfinite)
        lightSample = infiniteSample;
    else
        lightSample = localSample;

    // --- Optional initial visibility ---
    if (sampleParams.enableInitialVisibility && RTXDI_IsValidDIReservoir(reservoir))
    {
        if (!RAB_GetConservativeVisibility(surface, lightSample))
        {
            RTXDI_StoreVisibilityInDIReservoir(reservoir, 0, true);
        }
    }
    // ---------------------------------------------------------------------
    // End inlined RTXDI_SampleLightsForSurface
    // ---------------------------------------------------------------------

    RTXDI_StoreDIReservoir(reservoir, g_Const.restirDI.reservoirBufferParams, GlobalIndex, g_Const.restirDI.bufferIndices.initialSamplingOutputBufferIndex);

    if (RTXDI_IsValidDIReservoir(reservoir)){
        u_DirectLightingRaw[pixelPosition] = float4(0, 0, 0, 0);
    }else{
        u_DirectLightingRaw[pixelPosition] = float4(1, 0, 0, 0);
    }
}