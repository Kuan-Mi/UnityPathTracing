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

    // Per-pixel statistics for the local-light sampling loop.
    uint  dbg_loopCount        = 0;   // number of local-light samples attempted
    uint  dbg_invSrcPdfZero    = 0;   // candidates with invSourcePdf <= 0 (bad tile entry)
    uint  dbg_blendedPdfZero   = 0;   // candidates skipped by MIS (blendedSourcePdf == 0)
    uint  dbg_targetPdfZero    = 0;   // candidates whose target pdf was 0 for this surface
    uint  dbg_streamSelected   = 0;   // times RTXDI_StreamSample picked a new candidate
    float dbg_maxTargetPdf     = 0.0; // largest targetPdf seen across all candidates
    float dbg_maxInvSourcePdf  = 0.0; // largest invSourcePdf seen (rough cell occupancy hint)

    // Extra ReGIR-specific diagnostics
    uint  dbg_firstLightIndex   = 0xFFFFFFFFu; // index of the first candidate light
    uint  dbg_uniqueLightCount  = 0;           // how many distinct lightIndex values were seen
    uint  dbg_lastLightIndex    = 0xFFFFFFFFu;
    uint  dbg_radianceNonZero   = 0;           // candidates whose sampled radiance != 0
    uint  dbg_nDotLNonPos       = 0;           // candidates with N.L <= 0 (back-facing light dir)
    float dbg_maxRadiance       = 0.0;

    // Even deeper "purple" diagnostics: why is every candidate radiance == 0?
    uint  dbg_lightIndexOOB     = 0;  // lightIndex >= localLightBufferRegion's range
    uint  dbg_invalidLightInfo  = 0;  // lightInfo itself looks empty / zero radiance source
    uint  dbg_zeroFromSampling  = 0;  // lightInfo OK but RAB_SamplePolymorphicLight returned 0
    uint  dbg_minLightIndex     = 0xFFFFFFFFu;
    uint  dbg_maxLightIndex     = 0;

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

        // ----- Sampling loop (inlined from RTXDI_SampleLocalLightsInternal,
        //       further inlined RTXDI_StreamLocalLightAtUVIntoReservoir so
        //       we can observe per-candidate quantities) -----
        dbg_loopCount        = sampleParams.numLocalLightSamples;
        dbg_invSrcPdfZero    = 0;
        dbg_blendedPdfZero   = 0;
        dbg_targetPdfZero    = 0;
        dbg_streamSelected   = 0;
        dbg_maxTargetPdf     = 0.0;
        dbg_maxInvSourcePdf  = 0.0;

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

            if (invSourcePdf <= 0.0) { dbg_invSrcPdfZero++; }
            dbg_maxInvSourcePdf = max(dbg_maxInvSourcePdf, invSourcePdf);

            float2 uv = RTXDI_RandomlySelectLocalLightUV(rng);

            // --- Inlined RTXDI_StreamLocalLightAtUVIntoReservoir ---
            RAB_LightSample candidateSample = RAB_SamplePolymorphicLight(lightInfo, surface, uv);
            float blendedSourcePdf = RTXDI_LightBrdfMisWeight(
                surface, candidateSample, 1.0 / invSourcePdf,
                misData.localLightMisWeight, false,
                misData.brdfMisWeight, sampleParams.brdfCutoff);
            float targetPdf = RAB_GetLightSampleTargetPdfForSurface(candidateSample, surface);
            float risRnd    = RTXDI_GetNextRandom(rng);

            dbg_maxTargetPdf = max(dbg_maxTargetPdf, targetPdf);
            if (targetPdf == 0.0)         dbg_targetPdfZero++;
            if (blendedSourcePdf == 0.0)  { dbg_blendedPdfZero++; continue; }

            // ----- ReGIR per-candidate probes -----
            if (i == 0) dbg_firstLightIndex = lightIndex;
            if (lightIndex != dbg_lastLightIndex) { dbg_uniqueLightCount++; dbg_lastLightIndex = lightIndex; }

            dbg_minLightIndex = min(dbg_minLightIndex, lightIndex);
            dbg_maxLightIndex = max(dbg_maxLightIndex, lightIndex);

            // Is lightIndex inside the local light buffer region?
            {
                uint first = lightBufferParams.localLightBufferRegion.firstLightIndex;
                uint last  = first + lightBufferParams.localLightBufferRegion.numLights;
                if (lightIndex < first || lightIndex >= last)
                    dbg_lightIndexOOB++;
            }

            // Re-sample with a fixed uv at the light center to factor out uv randomness.
            // If radiance is still 0 with uv=(0.5,0.5), the lightInfo itself is dead.
            {
                RAB_LightSample centerSample = RAB_SamplePolymorphicLight(lightInfo, surface, float2(0.5, 0.5));
                float centerR = max(centerSample.radiance.x,
                                    max(centerSample.radiance.y, centerSample.radiance.z));
                float candR2  = max(candidateSample.radiance.x,
                                    max(candidateSample.radiance.y, candidateSample.radiance.z));
                if (centerR == 0.0 && candR2 == 0.0)
                    dbg_invalidLightInfo++;
                else if (candR2 == 0.0)
                    dbg_zeroFromSampling++;
            }

            float candR = max(candidateSample.radiance.x,
                              max(candidateSample.radiance.y, candidateSample.radiance.z));
            dbg_maxRadiance = max(dbg_maxRadiance, candR);
            if (candR > 0.0) dbg_radianceNonZero++;

            // Check N.L sign (does the cell-chosen light even sit on the front side?)
            {
                float3 lDir; float lDist;
                RAB_GetLightDirDistance(surface, candidateSample, lDir, lDist);
                if (dot(lDir, surface.geoNormal) <= 0.0) dbg_nDotLNonPos++;
            }

            bool selected = RTXDI_StreamSample(localReservoir, lightIndex, uv, risRnd,
                                               targetPdf, 1.0 / blendedSourcePdf);
            if (selected)
            {
                dbg_streamSelected++;
                localSample = candidateSample;
            }
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

    // Only visualize the previously-purple region:
    //   ReGIR cell hit, every candidate had radiance == 0.
    // Everything else stays black.
    if (!RTXDI_IsValidDIReservoir(reservoir) &&
        dbg_localSampleMode == 2 &&
        dbg_loopCount > 0 &&
        dbg_blendedPdfZero != dbg_loopCount &&
        dbg_targetPdfZero  == dbg_loopCount &&
        dbg_radianceNonZero == 0)
    {
        if (dbg_lightIndexOOB == dbg_loopCount)
            // ALL candidate lightIndex values are outside the local light range.
            // -> Presample is writing wrong indices, OR shader is reading from the
            //    wrong tile / wrong buffer offset (risBufferOffset / lightsPerCell
            //    mismatch between CPU and GPU side).
            u_DirectLightingRaw[pixelPosition] = float4(1, 0, 0, 0);     // red
        else if (dbg_lightIndexOOB > 0)
            // Mixed: some valid, some OOB. Still indicates a tile-layout bug.
            u_DirectLightingRaw[pixelPosition] = float4(1, 0.5, 0, 0);   // orange
        else if (dbg_invalidLightInfo == dbg_loopCount)
            // lightIndex is in range, but the loaded RAB_LightInfo is "dead":
            // even with uv=(0.5,0.5) the radiance is 0.
            // -> Presample picked lights that exist in the buffer but are off
            //    (zero intensity / disabled / wrong type).
            u_DirectLightingRaw[pixelPosition] = float4(0.5, 0, 0.5, 0); // purple
        else if (dbg_zeroFromSampling == dbg_loopCount)
            // lightInfo is alive at uv=(0.5,0.5), but every random uv produced
            // radiance == 0. Unusual: triangle/mesh light with degenerate area,
            // or a uv-dependent mask.
            u_DirectLightingRaw[pixelPosition] = float4(1, 0, 1, 0);     // magenta
        else
            // Some other mix (e.g. some candidates alive but masked out).
            u_DirectLightingRaw[pixelPosition] = float4(1, 1, 1, 0);     // white
    }
    else
    {
        u_DirectLightingRaw[pixelPosition] = float4(0, 0, 0, 0);
    }
}