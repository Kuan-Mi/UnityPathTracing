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

#ifndef RTXDI_DI_TEMPORAL_RESAMPLING_HLSLI
#define RTXDI_DI_TEMPORAL_RESAMPLING_HLSLI

#include "Assets/Shaders/Rtxdi/DI/Reservoir.hlsl"
#include <Assets/Shaders/Rtxdi/DI/ReservoirStorage.hlsl>
#include "Assets/Shaders/Rtxdi/Utils/Checkerboard.hlsl"

// Temporal resampling pass.
// Takes the previous G-buffer, motion vectors, and two light reservoir buffers as inputs.
// Tries to match the surfaces in the current frame to surfaces in the previous frame.
// If a match is found for a given pixel, the current and previous reservoirs are 
// combined. An optional visibility ray may be cast if enabled, to reduce the resampling bias.
// That visibility ray should ideally be traced through the previous frame BVH, but
// can also use the current frame BVH if the previous is not available - that will produce more bias.
// The selectedLightSample parameter is used to update and return the selected sample; it's optional,
// and it's safe to pass a null structure there and ignore the result.
RTXDI_DIReservoir RTXDI_DITemporalResampling(
    uint2 pixelPosition,
    RAB_Surface surface,
    RTXDI_DIReservoir curSample,
    inout RTXDI_RandomSamplerState rng,
    RTXDI_RuntimeParameters params,
    RTXDI_ReservoirBufferParameters reservoirParams,
	float3 screenSpaceMotion,
	uint sourceBufferIndex,
    RTXDI_DITemporalResamplingParameters tparams,
    out int2 temporalSamplePixelPos,
    inout RAB_LightSample selectedLightSample)
{
    uint historyLimit = min(RTXDI_PackedDIReservoir_MaxM, uint(tparams.maxHistoryLength * curSample.M));

    int selectedLightPrevID = -1;

    if (RTXDI_IsValidDIReservoir(curSample))
    {
        selectedLightPrevID = RAB_TranslateLightIndex(RTXDI_GetDIReservoirLightIndex(curSample), true);
    }

    temporalSamplePixelPos = int2(-1, -1);

    RTXDI_DIReservoir state = RTXDI_EmptyDIReservoir();
    RTXDI_CombineDIReservoirs(state, curSample, /* random = */ 0.5, curSample.targetPdf);

    // Backproject this pixel to last frame
    float3 motion = screenSpaceMotion;
    
    if (!tparams.enablePermutationSampling)
    {
        motion.xy += float2(RTXDI_GetNextRandom(rng), RTXDI_GetNextRandom(rng)) - 0.5;
    }

    float2 reprojectedSamplePosition = float2(pixelPosition) + motion.xy;
    int2 prevPos = int2(round(reprojectedSamplePosition));

    float expectedPrevLinearDepth = RAB_GetSurfaceLinearDepth(surface) + motion.z;

    RAB_Surface temporalSurface = RAB_EmptySurface();
    bool foundNeighbor = false;
    const float radius = (params.activeCheckerboardField == 0) ? 4 : 8;
    int2 spatialOffset = int2(0, 0);

    // Try to find a matching surface in the neighborhood of the reprojected pixel
    for(int i = 0; i < 9; i++)
    {
        int2 offset = int2(0, 0);
        if(i > 0)
        {
            offset.x = int((RTXDI_GetNextRandom(rng) - 0.5) * radius);
            offset.y = int((RTXDI_GetNextRandom(rng) - 0.5) * radius);
        }

        int2 idx = prevPos + offset;
        if (tparams.enablePermutationSampling && i == 0)
        {
            RTXDI_ApplyPermutationSampling(idx, tparams.uniformRandomNumber);
        }

        RTXDI_ActivateCheckerboardPixel(idx, true, params.activeCheckerboardField);

        // Grab shading / g-buffer data from last frame
        temporalSurface = RAB_GetGBufferSurface(idx, true);
        if (!RAB_IsSurfaceValid(temporalSurface))
            continue;
        
        // Test surface similarity, discard the sample if the surface is too different.
        if (!RTXDI_IsValidNeighbor(
            RAB_GetSurfaceNormal(surface), RAB_GetSurfaceNormal(temporalSurface), 
            expectedPrevLinearDepth, RAB_GetSurfaceLinearDepth(temporalSurface), 
            tparams.normalThreshold, tparams.depthThreshold))
            continue;

        spatialOffset = idx - prevPos;
        prevPos = idx;
        foundNeighbor = true;

        break;
    }

    bool selectedPreviousSample = false;
    float previousM = 0;

    if (foundNeighbor)
    {
        // Resample the previous frame sample into the current reservoir, but reduce the light's weight
        // according to the bilinear weight of the current pixel
        uint2 prevReservoirPos = RTXDI_PixelPosToReservoirPos(prevPos, params.activeCheckerboardField);
        RTXDI_DIReservoir prevSample = RTXDI_LoadDIReservoir(reservoirParams,
            prevReservoirPos, sourceBufferIndex);
        prevSample.M = min(prevSample.M, historyLimit);
        prevSample.spatialDistance += spatialOffset;
        prevSample.age += 1;

        uint originalPrevLightID = RTXDI_GetDIReservoirLightIndex(prevSample);

        // Map the light ID from the previous frame into the current frame, if it still exists
        if (RTXDI_IsValidDIReservoir(prevSample))
        {
            if (prevSample.age <= 1)
            {
                temporalSamplePixelPos = prevPos;
            }

            int mappedLightID = RAB_TranslateLightIndex(RTXDI_GetDIReservoirLightIndex(prevSample), false);

            if (mappedLightID < 0)
            {
                // Kill the reservoir
                prevSample.weightSum = 0;
                prevSample.lightData = 0;
            }
            else
            {
                // Sample is valid - modify the light ID stored
                prevSample.lightData = mappedLightID | RTXDI_DIReservoir_LightValidBit;
            }
        }

        previousM = prevSample.M;

        float weightAtCurrent = 0;
        RAB_LightSample candidateLightSample = RAB_EmptyLightSample();
        if (RTXDI_IsValidDIReservoir(prevSample))
        {
            const RAB_LightInfo candidateLight = RAB_LoadLightInfo(RTXDI_GetDIReservoirLightIndex(prevSample), false);

            candidateLightSample = RAB_SamplePolymorphicLight(
                candidateLight, surface, RTXDI_GetDIReservoirSampleUV(prevSample));

            weightAtCurrent = RAB_GetLightSampleTargetPdfForSurface(candidateLightSample, surface);
        }

        bool sampleSelected = RTXDI_CombineDIReservoirs(state, prevSample, RTXDI_GetNextRandom(rng), weightAtCurrent);
        if(sampleSelected)
        {
            selectedPreviousSample = true;
            selectedLightPrevID = int(originalPrevLightID);
            selectedLightSample = candidateLightSample;
        }
    }

#if RTXDI_ALLOWED_BIAS_CORRECTION >= RTXDI_BIAS_CORRECTION_BASIC
    if (tparams.biasCorrectionMode >= RTXDI_BIAS_CORRECTION_BASIC)
    {
        // Compute the unbiased normalization term (instead of using 1/M)
        float pi = state.targetPdf;
        float piSum = state.targetPdf * curSample.M;
        
        if (RTXDI_IsValidDIReservoir(state) && selectedLightPrevID >= 0 && previousM > 0)
        {
            float temporalP = 0;

            const RAB_LightInfo selectedLightPrev = RAB_LoadLightInfo(selectedLightPrevID, true);

            // Get the PDF of the sample RIS selected in the first loop, above, *at this neighbor* 
            const RAB_LightSample selectedSampleAtTemporal = RAB_SamplePolymorphicLight(
                selectedLightPrev, temporalSurface, RTXDI_GetDIReservoirSampleUV(state));
        
            temporalP = RAB_GetLightSampleTargetPdfForSurface(selectedSampleAtTemporal, temporalSurface);

#if RTXDI_ALLOWED_BIAS_CORRECTION >= RTXDI_BIAS_CORRECTION_RAY_TRACED
            if (tparams.biasCorrectionMode == RTXDI_BIAS_CORRECTION_RAY_TRACED && temporalP > 0 && (!selectedPreviousSample || !tparams.enableVisibilityShortcut))
            {
                if (!RAB_GetTemporalConservativeVisibility(surface, temporalSurface, selectedSampleAtTemporal))
                {
                    temporalP = 0;
                }
            }
#endif

            pi = selectedPreviousSample ? temporalP : pi;
            piSum += temporalP * previousM;
        }

        RTXDI_FinalizeResampling(state, pi, piSum);
    }
    else
#endif
    {
        RTXDI_FinalizeResampling(state, 1.0, state.M);
    }

    return state;
}

#endif // RTXDI_DI_TEMPORAL_RESAMPLING_HLSLI