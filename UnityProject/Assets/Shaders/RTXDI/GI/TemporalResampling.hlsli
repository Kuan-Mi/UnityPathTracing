/***************************************************************************
 # Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

#ifndef RTXDI_GI_TEMPORAL_RESAMPLING_HLSLI
#define RTXDI_GI_TEMPORAL_RESAMPLING_HLSLI

#include "Rtxdi/GI/JacobianMath.hlsli"
#include "Rtxdi/GI/Reservoir.hlsli"
#include "Rtxdi/Utils/Checkerboard.hlsli"
#include "Rtxdi/Utils/ReservoirAddressing.hlsli"

// Enabled by default. Application code need to define those macros appropriately to optimize shaders.
#ifndef RTXDI_GI_ALLOWED_BIAS_CORRECTION
#define RTXDI_GI_ALLOWED_BIAS_CORRECTION RTXDI_BIAS_CORRECTION_RAY_TRACED
#endif

 // Generates a pattern of offsets for looking closely around a given pixel.
// The pattern places 'sampleIdx' at the following locations in screen space around pixel (x):
//   0 4 3
//   6 x 7
//   2 5 1
int2 RTXDI_CalculateTemporalResamplingOffset(int sampleIdx, int radius)
{
    sampleIdx &= 7;

    int mask2 = sampleIdx >> 1 & 0x01;       // 0, 0, 1, 1, 0, 0, 1, 1
    int mask4 = 1 - (sampleIdx >> 2 & 0x01); // 1, 1, 1, 1, 0, 0, 0, 0
    int tmp0 = -1 + 2 * (sampleIdx & 0x01);  // -1, 1,....
    int tmp1 = 1 - 2 * mask2;                // 1, 1,-1,-1, 1, 1,-1,-1
    int tmp2 = mask4 | mask2;                // 1, 1, 1, 1, 0, 0, 1, 1
    int tmp3 = mask4 | (1 - mask2);          // 1, 1, 1, 1, 1, 1, 0, 0

    return int2(tmp0, tmp0 * tmp1) * int2(tmp2, tmp3) * radius;
}

// A structure that groups the application-provided settings for spatio-temporal resampling.
struct RTXDI_GITemporalResamplingParameters
{
    // Screen-space motion vector, computed as (previousPosition - currentPosition).
    // The X and Y components are measured in pixels.
    // The Z component is in linear depth units.
    float3 screenSpaceMotion;

    // The index of the reservoir buffer to pull the temporal samples from.
    uint sourceBufferIndex;

    // Maximum history length for reuse, measured in frames.
    // Higher values result in more stable and high quality sampling, at the cost of slow reaction to changes.
    uint maxHistoryLength;

    // Controls the bias correction math for temporal reuse. Depending on the setting, it can add
    // some shader cost and one approximate shadow ray per pixel (or per two pixels if checkerboard sampling is enabled).
    // Ideally, these rays should be traced through the previous frame's BVH to get fully unbiased results.
    uint biasCorrectionMode;

    // Surface depth similarity threshold for temporal reuse.
    // If the previous frame surface's depth is within this threshold from the current frame surface's depth,
    // the surfaces are considered similar. The threshold is relative, i.e. 0.1 means 10% of the current depth.
    // Otherwise, the pixel is not reused, and the resampling shader will look for a different one.
    float depthThreshold;

    // Surface normal similarity threshold for temporal reuse.
    // If the dot product of two surfaces' normals is higher than this threshold, the surfaces are considered similar.
    // Otherwise, the pixel is not reused, and the resampling shader will look for a different one.
    float normalThreshold;

    // Discard the reservoir if its age exceeds this value.
    uint maxReservoirAge;

    // Enables permuting the pixels sampled from the previous frame in order to add temporal
    // variation to the output signal and make it more denoiser friendly.
    bool enablePermutationSampling;

    // Enables resampling from a location around the current pixel instead of what the motion vector points at,
    // in case no surface near the motion vector matches the current surface (e.g. disocclusion).
    // This behavoir makes disocclusion areas less noisy but locally biased, usually darker.
    bool enableFallbackSampling;

    // Random number for permutation sampling that is the same for all pixels in the frame
    uint uniformRandomNumber;
};

// Temporal resampling for GI reservoir pass.
RTXDI_GIReservoir RTXDI_GITemporalResampling(
    const uint2 pixelPosition,
    const RAB_Surface surface,
    const RTXDI_GIReservoir inputReservoir,
    inout RAB_RandomSamplerState rng,
    const RTXDI_RuntimeParameters params,
    const RTXDI_ReservoirBufferParameters reservoirParams,
    const RTXDI_GITemporalResamplingParameters tparams)
{
    // Backproject this pixel to last frame
    int2 prevPos = int2(round(float2(pixelPosition) + tparams.screenSpaceMotion.xy));
    const float expectedPrevLinearDepth = RAB_GetSurfaceLinearDepth(surface) + tparams.screenSpaceMotion.z;
    const int radius = (params.activeCheckerboardField == 0) ? 1 : 2;

    RTXDI_GIReservoir temporalReservoir;
    bool foundTemporalReservoir = false;

    const int temporalSampleStartIdx = int(RAB_GetNextRandom(rng) * 8);

    RAB_Surface temporalSurface = RAB_EmptySurface();

    // Try to find a matching surface in the neighborhood of the reprojected pixel
    const int temporalSampleCount = 5;
    const int sampleCount = temporalSampleCount + (tparams.enableFallbackSampling ? 1 : 0);
    for (int i = 0; i < sampleCount; i++)
    {
        const bool isFirstSample = i == 0;
        const bool isFallbackSample = i == temporalSampleCount;

        int2 offset = int2(0, 0);
        if (isFallbackSample)
        {
            // Last sample is a fallback for disocclusion areas: use zero motion vector.
            prevPos = int2(pixelPosition);
        }
        else if (!isFirstSample)
        {
            offset = RTXDI_CalculateTemporalResamplingOffset(temporalSampleStartIdx + i, radius);
        }

        int2 idx = prevPos + offset;
        if ((tparams.enablePermutationSampling && isFirstSample) || isFallbackSample)
        {
            // Apply permutation sampling for the first (non-jittered) sample,
            // also for the last (fallback) sample to prevent visible repeating patterns in disocclusions.
            RTXDI_ApplyPermutationSampling(idx, tparams.uniformRandomNumber);
        }

        RTXDI_ActivateCheckerboardPixel(idx, true, params.activeCheckerboardField);
        
        // Grab shading / g-buffer data from last frame
        temporalSurface = RAB_GetGBufferSurface(idx, true);

        if (!RAB_IsSurfaceValid(temporalSurface))
        {
            continue;
        }

        // Test surface similarity, discard the sample if the surface is too different.
        // Skip this test for the last (fallback) sample.
        if (!isFallbackSample && !RTXDI_IsValidNeighbor(
            RAB_GetSurfaceNormal(surface), RAB_GetSurfaceNormal(temporalSurface),
            expectedPrevLinearDepth, RAB_GetSurfaceLinearDepth(temporalSurface),
            tparams.normalThreshold, tparams.depthThreshold))
        {
            continue;
        }

        // Test material similarity and perform any other app-specific tests.
        if (!RAB_AreMaterialsSimilar(RAB_GetMaterial(surface), RAB_GetMaterial(temporalSurface)))
        {
            continue;
        }

        // Read temporal reservoir.
        uint2 prevReservoirPos = RTXDI_PixelPosToReservoirPos(idx, params.activeCheckerboardField);
        temporalReservoir = RTXDI_LoadGIReservoir(reservoirParams, prevReservoirPos, tparams.sourceBufferIndex);

        // Check if the reservoir is a valid one.
        if (!RTXDI_IsValidGIReservoir(temporalReservoir))
        {
            continue;
        }

        foundTemporalReservoir = true;
        break;
    }

    RTXDI_GIReservoir curReservoir = RTXDI_EmptyGIReservoir();

    float selectedTargetPdf = 0;
    if (RTXDI_IsValidGIReservoir(inputReservoir))
    {
        selectedTargetPdf = RAB_GetGISampleTargetPdfForSurface(inputReservoir.position, inputReservoir.radiance, surface);

        RTXDI_CombineGIReservoirs(curReservoir, inputReservoir, /* random = */ 0.5, selectedTargetPdf);
    }
    
    if (foundTemporalReservoir)
    {
        // Found a valid temporal surface and its GI reservoir.

        // Calculate Jacobian determinant to adjust weight.
        float jacobian = RTXDI_CalculateJacobian(RAB_GetSurfaceWorldPos(surface), RAB_GetSurfaceWorldPos(temporalSurface), temporalReservoir);

        if (!RAB_ValidateGISampleWithJacobian(jacobian))
            foundTemporalReservoir = false;

        temporalReservoir.weightSum *= jacobian;
        
        // Clamp history length
        temporalReservoir.M = min(temporalReservoir.M, tparams.maxHistoryLength);

        // Make the sample older
        ++temporalReservoir.age;

        if (temporalReservoir.age > tparams.maxReservoirAge)
            foundTemporalReservoir = false;
    }

    bool selectedPreviousSample = false;
    if (foundTemporalReservoir)
    {
        // Reweighting and denormalize the temporal sample with the current surface.
        float targetPdf = RAB_GetGISampleTargetPdfForSurface(temporalReservoir.position, temporalReservoir.radiance, surface);
        
        // Combine the temporalReservoir into the curReservoir
        selectedPreviousSample = RTXDI_CombineGIReservoirs(curReservoir, temporalReservoir, RAB_GetNextRandom(rng), targetPdf);
        if (selectedPreviousSample)
        {
            selectedTargetPdf = targetPdf;
        }
    }

#if RTXDI_GI_ALLOWED_BIAS_CORRECTION >= RTXDI_BIAS_CORRECTION_BASIC
    if (tparams.biasCorrectionMode >= RTXDI_BIAS_CORRECTION_BASIC)
    {
        float pi = selectedTargetPdf;
        float piSum = selectedTargetPdf * inputReservoir.M;

        if (RTXDI_IsValidGIReservoir(curReservoir) && foundTemporalReservoir)
        {
            float temporalP = RAB_GetGISampleTargetPdfForSurface(curReservoir.position, curReservoir.radiance, temporalSurface);

#if RTXDI_GI_ALLOWED_BIAS_CORRECTION >= RTXDI_BIAS_CORRECTION_RAY_TRACED
            if (tparams.biasCorrectionMode == RTXDI_BIAS_CORRECTION_RAY_TRACED && temporalP > 0)
            {
                if (!RAB_GetTemporalConservativeVisibility(surface, temporalSurface, curReservoir.position))
                {
                    temporalP = 0;
                }
            }
#endif

            pi = selectedPreviousSample ? temporalP : pi;
            piSum += temporalP * temporalReservoir.M;
        }

        // Normalizing
        float normalizationNumerator = pi;
        float normalizationDenominator = piSum * selectedTargetPdf;
        RTXDI_FinalizeGIResampling(curReservoir, normalizationNumerator, normalizationDenominator);
    }
    else
#endif
    {
        // Normalizing
        float normalizationNumerator = 1.0;
        float normalizationDenominator = selectedTargetPdf * curReservoir.M;
        RTXDI_FinalizeGIResampling(curReservoir, normalizationNumerator, normalizationDenominator);
    }

    return curReservoir;
}

#endif // RTXDI_GI_TEMPORAL_RESAMPLING_HLSLI
