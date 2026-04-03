/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef RTXDI_PT_TEMPORAL_RESAMPLING_HLSLI
#define RTXDI_PT_TEMPORAL_RESAMPLING_HLSLI

#include "Assets/Shaders/Rtxdi/Utils/Math.hlsl"
#define RTXDI_RESTIR_PT_HYBRID_SHIFT
#include "Assets/Shaders/Rtxdi/PT/HybridShift.hlsl"
#include "Assets/Shaders/Rtxdi/Utils/Checkerboard.hlsl"

#define NEIGHBOR_FIND_MODE_GI 1
#define NEIGHBOR_FIND_MODE_IDENTITY 2

#define NEIGHBOR_FIND_MODE NEIGHBOR_FIND_MODE_GI

struct RTXDI_PTTemporalResamplingRuntimeParameters
{
    uint2 pixelPosition;
    uint2 reservoirPosition;
    float3 motionVector;

    float3 cameraPos;
    float3 prevCameraPos;
    float3 prevPrevCameraPos;
};

RTXDI_PTTemporalResamplingRuntimeParameters RTXDI_EmptyPTTemporalResamplingRuntimeParameters()
{
    return (RTXDI_PTTemporalResamplingRuntimeParameters)0;
}

bool IsValidNeighborSurface(RAB_Surface surfA, RAB_Surface surfB, float normalThreshold, float depthThreshold)
{
    return RTXDI_IsValidNeighbor(RAB_GetSurfaceNormal(surfA), RAB_GetSurfaceNormal(surfB), RAB_GetSurfaceLinearDepth(surfA), RAB_GetSurfaceLinearDepth(surfB), normalThreshold, depthThreshold);
}

#if NEIGHBOR_FIND_MODE == NEIGHBOR_FIND_MODE_GI

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

int2 CalculateTemporalSurfacePosition(RTXDI_PTTemporalResamplingParameters tParams, RTXDI_RuntimeParameters params, int i, int temporalSampleStartIdx, int2 pixelPos, float radius, int2 prevPos, bool isFallbackSample)
{
    const bool isFirstSample = i == 0;
    int2 offset = int2(0, 0);
    if (isFallbackSample)
    {
        // Last sample is a fallback for disocclusion areas: use zero motion vector.
        prevPos = int2(pixelPos);
    }
    else if (!isFirstSample)
    {
        offset = RTXDI_CalculateTemporalResamplingOffset(temporalSampleStartIdx + i, radius);
    }

    int2 idx = prevPos + offset;
    if ((tParams.enablePermutationSampling && isFirstSample) || isFallbackSample)
    {
        // Apply permutation sampling for the first (non-jittered) sample,
        // also for the last (fallback) sample to prevent visible repeating patterns in disocclusions.
        RTXDI_ApplyPermutationSampling(idx, tParams.uniformRandomNumber);
    }

    RTXDI_ActivateCheckerboardPixel(idx, true, params.activeCheckerboardField);

    return idx;
}

bool ValidTemporalSurface(RTXDI_PTTemporalResamplingParameters tParams,RAB_Surface surface, RAB_Surface temporalSurface, float expectedPrevLinearDepth, bool isFallbackSample)
{
    if (!RAB_IsSurfaceValid(temporalSurface))
    {
        return false;
    }

    if (!isFallbackSample && !RTXDI_IsValidNeighbor(
        RAB_GetSurfaceNormal(surface), RAB_GetSurfaceNormal(temporalSurface),
        expectedPrevLinearDepth, RAB_GetSurfaceLinearDepth(temporalSurface),
        tParams.normalThreshold, tParams.depthThreshold))
    {
        return false;
    }

    if (!RAB_AreMaterialsSimilar(RAB_GetMaterial(surface), RAB_GetMaterial(temporalSurface)))
    {
        return false;
    }

    return true;
}

bool FindTemporalNeighbor_GI(const RTXDI_PTTemporalResamplingParameters tParams,
                             const RTXDI_RuntimeParameters params,
                             const RTXDI_ReservoirBufferParameters reservoirBufferParams,
                             const RTXDI_PTBufferIndices bufferIndices,
                             inout RTXDI_RandomSamplerState rng,
                             const int2 pixelPos,
                             const int2 prevPos,
                             const RAB_Surface Surface,
                             const float expectedPrevLinearDepth,
                             inout RAB_Surface temporalSurface,
                             inout RTXDI_PTReservoir temporalReservoir)
{
    // Try to find a matching surface in the neighborhood of the reprojected pixel
    const int temporalSampleCount = 5;
    const int sampleCount = temporalSampleCount + (tParams.enableFallbackSampling ? 1 : 0);
    const int radius = (params.activeCheckerboardField == 0) ? 1 : 2;
    const int temporalSampleStartIdx = int(RTXDI_GetNextRandom(rng) * 8);
    for (int i = 0; i < sampleCount; i++)
    {
        const bool isFallbackSample = i == temporalSampleCount;

        int2 temporalSurfacePos = CalculateTemporalSurfacePosition(tParams, params, i, temporalSampleStartIdx, pixelPos, radius, prevPos, isFallbackSample);
        temporalSurface = RAB_GetGBufferSurface(temporalSurfacePos, true);

        if(!ValidTemporalSurface(tParams, Surface, temporalSurface, expectedPrevLinearDepth, isFallbackSample))
        {
            continue;
        }

        uint2 prevReservoirPos = RTXDI_PixelPosToReservoirPos(temporalSurfacePos, params.activeCheckerboardField);
        temporalReservoir = RTXDI_LoadPTReservoir(reservoirBufferParams, prevReservoirPos, bufferIndices.temporalResamplingInputBufferIndex);

        if (!RTXDI_IsValidPTReservoir(temporalReservoir))
        {
            continue;
        }

        return true;
    }

    return false;
}

#elif NEIGHBOR_FIND_MODE == NEIGHBOR_FIND_MODE_IDENTITY

bool FindTemporalNeighbor_Identity(RTXDI_RuntimeParameters runtimeParams,
                                   RTXDI_ReservoirBufferParameters reservoirBufferParams,
                                   RTXDI_PTBufferIndices bufferIndices,
                                   inout int2 prevReservoirPos,
                                   const int2 pixelPosition,
                                   const RAB_Surface curSurface,
                                   inout RAB_Surface temporalSurface,
                                   inout RTXDI_PTReservoir temporalReservoir)
{
    int2 temporalSampleIdx = pixelPosition;

    temporalSurface = RAB_GetGBufferSurface(temporalSampleIdx, true);
    if(!RAB_IsSurfaceValid(temporalSurface))
    {
        return false;
    }

    prevReservoirPos = RTXDI_PixelPosToReservoirPos(temporalSampleIdx, runtimeParams.activeCheckerboardField);
    temporalReservoir = RTXDI_LoadPTReservoir(reservoirBufferParams, prevReservoirPos, bufferIndices.temporalResamplingInputBufferIndex);
    if (!RTXDI_IsValidPTReservoir(temporalReservoir))
    {
        return false;
    }

    return true;
}

#endif

bool FindTemporalNeighbor(RTXDI_PTTemporalResamplingParameters tParams, RTXDI_RuntimeParameters params, RTXDI_ReservoirBufferParameters bufferParams, RTXDI_PTBufferIndices bufferIndices, inout RTXDI_RandomSamplerState rng, int2 pixelPos, int2 prevPos, RAB_Surface Surface, float expectedPrevLinearDepth, inout RAB_Surface temporalSurface, inout RTXDI_PTReservoir PrevSample)
{
#if NEIGHBOR_FIND_MODE == NEIGHBOR_FIND_MODE_GI
    return FindTemporalNeighbor_GI(tParams, params, bufferParams, bufferIndices, rng, pixelPos, prevPos, Surface, expectedPrevLinearDepth, temporalSurface, PrevSample);
#elif NEIGHBOR_FIND_MODE == NEIGHBOR_FIND_MODE_IDENTITY
    return FindTemporalNeighbor_Identity(params, bufferParams, bufferIndices, prevPos, pixelPos, Surface, temporalSurface, PrevSample);
#endif
}

bool ValidateTemporalNeighbor(const RTXDI_PTTemporalResamplingParameters tParams,
                              inout RTXDI_PTReservoir PrevSample,
                              const RAB_Surface Surface,
                              const RAB_Surface PrevSurface,
                              inout int ReducedMaxTemporalHistory)
{
    PrevSample.M = min(PrevSample.M, min(RTXDI_PTReservoir::MaxM, ReducedMaxTemporalHistory));
    ++PrevSample.Age;

    bool FoundNeighbor = true;
    if (PrevSample.Age > min(RTXDI_PTRESERVOIR_AGE_MAX, tParams.maxReservoirAge))
    {
        FoundNeighbor = false;
    }

    return FoundNeighbor;
}

bool ShouldEvaluateTemporalReconnection(RTXDI_PTReservoir prevSample)
{
    return !(((prevSample.RcVertexLength == prevSample.PathLength) && RTXDI_ConnectsToNeeLight(prevSample)) || (prevSample.RcVertexLength > prevSample.PathLength));
}

void EvaluateReconnection(const RTXDI_PTHybridShiftPerFrameParameters hspfParams,
                          inout float3 TargetFunction,
                          const RAB_Surface Surface,
                          const RTXDI_PTReservoir PrevSample,
                          const bool checkVisibility)
{
    TargetFunction *= RAB_GetPTSampleTargetPdfForSurface(PrevSample.TranslatedWorldPosition, PrevSample.Radiance, Surface);

    if (checkVisibility && any(TargetFunction > 0.f))
    {
        TargetFunction *= RAB_GetConservativeVisibility(Surface, PrevSample.TranslatedWorldPosition);
    }
}

RTXDI_PTHybridShiftRuntimeParameters BuildHSRShiftParams(const RTXDI_PTTemporalResamplingRuntimeParameters trrParams)
{
    RTXDI_PTHybridShiftRuntimeParameters hsrParams = (RTXDI_PTHybridShiftRuntimeParameters)0;
    hsrParams.IsBasePathInPrevFrame = true;
    hsrParams.IsPrevFrame = false;
    hsrParams.cameraPos = trrParams.cameraPos;
    hsrParams.prevCameraPos = trrParams.prevCameraPos;
    hsrParams.prevPrevCameraPos = trrParams.prevPrevCameraPos;
    return hsrParams;
}

RTXDI_PTHybridShiftRuntimeParameters BuildHSRInverseShiftParams(const RTXDI_PTTemporalResamplingRuntimeParameters trrParams)
{
    RTXDI_PTHybridShiftRuntimeParameters hsrParams = (RTXDI_PTHybridShiftRuntimeParameters)0;
    hsrParams.IsBasePathInPrevFrame = false;
    hsrParams.IsPrevFrame = true;
    hsrParams.cameraPos = trrParams.cameraPos;
    hsrParams.prevCameraPos = trrParams.prevCameraPos;
    hsrParams.prevPrevCameraPos = trrParams.prevPrevCameraPos;
    return hsrParams;
}

#if RTXDI_DEBUG == 1
void TemporalRetrace(const RTXDI_PTTemporalResamplingRuntimeParameters trrParams,
                     const RTXDI_PTHybridShiftPerFrameParameters hspfParams,
                     RTXDI_PTReconnectionParameters rParams,
                     RAB_Surface prevSurfCopy,
                     RTXDI_PTReservoir prevSampleCopy,
                     inout RAB_PathTracerUserData ptud)
{
    float3 TargetFunction = float3(0.0f, 0.0f, 0.0f);
    float Jacobian = 0;
    prevSampleCopy.PathLength = max(prevSampleCopy.PathLength, hspfParams.maxBounceDepth);
    prevSampleCopy.RcVertexLength = prevSampleCopy.PathLength + 1;
    rParams.roughnessThreshold = 1.0;
    rParams.distanceThreshold = 1e6;
    rParams.reconnectionMode = RTXDI_RESTIRPT_RECONNECTION_MODE_FIXED_THRESHOLD;
    RAB_PathTracerUserDataSetPathType(ptud, RTXDI_PTPathTraceInvocationType_DebugTemporalRetrace);
    ComputeHybridShift(prevSurfCopy, prevSampleCopy, hspfParams, BuildHSRShiftParams(trrParams), rParams, TargetFunction, Jacobian, ptud);
}
#endif

// Shift previous sample - random replay to new location
void ResampleTemporalNeighbor(const RTXDI_PTTemporalResamplingParameters tParams,
                              const RTXDI_PTTemporalResamplingRuntimeParameters trrParams,
                              const RTXDI_PTHybridShiftPerFrameParameters hspfParams,
                              const RTXDI_PTReconnectionParameters rParams,
                              inout RTXDI_RandomSamplerState rng,
                              inout RTXDI_PTReservoir targetReservoir,
                              inout RAB_Surface surface,
                              inout RTXDI_PTReservoir PrevSample,
                              inout float Jacobian,
                              inout float3 SelectedTargetFunction,
                              inout bool selectedPrevSample,
                              inout RAB_PathTracerUserData ptud)
{
    float3 TargetFunction = float3(1.0f, 1.0f, 1.0f);
    bool shouldEvaluateReconnection = ShouldEvaluateTemporalReconnection(PrevSample);

    // The random replay (prefix tracing) pass taken by hybrid shift
    // this will update Surface to the vertex before reconnection
    // If the reconnection vertex is a NEE-sampled light vertex (UseRTXDILight), it will be handled inside this function
    RAB_PathTracerUserDataSetPathType(ptud, RTXDI_PTPathTraceInvocationType_Temporal);
    ComputeHybridShift(surface, PrevSample, hspfParams, BuildHSRShiftParams(trrParams), rParams, TargetFunction, Jacobian, ptud);

    PrevSample.WeightSum *= Jacobian;

    if (shouldEvaluateReconnection)
    {
        EvaluateReconnection(hspfParams, TargetFunction, surface, PrevSample, tParams.enableVisibilityBeforeCombine);
    }

    if (CombineReservoirs(targetReservoir, PrevSample, RTXDI_GetNextRandom(rng), TargetFunction))
    {
        selectedPrevSample = true;
        SelectedTargetFunction = TargetFunction;
    }
}

void BiasCorrection(const RTXDI_PTTemporalResamplingRuntimeParameters trrParams,
                    const RTXDI_PTHybridShiftPerFrameParameters hspfParams,
                    const RTXDI_PTReconnectionParameters rParams,
                    inout RTXDI_RandomSamplerState rng,
                    inout RTXDI_PTReservoir targetReservoir,
                    inout RAB_Surface surface,
                    inout RAB_Surface PrevSurface,
                    inout RTXDI_PTReservoir PrevSample,
                    const bool selectedPrevSample,
                    inout float Pi,
                    inout float PiSum,
                    inout RAB_PathTracerUserData ptud)
{
    float3 TargetFunction = float3(1.0f, 1.0f, 1.0f);
    bool shouldEvaluateReconnection = ShouldEvaluateTemporalReconnection(targetReservoir);

    float Jacobian = RTXDI_CalculateJacobian(RAB_GetSurfaceWorldPos(PrevSurface), RAB_GetSurfaceWorldPos(surface), targetReservoir.TranslatedWorldPosition, targetReservoir.WorldNormal);
    float backupPartialJacobian = targetReservoir.PartialJacobian; // for MIS weight computation, we don't want targetReservoir value to be eventually overwritten

    RAB_PathTracerUserDataSetPathType(ptud, RTXDI_PTPathTraceInvocationType_TemporalInverse);
    ComputeHybridShift(PrevSurface, targetReservoir, hspfParams, BuildHSRInverseShiftParams(trrParams), rParams, TargetFunction, Jacobian, ptud);

    targetReservoir.PartialJacobian = backupPartialJacobian;

    if (shouldEvaluateReconnection)
    {
        EvaluateReconnection(hspfParams, TargetFunction, PrevSurface, targetReservoir, true);
    }

    float TemporalP = RTXDI_Luminance(TargetFunction) * Jacobian;

    Pi = selectedPrevSample ? TemporalP : Pi;
    PiSum += TemporalP * PrevSample.M;
}

RTXDI_PTReservoir RTXDI_PTTemporalResampling(RTXDI_PTTemporalResamplingParameters tParams,
                                             RTXDI_PTTemporalResamplingRuntimeParameters trrParams,
                                             RTXDI_PTHybridShiftPerFrameParameters hspfParams,
                                             RTXDI_PTReconnectionParameters rParams,
                                             RTXDI_RuntimeParameters runtimeParams,
                                             RTXDI_ReservoirBufferParameters bufferParams,
                                             RTXDI_RandomSamplerState rng,
                                             RTXDI_PTBufferIndices bufferIndices,
                                             inout bool selectedPrevSample,
                                             inout RAB_PathTracerUserData ptud)
{
    RAB_Surface Surface = RAB_GetGBufferSurface(trrParams.pixelPosition, false);
    if (!RAB_IsSurfaceValid(Surface))
    {
        return RTXDI_EmptyPTReservoir();
    }

    RTXDI_PTReservoir targetReservoir = RTXDI_EmptyPTReservoir();
    RTXDI_PTReservoir CurSample = RTXDI_LoadPTReservoir(bufferParams, trrParams.reservoirPosition, bufferIndices.initialPathTracerOutputBufferIndex);
    float3 SelectedTargetFunction = float3(0.0f, 0.0f, 0.0f);

    if (RTXDI_IsValidPTReservoir(CurSample))
    {
        if (CombineReservoirs(targetReservoir, CurSample, /* random = */ 0.5, CurSample.TargetFunction))
        {
            SelectedTargetFunction = CurSample.TargetFunction;
        }
    }

    int2 prevPos = int2(round(float2(trrParams.pixelPosition) + trrParams.motionVector.xy));
    float ExpectedPrevLinearDepth = RAB_GetSurfaceLinearDepth(Surface) + trrParams.motionVector.z;
    RTXDI_PTReservoir PrevSample = RTXDI_EmptyPTReservoir();
    RAB_Surface PrevSurface = RAB_EmptySurface();

    bool FoundNeighbor = FindTemporalNeighbor(tParams, runtimeParams, bufferParams, bufferIndices, rng, trrParams.pixelPosition, prevPos, Surface, ExpectedPrevLinearDepth, PrevSurface, PrevSample);

#if RTXDI_DEBUG == 1
    if(FoundNeighbor)
    {
        TemporalRetrace(trrParams, hspfParams, rParams, PrevSurface, PrevSample, ptud);
    }
#endif

    int ReducedMaxTemporalHistory = tParams.maxHistoryLength;
    // Duplication-based history reduction: reduce MCap where many pixels share the same sample ID
    if (tParams.duplicationBasedHistoryReduction && FoundNeighbor && tParams.historyReductionStrength > 0.f)
    {
        uint dupCount = RAB_GetDuplicationMapCount(prevPos);
        float impoverishment = saturate(float(dupCount) / 288.0);
        float powerFactor = 0.1 * pow(2,6 * (1.f - tParams.historyReductionStrength) - 3);
        float t = pow(impoverishment, powerFactor);
        ReducedMaxTemporalHistory = max(1, (int)lerp((float)tParams.maxHistoryLength, 1.0, t));
    }
    float Jacobian = 1.f;
    if (FoundNeighbor)
    {
        Jacobian = RTXDI_CalculateJacobian(RAB_GetSurfaceWorldPos(Surface), RAB_GetSurfaceWorldPos(PrevSurface), PrevSample.TranslatedWorldPosition, PrevSample.WorldNormal);
        FoundNeighbor = ValidateTemporalNeighbor(tParams, PrevSample, Surface, PrevSurface, ReducedMaxTemporalHistory);
    }

    selectedPrevSample = false;
    if (FoundNeighbor)
    {
        ResampleTemporalNeighbor(tParams, trrParams, hspfParams, rParams, rng, targetReservoir, Surface, PrevSample, Jacobian, SelectedTargetFunction, selectedPrevSample, ptud);
    }

    float Pi = RTXDI_Luminance(SelectedTargetFunction);
    float PiSum = RTXDI_Luminance(SelectedTargetFunction) * CurSample.M;
    if (RTXDI_IsValidPTReservoir(targetReservoir) && FoundNeighbor)
    {
        BiasCorrection(trrParams, hspfParams, rParams, rng, targetReservoir, Surface, PrevSurface, PrevSample, selectedPrevSample, Pi, PiSum, ptud);
    }

    const float NormalizationNumerator = Pi;
    const float NormalizationDenominator = PiSum * RTXDI_Luminance(SelectedTargetFunction);
    RTXDI_FinalizeResampling(targetReservoir, NormalizationNumerator, NormalizationDenominator);
    if (tParams.duplicationBasedHistoryReduction)
    {
        targetReservoir.ShouldBoostSpatialSamples = ReducedMaxTemporalHistory > targetReservoir.M;
    }

    return targetReservoir;
}

#endif // RTXDI_PT_TEMPORAL_RESAMPLING_HLSLI