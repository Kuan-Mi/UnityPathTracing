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

#ifndef RTXDI_PT_SPATIAL_RESAMPLING_HLSLI
#define RTXDI_PT_SPATIAL_RESAMPLING_HLSLI

#include "Rtxdi/PT/HybridShift.hlsli"
#include "Rtxdi/PT/Reservoir.hlsli"
#include "Rtxdi/Utils/Math.hlsli"

// using uint mask to track samples, so absolute limit is 32
static const uint RTXDI_MAX_SPATIAL_RESAMPLING_COUNT = 32;

struct RTXDI_PTSpatialResamplingRuntimeParameters
{
    uint2 PixelPosition;
    uint2 ReservoirPosition;

    float3 cameraPos;
    float3 prevCameraPos;
    float3 prevPrevCameraPos;
};

RTXDI_PTSpatialResamplingRuntimeParameters RTXDI_EmptyPTSpatialResamplingRuntimeParameters()
{
    return (RTXDI_PTSpatialResamplingRuntimeParameters)0;
}

uint2 CalculateNeighborSurfacePixelPosition(RTXDI_PTSpatialResamplingParameters SpatialParams, RTXDI_RuntimeParameters RuntimeParams, uint2 pixelPosition, uint i, uint StartIdx)
{
    uint sampleIdx = (StartIdx + i) & RuntimeParams.neighborOffsetMask;
    int2 spatialOffset = int2(float2(RTXDI_NEIGHBOR_OFFSETS_BUFFER[sampleIdx].xy) * SpatialParams.samplingRadius);
    int2 idx = int2(pixelPosition) + spatialOffset;
    idx = RAB_ClampSamplePositionIntoView(idx, false);
    RTXDI_ActivateCheckerboardPixel(idx, false, RuntimeParams.activeCheckerboardField);
    return idx;
}

bool IsValidNeighborSurface(RTXDI_PTSpatialResamplingParameters SpatialParams, RAB_Surface Surface, RAB_Surface neighborSurface)
{
    if (!RAB_IsSurfaceValid(neighborSurface))
    {
        return false;
    }

    if (!RTXDI_IsValidNeighbor(RAB_GetSurfaceNormal(Surface), RAB_GetSurfaceNormal(neighborSurface),
        RAB_GetSurfaceLinearDepth(Surface), RAB_GetSurfaceLinearDepth(neighborSurface),
        SpatialParams.normalThreshold, SpatialParams.depthThreshold))
    {
        return false;
    }

    //if (spatialParams.enableMaterialSimilarityTest && !RAB_AreMaterialsSimilar(RAB_GetMaterial(centerSurface), RAB_GetMaterial(neighborSurface)))
    if (!RAB_AreMaterialsSimilar(RAB_GetMaterial(Surface), RAB_GetMaterial(neighborSurface)))
    {
        return false;
    }

    return true;
}

bool ShouldApplyDisocclusionBoost(RTXDI_PTSpatialResamplingParameters SpatialParams, RTXDI_PTReservoir Reservoir, RTXDI_PTReservoir CurSample)
{
    return (Reservoir.RcVertexLength == 2 && ((!SpatialParams.duplicationBasedHistoryReduction && SpatialParams.maxTemporalHistory > CurSample.M) || (SpatialParams.duplicationBasedHistoryReduction && CurSample.ShouldBoostSpatialSamples)));
}

uint CalculateNumSamples(RTXDI_PTSpatialResamplingParameters SpatialParams, RTXDI_PTReservoir Reservoir, RTXDI_PTReservoir CurSample)
{
    int NumSamples = SpatialParams.numSpatialSamples;

    // doing sample boost for random replay (RcVertexLength>2) is pretty expensive. Let's disable that for now.
    if (ShouldApplyDisocclusionBoost(SpatialParams, Reservoir, CurSample))
    {
        NumSamples = max(NumSamples, SpatialParams.numDisocclusionBoostSamples);
    }

    NumSamples = min(NumSamples, RTXDI_MAX_SPATIAL_RESAMPLING_COUNT);
    return NumSamples;
}

RTXDI_PTHybridShiftRuntimeParameters BuildHSParams(const RTXDI_PTSpatialResamplingRuntimeParameters srrParams)
{
    RTXDI_PTHybridShiftRuntimeParameters hsrParams = (RTXDI_PTHybridShiftRuntimeParameters)0;
    hsrParams.IsBasePathInPrevFrame = false;
    hsrParams.IsPrevFrame = false;

    hsrParams.cameraPos = srrParams.cameraPos;
    hsrParams.prevCameraPos = srrParams.prevCameraPos;
    hsrParams.prevPrevCameraPos = srrParams.prevPrevCameraPos;
    return hsrParams;
}

#if RTXDI_DEBUG == 1
void SpatialRetrace(const RTXDI_PTSpatialResamplingRuntimeParameters srrParams,
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
    RAB_PathTracerUserDataSetPathType(ptud, RTXDI_PTPathTraceInvocationType_DebugSpatialRetrace);
    ComputeHybridShift(prevSurfCopy, prevSampleCopy, hspfParams, BuildHSParams(srrParams), rParams, TargetFunction, Jacobian, ptud);
}
#endif

bool ShouldEvaluateSpatialReconnection(RTXDI_PTReservoir NeighborReservoir)
{
    return !(((NeighborReservoir.RcVertexLength == NeighborReservoir.PathLength) && RTXDI_ConnectsToNeeLight(NeighborReservoir)) || (NeighborReservoir.RcVertexLength > NeighborReservoir.PathLength));
}

void EvaluateReconnection(RTXDI_PTHybridShiftPerFrameParameters hspfParams, inout float3 TargetFunction, RAB_Surface Surface, RTXDI_PTReservoir NeighborReservoir)
{
    TargetFunction *= RAB_GetPTSampleTargetPdfForSurface(NeighborReservoir.TranslatedWorldPosition, NeighborReservoir.Radiance, Surface);

    // we don't want to skip this ReSTIR PT, otherwise, we have to do random replay in final shading
    // as well to fetch the vertex before reconnection to do a visibility test
    if (any(TargetFunction > 0.f))
    {
        TargetFunction *= RAB_GetConservativeVisibility(Surface, NeighborReservoir.TranslatedWorldPosition);
    }
}

bool ResampleNeighbors(const RTXDI_PTSpatialResamplingRuntimeParameters srrParams,
                       const RTXDI_PTSpatialResamplingParameters SpatialParams,
                       const RTXDI_PTHybridShiftPerFrameParameters hspfParams,
                       const RTXDI_PTReconnectionParameters rParams,
                       const RTXDI_ReservoirBufferParameters ReservoirBufferParams,
                       const RTXDI_PTBufferIndices BufferIndices,
                       RTXDI_RuntimeParameters RuntimeParams,
                       const uint StartIdx,
                       inout uint SelectedIndex,
                       inout RTXDI_RandomSamplerState rng,
                       const RAB_Surface Surface,
                       inout RTXDI_PTReservoir targetReservoir,
                       inout float3 SelectedTargetFunction,
                       const uint NumSamples,
                       inout uint CachedResult,
                       inout RAB_PathTracerUserData ptud)
{
    bool resampled = false;
    for (uint i = 0; i < NumSamples; ++i)
    {
        uint2 neighborSurfacePos = CalculateNeighborSurfacePixelPosition(SpatialParams, RuntimeParams, srrParams.PixelPosition, i, StartIdx);
        RAB_Surface neighborSurface = RAB_GetGBufferSurface(neighborSurfacePos, false);

        if (!IsValidNeighborSurface(SpatialParams, Surface, neighborSurface))
        {
            continue;
        }

        uint2 neighborReservoirPos = RTXDI_PixelPosToReservoirPos(neighborSurfacePos, RuntimeParams.activeCheckerboardField);
        RTXDI_PTReservoir NeighborReservoir = RTXDI_LoadPTReservoir(ReservoirBufferParams, neighborReservoirPos, BufferIndices.spatialResamplingInputBufferIndex);

#if RTXDI_DEBUG == 1
        SpatialRetrace(srrParams, hspfParams, rParams, neighborSurface, NeighborReservoir, ptud);
#endif

        float Jacobian = RTXDI_CalculateJacobian(RAB_GetSurfaceWorldPos(Surface), RAB_GetSurfaceWorldPos(neighborSurface), NeighborReservoir.TranslatedWorldPosition, NeighborReservoir.WorldNormal);
        float3 TargetFunction = float3(1.0f, 1.0f, 1.0f);

        RAB_Surface SurfaceForResampling = Surface;

        // The random replay (prefix tracing) pass taken by hybrid shift
        // this will update Surface to the vertex before reconnection
        // If the reconnection vertex is a NEE-sampled light vertex (UseRTXDILight), it will be handled inside this function
        bool shouldEvaluateReconnection = ShouldEvaluateSpatialReconnection(NeighborReservoir);
        RAB_PathTracerUserDataSetPathType(ptud, RTXDI_PTPathTraceInvocationType_Spatial);
        ComputeHybridShift(SurfaceForResampling, NeighborReservoir, hspfParams, BuildHSParams(srrParams), rParams, TargetFunction, Jacobian, ptud);

        if (shouldEvaluateReconnection)
        {
            EvaluateReconnection(hspfParams, TargetFunction, SurfaceForResampling, NeighborReservoir);
        }

        NeighborReservoir.WeightSum *= Jacobian;

        CachedResult |= (1u << uint(i));

        if (CombineReservoirs(targetReservoir, NeighborReservoir, RTXDI_GetNextRandom(rng), TargetFunction))
        {
            SelectedTargetFunction = TargetFunction;
            SelectedIndex = i;
            resampled = true;
        }
    }
    return resampled;
}

void BiasCorrection(const RTXDI_PTSpatialResamplingRuntimeParameters srrParams,
                    const RTXDI_PTSpatialResamplingParameters SpatialParams,
                    const RTXDI_PTHybridShiftPerFrameParameters hspfParams,
                    const RTXDI_PTReconnectionParameters rParams,
                    const RTXDI_ReservoirBufferParameters ReservoirBufferParams,
                    const RTXDI_PTBufferIndices BufferIndices,
                    RTXDI_RuntimeParameters RuntimeParams,
                    const uint StartIdx,
                    const uint SelectedIndex,
                    const RAB_Surface Surface,
                    inout RTXDI_PTReservoir targetReservoir,
                    const uint NumSamples,
                    const uint CachedResult,
                    inout float Pi,
                    inout float PiSum,
                    inout RAB_PathTracerUserData ptud)
{
    for (uint i = 0; i < NumSamples; ++i)
    {
        // If we skipped this neighbor above, do so again.
        if ((CachedResult & (1u << uint(i))) == 0)
            continue;

        uint2 neighborSurfacePos = CalculateNeighborSurfacePixelPosition(SpatialParams, RuntimeParams, srrParams.PixelPosition, i, StartIdx);
        RAB_Surface NeighborSurface = RAB_GetGBufferSurface(neighborSurfacePos, false);

        uint2 neighborReservoirPos = RTXDI_PixelPosToReservoirPos(neighborSurfacePos, RuntimeParams.activeCheckerboardField);
        RTXDI_PTReservoir NeighborReservoir = RTXDI_LoadPTReservoir(ReservoirBufferParams, neighborReservoirPos, BufferIndices.spatialResamplingInputBufferIndex);

        float3 TargetFunction = float3(1.0f, 1.0f, 1.0f);

        float Jacobian = RTXDI_CalculateJacobian(RAB_GetSurfaceWorldPos(NeighborSurface), RAB_GetSurfaceWorldPos(Surface), targetReservoir.TranslatedWorldPosition, targetReservoir.WorldNormal);

        // The random replay (prefix tracing) pass taken by hybrid shift
        // this will update Surface to the vertex before reconnection
        // If the reconnection vertex is a NEE-sampled light vertex (UseRTXDILight), it will be handled inside this function
        float BackupPartialJacobian = targetReservoir.PartialJacobian; // for MIS weight computation, we don't want Reservoir value to be eventually overwritten
        bool shouldEvaluateReconnection = ShouldEvaluateSpatialReconnection(targetReservoir);
        RAB_PathTracerUserDataSetPathType(ptud, RTXDI_PTPathTraceInvocationType_SpatialInverse);
        ComputeHybridShift(NeighborSurface, targetReservoir, hspfParams, BuildHSParams(srrParams), rParams, TargetFunction, Jacobian, ptud);
        targetReservoir.PartialJacobian = BackupPartialJacobian;

        if (shouldEvaluateReconnection)
        {
            EvaluateReconnection(hspfParams, TargetFunction, NeighborSurface, targetReservoir);
        }

        float Ps = RTXDI_Luminance(TargetFunction) * Jacobian;

        // Select this sample for the (normalization) numerator if this particular neighbor pixel
        // was the one we selected via RIS in the first loop, above.
        Pi = (SelectedIndex == i) ? Ps : Pi;

        // Add to the sums of weights for the (normalization) denominator
        PiSum += Ps * NeighborReservoir.M;
    }
}

void FinalizeResampling(inout RTXDI_PTReservoir targetReservoir, float3 SelectedTargetFunction, float Pi, float PiSum)
{
    // "MIS-like" normalization
    // {wSum * (pi/piSum)} * 1/selectedTargetPdf
    const float NormalizationNumerator = Pi;
    const float NormalizationDenominator = RTXDI_Luminance(SelectedTargetFunction) * PiSum;
    RTXDI_FinalizeResampling(targetReservoir, NormalizationNumerator, NormalizationDenominator);
}

RTXDI_PTReservoir RTXDI_PTSpatialResampling(RTXDI_PTSpatialResamplingRuntimeParameters srrParams,
                                            RTXDI_PTSpatialResamplingParameters SpatialParams,
                                            RTXDI_PTHybridShiftPerFrameParameters hspfParams,
                                            RTXDI_PTReconnectionParameters rParams,
                                            RTXDI_ReservoirBufferParameters ReservoirBufferParams,
                                            RTXDI_PTBufferIndices BufferIndices,
                                            RTXDI_RuntimeParameters RuntimeParams,
                                            RTXDI_RandomSamplerState rng,
                                            inout bool resampled,
                                            inout RAB_PathTracerUserData ptud)
{
    RAB_Surface Surface = RAB_GetGBufferSurface(srrParams.PixelPosition, false);
    if (!RAB_IsSurfaceValid(Surface))
    {
        return RTXDI_EmptyPTReservoir();
    }

    RTXDI_PTReservoir targetReservoir = RTXDI_EmptyPTReservoir();
    RTXDI_PTReservoir CurSample = RTXDI_LoadPTReservoir(ReservoirBufferParams, srrParams.ReservoirPosition, BufferIndices.spatialResamplingInputBufferIndex);
    float3 SelectedTargetFunction = float3(0.0f, 0.0f, 0.0f);

    if (RTXDI_IsValidPTReservoir(CurSample))
    {
        if (CombineReservoirs(targetReservoir, CurSample, /* random = */ 0.5, CurSample.TargetFunction))
        {
            SelectedTargetFunction = CurSample.TargetFunction;
        }
    }

    int NumSamples = CalculateNumSamples(SpatialParams, targetReservoir, CurSample);
    const uint StartIdx = RTXDI_GetNextRandom(rng) * RuntimeParams.neighborOffsetMask;
    uint SelectedIndex = NumSamples + 1;
    uint CachedResult = 0;

    resampled = ResampleNeighbors(srrParams, SpatialParams, hspfParams, rParams, ReservoirBufferParams, BufferIndices, RuntimeParams, StartIdx, SelectedIndex, rng, Surface, targetReservoir, SelectedTargetFunction, NumSamples, CachedResult, ptud);

    float Pi = RTXDI_Luminance(SelectedTargetFunction);
    float PiSum = RTXDI_Luminance(SelectedTargetFunction) * CurSample.M;

    BiasCorrection(srrParams, SpatialParams, hspfParams, rParams, ReservoirBufferParams, BufferIndices, RuntimeParams, StartIdx, SelectedIndex, Surface, targetReservoir, NumSamples, CachedResult, Pi, PiSum, ptud);

    FinalizeResampling(targetReservoir, SelectedTargetFunction, Pi, PiSum);
    return targetReservoir;
}

#endif // RTXDI_PT_SPATIAL_RESAMPLING_HLSLI
