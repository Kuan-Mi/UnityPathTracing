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

#ifndef RTXDI_PT_PATH_TRACING_STATE_HLSLI
#define RTXDI_PT_PATH_TRACING_STATE_HLSLI

#include "Rtxdi/Utils/BrdfRaySample.hlsli"
#include "Rtxdi/Utils/SampledLightData.hlsli"

// Path Tracing flag bits
#define PT_STATE_LAST_VERTEX_FAR           0x02
#define PT_STATE_LAST_VERTEX_ROUGH         0x04
#define PT_STATE_PATH_TERMINATION          0x08
#define PT_STATE_LAST_VERTEX_NEARDELTA     0x10

class RTXDI_PathTracerState
{
    // Base state

    bool IsSecondaryBounce()
    {
        return bounceDepth == 2;
    }

    // Reset partial state for the next path
    void BeginPathState()
    {
        LastHitT = RAB_RayPayloadGetCommittedHitT(TraceResult);

        // Compiler should be able to optimize all of these instructions below, hopefully.
        brdfRaySample = (RTXDI_BrdfRaySample)0;
        TraceResult = (RAB_RayPayload)0;
        ContinuationRay = (RayDesc)0;
    }

    // Hybrid shift functions

    void SetPathTermination(bool Value)
    {
        PTStateFlags &= ~PT_STATE_PATH_TERMINATION;
        PTStateFlags |= Value ? PT_STATE_PATH_TERMINATION : 0x0;
    }

    bool IsPathTermination()
    {
        return (PTStateFlags & PT_STATE_PATH_TERMINATION) != 0;
    }

    // PT Base Functions

    void SetIsLastVertexFar(bool Value)
    {
        PTStateFlags &= ~PT_STATE_LAST_VERTEX_FAR;
        PTStateFlags |= Value ? PT_STATE_LAST_VERTEX_FAR : 0x0;
    }

    bool IsLastVertexFar()
    {
        return (PTStateFlags & PT_STATE_LAST_VERTEX_FAR) != 0;
    }

    void SetIsLastVertexRough(bool Value)
    {
        PTStateFlags &= ~PT_STATE_LAST_VERTEX_ROUGH;
        PTStateFlags |= Value ? PT_STATE_LAST_VERTEX_ROUGH : 0x0;
    }

    bool IsLastVertexRough()
    {
        return (PTStateFlags & PT_STATE_LAST_VERTEX_ROUGH) != 0;
    }

    bool NoMainPathRcVertex()
    {
        return RcVertexLength >= bounceDepth; // change > to >= because maxRcVertex can be small
    }

    //
    // Base state
    //

    // 0 = camera, 1 = primary hit, 2 = secondary hit, etc.
    uint16_t initialBounceDepth;

    // Max number of bounces from camera
    uint16_t maxPathBounce;

    // BounceDepth means the number of vertices in the path sample, which excludes the eye point and light sample.
    uint16_t bounceDepth;

    // Current path throughput divided by part of the pdfs, it doesn't count the pdf of producing secondary ray though.
    float3 pathThroughput;

    // Outgoing direct and PDF from surface n to surface n+1.
    RTXDI_BrdfRaySample brdfRaySample;

    // The ray for the current bounce
    RayDesc ContinuationRay;

    // The vertex in the path, depending on different state of the path tracing, this means different things.
    // Check the implementation for further details
    RAB_Surface intersectionSurface;

    // See PT_STATE_* defines
    uint16_t PTStateFlags;

    // Continuation ray
    float3 continuationRayBrdfOverPdf;

    // Intersection payload
    RAB_RayPayload TraceResult;

    RAB_LightSample lightSampleForDI;

    //
    // PT Base state
    //

    float LastHitT;

    //
    // Replay state
    //

    float3 PreviousNormal;
    // float   LastHitT;

    //
    // ReSTIR PT state
    //

    float3 RcPathThroughput;

    uint16_t RcVertexLength;

    RTXDI_SampledLightData sampledLightDataForDI;

    float CandidateReservoirWeightSum;
};

RTXDI_PathTracerState RTXDI_EmptyPathTracerState()
{
    return (RTXDI_PathTracerState)0;
}

RTXDI_PathTracerState RTXDI_InitializePathTracerState(RAB_Surface surface, uint maxBounces)
{
    RTXDI_PathTracerState ptState = RTXDI_EmptyPathTracerState();
    ptState.initialBounceDepth = 2; // Starting surface is primary hit (1), but initialBounce starts at the surface after the scattering event, which is the secondary surface (2)
    ptState.bounceDepth = ptState.initialBounceDepth;
    ptState.maxPathBounce = (uint16_t)maxBounces;
    ptState.pathThroughput = float3(1.0f, 1.0f, 1.0f);
    ptState.intersectionSurface = surface;
    return ptState;
}

#endif // RTXDI_PT_PATH_TRACING_STATE_HLSLI
