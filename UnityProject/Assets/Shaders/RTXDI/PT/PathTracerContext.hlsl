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

#ifndef RTXDI_PT_PATH_TRACING_CONTEXT_HLSLI
#define RTXDI_PT_PATH_TRACING_CONTEXT_HLSLI

#include "Rtxdi/DI/Reservoir.hlsli"
#include "Rtxdi/PT/PathTracerContextParameters.hlsli"
#include "Rtxdi/Utils/BrdfRaySample.hlsli"
#include "Rtxdi/Utils/Math.hlsli"
#include "Rtxdi/Utils/SampledLightData.hlsli"

//
// define-style interface for broad compatibility
// RTXDI_PathTracerContext lays out the interface
// The InitialSampling and HybridShift variants contain the specialized implementations
//

#if defined(RTXDI_RESTIR_PT_INITIAL_SAMPLING)
#include "InitialSamplingPathTracerContext.hlsli"
#elif defined(RTXDI_RESTIR_PT_HYBRID_SHIFT)
#include "HybridShiftPathTracerContext.hlsli"
#else
#define RTXDI_NO_RESTIR_PT_CONTEXT
#endif

#if !defined(RTXDI_NO_RESTIR_PT_CONTEXT)
//
// Principal interface between the path tracer and ReSTIR PT
// Responsible for recording information about the path and
//    the reconnectibility of vertices along it.
//
// Functions are laid out in the order they are called from
//    the path tracing loop
//
struct RTXDI_PathTracerContext
{
    //
    // The RTXDI_BrdfRaySample contains all the information about
    // the outgoing ray from surface N to surface N+1
    //
    void SetBrdfRaySample(RTXDI_BrdfRaySample brs)
    {
        ctx.SetBrdfRaySample(brs);
    }

    void SetMaxPathBounce(uint16_t newMax)
    {
        ctx.SetMaxPathBounce(newMax);
    }

    //
    // Updates the maximum reconnection vertex length only if
    // the path so far has not produced a reconnection vertex
    // Thus, if an RC vertex has already been found, this function
    // does nothing. But if an RC vertex has not yet been found,
    // the maximum allowable RC vertex length will be set to the
    // newMax passed in.
    //
    // Useful for selectively increasing the rc vertex length
    // for mirror bounces.
    //
    void SetMaxRcVertexLengthIfUnset(uint16_t newMax)
    {
        ctx.SetMaxRcVertexLengthIfUnset(newMax);
    }

    //
    // Enabled for initial sampling
    // Disabled for hybrid shift
    //
    bool ShouldRunRussianRoulette()
    {
        return ctx.ShouldRunRussianRoulette();
    }

    //
    // The context needs to know about the Russian roulette
    // continuation probability to keep resampling unbiased
    //
    void RecordRussianRouletteProbability(float RRProb)
    {
        ctx.RecordRussianRouletteProbability(RRProb);
    }

    //
    // Multiplies the path throughput by the given multiplication factor
    //
    // Call after CalculateBRDFOverPDF with the appropriate result
    //
    void MultiplyPathThroughput(float3 multiplicationFactor)
    {
        ctx.MultiplyPathThroughput(multiplicationFactor);
    }

    //
    // Sets the outgoing ray direction for the path tracer from the
    //     current surface.
    //
    void SetContinuationRay(RayDesc cr)
    {
        ctx.SetContinuationRay(cr);
    }

    //
    // Determines reconnectibility state for the vertex from which the
    //     outgoing ray is departing based off of the ray footprint,
    //     bounce depth, and other reconnectibility state
    //
    // Call after the ray from the surface has been created and before it
    //     is actually traced with RAB_TraceNextBounce
    //
    // Returns true if the path is valid and RAB_TraceNextBounce should
    //     be called.
    // Returns fails if the path is invalid and the path tracer should
    //     exit.
    //
    bool AnalyzePathReconnectibilityBeforeTrace()
    {
        return ctx.AnalyzePathReconnectibilityBeforeTrace();
    }

    //
    // Call after AnalyzePathReconnectibilityBeforeTrace
    //
    void SetTraceResult(RAB_RayPayload rp)
    {
        ctx.SetTraceResult(rp);
    }

    //
    // Updates the current intersection surface tracked by the context
    //
    // Call this function when the ray traced from surface N hits surface N + 1
    // Do not call this function when the ray misses
    //
    void RecordPathIntersection(const RAB_Surface intersectionSurface)
    {
        ctx.RecordPathIntersection(intersectionSurface);
    }

    //
    // The path tracer can use MIS to balance between sampling lights directly via
    //     NEE or casting BRDF rays towards emissive surfaces.
    // Returns true when the context is set to EMISSIVE_ONLY or MIS mode and signals
    //     that the emissive component of intersection surfaces should be sampled
    // Returns false when the context is set to EMISSIVE_ONLY or during hybrid shift
    //
    bool ShouldSampleEmissiveSurfaces()
    {
        return ctx.ShouldSampleEmissiveSurfaces();
    }

    //
    // Records an emissive light sample
    // Assumes the emissive surface is the intersection surface called by
    //     RecordPathIntersection() for tracking reconnection data.
    // Call after RecordPathIntersection()
    //
    bool RecordEmissiveLightSample(float3 radianceFromEmissiveSurface, RAB_Surface prevSurface, inout RTXDI_RandomSamplerState RandContext)
    {
        return ctx.RecordEmissiveLightSample(radianceFromEmissiveSurface, prevSurface, RandContext);
    }

    //
    // The path tracer can use MIS to balance between sampling lights directly via
    //     NEE or casting BRDF rays towards emissive surfaces.
    // Returns true when the context is set to NEE_ONLY or MIS mode and signals that
    //     lights should be sampled via NEE.
    // Returns false when the context is set to EMISSIVE_ONLY or during hybrid shift
    //
    bool ShouldSampleNee()
    {
        return ctx.ShouldSampleNee();
    }

    //
    // Call after RecordPathIntersection with the results of the path tracer's NEE routine
    //
    bool RecordNeeLightSample(in const RTXDI_SampledLightData sampledLightData,
                              in const float3 radianceFromLights,
                              in const float neePdf,
                              in const float scatterPdf,
                              in const RAB_LightSample lightSample,
                              inout RTXDI_RandomSamplerState randContext)
    {
        return ctx.RecordNeeLightSample(sampledLightData, radianceFromLights, neePdf, scatterPdf, lightSample, randContext);
    }

    void RecordPathRadianceMiss(inout RTXDI_RandomSamplerState rng)
    {
        ctx.RecordPathRadianceMiss(rng);
    }

    bool RecordEnvironmentMapLightSample(const float3 environmentMapRadiance,
                                         RAB_Surface prevSurface,
                                         inout RTXDI_RandomSamplerState rng)
    {
        return ctx.RecordEnvironmentMapLightSample(environmentMapRadiance, prevSurface, rng);
    }

    float calculateRISWeight(float3 targetFunctionOverP)
    {
        return ctx.calculateRISWeight(targetFunctionOverP);
    }

    bool RISStreamPathSample(uint PathLength,
                             float3 CurrentLightRadiance,
                             inout RTXDI_RandomSamplerState rng,
                             uint RcVertexLength,
                             bool IsNEE,
                             float NeePdf,
                             float3 PackedLightData,
                             float PartialJacobian)
    {
        return ctx.RISStreamPathSample(PathLength, CurrentLightRadiance, rng, RcVertexLength, IsNEE, NeePdf, PackedLightData, PartialJacobian);
    }

    bool IsPathTerminated()
    {
        return ctx.IsPathTerminated();
    }

    //
    // PTState functions
    //

    uint GetBounceDepth()
    {
        return ctx.GetBounceDepth();
    }

    uint GetMaxPathBounce()
    {
        return ctx.GetMaxPathBounce();
    }

    RTXDI_BrdfRaySample GetBrdfRaySample()
    {
        return ctx.GetBrdfRaySample();
    }

    RAB_Surface GetIntersectionSurface()
    {
        return ctx.GetIntersectionSurface();
    }

    float3 GetContinuationRayBrdfOverPdf()
    {
        return ctx.GetContinuationRayBrdfOverPdf();
    }

    RayDesc GetContinuationRay()
    {
        return ctx.GetContinuationRay();
    }

    float3 GetPathThroughput()
    {
        return ctx.GetPathThroughput();
    }

    RAB_RayPayload GetTraceResult()
    {
        return ctx.GetTraceResult();
    }

    void IncreaseBounceDepth()
    {
        ctx.IncreaseBounceDepth();
    }

    bool ValidContinuationRayBrdfOverPdf()
    {
        return ctx.ValidContinuationRayBrdfOverPdf();
    }

    uint16_t GetRcVertexLength()
    {
        return ctx.GetRcVertexLength();
    }

    //
    //
    //

    float3 GetRadiance()
    {
        return ctx.GetRadiance();
    }

    float GetRunningWeightSum()
    {
        return ctx.RunningWeightSum;
    }

    float GetSelectedPartialJacobian()
    {
        return ctx.SelectedPartialJacobian;
    }

    uint GetSelectedPathLength()
    {
        return ctx.SelectedPathLength;
    }

    RAB_Surface GetSelectedRcSurface()
    {
        return ctx.SelectedRcSurface;
    }

    uint GetSelectedRcVertexLength()
    {
        return ctx.SelectedRcVertexLength;
    }

    float GetSelectedRcWiPdf()
    {
        return ctx.SelectedRcWiPdf;
    }

    float3 GetSelectedTargetFunction()
    {
        return ctx.GetSelectedTargetFunction();
    }

    //
    // Reconnection Parameters
    //

    RTXDI_PTReconnectionMode GetReconnectionMode()
    {
        return ctx.GetReconnectionMode();
    }

    // Fixed cutoff mode reconnection thrsholds
    float GetRoughnessThreshold()
    {
        return ctx.GetRoughnessThreshold();
    }

    float GetDistanceThreshold()
    {
        return ctx.GetDistanceThreshold();
    }

    // Footprint mode reconnection thresholds
    float GetFootprintThreshold()
    {
        return ctx.GetFootprintThreshold();
    }

    float GetPdfThreshold()
    {
        return ctx.GetPdfThreshold();
    }

    //
    // Hybrid-Shift only functions
    //

    float GetCheckPreRcInverseGeoTerm()
    {
        return ctx.GetCheckPreRcInverseGeoTerm();
    }

    RAB_Surface GetRcPrevSurface()
    {
        return ctx.GetRcPrevSurface();
    }

    void SetRcVertexLength(uint length)
    {
        ctx.SetRcVertexLength(length);
    }

    void SetSelectedPathLength(uint length)
    {
        ctx.SetSelectedPathLength(length);
    }

    //
    // PTState functions
    //

    bool IsSecondaryBounce()
    {
        return ctx.IsSecondaryBounce();
    }

    // Reset partial state for the next path
    void BeginPathState()
    {
        ctx.BeginPathState();
    }

    // Hybrid shift functions

    void SetPathTermination(bool Value)
    {
        ctx.SetPathTermination(Value);
    }

    bool IsPathTermination()
    {
        return ctx.IsPathTermination();
    }

    // PT Base Functions

    void SetIsLastVertexFar(bool Value)
    {
        ctx.SetIsLastVertexFar(Value);
    }

    bool IsLastVertexFar()
    {
        return ctx.IsLastVertexFar();
    }

    void SetIsLastVertexRough(bool Value)
    {
        ctx.SetIsLastVertexRough(Value);
    }

    bool IsLastVertexRough()
    {
        return ctx.IsLastVertexRough();
    }

    bool NoMainPathRcVertex()
    {
        return ctx.NoMainPathRcVertex();
    }

#if defined(RTXDI_RESTIR_PT_INITIAL_SAMPLING)
    RTXDI_InitialSamplingPathTracerContext ctx;
#elif defined(RTXDI_RESTIR_PT_HYBRID_SHIFT)
    RTXDI_HybridShiftPathTracerContext ctx;
#endif
};

RTXDI_PathTracerContext RTXDI_InitializePathTracerContext(
    const RTXDI_PathTracerContextParameters ptParams,
    const RAB_Surface primarySurface,
    inout RTXDI_PathTracerRandomContext ptRandContext)
{
    RTXDI_PathTracerContext context = (RTXDI_PathTracerContext)0;
#if defined(RTXDI_RESTIR_PT_INITIAL_SAMPLING)
    context.ctx = RTXDI_InitializeInitialSamplingPathTracerContext(ptParams, primarySurface, ptRandContext);
#elif defined(RTXDI_RESTIR_PT_HYBRID_SHIFT)
    context.ctx = RTXDI_InitializeHybridShiftPathTracerContext(ptParams, primarySurface, ptRandContext);
#endif
    return context;
}

#endif // !defined(RTXDI_RESTIR_PT_HYBRID_SHIFT)

#endif // RTXDI_PT_PATH_TRACING_CONTEXT_HLSLI
