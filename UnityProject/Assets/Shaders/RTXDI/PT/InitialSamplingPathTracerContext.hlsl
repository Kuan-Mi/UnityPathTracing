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

#ifndef RTXDI_INITIAL_SAMPLING_PATH_TRACING_CONTEXT_HLSLI
#define RTXDI_INITIAL_SAMPLING_PATH_TRACING_CONTEXT_HLSLI

#include "PathReconnectibility.hlsl"
#include "PathTracerContextParameters.hlsl"
#include "PathTracerState.hlsl"
#include "Assets/Shaders/Rtxdi/Utils/Color.hlsl"

struct RTXDI_InitialSamplingPathTracerContext
{
    void SetBrdfRaySample(RTXDI_BrdfRaySample raySample)
    {
        ptState.brdfRaySample = raySample;
        SetContinuationRayBrdfOverPdf(raySample.BrdfTimesNoL / raySample.OutPdf);
    }

    void SetMaxPathBounce(uint16_t newMax)
    {
        ptState.maxPathBounce = newMax;
    }

    void SetMaxRcVertexLengthIfUnset(uint16_t newMax)
    {
        if(!FoundRcVertex)
            ptState.RcVertexLength = newMax;
    }

    bool ShouldRunRussianRoulette()
    {
        return ptState.bounceDepth != ptState.initialBounceDepth;
    }

    void RecordRussianRouletteProbability(float RRProb)
    {
        RussianRoulettePdf *= RRProb;
    }
    
    void MultiplyPathThroughput(float3 multiplicationFactor)
    {
        ptState.pathThroughput *= multiplicationFactor;
    }

    void SetContinuationRay(RayDesc cr)
    {
        ptState.ContinuationRay = cr;
    }

    bool AnalyzePathReconnectibilityBeforeTrace()
    {
        if (ptState.NoMainPathRcVertex())
        {
            bool IsCurrentVertexRoughForConnection = false;
            if(ReconnectionMode == RTXDI_RESTIRPT_RECONNECTION_MODE_FIXED_THRESHOLD)
            {
                IsCurrentVertexRoughForConnection = RAB_GetSurfaceRoughness(ptState.intersectionSurface) > RoughnessThreshold;
            }
            else if(ReconnectionMode == RTXDI_RESTIRPT_RECONNECTION_MODE_FOOTPRINT)
            {
                const float InverseFootprint = RTXDI_CalculateRayFootprint(RAB_GetSurfaceNormal(MainPathRcVertexSurface), RAB_GetSurfaceViewDir(ptState.intersectionSurface), ptState.LastHitT, ptState.brdfRaySample);
                IsCurrentVertexRoughForConnection = InverseFootprint > FootprintThreshold;
            }
            // If the conditions are meet, we set the current vertex to be the rcVertex by marking ptState.RcVertexLength = BouceDepth - 1
            // It is BouceDepth - 1, because we haven't traced the ray yet
            if(IsCurrentVertexRoughForConnection && ptState.IsLastVertexRough() && ptState.IsLastVertexFar())
            {
                ptState.RcVertexLength = ptState.bounceDepth - 1;
                FoundRcVertex = true;
            }
        }

        // mark the current vertex to be rough for next vertex's reconnection condition test
        // the current vertex is "last vertex" in the next vertex's eyes
        if(ReconnectionMode == RTXDI_RESTIRPT_RECONNECTION_MODE_FIXED_THRESHOLD)
        {
            ptState.SetIsLastVertexRough(RAB_GetSurfaceRoughness(ptState.intersectionSurface) > RoughnessThreshold);
        }
        else if(ReconnectionMode == RTXDI_RESTIRPT_RECONNECTION_MODE_FOOTPRINT)
        {
            ptState.SetIsLastVertexRough(ptState.brdfRaySample.OutPdf <= PdfThreshold && !ptState.brdfRaySample.properties.IsDelta());
        }

        // We memorize scatterPdf because we need to check invertibility during spatial/temporal reuse (this can change which affects the inverse footprint of the shifted path)
        if (ptState.RcVertexLength == ptState.bounceDepth - 1)
        {
            MainPathRcWiPdf = ptState.brdfRaySample.OutPdf;
            MainPathPartialJacobian = RTXDI_CalculatePartialJacobian(ptState.LastHitT, -RAB_GetSurfaceViewDir(ptState.intersectionSurface), RAB_GetSurfaceNormal(ptState.intersectionSurface));
        }

        // update MainPathRcVertex (main path RcVertex)
        // we also update this before finding the MainPathRcVertex because we use it to get previous vertex normal
        if (ptState.RcVertexLength >= ptState.bounceDepth - 1)
        {
            MainPathRcVertexSurface = ptState.intersectionSurface;
        }

        // Memorize BrdfPdf before RcVertex to convert path parameterization
        // This is useful even if MainPathRcVertex is not found (connecting directly to a light)
        if (ptState.RcVertexLength > ptState.bounceDepth - 1)
        {
            BrdfPdfBeforeRcVertex = ptState.brdfRaySample.OutPdf;
        }

        // the "suffix" path throughput only happens after rcVertexLength
        if (ptState.bounceDepth > ptState.RcVertexLength)
        {
            ptState.RcPathThroughput *= ptState.continuationRayBrdfOverPdf;
        }

        return !IsPathTerminated();
    }

    void SetTraceResult(RAB_RayPayload rp)
    {
        ptState.TraceResult = rp;
    }

    void SetIntersectionSurface(const RAB_Surface intersectionSurface)
    {
        ptState.intersectionSurface = intersectionSurface;
        RAB_SetSurfaceNormal(ptState.intersectionSurface, normalize(RAB_GetSurfaceNormal(ptState.intersectionSurface)));
    }

    void UpdateReconnectionStateForPathIntersection()
    {
        // compute the V_prev->V ray footprint, if it is above threshold, we mark last vertex as "far"
        // this is used when testing the current vertex is connectible given the location on the path sample
        if(ReconnectionMode == RTXDI_RESTIRPT_RECONNECTION_MODE_FIXED_THRESHOLD)
        {
            ptState.SetIsLastVertexFar(RAB_RayPayloadGetCommittedHitT(ptState.TraceResult) > DistanceThreshold && !ptState.brdfRaySample.properties.IsDelta());
        }
        else if(ReconnectionMode == RTXDI_RESTIRPT_RECONNECTION_MODE_FOOTPRINT)
        {
            const float RayFootprint = RTXDI_CalculateRayFootprint(RAB_GetSurfaceNormal(ptState.intersectionSurface), RAB_GetSurfaceViewDir(ptState.intersectionSurface), RAB_RayPayloadGetCommittedHitT(ptState.TraceResult), ptState.brdfRaySample);
            ptState.SetIsLastVertexFar(RayFootprint > FootprintThreshold);
        }
    }

    void RecordPathIntersection(const RAB_Surface intersectionSurface)
    {
        SetIntersectionSurface(intersectionSurface);
        UpdateReconnectionStateForPathIntersection();
    }

    bool ShouldSampleEmissiveSurfaces()
    {
        return true;
    }

    bool RecordEmissiveLightSample(float3 radianceFromEmissiveSurface, RAB_Surface prevSurface, inout RTXDI_RandomSamplerState RandContext)
    {
        const uint RcVertexLength = (ptState.NoMainPathRcVertex() && ptState.IsLastVertexFar()) ? ptState.bounceDepth : ptState.RcVertexLength;

        float PartialJacobianToLight = 0.f;
        // if the current vertex is the rcVertex, we need to compute a partial Jacobian (which is different from the main path partial Jacobian)
        if (RcVertexLength == ptState.bounceDepth)
        {
            PartialJacobianToLight = RTXDI_CalculatePartialJacobian(RAB_RayPayloadGetCommittedHitT(ptState.TraceResult), ptState.ContinuationRay.Direction, RAB_GetSurfaceNormal(ptState.intersectionSurface));
        }
        
        
        // RIS stream the current path sample, which is only the emissive portion of the intersection surface.
        float3 Le = radianceFromEmissiveSurface;

        bool accepted = RISStreamPathSample(ptState.bounceDepth, Le, RandContext, RcVertexLength, false, 1.f, ptState.brdfRaySample.OutDirection, PartialJacobianToLight);
        return accepted;
    }

    bool ShouldSampleNee()
    {
        return true;
    }

    bool RecordNeeLightSample(in const RTXDI_SampledLightData sampledLightData,
                              in const float3 radianceFromLights,
                              in const float neePdf,
                              in const float scatterPdf,
                              in const RAB_LightSample lightSample,
                              inout RTXDI_RandomSamplerState RandContext)
    {
        bool Selected = false;
        ptState.sampledLightDataForDI = sampledLightData;
        ptState.lightSampleForDI = lightSample;
        if (any(radianceFromLights > 0.f) && RTXDI_SampledLightData_IsValidLightData(ptState.sampledLightDataForDI))
        {
            RTXDI_SampledLightData lightData = ptState.sampledLightDataForDI; // UV and index
            RAB_LightSample lightSample = ptState.lightSampleForDI; // Orientation/location info for the light, plus the radiance solidAnglePdf, and polymorphicLightType

            const uint pathLength = ptState.bounceDepth + 1; // Include light vertex

            // make sure bounce count match between NEE and emissive modes
            if (pathLength > GetMaxPathBounce()) return false;
            
            // whether the current vertex is connectible (Case 2)
            bool isCurrentVertexRoughForConnection = false;
            if (ptState.NoMainPathRcVertex())
            {
                if(ReconnectionMode == RTXDI_RESTIRPT_RECONNECTION_MODE_FIXED_THRESHOLD)
                {
                    isCurrentVertexRoughForConnection = RAB_GetSurfaceRoughness(ptState.intersectionSurface) > RoughnessThreshold;
                }
                else if(ReconnectionMode == RTXDI_RESTIRPT_RECONNECTION_MODE_FOOTPRINT)
                {
                    RTXDI_BrdfRaySample brs = (RTXDI_BrdfRaySample)0;
                    brs.OutPdf = scatterPdf;
                    brs.properties.SetContinuous();
                    float inverseFootprint = RTXDI_CalculateRayFootprint(RAB_GetSurfaceNormal(MainPathRcVertexSurface), RAB_GetSurfaceViewDir(ptState.intersectionSurface), RAB_RayPayloadGetCommittedHitT(ptState.TraceResult), brs);
                    isCurrentVertexRoughForConnection = inverseFootprint > FootprintThreshold;
                }
            }

            // Three cases
            // 1) there is a MainPathRcVertex before
            // 2) not 1) and the current vertex is connectible
            // 3) not 1) and not 2) and the NEE-sampled light vertex is connectible
            // we don't allow random replay to the NEE sample (NEE sample is always reconnected if the path is not reconnected before)
            const uint rcVertexLength = ptState.NoMainPathRcVertex() ?
                (ptState.IsLastVertexRough() && ptState.IsLastVertexFar() && isCurrentVertexRoughForConnection ? pathLength - 1 : pathLength) :
                ptState.RcVertexLength;

            // in case 3), partial Jacobian is the light sample's solid angle PDF
            // in case 2), partial Jacobian is the ordinary G term

            float PartialJacobian = rcVertexLength == pathLength ? RAB_LightSampleSolidAnglePdf(lightSample) : 0.f;
            if (rcVertexLength == pathLength - 1)
            {
                PartialJacobian = RTXDI_CalculatePartialJacobian(RAB_RayPayloadGetCommittedHitT(ptState.TraceResult), ptState.ContinuationRay.Direction, RAB_GetSurfaceNormal(ptState.intersectionSurface));
            }
            
            float3 Le = radianceFromLights;

            // if the rcVertex is the light vertex, we packed lightData to store in the "Radiance" member in the reservoir (x is indicated by INF)
            // (we can re-evaluate the radiance by using RTXDI functions)

            Selected = RISStreamPathSample(pathLength, Le, RandContext,
                rcVertexLength, true, neePdf,
                float3((rcVertexLength == pathLength ? 1.f / 0.f : 1.f), asfloat(ptState.sampledLightDataForDI.lightData), asfloat(ptState.sampledLightDataForDI.uvData)),
                PartialJacobian);

            // we memorize scatterPdf because we need to check invertibility during spatial/temporal reuse (this can change which affects the inverse footprint of the shifted path)
            if (Selected && rcVertexLength == pathLength - 1)
            {
                SelectedRcWiPdf = scatterPdf;
            }
        }
        
        return Selected;
    }

    void RecordPathRadianceMiss(inout RTXDI_RandomSamplerState rng)
    {
        ptState.SetPathTermination(true);
    }

    bool RecordEnvironmentMapLightSample(const float3 environmentMapRadiance,
                                         RAB_Surface prevSurface,
                                         inout RTXDI_RandomSamplerState rng)
    {
        const float FakeDistanceHitT = RAB_DISTANT_LIGHT_DISTANCE;
        const float3 FakeWorldPosition = RAB_GetSurfaceWorldPos(ptState.intersectionSurface) + FakeDistanceHitT * ptState.ContinuationRay.Direction;

        float3 Le = environmentMapRadiance;

        const uint RcVertexLength = (ptState.NoMainPathRcVertex() && ptState.IsLastVertexRough()) ? ptState.bounceDepth : ptState.RcVertexLength;

        // if the current vertex is the rcVertex, we need to compute a partial Jacobian (which is different from the main path partial Jacobian)
        float PartialJacobianToLight = 0.f;
        if (RcVertexLength == ptState.bounceDepth)
        {
            RAB_SetSurfaceWorldPos(ptState.intersectionSurface, FakeWorldPosition);
            RAB_SetSurfaceNormal(ptState.intersectionSurface, -ptState.ContinuationRay.Direction);
            PartialJacobianToLight = RTXDI_CalculatePartialJacobian(FakeDistanceHitT, ptState.ContinuationRay.Direction, RAB_GetSurfaceNormal(ptState.intersectionSurface));
        }

        bool Selected = RISStreamPathSample(ptState.bounceDepth, Le, rng, RcVertexLength, false, 1.f, ptState.ContinuationRay.Direction, PartialJacobianToLight);
        return Selected;
    }

    float calculateRISWeight(float3 targetFunctionOverP)
    {
        float risWeight = RTXDI_Luminance(targetFunctionOverP);
        risWeight = (isnan(risWeight) || isinf(risWeight)) ? 0.f : risWeight;
        return risWeight;
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
        const float3 TargetFunctionOverP = CurrentLightRadiance * ptState.pathThroughput;

        float RISWeight = RTXDI_Luminance(TargetFunctionOverP);

        RISWeight = (isnan(RISWeight) || isinf(RISWeight)) ? 0.f : RISWeight;

        RunningWeightSum += RISWeight;
        
        if (RunningWeightSum * RTXDI_GetNextRandom(rng) < RISWeight)
        {
            bool connectBeforeLight = RcVertexLength == PathLength - 1; //connect to vertex prior to the light vertex
            bool connectToLight = RcVertexLength == PathLength; //connect to the light vertex

            SelectedPathLength = PathLength;
            Radiance = CurrentLightRadiance * ptState.RcPathThroughput;

            // Russian roulette PDF and the BSDF sampling PDF before RcVertex is baked in the path throughput
            // we want to exclude it from our target function
            // In ReSTIR PT, the target function is the path contribution in the mixed PSS-solid angle measure
            // where only the bounce V_prev_prev->V_prev->rcVertex uses solid angle measure (only f)
            // all other bounces uses PSS (f/p)
            // if rcVertexLength > pathLength, there is no rcVertex on this path, so "BSDF sampling PDF before RcVertex" should be 1
            float pdfToBeExcludedFromTargetFunction = RussianRoulettePdf * ((connectToLight && IsNEE) ? NeePdf : (RcVertexLength > PathLength ? 1.f : BrdfPdfBeforeRcVertex));
            SelectedTargetFunction = pdfToBeExcludedFromTargetFunction * CurrentLightRadiance * ptState.pathThroughput; //pHat

            SelectedRcVertexLength = RcVertexLength;
            SelectedRcWiPdf = MainPathRcWiPdf;
            
            // for these two scenarios, the rcVertex can be different from the main path
            if ((connectToLight && !IsNEE) || (connectBeforeLight && IsNEE))
            {
                SelectedRcSurface = ptState.intersectionSurface;
            }
            else
            {
                SelectedRcSurface = MainPathRcVertexSurface;
            }

            SelectedPartialJacobian = PartialJacobian == 0.f ? MainPathPartialJacobian : PartialJacobian;

            Radiance = isinf(PackedLightData.x) ? PackedLightData : Radiance; // store RTXDI sampleRef

            return true;
        }

        return false;
    }


    // Offer a chance for early termination in RAB_PathTracer.hlsli
    bool IsPathTerminated()
    {
        return ptState.IsPathTermination();
    }

    float3 GetSelectedTargetFunction()
    {
        return SelectedTargetFunction;
    }

    //
    // Reconnection Parameters
    //

    RTXDI_PTReconnectionMode GetReconnectionMode()
    {
        return ReconnectionMode;
    }

    // Fixed cutoff mode reconnection thrsholds
    float GetRoughnessThreshold()
    {
        return RoughnessThreshold;
    }

    float GetDistanceThreshold()
    {
        return DistanceThreshold;
    }

    // Footprint mode reconnection thresholds
    float GetFootprintThreshold()
    {
        return FootprintThreshold;
    }

    float GetPdfThreshold()
    {
        return PdfThreshold;
    }

    //
    // Hybrid-shift only functions
    //

    float GetCheckPreRcInverseGeoTerm()
    {
        return 0.0;
    }

    RAB_Surface GetRcPrevSurface()
    {
        return RAB_EmptySurface();
    }

    void SetRcVertexLength(uint length)
    {

    }

    void SetSelectedPathLength(uint length)
    {

    }

    float3 GetRadiance()
    {
        return Radiance;
    }

    //
    //
    // PTState functions
    //
    //

    //
    // Accessors for internal state
    //

    uint GetBounceDepth()
    {
        return ptState.bounceDepth;
    }

    uint GetMaxPathBounce()
    {
        return ptState.maxPathBounce;
    }

    RTXDI_BrdfRaySample GetBrdfRaySample()
    {
        return ptState.brdfRaySample;
    }

    RAB_Surface GetIntersectionSurface()
    {
        return ptState.intersectionSurface;
    }


    void SetContinuationRayBrdfOverPdf(float3 brdfOverPdf)
    {
        ptState.continuationRayBrdfOverPdf = brdfOverPdf;
    }

    float3 GetContinuationRayBrdfOverPdf()
    {
        return ptState.continuationRayBrdfOverPdf;
    }

    RayDesc GetContinuationRay()
    {
        return ptState.ContinuationRay;
    }

    float3 GetPathThroughput()
    {
        return ptState.pathThroughput;
    }



    RAB_RayPayload GetTraceResult()
    {
        return ptState.TraceResult;
    }



    void IncreaseBounceDepth()
    {
        ptState.bounceDepth++;
    }

    bool ValidContinuationRayBrdfOverPdf()
    {
        float lum = RTXDI_Luminance(ptState.continuationRayBrdfOverPdf);
        return lum > 0 && !isinf(lum) && !isnan(lum);
    }

    uint16_t GetRcVertexLength()
    {
        return ptState.RcVertexLength;
    }

    //
    // Original Functions
    //

    bool IsSecondaryBounce()
    {
        return ptState.IsSecondaryBounce();
    }

    // Reset partial state for the next path
    void BeginPathState()
    {
        ptState.BeginPathState();
    }

    // Hybrid shift functions

    void SetPathTermination(bool Value)
    {
        ptState.SetPathTermination(Value);
    }

    bool IsPathTermination()
    {
        return ptState.IsPathTermination();
    }

    // PT Base Functions

    void SetIsLastVertexFar(bool Value)
    {
        ptState.SetIsLastVertexFar(Value);
    }

    bool IsLastVertexFar()
    {
        return ptState.IsLastVertexFar();
    }

    void SetIsLastVertexRough(bool Value)
    {
        ptState.SetIsLastVertexRough(Value);
    }

    bool IsLastVertexRough()
    {
        return ptState.IsLastVertexRough();
    }

    // ReSTIR PT state from FReSTIRPTPathTracerState
    bool NoMainPathRcVertex()
    {
        return ptState.NoMainPathRcVertex();
    }

    // Total radiance carried from the path tree to the primary intersection divided by 
    // the pdfs of producing the path tree
    float3              Radiance;

    // RC vertex surface
    RAB_Surface         MainPathRcVertexSurface;
    float               MainPathRcWiPdf;
    float               MainPathPartialJacobian;

    // The brdf pdf w.r.t solid angle prior to the rc vertex
    float               BrdfPdfBeforeRcVertex;

    float               RunningWeightSum;
    float3              SelectedTargetFunction;
    float               SelectedRcWiPdf;
    uint                SelectedPathLength;
    uint                SelectedRcVertexLength;
    RAB_Surface         SelectedRcSurface;
    float               SelectedPartialJacobian;
    float               RussianRoulettePdf;
    float               FootprintThreshold;
    float               PdfThreshold;

    RTXDI_PTReconnectionMode ReconnectionMode;
    float               RoughnessThreshold;
    float               DistanceThreshold;

    bool                EvaluateEnvMapOnMiss;
    uint                LightSamplingMode;

    bool                FoundRcVertex;

    RTXDI_PathTracerState ptState;
};

RTXDI_InitialSamplingPathTracerContext RTXDI_InitializeInitialSamplingPathTracerContext(
    RTXDI_PathTracerContextParameters ptParams,
    const RAB_Surface primarySurface,
    inout RTXDI_PathTracerRandomContext ptRandContext)
{
    RTXDI_InitialSamplingPathTracerContext ctx = (RTXDI_InitialSamplingPathTracerContext)0;

    ctx.ptState.initialBounceDepth = 2;
    ctx.ptState.bounceDepth = ctx.ptState.initialBounceDepth;
    ctx.ptState.maxPathBounce = (uint16_t)ptParams.maxBounces;
    ctx.ptState.pathThroughput = float3(1.0f, 1.0f, 1.0f);
    ctx.ptState.intersectionSurface = primarySurface;

    ctx.RunningWeightSum = 0.f;
    ctx.SelectedPathLength = 0;
    ctx.BrdfPdfBeforeRcVertex = 1.f;
    ctx.RussianRoulettePdf = 1.f;
    ctx.MainPathRcVertexSurface = RAB_EmptySurface();
    ctx.SelectedRcSurface = RAB_EmptySurface();

    // "suffix" throughput to compute the radiance coming out of the reconnection vertex
    ctx.ptState.RcPathThroughput = float3(1.f, 1.f, 1.f);

    // the reconnection vertex length. Initially we set it to maximum, as we find the proper vertex we reduce the length
    ctx.ptState.RcVertexLength = ptParams.maxRcVertexLength;

    ctx.FoundRcVertex = false;

    ctx.ReconnectionMode = ptParams.rParams.reconnectionMode;
    RTXDI_CalculateFootprintParameters(ptParams.rParams, ptParams.rrParams, primarySurface, ptRandContext, ctx.FootprintThreshold, ctx.PdfThreshold);
    ctx.DistanceThreshold = ptParams.rParams.distanceThreshold;
    ctx.RoughnessThreshold = ptParams.rParams.roughnessThreshold;

    return ctx;
}

#endif // RTXDI_INITIAL_SAMPLING_PATH_TRACING_CONTEXT_HLSLI