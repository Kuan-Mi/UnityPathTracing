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

#ifndef RTXDI_PT_HYBRID_SHIFT_HLSLI
#define RTXDI_PT_HYBRID_SHIFT_HLSLI

//
// This define selects the correct PathTracerContext implementation
// to be used for hybrid shift used during the resampling passes
// The other implementation is used for initial sampling
//
#define RTXDI_RESTIR_PT_HYBRID_SHIFT

#include "Assets/Shaders/Rtxdi/PT/PathTracerContext.hlsl"
#include "Assets/Shaders/Rtxdi/PT/PathReconnectibility.hlsl"
#include "Assets/Shaders/Rtxdi/PT/Reservoir.hlsl"

struct RTXDI_PTHybridShiftRuntimeParameters
{
    bool IsPrevFrame;
    bool IsBasePathInPrevFrame;

    float3 cameraPos;
    float3 prevCameraPos;
    float3 prevPrevCameraPos;
};

RTXDI_PathTracerContext InitializePathTracerContext(const RTXDI_PTHybridShiftPerFrameParameters hspfParams,
                                                      const RTXDI_PTHybridShiftRuntimeParameters hsrParams,
                                                      const RTXDI_PTReconnectionParameters rParams,
                                                      const RTXDI_PTReservoir NeighborSample,
                                                      const RAB_Surface SurfaceForResampling,
                                                      inout RTXDI_PathTracerRandomContext ptRandContext)
{
    RTXDI_PathTracerContextParameters ptParams = (RTXDI_PathTracerContextParameters) 0;
    ptParams.maxBounces = (uint16_t) hspfParams.maxBounceDepth;
    ptParams.maxRcVertexLength = (uint16_t) hspfParams.maxRcVertexLength;
    ptParams.RcVertexLength = (uint16_t) NeighborSample.RcVertexLength;
    ptParams.SelectedPathLength = (uint16_t) NeighborSample.PathLength;
    ptParams.rParams = rParams;
    ptParams.rrParams.isPrevFrame = hsrParams.IsPrevFrame;
    ptParams.rrParams.cameraPos = hsrParams.cameraPos;
    ptParams.rrParams.prevCameraPos = hsrParams.prevCameraPos;
    ptParams.rrParams.prevPrevCameraPos = hsrParams.prevPrevCameraPos;

    RTXDI_PathTracerContext ctx = RTXDI_InitializePathTracerContext(ptParams, SurfaceForResampling, ptRandContext);

    return ctx;
}

bool NeedToRunRandomReplayPathTracer(const RTXDI_PTReservoir NeighborSample)
{
    return NeighborSample.RcVertexLength > 2;
}

void RandomReplay(inout RTXDI_PathTracerContext ctx,
                  const RTXDI_PTReservoir NeighborSample,
                  inout RAB_Surface surfaceForResampling,
                  inout RTXDI_PathTracerRandomContext ptRandContext,
                  inout float3 targetFunction,
                  inout RAB_PathTracerUserData ptud)
{
    // set the RcVertexLength and the path length here
    // such that we know to which bounce we need to replay our path to
    ctx.SetRcVertexLength(NeighborSample.RcVertexLength);
    ctx.SetSelectedPathLength(NeighborSample.PathLength);

    RAB_PathTrace(ctx, ptRandContext, ptud);

    // surfaceForResampling is now the surface of the vertex before rcVertex
    surfaceForResampling = ctx.GetRcPrevSurface();

    // This is now the "prefix" throughput
    targetFunction = ctx.GetRadiance();
}

void CalculatePartialJacobian(const float3 RecieverPos,
                              const float3 SamplePos,
                              const float3 SampleNormal,
                              inout float DistanceToSurfaceSqr,
                              inout float CosineEmissionAngle)
{
    const float3 Vec = RecieverPos - SamplePos;

    DistanceToSurfaceSqr = dot(Vec, Vec);
    CosineEmissionAngle = saturate(dot(SampleNormal, Vec * rsqrt(DistanceToSurfaceSqr)));
}

float CalculateJacobianWithCachedJacobian(const float3 ReceiverPos,
                                          inout RTXDI_PTReservoir NeighborReservoir)
{
    const float OriginalInversePartialJacobian = NeighborReservoir.PartialJacobian;
    float NewDistanceSqr = 0.0f;
    float NewCosine = 0.0f;
    CalculatePartialJacobian(ReceiverPos, NeighborReservoir.TranslatedWorldPosition, NeighborReservoir.WorldNormal, NewDistanceSqr, NewCosine);

    const float NewPartialJacobian = NewCosine / NewDistanceSqr;
    float Jacobian = NewPartialJacobian * OriginalInversePartialJacobian;

    NeighborReservoir.PartialJacobian = 1.f / NewPartialJacobian;

    if (isinf(Jacobian) || isnan(Jacobian))
    {
        Jacobian = 0;
    }

    return Jacobian;
}

bool ConnectsToNEELight(RTXDI_PTReservoir NeighborSample)
{
    // NeighborSample.Radiance.x == INF is used as an indicator for NEE-sample lights
    return (NeighborSample.RcVertexLength == NeighborSample.PathLength) &&
            isinf(NeighborSample.Radiance.x);
}

bool ShouldComputeReconnectionJacobian(const RTXDI_PTReservoir NeighborSample,
                                       const bool ConnectToRTXDILight)
{
    // We only need to execute the reconnection code below if the rcVertex exists
    // Exception is that if the rcVertex is a NEE-sampled light vertex, it requires some different code
    // and we handle it in another code block below

    // IE only need to compute reconnection jacobian under conditions explained above and implemented here
    return NeighborSample.RcVertexLength <= NeighborSample.PathLength && !ConnectToRTXDILight;
}

float ComputeReconnectionJacobian(const RTXDI_PathTracerContext ctx,
                                  inout RTXDI_PTReservoir NeighborSample,
                                  const RAB_Surface SurfaceForResampling,
                                  inout float Jacobian)
{
    // note that we don't need the RcPrevSurface of the base path, as the partial Jacobian (geometry term) is precomputed in NeighborSample.PartialJacobian
    if (NeighborSample.RcVertexLength <= NeighborSample.PathLength)
    {
        Jacobian = CalculateJacobianWithCachedJacobian(RAB_GetSurfaceWorldPos(SurfaceForResampling), NeighborSample);
    }

    return Jacobian;
}

void ValidateInvertibilityCondition(const RTXDI_PathTracerContext ctx,
                                    const RTXDI_PTReservoir NeighborSample,
                                    const RAB_Surface SurfaceForResampling,
                                    inout float Jacobian)
{
    // Test invertibility. We need to make sure that the RcVertex is connectible in the shifted path
    float3 ReconnectionDir = NeighborSample.TranslatedWorldPosition - RAB_GetSurfaceWorldPos(SurfaceForResampling);
    float ReconnectionDist = length(ReconnectionDir);
    ReconnectionDir /= ReconnectionDist;

    if (ctx.GetReconnectionMode() == RTXDI_RESTIRPT_RECONNECTION_MODE_FIXED_THRESHOLD)
    {
        if (NeighborSample.RcVertexLength == NeighborSample.PathLength)
        {
            // Connects to an RTXDI light. Always reconnectible.
            if (isinf(NeighborSample.Radiance.x))
            {
                return;
            }
            // Connects to an emissive surface. Only reconnectible if it
            // passes the roughness and distance thresholds
            else if (ReconnectionDist < ctx.GetDistanceThreshold() || RAB_GetSurfaceRoughness(SurfaceForResampling) < ctx.GetRoughnessThreshold())
            {
                Jacobian = 0.0f;
            }
        }
        // Mid-path reconnection.
        // If the previous vertex is not far or rough enough, then the current vertex is not reconnectible.
        else if (NeighborSample.RcVertexLength < NeighborSample.PathLength)
        {
            // If the previous vertex is neither far nor rough enough, this reconnection is not valid.
            if (ReconnectionDist < ctx.GetDistanceThreshold() || RAB_GetSurfaceRoughness(SurfaceForResampling) < ctx.GetRoughnessThreshold())
            {
                Jacobian = 0.0f;
            }
        }
    }
    else if (ctx.GetReconnectionMode() == RTXDI_RESTIRPT_RECONNECTION_MODE_FOOTPRINT)
    {
        float scatterPdf = 0.f;
        if (NeighborSample.RcVertexLength <= NeighborSample.PathLength)
        {
            scatterPdf = RAB_SurfaceEvaluateBrdfPdf(SurfaceForResampling, ReconnectionDir);
        }

        float geometryFactor = abs(dot(NeighborSample.WorldNormal, ReconnectionDir)) / (ReconnectionDist * ReconnectionDist);
        float rayfootprint = 1.f / (geometryFactor * scatterPdf);

        if (rayfootprint <= ctx.GetFootprintThreshold())
        {
            Jacobian = 0.f;
        }

        // This the missing check of whether RcPrevVertex is connectible (it must not be conenctible for the shift to be invertible)
        // i.e. check the inverse footprint for the case 2 in RandomReplayTracing
        if (ctx.GetCheckPreRcInverseGeoTerm() > 0.f && ctx.GetCheckPreRcInverseGeoTerm() / scatterPdf > ctx.GetFootprintThreshold())
        {
            Jacobian = 0.f;
        }

        // This checks the whether RcPrevVertex is "rough" enough such that rcVertex is connectible
        // Note that if rcVertex a light vertex sampled by bsdf, we don't need to check this as the we relax the connectibility condition in this case
        // to encourage connection to light vertices
        // (WARNING: if we support delta material, the scatterPdf it returns must be set to a large float)
        if (scatterPdf > ctx.GetPdfThreshold() && NeighborSample.RcVertexLength < NeighborSample.PathLength)
        {
            Jacobian = 0.f;
        }

        // This checks whether the inverse footprint of the ray RcVertex -> RcPrevVertex is above the threshold
        // if it is not, then the shift is non-invertible
        if (NeighborSample.RcVertexLength < NeighborSample.PathLength)
        {
            // The BSDF sampling PDF of RcVertex->RcNextVertex will actually change due to reconnecting from a different direction
            // However, we assume they are the same to avoid the expensive BSDF evaluation at the rcVertex (we don't store the material reference anyway)
            // this is likely to be a reasonable approximation, consider that the rcVertex is "far enough to be rough" when we set it up in initial sampling
            float rcScatterPdf = NeighborSample.RcWiPdf;
            float inverseGeometryFactor = abs(dot(RAB_GetSurfaceNormal(SurfaceForResampling), ReconnectionDir)) / (ReconnectionDist * ReconnectionDist);
            float inversefootprint = 1.f / (inverseGeometryFactor * rcScatterPdf);
            if (inversefootprint <= ctx.GetFootprintThreshold())
                Jacobian = 0.f;
        }
    }
}

void ValidateLightIDAndReservoir(const RTXDI_PTHybridShiftRuntimeParameters hsrParams,
                                 inout RTXDI_PTReservoir NeighborSample,
                                 inout RTXDI_SampledLightData sampledLightData,
                                 inout float3 TargetFunction)
{
    if (RTXDI_SampledLightData_IsValidLightData(sampledLightData))
    {
        int mappedLightID = -1;

        bool remapped = false;

        if (hsrParams.IsBasePathInPrevFrame)
        {
            mappedLightID = RAB_TranslateLightIndex(RTXDI_SampledLightData_GetLightIndex(sampledLightData), false);
            NeighborSample.Radiance.y = asfloat(RTXDI_LightIndexToLightData(mappedLightID));
            remapped = true;
        }

        if (remapped)
        {
            // invalid index
            if (mappedLightID == -1)
            {
                TargetFunction *= 0.f;
                sampledLightData = RTXDI_SampledLightData_CreateInvalidData();
            }
            else
            {
                RTXDI_SampledLightData_SetLightData(sampledLightData, mappedLightID);
            }
        }
    }
}

// A helper used for pairwise MIS computations.  This might be able to simplify code elsewhere, too.
float3 RTXDI_ComputeTargetFunctionWithLightPdf(const RTXDI_PTHybridShiftRuntimeParameters hsrParams,
                                               const RTXDI_SampledLightData sampledLightData,
                                               const RAB_Surface Surface,
                                               inout float lightPdf,
                                               inout float3 lightPos,
                                               inout RAB_LightSample lightSample)
{
    RAB_LightInfo lightInfo = RAB_LoadLightInfo(RTXDI_SampledLightData_GetLightIndex(sampledLightData), hsrParams.IsPrevFrame);
    lightSample = RAB_SamplePolymorphicLight(lightInfo, Surface, RTXDI_SampledLightData_GetUVDataFloat2(sampledLightData));

    const bool IsVisible = RAB_GetConservativeVisibility(Surface, lightSample);

    float3 TargetFunction = IsVisible ? RAB_GetReflectedBsdfRadianceForSurface(RAB_LightSamplePosition(lightSample), RAB_LightSampleRadiance(lightSample), Surface) : float3(0.0f, 0.0f, 0.0f);

    float3 surfaceToLightVector = RAB_LightSamplePosition(lightSample) - RAB_GetSurfaceWorldPos(Surface);
    float distanceToLight = length(surfaceToLightVector);
    float3 lightDir = surfaceToLightVector / distanceToLight;

    lightPdf = RAB_LightSampleSolidAnglePdf(lightSample); // pdf is inverse G
    lightPos = RAB_LightSamplePosition(lightSample);

    return TargetFunction / RAB_LightSampleSolidAnglePdf(lightSample);
}

void UpdateReconnectionForRTXDIConnectedLight(const RTXDI_PTHybridShiftRuntimeParameters hsrParams,
                                              const RTXDI_PathTracerContext ctx,
                                              const RAB_Surface SurfaceForResampling,
                                              inout float Jacobian,
                                              inout float3 TargetFunction,
                                              inout RTXDI_PTReservoir NeighborSample,
                                              inout RAB_PathTracerUserData ptud)
{
    // fetch sampledLightData packed in Radiance
    RTXDI_SampledLightData sampledLightData = RTXDI_GetSampledLightData(NeighborSample);
	
    ValidateLightIDAndReservoir(hsrParams, NeighborSample, sampledLightData, TargetFunction);

    // Compute target function and Jacobian using some RTXDI functions
    // Since we are using solid angle measure, we need solid angle light sampling PDF (which contains a geomtery term)
    // to compute Jacobian of reconnection
    float InvPartialJacobian = 1.f;
    float3 lightPos = float3(0.0f, 0.0f, 0.0f);
    RAB_LightSample lightSample = RAB_EmptyLightSample();
    float3 lightReflectedRadiance = RTXDI_ComputeTargetFunctionWithLightPdf(hsrParams, sampledLightData, SurfaceForResampling, InvPartialJacobian, lightPos, lightSample);
    
    const float OriginalInvPartialJacobian = NeighborSample.PartialJacobian; // yes, this stores the src sample light pdf
    TargetFunction *= lightReflectedRadiance;
    Jacobian = InvPartialJacobian == 0.f ? 0.f : OriginalInvPartialJacobian / InvPartialJacobian;
    NeighborSample.PartialJacobian = InvPartialJacobian; // override this reservoir member with the shifted path's partial jacobian

    const float3 LightDirection = normalize(lightPos - RAB_GetSurfaceWorldPos(SurfaceForResampling));
    
    RAB_LastBounceDenoiserCallback(lightPos, SurfaceForResampling, ptud);

    float ScatterPdf = 0.f;
    if (ctx.GetReconnectionMode() == RTXDI_RESTIRPT_RECONNECTION_MODE_FOOTPRINT)
        ScatterPdf = RAB_SurfaceEvaluateBrdfPdf(SurfaceForResampling, LightDirection);
        
    // invertibility check 
    // an NEE-sampled light vertex is by definition always connectible
    // We only need to ensure that the vertex before rcVertex is not connectible
    // basically we just need to complete the missing check of inverse footprint for case 2 in RandomReplayTracing
    if (ctx.GetReconnectionMode() == RTXDI_RESTIRPT_RECONNECTION_MODE_FOOTPRINT)
    {
        if (ctx.GetCheckPreRcInverseGeoTerm() > 0.f && ctx.GetCheckPreRcInverseGeoTerm() / ScatterPdf > ctx.GetFootprintThreshold())
        {
            Jacobian = 0.f;
        }
    }
    
    TargetFunction *= RAB_GetMISWeightForNEE(RTXDI_SampledLightData_GetLightIndex(sampledLightData), lightSample, LightDirection, InvPartialJacobian, ScatterPdf);
}

void ComputeHybridShift(inout RAB_Surface SurfaceForResampling,
                        inout RTXDI_PTReservoir NeighborSample,
                        RTXDI_PTHybridShiftPerFrameParameters hspfParams,
                        RTXDI_PTHybridShiftRuntimeParameters hsrParams,
                        RTXDI_PTReconnectionParameters rParams,
						inout float3 TargetFunction,
                        inout float Jacobian,
                        inout RAB_PathTracerUserData ptud)
{
    Jacobian = 1.f;
    TargetFunction = float3(1.0f, 1.0f, 1.0f);

    RTXDI_PathTracerRandomContext PTRandContext = (RTXDI_PathTracerRandomContext) 0;
    PTRandContext.replayRandomSamplerState = RTXDI_CreateRandomSamplerFromDirectSeed(NeighborSample.RandomSeed, NeighborSample.RandomIndex);

    RTXDI_PathTracerContext ctx = InitializePathTracerContext(hspfParams, hsrParams, rParams, NeighborSample, SurfaceForResampling, PTRandContext);

    bool needToRunRandomReplayPathTracer = NeedToRunRandomReplayPathTracer(NeighborSample);

#if RTXDI_PATHTRACER_USE_REORDERING
    // this helps performance a lot. Basically all threads that don't need to do random replay will be grouped together
    NvReorderThread((needToRunRandomReplayPathTracer ? 1 : 0), 1);
#endif

    if (needToRunRandomReplayPathTracer)
    {
        RandomReplay(ctx, NeighborSample, SurfaceForResampling, PTRandContext, TargetFunction, ptud);
    }

    const bool ConnectToRTXDILight = ConnectsToNEELight(NeighborSample);
    if (ShouldComputeReconnectionJacobian(NeighborSample, ConnectToRTXDILight))
    {
        Jacobian = ComputeReconnectionJacobian(ctx, NeighborSample, SurfaceForResampling, Jacobian);
        ValidateInvertibilityCondition(ctx, NeighborSample, SurfaceForResampling, Jacobian);

        RAB_ReconnectionDenoiserCallback(NeighborSample, SurfaceForResampling, ptud);
    }

    // Handle the nee-sampled light vertex reconnection here
    if (ConnectToRTXDILight)
    {
        UpdateReconnectionForRTXDIConnectedLight(hsrParams, ctx, SurfaceForResampling, Jacobian, TargetFunction, NeighborSample, ptud);
    }
}

#endif // RTXDI_PT_HYBRID_SHIFT_HLSLI