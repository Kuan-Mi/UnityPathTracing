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

#ifndef RTXDI_PATH_RECONNECTIBILITY_HLSLI
#define RTXDI_PATH_RECONNECTIBILITY_HLSLI

#include "ReSTIRPTParameters.h"
#include "PathTracerRandomContext.hlsl"
#include "Assets/Shaders/Rtxdi/Utils/BrdfRaySample.hlsl"
#include "Assets/Shaders/Rtxdi/Utils/Math.hlsl"

struct RTXDI_PTReconnectionRuntimeParameters
{
    float3 cameraPos;
    float3 prevCameraPos;
    float3 prevPrevCameraPos;
    bool isPrevFrame;
};

float RTXDI_CalculatePrimaryRayFootprint(RTXDI_PTReconnectionRuntimeParameters rParams, RAB_Surface SurfaceForResampling)
{
    // Note that the footprint threshold is a ratio of the primary ray's footprint
    // If we shift a sample to the previous frame, we need to know previous frame's primary ray's footprint

    float3 PreViewTranslationOffset = (rParams.prevCameraPos - rParams.prevPrevCameraPos);
    float3 PrimaryHitDisp = RAB_GetSurfaceWorldPos(SurfaceForResampling) - (rParams.isPrevFrame ? (rParams.prevCameraPos + PreViewTranslationOffset) : rParams.cameraPos);

    return dot(PrimaryHitDisp, PrimaryHitDisp) * 4 * RTXDI_PI / abs(dot(RAB_GetSurfaceNormal(SurfaceForResampling), RAB_GetSurfaceViewDir(SurfaceForResampling)));
}

void RTXDI_CalculateFootprintParameters(
    const RTXDI_PTReconnectionParameters rParams,
    const RTXDI_PTReconnectionRuntimeParameters rrParams,
    const RAB_Surface surfaceForResampling,
    inout RTXDI_PathTracerRandomContext ptRandContext,
    inout float footprintThreshold,
    inout float pdfThreshold)
{
    const float LocalMinConnectionFootprint = RTXDI_GaussRand(rParams.minConnectionFootprint, rParams.minConnectionFootprintSigma, ptRandContext.replayRandomSamplerState);
    const float LocalMinPdfRoughness = RTXDI_GaussRand(rParams.minPdfRoughness, rParams.minPdfRoughnessSigma, ptRandContext.replayRandomSamplerState);
    float PrimaryRayFootprint = RTXDI_CalculatePrimaryRayFootprint(rrParams, surfaceForResampling);

    footprintThreshold = PrimaryRayFootprint * LocalMinConnectionFootprint * LocalMinConnectionFootprint;
    pdfThreshold = 1.f / (LocalMinPdfRoughness * LocalMinPdfRoughness);
}


float RTXDI_CalculateRayFootprint(const float3 surfNormal,
                                  const float3 surfViewDir,
                                  const float rayDist, // committedRayT
                                  const RTXDI_BrdfRaySample brs) // const float pdf)
{
    if(brs.properties.IsDelta())
        return 0.0;    
    const float pdf = brs.OutPdf;
    const float GeometryFactor = abs(dot(surfNormal, surfViewDir)) / (rayDist * rayDist);
    const float RayFootprint = 1.f / (GeometryFactor * pdf);
    return RayFootprint;
}

#endif // RTXDI_PATH_RECONNECTIBILITY_HLSLI