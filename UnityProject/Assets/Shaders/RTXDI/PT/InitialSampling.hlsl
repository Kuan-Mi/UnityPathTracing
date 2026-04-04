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

#ifndef RTXDI_RESTIR_PT_INITIAL_SAMPLING_HLSLI
#define RTXDI_RESTIR_PT_INITIAL_SAMPLING_HLSLI

//
// This define selects the correct PathTracerContext implementation
// to be used for the initial sampling pass
// The other implementation is used for hybrid shift during resampling
//
#define RTXDI_RESTIR_PT_INITIAL_SAMPLING

#include "Assets/Shaders/Rtxdi/PT/PathReconnectibility.hlsl"
#include "Assets/Shaders/Rtxdi/PT/PathTracerContext.hlsl"
#include "Assets/Shaders/Rtxdi/PT/PathTracerRandomContext.hlsl"
#include "Assets/Shaders/Rtxdi/PT/Reservoir.hlsl"

struct RTXDI_PTInitialSamplingRuntimeParameters
{
    float3 cameraPos;
    float3 prevCameraPos;
    float3 prevPrevCameraPos;
};

RTXDI_PTInitialSamplingRuntimeParameters RTXDI_EmptyPTInitialSamplingRuntimeParameters()
{
    return (RTXDI_PTInitialSamplingRuntimeParameters)0;
}

RTXDI_PTReservoir InitializePTReservoirFromPTContext(RTXDI_PathTracerContext ctx, RTXDI_RandomSamplerState cachedReplayRng)
{
    RTXDI_PTReservoir reservoir = RTXDI_MakePTReservoir(
            ctx.GetSelectedTargetFunction(), // target function (to be filled later)
            cachedReplayRng.seed,
            cachedReplayRng.index,
            ctx.GetSelectedRcVertexLength(),
            ctx.GetSelectedPathLength(),
            ctx.GetSelectedPartialJacobian(),
            ctx.GetSelectedRcWiPdf(),
            RAB_GetSurfaceWorldPos(ctx.GetSelectedRcSurface()),
            RAB_GetSurfaceNormal(ctx.GetSelectedRcSurface()),
            ctx.GetRadiance(),
            0.f);

    const float PHat = RTXDI_Luminance(ctx.GetSelectedTargetFunction());
    reservoir.WeightSum = (PHat == 0.f) ? 0.f : (ctx.GetRunningWeightSum() / PHat);

    return reservoir;
}

void FinalizeResampling(inout RTXDI_PTReservoir selectedReservoir, float3 selectedTargetFunction)
{
    const float NormalizationNumerator = 1.0;
    const float NormalizationDenominator = RTXDI_Luminance(selectedTargetFunction) * selectedReservoir.M;
    RTXDI_FinalizeResampling(selectedReservoir, NormalizationNumerator, NormalizationDenominator);
}

RTXDI_PathTracerContextParameters GetPTContextParams(
    RTXDI_PTInitialSamplingParameters initialSamplingParams,
    RTXDI_PTInitialSamplingRuntimeParameters isrParams,
    RTXDI_PTReconnectionParameters rParams)
{
    RTXDI_PathTracerContextParameters ctxParams = (RTXDI_PathTracerContextParameters)0;
    ctxParams.maxBounces = (uint16_t)initialSamplingParams.maxBounceDepth;
    ctxParams.maxRcVertexLength = (uint16_t)initialSamplingParams.maxRcVertexLength;
    ctxParams.rParams = rParams;
    ctxParams.rrParams.cameraPos = isrParams.cameraPos;
    ctxParams.rrParams.prevCameraPos = isrParams.prevCameraPos;
    ctxParams.rrParams.prevPrevCameraPos = isrParams.prevPrevCameraPos;
    ctxParams.rrParams.isPrevFrame = false;
    return ctxParams;
}

RTXDI_PTReservoir GenerateInitialSamples(RTXDI_PTInitialSamplingParameters initialSamplingParams,
                                         RTXDI_PTInitialSamplingRuntimeParameters isrParams,
                                         RTXDI_PTReconnectionParameters rParams,
                                         RTXDI_PathTracerRandomContext ptRandContext,
                                         RAB_Surface surface,
                                         inout RAB_PathTracerUserData ptud)
{
    RTXDI_PTReservoir selectedReservoir = RTXDI_EmptyPTReservoir();

    RAB_PathTracerUserDataSetPathType(ptud, RTXDI_PTPathTraceInvocationType_Initial);

    float3 selectedTargetFunction = float3(0.0, 0.0, 0.0);

    for(int i = 0; i < initialSamplingParams.numInitialSamples; i++)
    {
        RTXDI_RandomSamplerState cachedRrRNG = ptRandContext.replayRandomSamplerState;

        RTXDI_PathTracerContextParameters ctxParams = GetPTContextParams(initialSamplingParams, isrParams, rParams);
        RTXDI_PathTracerContext ctx = RTXDI_InitializePathTracerContext(ctxParams, surface, ptRandContext);

        RAB_PathTrace(ctx, ptRandContext, ptud);
		
        RTXDI_PTReservoir candidateReservoir = InitializePTReservoirFromPTContext(ctx, cachedRrRNG);

        if(CombineReservoirs(selectedReservoir, candidateReservoir, RTXDI_GetNextRandom(ptRandContext.initialRandomSamplerState), ctx.GetSelectedTargetFunction()))
        {
            selectedTargetFunction = ctx.GetSelectedTargetFunction();
        }
    }

    FinalizeResampling(selectedReservoir, selectedTargetFunction);

    return selectedReservoir;
}

#endif // RTXDI_RESTIR_PT_INITIAL_SAMPLING_HLSLI