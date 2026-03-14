/***************************************************************************
 # Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

#ifndef RTXDI_GI_JACOBIAN_MATH_HLSLI
#define RTXDI_GI_JACOBIAN_MATH_HLSLI

#include "Reservoir.hlsli"

// Calculate the elements of the Jacobian to transform the sample's solid angle.
void RTXDI_CalculatePartialJacobian(const float3 recieverPos, const float3 samplePos, const float3 sampleNormal,
    out float distanceToSurface, out float cosineEmissionAngle)
{
    float3 vec = recieverPos - samplePos;

    distanceToSurface = length(vec);
    cosineEmissionAngle = saturate(dot(sampleNormal, vec / distanceToSurface));
}

// Calculates the full Jacobian for resampling neighborReservoir into a new receiver surface
float RTXDI_CalculateJacobian(float3 recieverPos, float3 neighborReceiverPos, const RTXDI_GIReservoir neighborReservoir)
{
    // Calculate Jacobian determinant to adjust weight.
    // See Equation (11) in the ReSTIR GI paper.
    float originalDistance, originalCosine;
    float newDistance, newCosine;
    RTXDI_CalculatePartialJacobian(recieverPos, neighborReservoir.position, neighborReservoir.normal, newDistance, newCosine);
    RTXDI_CalculatePartialJacobian(neighborReceiverPos, neighborReservoir.position, neighborReservoir.normal, originalDistance, originalCosine);

    float jacobian = (newCosine * originalDistance * originalDistance)
        / (originalCosine * newDistance * newDistance);

    if (isinf(jacobian) || isnan(jacobian))
        jacobian = 0;

    return jacobian;
}

#endif // RTXDI_GI_JACOBIAN_MATH_HLSLI
