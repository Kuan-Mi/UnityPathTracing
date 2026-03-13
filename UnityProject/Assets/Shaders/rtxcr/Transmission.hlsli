/*
* Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

#ifndef _RTXCR_SUBSURFACETRANSMISSION_HLSLI_
#define _RTXCR_SUBSURFACETRANSMISSION_HLSLI_

#include "utils/RtxcrBsdf.hlsli"

#include "SubsurfaceMaterial.hlsli"

float3 RTXCR_CalculateRefractionRay(
    in const RTXCR_SubsurfaceInteraction subsurfaceInteraction,
    in const float2 rand2)
{
    // Note: We are doing cosine lobe importance sampling by default, we don't need the pdf because it will be canceled out with BSDF
    //       In case you are using other refraction sampling methods, you need to write your own function to generate refraction ray and calculate PDF
    float bsdfSamplePdf = 0.0f;
    const float3 sampleDirectionLocal = RTXCR_SampleHemisphere(rand2, bsdfSamplePdf);

    const float3x3 tangentBasis = float3x3(subsurfaceInteraction.tangent, -subsurfaceInteraction.biTangent, -subsurfaceInteraction.normal);
    // Note: The tangentBasis is an orthogonal matrix, so we can just do transpose to get the inverse matrix.
    //        This also avoids the issue that HLSL doesn't have inverse matrix intrinsics.
    const float3x3 tangentToWorld = transpose(tangentBasis);
    const float3 refractedRayDirection = mul(tangentToWorld, sampleDirectionLocal);

    return refractedRayDirection;
}

float3 RTXCR_EvaluateBoundaryTerm(
    in const float3 normal,
    in const float3 vectorToLight,
    in const float3 refractedRayDirection,
    in const float3 backfaceNormal,
    in const float thickness,
    in const RTXCR_SubsurfaceMaterialCoefficients sssMaterialCoeffcients)
{
    const float3 boundaryBsdf = RTXCR_EvalLambertianBRDF(backfaceNormal, vectorToLight, sssMaterialCoeffcients.albedo);
    const float3 frontLambertBsdf = RTXCR_EvalLambertianBRDF(-normal, refractedRayDirection, sssMaterialCoeffcients.albedo);
    const float3 volumetricAttenuation = RTXCR_EvalBeerLambertAttenuation(sssMaterialCoeffcients.sigma_t, thickness);

    return boundaryBsdf * volumetricAttenuation * frontLambertBsdf;
}

float3 RTXCR_EvaluateSingleScattering(
    in const float3 vectorToLight,
    in const float3 scatteringBoundaryNormal,
    in const float totalScatteringDistance,
    in const RTXCR_SubsurfaceMaterialCoefficients sssMaterialCoeffcients)
{
    const float3 scatteringBoundaryBsdf = RTXCR_EvalLambertianBRDF(scatteringBoundaryNormal, vectorToLight, sssMaterialCoeffcients.albedo);
    const float3 volumetricAttenuation = RTXCR_EvalBeerLambertAttenuation(sssMaterialCoeffcients.sigma_t, totalScatteringDistance);
    return sssMaterialCoeffcients.sigma_s * scatteringBoundaryBsdf * volumetricAttenuation;
}

#endif
