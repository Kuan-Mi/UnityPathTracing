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

#ifndef _RTXCR_SUBSURFACEMATERIAL_HLSLI_
#define _RTXCR_SUBSURFACEMATERIAL_HLSLI_

#include "utils/RtxcrMath.hlsli"

#define MAX_SSS_SAMPLE_COUNT 4

#define SSS_METERS_UNIT 0.01f // 1cm

#define SSS_MIN_ALBEDO 0.01f

/************************************************
    Subsurface Material
************************************************/

struct RTXCR_SubsurfaceMaterialData
{
    float3 transmissionColor;
    float  g;

    float3 scatteringColor;
    float  scale;
};

struct RTXCR_SubsurfaceInteraction
{
    float3 centerPosition;

    float3 normal;
    float3 tangent;
    float3 biTangent;
};

struct RTXCR_SubsurfaceSample
{
    float3 samplePosition;
    float3 bssrdfWeight;
};

struct RTXCR_VolumeCoefficients
{
    float3 scattering;
    float3 absorption;
};

struct RTXCR_SubsurfaceMaterialCoefficients
{
    float3 sigma_s;
    float3 sigma_t;
    float3 albedo;
    float3 ssAlbedo;
};

// Helper functions
RTXCR_SubsurfaceMaterialData RTXCR_CreateDefaultSubsurfaceMaterialData()
{
    RTXCR_SubsurfaceMaterialData subsurfaceMaterialData;
    subsurfaceMaterialData.transmissionColor = float3(0.0f, 0.0f, 0.0f);
    subsurfaceMaterialData.scatteringColor = float3(0.0f, 0.0f, 0.0f);
    subsurfaceMaterialData.g = 0.0f;
    subsurfaceMaterialData.scale = 0.0f;
    return subsurfaceMaterialData;
}

RTXCR_SubsurfaceInteraction RTXCR_CreateSubsurfaceInteraction(
    const float3 centerPosition,
    const float3 normal,
    const float3 tangent,
    const float3 biTangent)
{
    RTXCR_SubsurfaceInteraction subsurfaceInteraction;
    subsurfaceInteraction.centerPosition = centerPosition;
    subsurfaceInteraction.normal = normal;
    subsurfaceInteraction.tangent = tangent;
    subsurfaceInteraction.biTangent = biTangent;

    return subsurfaceInteraction;
}

//https://blog.selfshadow.com/publications/s2017-shading-course/imageworks/s2017_pbs_imageworks_slides_v2.pdf
float3 RTXCR_ComputeTransmissionAlbedo(in const float3 transmissionColor)
{
    return float3(4.09712f, 4.09712f, 4.09712f) +
           (4.20863f * transmissionColor) -
           RTXCR_Sqrt0(9.59217f +
                       41.6808f * transmissionColor +
                       17.7126f * transmissionColor * transmissionColor);
}

RTXCR_VolumeCoefficients RTXCR_ComputeSubsurfaceVolumeCoefficients(in const RTXCR_SubsurfaceMaterialData sssData)
{
    const float3 s = RTXCR_ComputeTransmissionAlbedo(sssData.transmissionColor);
    const float3 alpha = (RTXCR_WhiteColor - s * s) / max(RTXCR_WhiteColor - sssData.g * (s * s), 1e-7f);
    const float scale = SSS_METERS_UNIT * sssData.scale;
    const float3 scatteringRadius = max(scale.rrr * sssData.scatteringColor, 1e-7f);

    RTXCR_VolumeCoefficients subsurfaceVolumeCoefficients;
    subsurfaceVolumeCoefficients.scattering = alpha / scatteringRadius;
    subsurfaceVolumeCoefficients.absorption =
        (RTXCR_WhiteColor / scatteringRadius) - subsurfaceVolumeCoefficients.scattering;

    return subsurfaceVolumeCoefficients;
}

RTXCR_SubsurfaceMaterialCoefficients RTXCR_ComputeSubsurfaceMaterialCoefficients(in const RTXCR_SubsurfaceMaterialData sssData)
{
    RTXCR_VolumeCoefficients volumeCoefficients = RTXCR_ComputeSubsurfaceVolumeCoefficients(sssData);
    const float3 sigma_a = volumeCoefficients.absorption;
    const float3 sigma_s = volumeCoefficients.scattering;
    const float3 sigma_t = max(sigma_a + sigma_s, 1e-7f);

    const float3 mfp = 1.0f.rrr / sigma_t;
    const float3 s = RTXCR_Sqrt0(sigma_a * mfp); // sigma_a / sigma_t

    // custom diffuse albedo prediction based on MC simulation of isotropic scattering, diffuse transmittance on entry
    // and Fresnel reflection back into the volume assuming ior = 1.4 (as if the air outside was denser)
    RTXCR_SubsurfaceMaterialCoefficients subsurfaceMaterialCoefficients;
    subsurfaceMaterialCoefficients.sigma_s = sigma_s;
    subsurfaceMaterialCoefficients.sigma_t = sigma_a + sigma_s;
    subsurfaceMaterialCoefficients.albedo = 0.88f * (1.0f - s) / (1.0f + 1.5535f * s);
    subsurfaceMaterialCoefficients.ssAlbedo = max(SSS_MIN_ALBEDO, sigma_s / sigma_t);

    return subsurfaceMaterialCoefficients;
}

#endif
