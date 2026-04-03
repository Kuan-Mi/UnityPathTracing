/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef INITIAL_SAMPLING_FUNCTIONS_HLSLI
#define INITIAL_SAMPLING_FUNCTIONS_HLSLI

#include "Rtxdi/DI/ReSTIRDIParameters.h"
#include "Rtxdi/DI/Reservoir.hlsli"
#include "Rtxdi/LightSampling/LocalLightSelection.hlsli"
#ifdef RTXDI_RIS_BUFFER_HLSLI
#include "Rtxdi/LightSampling/RISBuffer.hlsli"
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
#include "Rtxdi/ReGIR/ReGIRSampling.hlsli"
#endif
#endif // RTXDI_RIS_BUFFER_HLSLI
#include "Rtxdi/LightSampling/UniformSampling.hlsli"

#define RTXDI_STRATIFY_LOCAL_SAMPLING 1

//
// MIS functions
//

struct RTXDI_InitialSamplingMisData
{
	uint numMisSamples;
	float localLightMisWeight;
	float environmentMapMisWeight;
	float brdfMisWeight;
};

RTXDI_InitialSamplingMisData RTXDI_ComputeInitialSamplingMisData(RTXDI_DIInitialSamplingParameters initialSamplingParams)
{
	RTXDI_InitialSamplingMisData result;

	result.numMisSamples = initialSamplingParams.numLocalLightSamples +
		initialSamplingParams.numEnvironmentSamples +
		initialSamplingParams.numBrdfSamples;

	result.localLightMisWeight = float(initialSamplingParams.numLocalLightSamples) / result.numMisSamples;
	result.environmentMapMisWeight = float(initialSamplingParams.numEnvironmentSamples) / result.numMisSamples;
	result.brdfMisWeight = float(initialSamplingParams.numBrdfSamples) / result.numMisSamples;

	return result;
}

// Heuristic to determine a max visibility ray length from a PDF wrt. solid angle.
float RTXDI_BrdfMaxDistanceFromPdf(float brdfCutoff, float pdf)
{
    const float kRayTMax = 3.402823466e+38F; // FLT_MAX
    return brdfCutoff > 0.f ? sqrt((1.f / brdfCutoff - 1.f) * pdf) : kRayTMax;
}

// Computes the multi importance sampling pdf for brdf and light sample.
// For light and BRDF PDFs wrt solid angle, blend between the two.
//      lightSelectionPdf is a dimensionless selection pdf
float RTXDI_LightBrdfMisWeight(RAB_Surface surface, RAB_LightSample lightSample,
    float lightSelectionPdf, float lightMisWeight, bool isEnvironmentMap,
    float brdfMisWeight, float brdfCutoff)
{
    float lightSolidAnglePdf = RAB_LightSampleSolidAnglePdf(lightSample);
    if (brdfMisWeight == 0 || RAB_IsAnalyticLightSample(lightSample) ||
        lightSolidAnglePdf <= 0 || isinf(lightSolidAnglePdf) || isnan(lightSolidAnglePdf))
    {
        // BRDF samples disabled or we can't trace BRDF rays MIS with analytical lights
        return lightMisWeight * lightSelectionPdf;
    }

    float3 lightDir;
    float lightDistance;
    RAB_GetLightDirDistance(surface, lightSample, lightDir, lightDistance);

    // Compensate for ray shortening due to brdf cutoff, does not apply to environment map sampling
    float brdfPdf = RAB_SurfaceEvaluateBrdfPdf(surface, lightDir);
    float maxDistance = RTXDI_BrdfMaxDistanceFromPdf(brdfCutoff, brdfPdf);
    if (!isEnvironmentMap && lightDistance > maxDistance)
        brdfPdf = 0.f;

    // Convert light selection pdf (unitless) to a solid angle measurement
    float sourcePdfWrtSolidAngle = lightSelectionPdf * lightSolidAnglePdf;

    // MIS blending against solid angle pdfs.
    float blendedPdfWrtSolidangle = lightMisWeight * sourcePdfWrtSolidAngle + brdfMisWeight * brdfPdf;

    // Convert back, RTXDI divides shading again by this term later
    return blendedPdfWrtSolidangle / lightSolidAnglePdf;
}

//
// Local light UV selection and reservoir streaming
//

float2 RTXDI_RandomlySelectLocalLightUV(inout RTXDI_RandomSamplerState rng)
{
    float2 uv;
    uv.x = RTXDI_GetNextRandom(rng);
    uv.y = RTXDI_GetNextRandom(rng);
    return uv;
}

bool RTXDI_StreamLocalLightAtUVIntoReservoir(
    inout RTXDI_RandomSamplerState rng,
    RTXDI_InitialSamplingMisData misData,
    RAB_Surface surface,
	float brdfCutoff,
	float localLightMisWeight,
    uint lightIndex,
    float2 uv,
    float invSourcePdf,
    RAB_LightInfo lightInfo,
    inout RTXDI_DIReservoir state,
    inout RAB_LightSample o_selectedSample)
{
    RAB_LightSample candidateSample = RAB_SamplePolymorphicLight(lightInfo, surface, uv);
    float blendedSourcePdf = RTXDI_LightBrdfMisWeight(surface, candidateSample, 1.0 / invSourcePdf,
        misData.localLightMisWeight, false, misData.brdfMisWeight, brdfCutoff);
    float targetPdf = RAB_GetLightSampleTargetPdfForSurface(candidateSample, surface);
    float risRnd = RTXDI_GetNextRandom(rng);

    if (blendedSourcePdf == 0)
    {
        return false;
    }
    bool selected = RTXDI_StreamSample(state, lightIndex, uv, risRnd, targetPdf, 1.0 / blendedSourcePdf);

    if (selected) {
        o_selectedSample = candidateSample;
    }
    return true;
}

//
// ReGIR-based RIS for local lights. See PresampleReGIR.hlsl
//

#if RTXDI_ENABLE_PRESAMPLING
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED

int RTXDI_CalculateReGIRCellIndex(
    inout RTXDI_RandomSamplerState coherentRng,
    ReGIR_Parameters regirParams,
    RAB_Surface surface)
{
    int cellIndex = -1;
    float3 cellJitter = float3(
        RTXDI_GetNextRandom(coherentRng),
        RTXDI_GetNextRandom(coherentRng),
        RTXDI_GetNextRandom(coherentRng));
    cellJitter -= 0.5;

    float3 samplingPos = RAB_GetSurfaceWorldPos(surface);
    float jitterScale = RTXDI_ReGIR_GetJitterScale(regirParams, samplingPos);
    samplingPos += cellJitter * jitterScale;

    cellIndex = RTXDI_ReGIR_WorldPosToCellIndex(regirParams, samplingPos);
    return cellIndex;
}

RTXDI_RISTileInfo RTXDI_SelectLocalLightReGIRRISTile(
    int cellIndex,
    ReGIR_CommonParameters regirCommon)
{
    RTXDI_RISTileInfo tileInfo;
    uint cellBase = uint(cellIndex)*regirCommon.lightsPerCell;
    tileInfo.risTileOffset = cellBase + regirCommon.risBufferOffset;
    tileInfo.risTileSize = regirCommon.lightsPerCell;
    return tileInfo;
}

#endif // RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
#endif // RTXDI_ENABLE_PRESAMPLING

#if RTXDI_ENABLE_PRESAMPLING
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
RTXDI_LocalLightSelectionContext RTXDI_InitializeLocalLightSelectionContextReGIRRIS(
    inout RTXDI_RandomSamplerState coherentRng,
    RTXDI_LightBufferRegion localLightBufferRegion,
    RTXDI_RISBufferSegmentParameters localLightRISBufferSegmentParams,
    ReGIR_Parameters regirParams,
    RAB_Surface surface)
{
    RTXDI_LocalLightSelectionContext ctx;
    int cellIndex = RTXDI_CalculateReGIRCellIndex(coherentRng, regirParams, surface);
    if (cellIndex >= 0)
    {
        ctx = RTXDI_InitializeLocalLightSelectionContextRIS(RTXDI_SelectLocalLightReGIRRISTile(cellIndex, regirParams.commonParams));
    }
    else if (regirParams.commonParams.localLightSamplingFallbackMode == ReSTIRDI_LocalLightSamplingMode_POWER_RIS)
    {
        ctx = RTXDI_InitializeLocalLightSelectionContextRIS(coherentRng, localLightRISBufferSegmentParams);
    }
    else
    {
        ctx = RTXDI_InitializeLocalLightSelectionContextUniform(localLightBufferRegion);
    }
    return ctx;
}
#endif // RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
#endif // RTXDI_ENABLE_PRESAMPLING

RTXDI_LocalLightSelectionContext RTXDI_InitializeLocalLightSelectionContext(
    inout RTXDI_RandomSamplerState coherentRng,
    ReSTIRDI_LocalLightSamplingMode localLightSamplingMode,
    RTXDI_LightBufferRegion localLightBufferRegion
#if RTXDI_ENABLE_PRESAMPLING
    ,RTXDI_RISBufferSegmentParameters localLightRISBufferSegmentParams
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
    ,ReGIR_Parameters regirParams
    ,RAB_Surface surface
#endif
#endif
    )
{
    RTXDI_LocalLightSelectionContext ctx;
#if RTXDI_ENABLE_PRESAMPLING
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
    if (localLightSamplingMode == ReSTIRDI_LocalLightSamplingMode_REGIR_RIS)
    {
        ctx = RTXDI_InitializeLocalLightSelectionContextReGIRRIS(coherentRng, localLightBufferRegion, localLightRISBufferSegmentParams, regirParams, surface);
    }
    else
#endif // RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
    if (localLightSamplingMode == ReSTIRDI_LocalLightSamplingMode_POWER_RIS)
    {
        ctx = RTXDI_InitializeLocalLightSelectionContextRIS(coherentRng, localLightRISBufferSegmentParams);
    }
    else
#endif // RTXDI_ENABLE_PRESAMPLING
    {
        ctx = RTXDI_InitializeLocalLightSelectionContextUniform(localLightBufferRegion);
    }
    return ctx;
}

RTXDI_DIReservoir RTXDI_SampleLocalLightsInternal(
    inout RTXDI_RandomSamplerState rng,
    inout RTXDI_RandomSamplerState coherentRng,
    RAB_Surface surface,
    RTXDI_DIInitialSamplingParameters sampleParams,
	RTXDI_InitialSamplingMisData misData,
    ReSTIRDI_LocalLightSamplingMode localLightSamplingMode,
    RTXDI_LightBufferRegion localLightBufferRegion,
#if RTXDI_ENABLE_PRESAMPLING
    RTXDI_RISBufferSegmentParameters localLightRISBufferSegmentParams,
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
    ReGIR_Parameters regirParams,
#endif
#endif
    out RAB_LightSample o_selectedSample)
{
    RTXDI_DIReservoir state = RTXDI_EmptyDIReservoir();

    RTXDI_LocalLightSelectionContext lightSelectionContext = RTXDI_InitializeLocalLightSelectionContext(coherentRng, localLightSamplingMode, localLightBufferRegion
#if RTXDI_ENABLE_PRESAMPLING
    ,localLightRISBufferSegmentParams
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
    ,regirParams
    ,surface
#endif
#endif
    );

    for (uint i = 0; i < sampleParams.numLocalLightSamples; i++)
    {
        uint lightIndex;
        RAB_LightInfo lightInfo;
        float invSourcePdf;

        float rnd = RTXDI_GetNextRandom(rng);
#if RTXDI_STRATIFY_LOCAL_SAMPLING
        rnd = (rnd + i) / sampleParams.numLocalLightSamples;
#endif // RTXDI_STRATIFY_LOCAL_SAMPLING

        RTXDI_SelectNextLocalLight(lightSelectionContext, rnd, lightInfo, lightIndex, invSourcePdf);
        float2 uv = RTXDI_RandomlySelectLocalLightUV(rng);
        bool zeroPdf = RTXDI_StreamLocalLightAtUVIntoReservoir(rng, misData, surface, sampleParams.brdfCutoff, misData.localLightMisWeight, lightIndex, uv, invSourcePdf, lightInfo, state, o_selectedSample);

        if (zeroPdf)
            continue;
    }

    RTXDI_FinalizeResampling(state, 1.0, misData.numMisSamples);
    state.M = 1;

    return state;
}

//
// Local light sampling
//

RTXDI_DIReservoir RTXDI_SampleLocalLights(
    inout RTXDI_RandomSamplerState rng,
    inout RTXDI_RandomSamplerState coherentRng,
    RAB_Surface surface,
    RTXDI_DIInitialSamplingParameters sampleParams,
	RTXDI_InitialSamplingMisData misData,
    ReSTIRDI_LocalLightSamplingMode localLightSamplingMode,
    RTXDI_LightBufferRegion localLightBufferRegion,
#if RTXDI_ENABLE_PRESAMPLING
    RTXDI_RISBufferSegmentParameters localLightRISBufferSegmentParams,
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
    ReGIR_Parameters regirParams,
#endif
#endif
    out RAB_LightSample o_selectedSample)
{
    o_selectedSample = RAB_EmptyLightSample();

    if (localLightBufferRegion.numLights == 0)
        return RTXDI_EmptyDIReservoir();

    if (sampleParams.numLocalLightSamples == 0)
        return RTXDI_EmptyDIReservoir();

    return RTXDI_SampleLocalLightsInternal(rng, coherentRng, surface, sampleParams, misData, localLightSamplingMode, localLightBufferRegion,
#if RTXDI_ENABLE_PRESAMPLING
    localLightRISBufferSegmentParams,
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
    regirParams,
#endif
#endif
    o_selectedSample);
}

//
// Uniform sampling for infinite lights
//

float2 RTXDI_RandomlySelectInfiniteLightUV(inout RTXDI_RandomSamplerState rng)
{
    float2 uv;
    uv.x = RTXDI_GetNextRandom(rng);
    uv.y = RTXDI_GetNextRandom(rng);
    return uv;
}

void RTXDI_StreamInfiniteLightAtUVIntoReservoir(
    inout RTXDI_RandomSamplerState rng,
    RAB_LightInfo lightInfo,
    RAB_Surface surface,
    uint lightIndex,
    float2 uv,
    float invSourcePdf,
    inout RTXDI_DIReservoir state,
    inout RAB_LightSample o_selectedSample)
{
    RAB_LightSample candidateSample = RAB_SamplePolymorphicLight(lightInfo, surface, uv);
    float targetPdf = RAB_GetLightSampleTargetPdfForSurface(candidateSample, surface);
    float risRnd = RTXDI_GetNextRandom(rng);
    bool selected = RTXDI_StreamSample(state, lightIndex, uv, risRnd, targetPdf, invSourcePdf);

    if (selected)
    {
        o_selectedSample = candidateSample;
    }
}

RTXDI_DIReservoir RTXDI_SampleInfiniteLights(
    inout RTXDI_RandomSamplerState rng,
    RAB_Surface surface,
    uint numSamples,
    RTXDI_LightBufferRegion infiniteLightBufferRegion,
    inout RAB_LightSample o_selectedSample)
{
    RTXDI_DIReservoir state = RTXDI_EmptyDIReservoir();
    o_selectedSample = RAB_EmptyLightSample();

    if (infiniteLightBufferRegion.numLights == 0)
        return state;

    if (numSamples == 0)
        return state;

    for (uint i = 0; i < numSamples; i++)
    {
        float invSourcePdf;
        uint lightIndex;
        RAB_LightInfo lightInfo;
        float rnd = RTXDI_GetNextRandom(rng);

        RTXDI_RandomlySelectLightUniformly(rnd, infiniteLightBufferRegion, lightInfo, lightIndex, invSourcePdf);
        float2 uv = RTXDI_RandomlySelectInfiniteLightUV(rng);
        RTXDI_StreamInfiniteLightAtUVIntoReservoir(rng, lightInfo, surface, lightIndex, uv, invSourcePdf, state, o_selectedSample);
    }

    RTXDI_FinalizeResampling(state, 1.0, state.M);
    state.M = 1;

    return state;
}

#if RTXDI_ENABLE_PRESAMPLING

//
// Power based RIS for the environment map. See PresampleEnvironmentMap.hlsl
//

void RTXDI_UnpackEnvironmentLightDataFromRISData(
    uint2 tileData,
    out float2 uv,
    out float invSourcePdf
)
{
    uint packedUv = tileData.x;
    invSourcePdf = asfloat(tileData.y);
    uv = float2(packedUv & 0xffff, packedUv >> 16) / float(0xffff);
}

void RTXDI_RandomlySelectEnvironmentLightUVFromRISTile(
    inout RTXDI_RandomSamplerState rng,
    RTXDI_RISTileInfo risTileInfo,
    out float2 uv,
    out float invSourcePdf
)
{
    uint2 tileData;
    uint risBufferPtr;
    float rnd = RTXDI_GetNextRandom(rng);
    RTXDI_RandomlySelectLightDataFromRISTile(rnd, risTileInfo, tileData, risBufferPtr);
    RTXDI_UnpackEnvironmentLightDataFromRISData(tileData, uv, invSourcePdf);
}

void RTXDI_StreamEnvironmentLightAtUVIntoReservoir(
    inout RTXDI_RandomSamplerState rng,
    RTXDI_DIInitialSamplingParameters sampleParams,
	RTXDI_InitialSamplingMisData misData,
    RAB_Surface surface,
    RAB_LightInfo lightInfo,
    uint environmentLightIndex,
    float2 uv,
    float invSourcePdf,
    inout RTXDI_DIReservoir state,
    inout RAB_LightSample o_selectedSample)
{
    RAB_LightSample candidateSample = RAB_SamplePolymorphicLight(lightInfo, surface, uv);
    float blendedSourcePdf = RTXDI_LightBrdfMisWeight(surface, candidateSample, 1.0 / invSourcePdf,
        misData.environmentMapMisWeight, true, misData.brdfMisWeight, sampleParams.brdfCutoff);
    float targetPdf = RAB_GetLightSampleTargetPdfForSurface(candidateSample, surface);
    float risRnd = RTXDI_GetNextRandom(rng);

    bool selected = RTXDI_StreamSample(state, environmentLightIndex, uv, risRnd, targetPdf, 1.0 / blendedSourcePdf);

    if (selected) {
        o_selectedSample = candidateSample;
    }
}

RTXDI_DIReservoir RTXDI_SampleEnvironmentMap(
    inout RTXDI_RandomSamplerState rng,
    inout RTXDI_RandomSamplerState coherentRng,
    RAB_Surface surface,
    RTXDI_DIInitialSamplingParameters sampleParams,
	RTXDI_InitialSamplingMisData misData,
    RTXDI_EnvironmentLightBufferParameters params,
    RTXDI_RISBufferSegmentParameters risBufferSegmentParams,
    out RAB_LightSample o_selectedSample)
{
    RTXDI_DIReservoir state = RTXDI_EmptyDIReservoir();
    o_selectedSample = RAB_EmptyLightSample();

    if (params.lightPresent == 0)
        return state;

    if (sampleParams.numEnvironmentSamples == 0)
        return state;

    RTXDI_RISTileInfo risTileInfo = RTXDI_RandomlySelectRISTile(coherentRng, risBufferSegmentParams);

    RAB_LightInfo lightInfo = RAB_LoadLightInfo(params.lightIndex, false);

    for (uint i = 0; i < sampleParams.numEnvironmentSamples; i++)
    {
        float2 uv;
        float invSourcePdf;
        RTXDI_RandomlySelectEnvironmentLightUVFromRISTile(rng, risTileInfo, uv, invSourcePdf);
        RTXDI_StreamEnvironmentLightAtUVIntoReservoir(rng, sampleParams, misData, surface, lightInfo, params.lightIndex, uv, invSourcePdf, state, o_selectedSample);
    }

    RTXDI_FinalizeResampling(state, 1.0, misData.numMisSamples);
    state.M = 1;

    return state;
}

#endif // RTXDI_ENABLE_PRESAMPLING

//
// BRDF sampling: Samples from the BRDF defined by the given surface
//

RTXDI_DIReservoir RTXDI_SampleBrdf(
    inout RTXDI_RandomSamplerState rng,
    RAB_Surface surface,
	uint numBrdfSamples,
	float brdfCutoff,
	float brdfRayMinT,
	RTXDI_InitialSamplingMisData misData,
	inout RTXDI_RandomSamplerState coherentRng,
    RTXDI_LightBufferParameters lightBufferParams,
    out RAB_LightSample o_selectedSample)
{
    RTXDI_DIReservoir state = RTXDI_EmptyDIReservoir();
    
    for (uint i = 0; i < numBrdfSamples; ++i)
    {
        float lightSourcePdf = 0;
        float3 sampleDir;
        uint lightIndex = RTXDI_InvalidLightIndex;
        float2 randXY = float2(0, 0);
        RAB_LightSample candidateSample = RAB_EmptyLightSample();

        if (RAB_SurfaceImportanceSampleBrdf(surface, rng, sampleDir))
        {
            float brdfPdf = RAB_SurfaceEvaluateBrdfPdf(surface, sampleDir);
            float maxDistance = RTXDI_BrdfMaxDistanceFromPdf(brdfCutoff, brdfPdf);
            
            bool hitAnything = RAB_TraceRayForLocalLight(RAB_GetSurfaceWorldPos(surface), sampleDir,
                brdfRayMinT, maxDistance, lightIndex, randXY);

            if (lightIndex != RTXDI_InvalidLightIndex)
            {
                RAB_LightInfo lightInfo = RAB_LoadLightInfo(lightIndex, false);
                candidateSample = RAB_SamplePolymorphicLight(lightInfo, surface, randXY);
                    
                if (brdfCutoff > 0.f)
                {
                    // If Mis cutoff is used, we need to evaluate the sample and make sure it actually could have been
                    // generated by the area sampling technique. This is due to numerical precision.
                    float3 lightDir;
                    float lightDistance;
                    RAB_GetLightDirDistance(surface, candidateSample, lightDir, lightDistance);

                    float brdfPdf = RAB_SurfaceEvaluateBrdfPdf(surface, lightDir);
                    float maxDistance = RTXDI_BrdfMaxDistanceFromPdf(brdfCutoff, brdfPdf);
                    if (lightDistance > maxDistance)
                        lightIndex = RTXDI_InvalidLightIndex;
                }

                if (lightIndex != RTXDI_InvalidLightIndex)
                {
                    lightSourcePdf = RAB_EvaluateLocalLightSourcePdf(lightIndex);
                }
            }
            else if (!hitAnything && (lightBufferParams.environmentLightParams.lightPresent != 0))
            {
                // sample environment light
                lightIndex = lightBufferParams.environmentLightParams.lightIndex;
                RAB_LightInfo lightInfo = RAB_LoadLightInfo(lightIndex, false);
                randXY = RAB_GetEnvironmentMapRandXYFromDir(sampleDir);
                candidateSample = RAB_SamplePolymorphicLight(lightInfo, surface, randXY);
                lightSourcePdf = RAB_EvaluateEnvironmentMapSamplingPdf(sampleDir);
            }
        }

        if (lightSourcePdf == 0)
        {
            // Did not hit a visible light
            continue;
        }

        bool isEnvMapSample = lightIndex == lightBufferParams.environmentLightParams.lightIndex;
        float targetPdf = RAB_GetLightSampleTargetPdfForSurface(candidateSample, surface);
        float blendedSourcePdf = RTXDI_LightBrdfMisWeight(surface, candidateSample, lightSourcePdf,
            isEnvMapSample ? misData.environmentMapMisWeight : misData.localLightMisWeight, 
            isEnvMapSample,
            misData.brdfMisWeight, brdfCutoff);
        float risRnd = RTXDI_GetNextRandom(rng);

        bool selected = RTXDI_StreamSample(state, lightIndex, randXY, risRnd, targetPdf, 1.0f / blendedSourcePdf);
        if (selected) {
            o_selectedSample = candidateSample;
        }
    }

    RTXDI_FinalizeResampling(state, 1.0, misData.numMisSamples);
    state.M = 1;

    return state;
}

// Samples the local, infinite, and environment lights for a given surface
RTXDI_DIReservoir RTXDI_SampleLightsForSurface(
    inout RTXDI_RandomSamplerState rng,
    inout RTXDI_RandomSamplerState coherentRng,
    RAB_Surface surface,
    RTXDI_DIInitialSamplingParameters sampleParams,
    RTXDI_LightBufferParameters lightBufferParams,
#if RTXDI_ENABLE_PRESAMPLING
    RTXDI_RISBufferSegmentParameters localLightRISBufferSegmentParams,
    RTXDI_RISBufferSegmentParameters environmentLightRISBufferSegmentParams,
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
    ReGIR_Parameters regirParams,
#endif
#endif
    out RAB_LightSample o_lightSample)
{
    o_lightSample = RAB_EmptyLightSample();

    RTXDI_DIReservoir localReservoir;
    RAB_LightSample localSample = RAB_EmptyLightSample();

	RTXDI_InitialSamplingMisData misData = RTXDI_ComputeInitialSamplingMisData(sampleParams);

    localReservoir = RTXDI_SampleLocalLights(rng, coherentRng, surface,
        sampleParams, misData, sampleParams.localLightSamplingMode, lightBufferParams.localLightBufferRegion,
#if RTXDI_ENABLE_PRESAMPLING
    localLightRISBufferSegmentParams,
#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
    regirParams,
#endif
#endif
localSample);

    RAB_LightSample infiniteSample = RAB_EmptyLightSample();  
    RTXDI_DIReservoir infiniteReservoir = RTXDI_SampleInfiniteLights(rng, surface,
        sampleParams.numInfiniteLightSamples, lightBufferParams.infiniteLightBufferRegion, infiniteSample);

#if RTXDI_ENABLE_PRESAMPLING
    RAB_LightSample environmentSample = RAB_EmptyLightSample();
    RTXDI_DIReservoir environmentReservoir = RTXDI_SampleEnvironmentMap(rng, coherentRng, surface,
        sampleParams, misData, lightBufferParams.environmentLightParams, environmentLightRISBufferSegmentParams, environmentSample);
#endif // RTXDI_ENABLE_PRESAMPLING

    RAB_LightSample brdfSample = RAB_EmptyLightSample();
    RTXDI_DIReservoir brdfReservoir = RTXDI_SampleBrdf(rng, surface, sampleParams.numBrdfSamples, sampleParams.brdfCutoff, sampleParams.brdfRayMinT, misData, coherentRng, lightBufferParams, brdfSample);

    RTXDI_DIReservoir state = RTXDI_EmptyDIReservoir();
    RTXDI_CombineDIReservoirs(state, localReservoir, 0.5, localReservoir.targetPdf);
    bool selectInfinite = RTXDI_CombineDIReservoirs(state, infiniteReservoir, RTXDI_GetNextRandom(rng), infiniteReservoir.targetPdf);
#if RTXDI_ENABLE_PRESAMPLING
    bool selectEnvironment = RTXDI_CombineDIReservoirs(state, environmentReservoir, RTXDI_GetNextRandom(rng), environmentReservoir.targetPdf);
#endif // RTXDI_ENABLE_PRESAMPLING
    bool selectBrdf = RTXDI_CombineDIReservoirs(state, brdfReservoir, RTXDI_GetNextRandom(rng), brdfReservoir.targetPdf);
    
    RTXDI_FinalizeResampling(state, 1.0, 1.0);
    state.M = 1;

    if (selectBrdf)
        o_lightSample = brdfSample;
    else
#if RTXDI_ENABLE_PRESAMPLING
    if (selectEnvironment)
        o_lightSample = environmentSample;
    else
#endif // RTXDI_ENABLE_PRESAMPLING
    if (selectInfinite)
        o_lightSample = infiniteSample;
    else
        o_lightSample = localSample;

	if(sampleParams.enableInitialVisibility && RTXDI_IsValidDIReservoir(state))
	{
		if (!RAB_GetConservativeVisibility(surface, o_lightSample))
        {
            RTXDI_StoreVisibilityInDIReservoir(state, 0, true);
        }

	}

    return state;
}

#endif