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

#ifndef RTXDI_RESTIRDI_PARAMETERS_H
#define RTXDI_RESTIRDI_PARAMETERS_H

#include "Assets/Shaders/Rtxdi/RtxdiParameters.h"
#include "Assets/Shaders/Rtxdi/RtxdiTypes.h"

#ifdef __cplusplus
enum class ReSTIRDI_LocalLightSamplingMode : uint32_t
{
    Uniform = ReSTIRDI_LocalLightSamplingMode_UNIFORM,
    Power_RIS = ReSTIRDI_LocalLightSamplingMode_POWER_RIS,
    ReGIR_RIS = ReSTIRDI_LocalLightSamplingMode_REGIR_RIS
};

enum class ReSTIRDI_TemporalBiasCorrectionMode : uint32_t
{
    Off = RTXDI_BIAS_CORRECTION_OFF,
    Basic = RTXDI_BIAS_CORRECTION_BASIC,
    Raytraced = RTXDI_BIAS_CORRECTION_RAY_TRACED
};

enum class ReSTIRDI_SpatialBiasCorrectionMode : uint32_t
{
    Off = RTXDI_BIAS_CORRECTION_OFF,
    Basic = RTXDI_BIAS_CORRECTION_BASIC,
    Pairwise = RTXDI_BIAS_CORRECTION_PAIRWISE,
    Raytraced = RTXDI_BIAS_CORRECTION_RAY_TRACED
};

enum class ReSTIRDI_SpatioTemporalBiasCorrectionMode : uint32_t
{
    Off = RTXDI_BIAS_CORRECTION_OFF,
    Basic = RTXDI_BIAS_CORRECTION_BASIC,
    Pairwise = RTXDI_BIAS_CORRECTION_PAIRWISE,
    Raytraced = RTXDI_BIAS_CORRECTION_RAY_TRACED
};
#else
#define ReSTIRDI_LocalLightSamplingMode uint32_t
#define ReSTIRDI_TemporalBiasCorrectionMode uint32_t
#define ReSTIRDI_SpatialBiasCorrectionMode uint32_t
#define ReSTIRDI_SpatioTemporalBiasCorrectionMode uint32_t
#endif

struct RTXDI_DIBufferIndices
{
    uint32_t initialSamplingOutputBufferIndex;
    uint32_t temporalResamplingInputBufferIndex;
    uint32_t temporalResamplingOutputBufferIndex;
    uint32_t spatialResamplingInputBufferIndex;

    uint32_t spatialResamplingOutputBufferIndex;
    uint32_t shadingInputBufferIndex;
    uint32_t pad1;
    uint32_t pad2;
};

struct RTXDI_DIInitialSamplingParameters
{
    uint32_t numLocalLightSamples;
    uint32_t numInfiniteLightSamples;
    uint32_t numEnvironmentSamples;
    uint32_t numBrdfSamples;

    float brdfCutoff;
    float brdfRayMinT;
    ReSTIRDI_LocalLightSamplingMode localLightSamplingMode;
    uint32_t enableInitialVisibility;

    uint32_t environmentMapImportanceSampling; // Only used in InitialSamplingFunctions.hlsli via RAB_EvaluateEnvironmentMapSamplingPdf
    uint32_t pad1;
    uint32_t pad2;
    uint32_t pad3;
};

struct RTXDI_DITemporalResamplingParameters
{
    // Maximum history length for temporal reuse, measured in frames.
    // Higher values result in more stable and high quality sampling, at the cost of slow reaction to changes.
    uint32_t maxHistoryLength;

    // Controls the bias correction math for temporal reuse. Depending on the setting, it can add
    // some shader cost and one approximate shadow ray per pixel (or per two pixels if checkerboard sampling is enabled).
    // Ideally, these rays should be traced through the previous frame's BVH to get fully unbiased results.
    ReSTIRDI_TemporalBiasCorrectionMode biasCorrectionMode;

    // Surface depth similarity threshold for temporal reuse.
    // If the previous frame surface's depth is within this threshold from the current frame surface's depth,
    // the surfaces are considered similar. The threshold is relative, i.e. 0.1 means 10% of the current depth.
    // Otherwise, the pixel is not reused, and the resampling shader will look for a different one.
    float depthThreshold;

    // Surface normal similarity threshold for temporal reuse.
    // If the dot product of two surfaces' normals is higher than this threshold, the surfaces are considered similar.
    // Otherwise, the pixel is not reused, and the resampling shader will look for a different one.
    float normalThreshold;

    // Allows the temporal resampling logic to skip the bias correction ray trace for light samples
    // reused from the previous frame. Only safe to use when invisible light samples are discarded
    // on the previous frame, then any sample coming from the previous frame can be assumed visible.
    uint32_t enableVisibilityShortcut;

    // Enables permuting the pixels sampled from the previous frame in order to add temporal
    // variation to the output signal and make it more denoiser friendly.
    uint32_t enablePermutationSampling;

    // Random number for permutation sampling that is the same for all pixels in the frame
    uint32_t uniformRandomNumber;

    float permutationSamplingThreshold; // Not used in TemporalResampling.hlsli
};

struct RTXDI_DISpatialResamplingParameters
{
    // Number of neighbor pixels considered for resampling (1-32)
    // Some of the may be skipped if they fail the surface similarity test.
    uint32_t numSamples;

    // Number of neighbor pixels considered when there is not enough history data (1-32)
    // Setting this parameter equal or lower than `numSpatialSamples` effectively
    // disables the disocclusion boost.
    uint32_t numDisocclusionBoostSamples;

    // Screen-space radius for spatial resampling, measured in pixels.
    float samplingRadius;

    // Controls the bias correction math for spatial reuse. Depending on the setting, it can add
    // some shader cost and one approximate shadow ray *per every spatial sample* per pixel
    // (or per two pixels if checkerboard sampling is enabled).
    ReSTIRDI_SpatialBiasCorrectionMode biasCorrectionMode;

    // Surface depth similarity threshold for spatial reuse.
    // See 'RTXDI_DITemporalResamplingParameters::depthThreshold' for more information.
    float depthThreshold;

    // Surface normal similarity threshold for spatial reuse.
    // See 'RTXDI_DITemporalResamplingParameters::normalThreshold' for more information.
    float normalThreshold;

    // Disocclusion boost is activated when the current reservoir's M value
    // is less than targetHistoryLength.
    uint32_t targetHistoryLength;

    // Enables the comparison of surface materials before taking a surface into resampling.
    uint32_t enableMaterialSimilarityTest;

    // Prevents samples which are from the current frame or have no reasonable temporal history merged being spread to neighbors
    uint32_t discountNaiveSamples;

    uint32_t pad1;
    uint32_t pad2;
    uint32_t pad3;
};

struct RTXDI_DISpatioTemporalResamplingParameters
{
    // Common parameters, see RTXDI_DITemporal* or RTXDI_DISpatialResamplingParameters

    float depthThreshold;

    float normalThreshold;

    ReSTIRDI_SpatioTemporalBiasCorrectionMode biasCorrectionMode;

    uint32_t maxHistoryLength;

    // Temporal parameters, see RTXDI_DITemporalResamplingParameters

    uint32_t enablePermutationSampling;

    uint32_t uniformRandomNumber;

    uint32_t enableVisibilityShortcut;

    // Spatial parameters, see RTXDI_DISpatialResamplingParameters

    uint32_t numSamples;

    uint32_t numDisocclusionBoostSamples;

    float samplingRadius;

    uint32_t enableMaterialSimilarityTest;

    uint32_t discountNaiveSamples;
};

struct RTXDI_ShadingParameters
{
    uint32_t enableFinalVisibility;
    uint32_t reuseFinalVisibility;
    uint32_t finalVisibilityMaxAge;
    float finalVisibilityMaxDistance;

    uint32_t enableDenoiserInputPacking;
    uint32_t pad1;
    uint32_t pad2;
    uint32_t pad3;
};

struct RTXDI_Parameters
{
    RTXDI_ReservoirBufferParameters reservoirBufferParams;
    RTXDI_DIBufferIndices bufferIndices;
    RTXDI_DIInitialSamplingParameters initialSamplingParams;
    RTXDI_DITemporalResamplingParameters temporalResamplingParams;
    RTXDI_BoilingFilterParameters boilingFilterParams;
    RTXDI_DISpatialResamplingParameters spatialResamplingParams;
    RTXDI_DISpatioTemporalResamplingParameters spatioTemporalResamplingParams;
    RTXDI_ShadingParameters shadingParams;
};

#endif // RTXDI_RESTIRDI_PARAMETERS_H