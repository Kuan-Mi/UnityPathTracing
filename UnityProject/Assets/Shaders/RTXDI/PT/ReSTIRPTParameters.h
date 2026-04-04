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

#ifndef RTXDI_PT_PARAMETERS_H
#define RTXDI_PT_PARAMETERS_H

#include "Rtxdi/RtxdiParameters.h"
#include "Rtxdi/RtxdiTypes.h"

#define RTXDI_PTPathTraceInvocationType_Initial 1
#define RTXDI_PTPathTraceInvocationType_Temporal 2
#define RTXDI_PTPathTraceInvocationType_TemporalInverse 3
#define RTXDI_PTPathTraceInvocationType_Spatial 4
#define RTXDI_PTPathTraceInvocationType_SpatialInverse 5
#define RTXDI_PTPathTraceInvocationType_DebugTemporalRetrace 11
#define RTXDI_PTPathTraceInvocationType_DebugSpatialRetrace 12

#ifdef __cplusplus
enum class RTXDI_PTReconnectionMode : uint16_t
{
    FixedThreshold = RTXDI_RESTIRPT_RECONNECTION_MODE_FIXED_THRESHOLD,
    Footprint = RTXDI_RESTIRPT_RECONNECTION_MODE_FOOTPRINT,
};

//
// Enums set by RTXDI before each invocation of
// the RAB_PathTrace function.
//
enum class RTXDI_PTPathTraceInvocationType : uint16_t
{
    Initial = RTXDI_PTPathTraceInvocationType_Initial,
    Temporal = RTXDI_PTPathTraceInvocationType_Temporal,
    TemporalInverse = RTXDI_PTPathTraceInvocationType_TemporalInverse,
    Spatial = RTXDI_PTPathTraceInvocationType_Spatial,
    SpatialInverse = RTXDI_PTPathTraceInvocationType_SpatialInverse,
    DebugTemporalRetrace = RTXDI_PTPathTraceInvocationType_DebugTemporalRetrace,
    DebugSpatialRetrace = RTXDI_PTPathTraceInvocationType_DebugSpatialRetrace
};

#else
#define RTXDI_PTReconnectionMode uint16_t
#define RTXDI_PTPathTraceInvocationType uint16_t
#endif

struct RTXDI_PackedPTReservoir
{
#ifdef __cplusplus
    using uint = uint32_t;
    using uint4 = uint32_t[4];
    using float3 = float[3];
#endif
    uint4 Data0;
    uint4 Data1;
    uint4 Data2;
    uint4 Data3;
};

struct RTXDI_PTBufferIndices
{
    uint32_t initialPathTracerOutputBufferIndex;
    uint32_t temporalResamplingInputBufferIndex;
    uint32_t temporalResamplingOutputBufferIndex;
    uint32_t spatialResamplingInputBufferIndex;

    uint32_t spatialResamplingOutputBufferIndex;
    uint32_t finalShadingInputBufferIndex;
    uint32_t pad1;
    uint32_t pad2;
};

struct RTXDI_PTInitialSamplingParameters
{
    uint32_t numInitialSamples;
    uint32_t maxBounceDepth;
    uint32_t maxRcVertexLength;
    uint32_t pad;
};

struct RTXDI_PTReconnectionParameters
{
    // Footprint method
    float minConnectionFootprint;
    float minConnectionFootprintSigma;
    float minPdfRoughness;
    float minPdfRoughnessSigma;

    // Threshold method
    float roughnessThreshold;
    float distanceThreshold;
    // Footprint vs threshold mode select
    RTXDI_PTReconnectionMode reconnectionMode;
    uint32_t pad1;
};

struct RTXDI_PTHybridShiftPerFrameParameters
{
    uint32_t maxBounceDepth;
    uint32_t maxRcVertexLength;
    uint32_t pad1;
    uint32_t pad2;
};

struct RTXDI_PTTemporalResamplingParameters
{
    float    depthThreshold;
    float    normalThreshold;
    uint32_t enablePermutationSampling;
    uint32_t maxHistoryLength;

    uint32_t maxReservoirAge;
    uint32_t enableFallbackSampling;
    uint32_t enableVisibilityBeforeCombine;
    uint32_t uniformRandomNumber;

    uint32_t duplicationBasedHistoryReduction;
    float    historyReductionStrength;
    uint32_t pad1;
    uint32_t pad2;
};

struct RTXDI_PTSpatialResamplingParameters
{
    uint32_t numSpatialSamples;
    uint32_t numDisocclusionBoostSamples;
    uint32_t maxTemporalHistory;
    uint32_t duplicationBasedHistoryReduction;

    float    samplingRadius;
    float    normalThreshold;
    float    depthThreshold;
    uint32_t pad1;
};

struct RTXDI_PTParameters
{
    RTXDI_ReservoirBufferParameters reservoirBuffer;
    RTXDI_PTBufferIndices bufferIndices;
    RTXDI_PTInitialSamplingParameters initialSampling;
    RTXDI_PTReconnectionParameters reconnection;
    RTXDI_PTTemporalResamplingParameters temporalResampling;
    RTXDI_PTHybridShiftPerFrameParameters hybridShift;
    RTXDI_BoilingFilterParameters boilingFilter;
    RTXDI_PTSpatialResamplingParameters spatialResampling;
};

#endif // RTXDI_PT_PARAMETERS_H