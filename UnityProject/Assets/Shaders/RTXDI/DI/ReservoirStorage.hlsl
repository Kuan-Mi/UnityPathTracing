/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef RTXDI_DI_RESERVOIR_STORAGE_HLSLI
#define RTXDI_DI_RESERVOIR_STORAGE_HLSLI

#include "Assets/Shaders/Rtxdi/DI/Reservoir.hlsl"
#include "Assets/Shaders/Rtxdi/Utils/ReservoirAddressing.hlsl"

// Define this macro to 0 if your shader needs read-only access to the reservoirs,
// to avoid compile errors in the RTXDI_StoreDIReservoir function
#ifndef RTXDI_ENABLE_STORE_RESERVOIR
#define RTXDI_ENABLE_STORE_RESERVOIR 1
#endif

#ifndef RTXDI_LIGHT_RESERVOIR_BUFFER
#error "RTXDI_LIGHT_RESERVOIR_BUFFER must be defined to point to a RWStructuredBuffer<RTXDI_PackedDIReservoir> type resource"
#endif

// Encoding helper constants for RTXDI_PackedDIReservoir.distanceAge
static const uint RTXDI_PackedDIReservoir_DistanceChannelBits = 8;
static const uint RTXDI_PackedDIReservoir_DistanceXShift = 0;
static const uint RTXDI_PackedDIReservoir_DistanceYShift = 8;
static const uint RTXDI_PackedDIReservoir_AgeShift = 16;
static const uint RTXDI_PackedDIReservoir_MaxAge = 0xff;
static const uint RTXDI_PackedDIReservoir_DistanceMask = (1u << RTXDI_PackedDIReservoir_DistanceChannelBits) - 1;
static const  int RTXDI_PackedDIReservoir_MaxDistance = int((1u << (RTXDI_PackedDIReservoir_DistanceChannelBits - 1)) - 1);

RTXDI_PackedDIReservoir RTXDI_PackDIReservoir(const RTXDI_DIReservoir reservoir)
{
    int2 clampedSpatialDistance = clamp(reservoir.spatialDistance, -RTXDI_PackedDIReservoir_MaxDistance, RTXDI_PackedDIReservoir_MaxDistance);
    uint clampedAge = clamp(reservoir.age, 0, RTXDI_PackedDIReservoir_MaxAge);

    RTXDI_PackedDIReservoir data;
    data.lightData = reservoir.lightData;
    data.uvData = reservoir.uvData;

    data.mVisibility = reservoir.packedVisibility
        | (min(uint(reservoir.M), RTXDI_PackedDIReservoir_MaxM) << RTXDI_PackedDIReservoir_MShift);

    data.distanceAge =
          ((clampedSpatialDistance.x & RTXDI_PackedDIReservoir_DistanceMask) << RTXDI_PackedDIReservoir_DistanceXShift)
        | ((clampedSpatialDistance.y & RTXDI_PackedDIReservoir_DistanceMask) << RTXDI_PackedDIReservoir_DistanceYShift)
        | (clampedAge << RTXDI_PackedDIReservoir_AgeShift);

    data.targetPdf = reservoir.targetPdf;
    data.weight = reservoir.weightSum;

    return data;
}

#if RTXDI_ENABLE_STORE_RESERVOIR
void RTXDI_StoreDIReservoir(
    const RTXDI_DIReservoir reservoir,
    RTXDI_ReservoirBufferParameters reservoirParams,
    uint2 reservoirPosition,
    uint reservoirArrayIndex)
{
    uint pointer = RTXDI_ReservoirPositionToPointer(reservoirParams, reservoirPosition, reservoirArrayIndex);
    RTXDI_LIGHT_RESERVOIR_BUFFER[pointer] = RTXDI_PackDIReservoir(reservoir);
}
#endif // RTXDI_ENABLE_STORE_RESERVOIR

RTXDI_DIReservoir RTXDI_UnpackDIReservoir(RTXDI_PackedDIReservoir data)
{
    RTXDI_DIReservoir res;
    res.lightData = data.lightData;
    res.uvData = data.uvData;
    res.targetPdf = data.targetPdf;
    res.weightSum = data.weight;
    res.M = (data.mVisibility >> RTXDI_PackedDIReservoir_MShift) & RTXDI_PackedDIReservoir_MaxM;
    res.packedVisibility = data.mVisibility & RTXDI_PackedDIReservoir_VisibilityMask;
    // Sign extend the shift values
    res.spatialDistance.x = int(data.distanceAge << (32 - RTXDI_PackedDIReservoir_DistanceXShift - RTXDI_PackedDIReservoir_DistanceChannelBits)) >> (32 - RTXDI_PackedDIReservoir_DistanceChannelBits);
    res.spatialDistance.y = int(data.distanceAge << (32 - RTXDI_PackedDIReservoir_DistanceYShift - RTXDI_PackedDIReservoir_DistanceChannelBits)) >> (32 - RTXDI_PackedDIReservoir_DistanceChannelBits);
    res.age = (data.distanceAge >> RTXDI_PackedDIReservoir_AgeShift) & RTXDI_PackedDIReservoir_MaxAge;
    res.canonicalWeight = 0.0f;

    // Discard reservoirs that have Inf/NaN
    if (isinf(res.weightSum) || isnan(res.weightSum)) {
        res = RTXDI_EmptyDIReservoir();
    }

    return res;
}

RTXDI_DIReservoir RTXDI_LoadDIReservoir(
    RTXDI_ReservoirBufferParameters reservoirParams,
    uint2 reservoirPosition,
    uint reservoirArrayIndex)
{
    uint pointer = RTXDI_ReservoirPositionToPointer(reservoirParams, reservoirPosition, reservoirArrayIndex);
    return RTXDI_UnpackDIReservoir(RTXDI_LIGHT_RESERVOIR_BUFFER[pointer]);
}

#endif // RTXDI_DI_RESERVOIR_STORAGE_HLSLI