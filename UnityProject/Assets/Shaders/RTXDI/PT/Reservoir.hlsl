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

#ifndef RTXDI_PT_RESERVOIR_HLSLI
#define RTXDI_PT_RESERVOIR_HLSLI

#ifndef ENABLE_RESTIR_PT_PATH_REPLAY
#define ENABLE_RESTIR_PT_PATH_REPLAY 0
#endif

#define TARGET_FUNCTION_TYPE float3

#include "../Utils/Color.hlsl"
#include "../Utils/RandomSamplerState.hlsl"
#include "../Utils/ReservoirAddressing.hlsl"
#include "../Utils/SampledLightData.hlsl"
#include "../Utils/Math.hlsl"

#ifndef RTXDI_PT_RESERVOIR_BUFFER
#error "RTXDI_PT_RESERVOIR_BUFFER must be defined to point to a RWStructuredBuffer<RTXDI_PackedPTReservoir> type resource"
#endif

#define RTXDI_PTRESERVOIR_AGE_MAX 31

// This structure represents a indirect lighting reservoir that stores the radiance and weight
// as well as its the position where the radiane come from.
struct RTXDI_PTReservoir
{
#ifdef __cplusplus
    using float3 = float[3];
    using uint = uint32_t;
#endif

    static const uint MaxM = 0xffff;

    // Position of the 2nd bounce surface (OR env map direction on miss?)
    float3 TranslatedWorldPosition;

    // Overloaded: represents RIS weight sum during streaming,
    // then reservoir weight (inverse PDF) after FinalizeResampling
    float WeightSum;

    // Normal vector of the 2nd bounce surface.
    float3 WorldNormal;

    // Number of samples considered for this reservoir
    float M;

    // Incoming radiance from the 2nd bounce surface.
    float3 Radiance;

    // Number of frames the chosen sample has survived.
    uint Age;

    // In case that we use duplication-based MCap reduction, we check this in temporal reuse
    int ShouldBoostSpatialSamples;

    float RcWiPdf;

    float PartialJacobian;

    // Reconnection vertex length
    uint RcVertexLength;

    // Length of path from primary surface to the final light
    uint PathLength;

    uint RandomSeed;

    uint RandomIndex;

    float3 TargetFunction;
};

RTXDI_PTReservoir RTXDI_EmptyPTReservoir()
{
    RTXDI_PTReservoir reservoir = (RTXDI_PTReservoir)0;
    return reservoir;
}

bool RTXDI_IsValidPTReservoir(const RTXDI_PTReservoir reservoir)
{
    return reservoir.M > 0;
}

// Adds `newReservoir` into `reservoir`, returns true if the new reservoir's sample was selected.
// This is a very general form, allowing input parameters to specfiy normalization and target function
// rather than computing them from `newReservoir`.  Named "internal" since these parameters take
// different meanings (e.g., in RTXDI_CombineReservoirs() or RTXDI_StreamNeighborWithPairwiseMIS())
bool InternalSimpleResample(inout RTXDI_PTReservoir targetReservoir,
                            const RTXDI_PTReservoir NewReservoir,
                            float Random,
                            TARGET_FUNCTION_TYPE NewTargetFunction, // Usually closely related to the sample normalization,
                            float SampleNormalization, //     typically off by some multiplicative factor
                            float SampleM // In its most basic form, should be newReservoir.M
)
{
    // What's the current weight (times any prior-step RIS normalization factor)
    float RisWeight = RTXDI_Luminance(NewTargetFunction) * SampleNormalization;
    RisWeight = (isnan(RisWeight) || isinf(RisWeight)) ? 0.0f : RisWeight;

    // Our *effective* candidate pool is the sum of our candidates plus those of our neighbors
    targetReservoir.M += SampleM;

    // Update the weight sum
    targetReservoir.WeightSum += RisWeight;

    // Decide if we will randomly pick this sample
    const bool SelectSample = (Random * targetReservoir.WeightSum < RisWeight);

    // If we did select this sample, update the relevant data
    if (SelectSample)
    {
        targetReservoir.TranslatedWorldPosition = NewReservoir.TranslatedWorldPosition;
        targetReservoir.Radiance = NewReservoir.Radiance;
        targetReservoir.WorldNormal = NewReservoir.WorldNormal;
        targetReservoir.Age = NewReservoir.Age;

        targetReservoir.TargetFunction = NewTargetFunction;
        targetReservoir.RandomSeed = NewReservoir.RandomSeed;
        targetReservoir.RandomIndex = NewReservoir.RandomIndex;
        targetReservoir.RcVertexLength = NewReservoir.RcVertexLength;
        targetReservoir.PathLength = NewReservoir.PathLength;
        targetReservoir.PartialJacobian = NewReservoir.PartialJacobian;
        targetReservoir.RcWiPdf = NewReservoir.RcWiPdf;
    }

    return SelectSample;
}

// Adds a reservoir with one sample into this reservoir.
// Algorithm (4) from the ReSTIR paper, Combining the streams of multiple reservoirs.
// Normalization - Equation (6) - is postponed until all reservoirs are combined.
bool CombineReservoirs(inout RTXDI_PTReservoir targetReservoir, RTXDI_PTReservoir NewReservoir, float Random, TARGET_FUNCTION_TYPE NewTargetFunction)
{
    return InternalSimpleResample(targetReservoir, NewReservoir, Random, NewTargetFunction, NewReservoir.WeightSum * NewReservoir.M, NewReservoir.M);
}

void RTXDI_FinalizeResampling(inout RTXDI_PTReservoir reservoir, in float Numerator, in float Denominator)
{
    reservoir.WeightSum = (Denominator > 0.0) ? (Numerator * reservoir.WeightSum / Denominator) : 0.f;
}

bool RTXDI_ConnectsToNeeLight(RTXDI_PTReservoir reservoir)
{
    return isinf(reservoir.Radiance.x);
}

RTXDI_SampledLightData RTXDI_GetSampledLightData(RTXDI_PTReservoir reservoir)
{
    RTXDI_SampledLightData sampledLightData;
    sampledLightData.lightData = asuint(reservoir.Radiance.y);
    sampledLightData.uvData = asuint(reservoir.Radiance.z);
    return sampledLightData;
}

/*
 * The RNG stored in the reservoir is used for fully replaying its path
 * Because the path tracer context uses it to generate reconnection parameters
 * before tracing, the RNG state is not the same state used for generating
 * the outgoing ray from the primary surface.
 * This function advances returns RNG that has been advanced to that state
 * so that its next use in sampling the BSDF for an outgoing ray as
 * done in RAB_PathTrace will produce the correct output.
 */
RTXDI_RandomSamplerState RTXDI_GetRngForShading(RTXDI_PTReservoir reservoir)
{
    return RTXDI_CreateRandomSamplerFromDirectSeed(reservoir.RandomSeed, reservoir.RandomIndex + 6);
}

// Creates a PT reservoir from a raw light sample.
// Note: the original sample PDF can be embedded into sampleRadiance, in which case the samplePdf parameter should be set to 1.0.
RTXDI_PTReservoir RTXDI_MakePTReservoir(const float3 TargetFunction,
                                        const uint RandomSeed,
                                        const uint RandomIndex,
                                        const uint RcVertexLength,
                                        const uint PathLength,
                                        const float PartialJacobian,
                                        const float RcWiPdf,

                                        const float3 TranslatedWorldPosition,
                                        const float3 WorldNormal,
                                        const float3 Radiance,
                                        const float SamplePdf)
{
    RTXDI_PTReservoir reservoir = (RTXDI_PTReservoir)0;
    reservoir.TranslatedWorldPosition = TranslatedWorldPosition;
    reservoir.WorldNormal = WorldNormal;
    reservoir.Radiance = Radiance;
    reservoir.WeightSum = SamplePdf > 0.0 ? 1.0 / SamplePdf : 0.0;
    reservoir.M = 1;
    reservoir.Age = 0;

    reservoir.TargetFunction = TargetFunction;
    reservoir.RandomSeed = RandomSeed;
    reservoir.RandomIndex = RandomIndex;
    reservoir.PartialJacobian = PartialJacobian;
    reservoir.RcVertexLength = RcVertexLength;
    reservoir.PathLength = PathLength;
    reservoir.RcWiPdf = RcWiPdf;

    return reservoir;
}

//
// Packing and Unpacking
//

uint2 EncodeSnorm3x16(float3 Value)
{
    const uint x = uint(round(clamp(Value.x, -1.0, 1.0) * 32767.0) + 32767.0);
    const uint y = uint(round(clamp(Value.y, -1.0, 1.0) * 32767.0) + 32767.0);
    const uint z = uint(round(clamp(Value.z, -1.0, 1.0) * 32767.0) + 32767.0);
    return uint2((x & 0x0000ffff) | (y << 16), (z & 0x0000ffff));
}

float3 DecodeSnorm3x16(uint2 PackedValue)
{
    return float3(clamp((float(PackedValue.x & 0xffff) - 32767.0f) * rcp(32767.0f), -1.0, 1.0), clamp((float(PackedValue.x >> 16u) - 32767.0f) * rcp(32767.0f), -1.0, 1.0),
                  clamp((float(PackedValue.y & 0xffff) - 32767.0f) * rcp(32767.0f), -1.0, 1.0));
}

RTXDI_PackedPTReservoir RTXDI_PackPTReservoir(RTXDI_PTReservoir reservoir)
{
    RTXDI_PackedPTReservoir PackedData = (RTXDI_PackedPTReservoir)0;
    PackedData.Data0.xyz = asuint(reservoir.TranslatedWorldPosition);
    PackedData.Data0.w = asuint(reservoir.WeightSum);
    PackedData.Data2.xyz = asuint(reservoir.Radiance);
    PackedData.Data2.w = (reservoir.Age & RTXDI_PTRESERVOIR_AGE_MAX);
    PackedData.Data2.w |= reservoir.ShouldBoostSpatialSamples ? 0x80 : 0x0;

    PackedData.Data1.xy = EncodeSnorm3x16(reservoir.WorldNormal);
    PackedData.Data1.y |= f32tof16(reservoir.M) << 16;
    PackedData.Data1.z = asuint(reservoir.PartialJacobian);
    PackedData.Data1.w = asuint(reservoir.RcWiPdf);
    PackedData.Data2.w = (PackedData.Data2.w & 0xff) | ((reservoir.RcVertexLength & 0xff) << 8) | ((reservoir.PathLength & 0xff) << 16) | ((reservoir.RandomIndex & 0xff) << 24);
    PackedData.Data3.xyz = asuint(reservoir.TargetFunction);
    PackedData.Data3.w = reservoir.RandomSeed;

    return PackedData;
}

// RTXDI_PTReservoir RTXDI_PTReservoirFromPackedData(in const RTXDI_PackedPTReservoir PackedData)
RTXDI_PTReservoir RTXDI_UnpackPTReservoir(in const RTXDI_PackedPTReservoir PackedData)
{
    RTXDI_PTReservoir Reservoir = (RTXDI_PTReservoir)0;

    Reservoir.TranslatedWorldPosition = asfloat(PackedData.Data0.xyz);
    Reservoir.WeightSum = asfloat(PackedData.Data0.w);
    Reservoir.Radiance = asfloat(PackedData.Data2.xyz);
    Reservoir.Age = PackedData.Data2.w & RTXDI_PTRESERVOIR_AGE_MAX;

    Reservoir.WorldNormal = normalize(DecodeSnorm3x16(PackedData.Data1.xy));
    Reservoir.M = f16tof32(PackedData.Data1.y >> 16);
    Reservoir.ShouldBoostSpatialSamples = ((PackedData.Data2.w & 0x80) != 0x0);
    Reservoir.TargetFunction = asfloat(PackedData.Data3.xyz);
    Reservoir.RcWiPdf = asfloat(PackedData.Data1.w);
    Reservoir.PartialJacobian = asfloat(PackedData.Data1.z);
    Reservoir.RcVertexLength = (PackedData.Data2.w >> 8) & 0xFF;
    Reservoir.PathLength = (PackedData.Data2.w >> 16) & 0xFF;
    Reservoir.RandomIndex = (PackedData.Data2.w >> 24) & 0xFF;
    Reservoir.RandomSeed = PackedData.Data3.w;

    return Reservoir;
}

//
// Loading and Storing
//

uint ComputePTReservoirAddress(uint2 PixelCoord, int2 BufferDim, int Slice)
{
    static const uint TileSize = 4;

    const int2 PaddedBufferDim = (BufferDim + TileSize - 1) / TileSize * TileSize;

    const uint RowStride = PaddedBufferDim.x * TileSize;

    int2 Tile = PixelCoord / TileSize;
    int2 TileCoord = PixelCoord % TileSize;

    uint Address = Slice * PaddedBufferDim.x * PaddedBufferDim.y;
    Address += Tile.y * RowStride;
    Address += Tile.x * TileSize * TileSize;
    Address += TileCoord.y * TileSize + TileCoord.x;

    return Address;
}

void RTXDI_StorePackedPTReservoir(const RTXDI_PackedPTReservoir packedPTReservoir, RTXDI_ReservoirBufferParameters reservoirParams, uint2 reservoirPosition, uint reservoirArrayIndex)
{
    uint pointer = RTXDI_ReservoirPositionToPointer(reservoirParams, reservoirPosition, reservoirArrayIndex);
    RTXDI_PT_RESERVOIR_BUFFER[pointer] = packedPTReservoir;
}

void RTXDI_StorePTReservoir(const RTXDI_PTReservoir reservoir, RTXDI_ReservoirBufferParameters reservoirParams, uint2 reservoirPosition, uint reservoirArrayIndex)
{
    RTXDI_PackedPTReservoir packedReservoir = RTXDI_PackPTReservoir(reservoir);
    RTXDI_StorePackedPTReservoir(packedReservoir, reservoirParams, reservoirPosition, reservoirArrayIndex);
}

RTXDI_PackedPTReservoir RTXDI_LoadPackedPTReservoir(RTXDI_ReservoirBufferParameters reservoirParams, uint2 reservoirPosition, uint reservoirArrayIndex)
{
    uint pointer = RTXDI_ReservoirPositionToPointer(reservoirParams, reservoirPosition, reservoirArrayIndex);
    return RTXDI_PT_RESERVOIR_BUFFER[pointer];
}

RTXDI_PTReservoir RTXDI_LoadPTReservoir(RTXDI_ReservoirBufferParameters reservoirParams, uint2 reservoirPosition, uint reservoirArrayIndex)
{
    RTXDI_PackedPTReservoir packedReservoir = RTXDI_LoadPackedPTReservoir(reservoirParams, reservoirPosition, reservoirArrayIndex);
    return RTXDI_UnpackPTReservoir(packedReservoir);
}

#endif // RTXDI_PT_RESERVOIR_HLSLI