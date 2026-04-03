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

#ifndef RTXDI_PT_BOILING_FILTER_HLSLI
#define RTXDI_PT_BOILING_FILTER_HLSLI

#include "Rtxdi/PT/Reservoir.hlsli"
#include "Rtxdi/Utils/BoilingFilter.hlsli"

#ifdef RTXDI_ENABLE_BOILING_FILTER

// Same as RTXDI_BoilingFilter but for PT reservoirs.
void RTXDI_PTBoilingFilter(
    uint2 LocalIndex,
    float filterStrength, // (0..1]
    inout RTXDI_PTReservoir reservoir)
{
    float weight = RTXDI_Luminance(reservoir.TargetFunction) * reservoir.WeightSum;

    if (RTXDI_BoilingFilterInternal(LocalIndex, filterStrength, weight))
        reservoir = RTXDI_EmptyPTReservoir();
}

#endif // RTXDI_ENABLE_BOILING_FILTER

#endif // RTXDI_PT_BOILING_FILTER_HLSLI
