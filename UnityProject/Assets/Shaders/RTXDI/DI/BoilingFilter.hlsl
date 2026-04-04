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

#ifndef RTXDI_DI_BOILING_FILTER_HLSLI
#define RTXDI_DI_BOILING_FILTER_HLSLI

#include "Assets/Shaders/Rtxdi/DI/Reservoir.hlsl"
#include "Assets/Shaders/Rtxdi/Utils/BoilingFilter.hlsl"

#ifdef RTXDI_ENABLE_BOILING_FILTER
// Boiling filter that should be applied at the end of the temporal resampling pass.
// Can be used inside the same shader that does temporal resampling if it's a compute shader,
// or in a separate pass if temporal resampling is a raygen shader.
// The filter analyzes the weights of all reservoirs in a thread group, and discards
// the reservoirs whose weights are very high, i.e. above a certain threshold.
void RTXDI_BoilingFilter(
    uint2 LocalIndex,
    float filterStrength, // (0..1]
    inout RTXDI_DIReservoir reservoir)
{
    if (RTXDI_BoilingFilterInternal(LocalIndex, filterStrength, reservoir.weightSum))
        reservoir = RTXDI_EmptyDIReservoir();
}
#endif // RTXDI_ENABLE_BOILING_FILTER

#endif
