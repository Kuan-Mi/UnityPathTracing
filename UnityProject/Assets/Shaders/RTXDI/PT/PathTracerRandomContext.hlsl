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

#ifndef RAB_PT_RANDOM_CONTEXT_HLSLI
#define RAB_PT_RANDOM_CONTEXT_HLSLI

#include "Rtxdi/RtxdiParameters.h"
#include "Rtxdi/Utils/RandomSamplerstate.hlsli"

struct RTXDI_PathTracerRandomContext
{
	RTXDI_RandomSamplerState initialRandomSamplerState;
	RTXDI_RandomSamplerState initialCoherentRandomSamplerState;
	
	RTXDI_RandomSamplerState replayRandomSamplerState;
};

RTXDI_PathTracerRandomContext RTXDI_InitializePathTracerRandomContext(uint2 pixel, uint frameIndex, uint initialSeed, uint randomReplaySeed)
{
	RTXDI_PathTracerRandomContext ptRandContext;

	ptRandContext.initialRandomSamplerState = RTXDI_InitRandomSampler(pixel, frameIndex, initialSeed);
	ptRandContext.initialCoherentRandomSamplerState = RTXDI_InitRandomSampler(pixel / RTXDI_TILE_SIZE_IN_PIXELS, frameIndex, initialSeed);
	ptRandContext.replayRandomSamplerState = RTXDI_InitRandomSampler(pixel, frameIndex, randomReplaySeed);

	return ptRandContext;
}

#endif // RAB_PT_RANDOM_CONTEXT_HLSLI