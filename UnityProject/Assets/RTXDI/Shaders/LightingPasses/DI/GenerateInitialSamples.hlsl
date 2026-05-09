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

#pragma pack_matrix(row_major)

#include "../RtxdiApplicationBridge/RtxdiApplicationBridge.hlsli"

#include <Rtxdi/DI/InitialSampling.hlsli>
#include <Rtxdi/DI/ReservoirStorage.hlsli>

#if USE_RAY_QUERY
[numthreads(RTXDI_SCREEN_SPACE_GROUP_SIZE, RTXDI_SCREEN_SPACE_GROUP_SIZE, 1)]
void main(uint2 GlobalIndex : SV_DispatchThreadID)
#else
[shader("raygeneration")]
void RayGen()
#endif
{
#if !USE_RAY_QUERY
    uint2 GlobalIndex = DispatchRaysIndex().xy;
#endif

    const RTXDI_RuntimeParameters params = g_Const.runtimeParams;

    uint2 pixelPosition = RTXDI_ReservoirPosToPixelPos(GlobalIndex, params.activeCheckerboardField);

    RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(pixelPosition, g_Const.runtimeParams.frameIndex, RTXDI_DI_GENERATE_INITIAL_SAMPLES_RANDOM_SEED);
    RTXDI_RandomSamplerState tileRng = RTXDI_InitRandomSampler(pixelPosition / RTXDI_TILE_SIZE_IN_PIXELS, g_Const.runtimeParams.frameIndex, RTXDI_DI_GENERATE_INITIAL_SAMPLES_RANDOM_SEED);

    RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);

    RAB_LightSample lightSample;
    RTXDI_DIReservoir reservoir = RTXDI_SampleLightsForSurface(rng, tileRng, surface,
        g_Const.restirDI.initialSamplingParams, g_Const.lightBufferParams,
#ifdef RTXDI_ENABLE_PRESAMPLING
        g_Const.localLightsRISBufferSegmentParams, g_Const.environmentLightRISBufferSegmentParams,
#if RTXDI_REGIR_MODE != RTXDI_REGIR_MODE_DISABLED
        g_Const.regir,
#endif
#endif
        lightSample);

    RTXDI_StoreDIReservoir(reservoir, g_Const.restirDI.reservoirBufferParams, GlobalIndex, g_Const.restirDI.bufferIndices.initialSamplingOutputBufferIndex);

    if (RTXDI_IsValidDIReservoir(reservoir)){


        u_DiffuseLighting[pixelPosition] = float4(lightSample.radiance, 0);
    }else{
        u_DiffuseLighting[pixelPosition] = float4(0, 0, 0, 0);
    }

    // debug 
    // u_DiffuseLighting[pixelPosition] = float4(1,0,0, 0);

    // float3 debugColor = g_Const.restirDI.initialSamplingParams.numLocalLightSamples / 2.0f;
    // u_DiffuseLighting[pixelPosition] = float4(debugColor, 0);

// if (g_Const.restirDI.initialSamplingParams.numLocalLightSamples == 1) 
//     u_DiffuseLighting[pixelPosition] = float4(0, 1, 0, 0); // 绿色表示确认为1
// else 
//     u_DiffuseLighting[pixelPosition] = float4(1, 0, 0, 0); // 红色表示不是1


    // u_DiffuseLighting[pixelPosition] = float4(lightSample.radiance, 0);
    // u_DiffuseLighting[pixelPosition] = float4(surface.material.diffuseAlbedo, 0);
}