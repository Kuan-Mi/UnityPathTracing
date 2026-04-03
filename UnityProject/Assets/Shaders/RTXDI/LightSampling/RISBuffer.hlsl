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

#ifndef RTXDI_RIS_BUFFER_HLSLI
#define RTXDI_RIS_BUFFER_HLSLI

struct RTXDI_RISTileInfo
{
    uint risTileOffset;
    uint risTileSize;
};

void RTXDI_RandomlySelectLightDataFromRISTile(
    float rnd,
    RTXDI_RISTileInfo bufferInfo,
    out uint2 tileData,
    out uint risBufferPtr)
{
    uint risSample = min(uint(floor(rnd * bufferInfo.risTileSize)), bufferInfo.risTileSize - 1);
    risBufferPtr = risSample + bufferInfo.risTileOffset;
    tileData = RTXDI_RIS_BUFFER[risBufferPtr];
}

RTXDI_RISTileInfo RTXDI_RandomlySelectRISTile(
    inout RTXDI_RandomSamplerState coherentRng,
    RTXDI_RISBufferSegmentParameters params)
{
    RTXDI_RISTileInfo risTileInfo;
    float tileRnd = RTXDI_GetNextRandom(coherentRng);
    uint tileIndex = uint(tileRnd * params.tileCount);
    risTileInfo.risTileOffset = tileIndex * params.tileSize + params.bufferOffset;
    risTileInfo.risTileSize = params.tileSize;
    return risTileInfo;
}

#endif // RTXDI_RIS_BUFFER_HLSLI
