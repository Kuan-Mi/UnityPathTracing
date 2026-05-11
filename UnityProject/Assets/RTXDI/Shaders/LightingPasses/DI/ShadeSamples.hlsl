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

#include "Rtxdi/DI/Reservoir.hlsli"
#include <Rtxdi/DI/ReservoirStorage.hlsli>

#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
#include "Rtxdi/ReGIR/ReGIRSampling.hlsli"
#endif

#ifdef WITH_NRD
#define NRD_HEADER_ONLY
#include <NRD.hlsli>
#endif

#include "../ShadingHelpers.hlsli"

#if USE_RAY_QUERY
[numthreads(RTXDI_SCREEN_SPACE_GROUP_SIZE, RTXDI_SCREEN_SPACE_GROUP_SIZE, 1)]
void main(uint2 GlobalIndex : SV_DispatchThreadID, uint2 LocalIndex : SV_GroupThreadID, uint2 GroupIdx : SV_GroupID)
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

    RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);

    RTXDI_DIReservoir reservoir = RTXDI_LoadDIReservoir(g_Const.restirDI.reservoirBufferParams, GlobalIndex, g_Const.restirDI.bufferIndices.shadingInputBufferIndex);

    float3 diffuse = 0;
    float3 specular = 0;
    float lightDistance = 0;
    float2 currLuminance = 0;

    if (RTXDI_IsValidDIReservoir(reservoir))
    {
        RAB_LightInfo lightInfo = RAB_LoadLightInfo(RTXDI_GetDIReservoirLightIndex(reservoir), false);

        RAB_LightSample lightSample = RAB_SamplePolymorphicLight(lightInfo,
            surface, RTXDI_GetDIReservoirSampleUV(reservoir));

        bool needToStore = ShadeSurfaceWithLightSample(reservoir, surface, g_Const.restirDI.shadingParams, lightSample,
            /* previousFrameTLAS = */ false, /* enableVisibilityReuse = */ true, g_Const.restirDI.temporalResamplingParams.enableVisibilityShortcut, diffuse, specular, lightDistance);

        currLuminance = float2(calcLuminance(diffuse * surface.material.diffuseAlbedo), calcLuminance(specular));

        specular = DemodulateSpecular(surface.material.specularF0, specular);

        if (needToStore)
        {
            RTXDI_StoreDIReservoir(reservoir, g_Const.restirDI.reservoirBufferParams, GlobalIndex, g_Const.restirDI.bufferIndices.shadingInputBufferIndex);
        }
    }

    // Store the sampled lighting luminance for the gradient pass.
    // Discard the pixels where the visibility was reused, as gradients need actual visibility.
    u_RestirLuminance[GlobalIndex] = currLuminance * (reservoir.age > 0 ? 0 : 1);

#if RTXDI_REGIR_MODE != RTXDI_REGIR_DISABLED
    if (g_Const.visualizeRegirCells)
    {
        diffuse *= RTXDI_VisualizeReGIRCells(g_Const.regir, RAB_GetSurfaceWorldPos(surface));
    }
#endif

    StoreShadingOutput(GlobalIndex, pixelPosition,
        surface.viewDepth, surface.material.roughness, diffuse, specular, lightDistance, true, g_Const.restirDI.shadingParams.enableDenoiserInputPacking);

    if (g_Const.debug.outputDebugDirectLighting)
    {
        u_DirectLightingRaw[pixelPosition] = float4(diffuse + specular, 1.0);
    }

// ── DEBUG TEMP: visualise PresampleReGIR output ───────────────────────────────
// Set REGIR_VIS_DEBUG 1 to override u_DirectLightingRaw with a RIS-weight heatmap.
// Every pixel maps row-major to one RIS slot inside the ReGIR region.
// Bright = high weight (light sampled); black = empty slot (PresampleReGIR didn't run or no lights).
// Restore to 0 when done.
#define REGIR_VIS_DEBUG 0
#if REGIR_VIS_DEBUG
    {
        uint regirOffset   = g_Const.regir.commonParams.risBufferOffset;
        uint lightsPerCell = g_Const.regir.commonParams.lightsPerCell;
        uint totalCells    = max(1u, g_Const.regir.gridParams.cellsX
                                   * g_Const.regir.gridParams.cellsY
                                   * g_Const.regir.gridParams.cellsZ);

        // Each cell is rendered as a square block of blockSize x blockSize pixels.
        // blockSize = ceil(sqrt(lightsPerCell)), so every pixel inside maps to one slot.
        uint blockSize     = (uint)ceil(sqrt((float)max(1u, lightsPerCell)));
        uint screenW       = (uint)g_Const.view.viewportSize.x;

        uint blockX        = pixelPosition.x / blockSize;
        uint blockY        = pixelPosition.y / blockSize;
        uint blocksPerRow  = max(1u, (screenW + blockSize - 1u) / blockSize);
        uint cellIdx       = blockY * blocksPerRow + blockX;

        // Pixel's local position inside its block
        uint lx            = pixelPosition.x % blockSize;
        uint ly            = pixelPosition.y % blockSize;
        uint slotInCell    = ly * blockSize + lx;

        float3 color = float3(0.0f, 0.0f, 0.0f);

        if (cellIdx < totalCells && slotInCell < lightsPerCell)
        {
            uint slotIdx  = regirOffset + cellIdx * lightsPerCell + slotInCell;
            uint2 entry   = u_RisBuffer[slotIdx];
            uint  lightIdx = entry.x & 0x7FFFFFFFu; // RTXDI_LIGHT_INDEX_MASK
            bool  hasLight = (entry.x != 0u) || (entry.y != 0u);

            if (hasLight)
            {
                // Hash light index → saturated RGB
                uint h = lightIdx;
                h ^= h >> 16; h *= 0x45d9f3bu;
                h ^= h >> 16; h *= 0x45d9f3bu;
                h ^= h >> 16;
                color = float3(
                    float((h      ) & 0xFFu) / 255.0f,
                    float((h >>  8) & 0xFFu) / 255.0f,
                    float((h >> 16) & 0xFFu) / 255.0f);
            }
            else
            {
                color = float3(0.04f, 0.04f, 0.04f); // dark grey = empty slot
            }
        }

        // Edge darkening: 0 at border pixel, 1 two pixels inward
        float edgeX   = min((float)lx, (float)(blockSize - 1u - lx));
        float edgeY   = min((float)ly, (float)(blockSize - 1u - ly));
        float edgeFade = saturate(min(edgeX, edgeY) * 0.5f); // ramp over 2px
        color *= lerp(0.15f, 1.0f, edgeFade);

        u_DirectLightingRaw[pixelPosition] = float4(color, 1.0f);
    }
#endif // REGIR_VIS_DEBUG

// ── DEBUG TEMP: visualise ReGIR Onion cell membership per pixel ───────────────
// Each pixel reads its surface world position, looks up which ReGIR Onion cell it
// belongs to via RTXDI_ReGIR_WorldPosToCellIndex, then hashes the cell ID to a
// saturated RGB colour.  Pixels outside the onion (cellIndex < 0) are shown in red.
// Set REGIR_CELL_DEBUG 1 to enable; restore to 0 when done.
#define REGIR_CELL_DEBUG 0
#if REGIR_CELL_DEBUG && (RTXDI_REGIR_MODE == RTXDI_REGIR_ONION)
    {
        float3 worldPos = RAB_GetSurfaceWorldPos(surface);
        int cellIndex   = RTXDI_ReGIR_WorldPosToCellIndex(g_Const.regir, worldPos);

        float3 cellColor;
        if (cellIndex < 0)
        {
            // Outside the onion structure
            cellColor = float3(1.0f, 0.0f, 0.0f);
        }
        else
        {
            // Hash cell ID → saturated colour
            uint h = (uint)cellIndex;
            h ^= h >> 16; h *= 0x45d9f3bu;
            h ^= h >> 16; h *= 0x45d9f3bu;
            h ^= h >> 16;
            cellColor = float3(
                float((h      ) & 0xFFu) / 255.0f,
                float((h >>  8) & 0xFFu) / 255.0f,
                float((h >> 16) & 0xFFu) / 255.0f);
        }

        u_DirectLightingRaw[pixelPosition] = float4(cellColor, 1.0f);
    }
#endif // REGIR_CELL_DEBUG

// ── DEBUG TEMP: visualise one random light per ReGIR Onion cell ───────────────
// For each pixel: find its Onion cell, then randomly pick one slot from that cell's
// RIS buffer region and show the light index as a hash colour.
// Black = empty slot (no light assigned). Red = outside onion.
// Set REGIR_LIGHT_DEBUG 1 to enable; restore to 0 when done.
#define REGIR_LIGHT_DEBUG 0
#if REGIR_LIGHT_DEBUG && (RTXDI_REGIR_MODE == RTXDI_REGIR_ONION)
    {
        float3 worldPos = RAB_GetSurfaceWorldPos(surface);
        int    cellIndex = RTXDI_ReGIR_WorldPosToCellIndex(g_Const.regir, worldPos);

        float3 lightColor;
        if (cellIndex < 0)
        {
            // Outside the onion
            lightColor = float3(1.0f, 0.0f, 0.0f);
        }
        else
        {
            uint lightsPerCell = g_Const.regir.commonParams.lightsPerCell;
            uint risOffset     = g_Const.regir.commonParams.risBufferOffset;

            // Pick a deterministic-random slot within this cell using pixel position as seed
            uint seed     = pixelPosition.x * 1664525u + pixelPosition.y * 22695477u + 1013904223u;
            uint slotInCell = seed % max(1u, lightsPerCell);

            uint slotIdx = risOffset + (uint)cellIndex * lightsPerCell + slotInCell;
            uint2 entry  = u_RisBuffer[slotIdx];
            bool hasLight = (entry.x != 0u) || (entry.y != 0u);

            if (hasLight)
            {
                uint lightIdx = entry.x & 0x7FFFFFFFu; // RTXDI_LIGHT_INDEX_MASK
                uint h = lightIdx;
                h ^= h >> 16; h *= 0x45d9f3bu;
                h ^= h >> 16; h *= 0x45d9f3bu;
                h ^= h >> 16;
                lightColor = float3(
                    float((h      ) & 0xFFu) / 255.0f,
                    float((h >>  8) & 0xFFu) / 255.0f,
                    float((h >> 16) & 0xFFu) / 255.0f);
            }
            else
            {
                lightColor = float3(0.0f, 0.0f, 0.0f); // empty slot
            }
        }

        u_DirectLightingRaw[pixelPosition] = float4(lightColor, 1.0f);
    }
#endif // REGIR_LIGHT_DEBUG

// ── DEBUG TEMP: shade pixel with one light sampled from its ReGIR Onion cell ──
// Finds the pixel's Onion cell, picks a slot deterministically, loads the light,
// samples it at the surface position, and writes diffuse + specular to u_DirectLightingRaw.
// No visibility ray is traced (shadow is ignored).
// Black = empty slot or outside onion.
// Set REGIR_SHADE_DEBUG 1 to enable; restore to 0 when done.
#define REGIR_SHADE_DEBUG 0
#if REGIR_SHADE_DEBUG && (RTXDI_REGIR_MODE == RTXDI_REGIR_ONION)
    {
        float3 worldPos  = RAB_GetSurfaceWorldPos(surface);
        int    cellIndex = RTXDI_ReGIR_WorldPosToCellIndex(g_Const.regir, worldPos);

        float3 shadedColor = float3(0.0f, 0.0f, 0.0f);

        if (cellIndex >= 0)
        {
            uint lightsPerCell = g_Const.regir.commonParams.lightsPerCell;
            uint risOffset     = g_Const.regir.commonParams.risBufferOffset;

            // Use the same RNG pattern as GenerateInitialSamples.hlsl
            RTXDI_RandomSamplerState shadeRng = RTXDI_InitRandomSampler(pixelPosition, g_Const.runtimeParams.frameIndex, 0x1234ABCDu);
            uint slotInCell = uint(RTXDI_GetNextRandom(shadeRng) * float(lightsPerCell));
            slotInCell = min(slotInCell, max(1u, lightsPerCell) - 1u);

            uint  slotIdx  = risOffset + (uint)cellIndex * lightsPerCell + slotInCell;
            uint2 entry    = u_RisBuffer[slotIdx];
            bool  hasLight = (entry.x != 0u) || (entry.y != 0u);

            if (hasLight)
            {
                uint lightIdx = entry.x & 0x7FFFFFFFu; // RTXDI_LIGHT_INDEX_MASK

                // Load light data. If compact bit is set the data is in RisLightDataBuffer;
                // RAB_LoadLightInfo always goes to the main buffer, which is fine for debug.
                RAB_LightInfo lightInfo = RAB_LoadLightInfo(lightIdx, false);

                // Use standard RNG for light UV sampling (same as InitialSampling path)
                float2 lightUV = float2(RTXDI_GetNextRandom(shadeRng), RTXDI_GetNextRandom(shadeRng));

                RAB_LightSample lightSample = RAB_SamplePolymorphicLight(lightInfo, surface, lightUV);

                if (lightSample.solidAnglePdf > 0.0f)
                {
                    // Evaluate unshadowed diffuse + specular BRDF response
                    float3 reflected = RAB_GetReflectedBrdfRadianceForSurface(
                        lightSample.position, lightSample.radiance, surface);
                    shadedColor = reflected / lightSample.solidAnglePdf;
                    shadedColor *= 10000;
                    // shadedColor = float3(0.0f, 0.0f, 1.0f); 
                }else{
                    shadedColor = float3(0.0f, 1.0f, 0.0f); 
                }
            }else
            {
                shadedColor = float3(1.0f, 0.0f, 0.0f); // empty slot
            }
        }

        u_DirectLightingRaw[pixelPosition] = float4(shadedColor, 1.0f);
    }
#endif // REGIR_SHADE_DEBUG

// ── DEBUG TEMP: visualise per-slot RIS weight (PDF proxy) in block layout ─────
// Same block layout as REGIR_VIS_DEBUG: each pixel = one RIS slot.
// Colour = heatmap of asfloat(entry.y) (the RIS weight stored by PresampleReGIR).
// Black = empty slot; blue→cyan→green→yellow→red = low→high weight.
// Set REGIR_PDF_DEBUG 1 to enable; restore to 0 when done.
#define REGIR_PDF_DEBUG 0
#if REGIR_PDF_DEBUG
    {
        uint regirOffset   = g_Const.regir.commonParams.risBufferOffset;
        uint lightsPerCell = g_Const.regir.commonParams.lightsPerCell;
        uint totalCells    = max(1u, g_Const.regir.gridParams.cellsX
                                   * g_Const.regir.gridParams.cellsY
                                   * g_Const.regir.gridParams.cellsZ);

        uint blockSize    = (uint)ceil(sqrt((float)max(1u, lightsPerCell)));
        uint screenW      = (uint)g_Const.view.viewportSize.x;

        uint blockX       = pixelPosition.x / blockSize;
        uint blockY       = pixelPosition.y / blockSize;
        uint blocksPerRow = max(1u, (screenW + blockSize - 1u) / blockSize);
        uint cellIdx      = blockY * blocksPerRow + blockX;

        uint lx           = pixelPosition.x % blockSize;
        uint ly           = pixelPosition.y % blockSize;
        uint slotInCell   = ly * blockSize + lx;

        float3 color = float3(0.0f, 0.0f, 0.0f);

        if (cellIdx < totalCells && slotInCell < lightsPerCell)
        {
            uint slotIdx = regirOffset + cellIdx * lightsPerCell + slotInCell;
            uint2 entry  = u_RisBuffer[slotIdx];
            bool hasLight = (entry.x != 0u) || (entry.y != 0u);

            if (hasLight)
            {
                float w = asfloat(entry.y); // RIS weight = weightSum / selectedTargetPdf

                // Log-scale normalise: map [0, 1e6] roughly to [0, 1]
                float t = saturate(log2(1.0f + max(0.0f, w)) / 20.0f);

                // Jet colormap: blue(0) → cyan(0.25) → green(0.5) → yellow(0.75) → red(1)
                color.r = saturate(1.5f - abs(4.0f * t - 3.0f));
                color.g = saturate(1.5f - abs(4.0f * t - 2.0f));
                color.b = saturate(1.5f - abs(4.0f * t - 1.0f));
            }
            else
            {
                color = float3(0.04f, 0.04f, 0.04f); // dark grey = empty
            }
        }

        // Same edge darkening as REGIR_VIS_DEBUG
        float edgeX   = min((float)lx, (float)(blockSize - 1u - lx));
        float edgeY   = min((float)ly, (float)(blockSize - 1u - ly));
        float edgeFade = saturate(min(edgeX, edgeY) * 0.5f);
        color *= lerp(0.15f, 1.0f, edgeFade);

        u_DirectLightingRaw[pixelPosition] = float4(color, 1.0f);
    }
#endif // REGIR_PDF_DEBUG
}
