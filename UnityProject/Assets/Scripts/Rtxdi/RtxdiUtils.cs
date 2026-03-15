// Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

using System;

namespace Rtxdi
{
    // Checkerboard sampling modes match those used in NRD, based on frameIndex:
    // Even frame(0)  Odd frame(1)   ...
    //     B W             W B
    //     W B             B W
    // BLACK and WHITE modes define cells with VALID data
    public enum CheckerboardMode : uint
    {
        Off   = 0,
        Black = 1,
        White = 2,
    }

    public static class RtxdiUtils
    {
        public static RTXDI_ReservoirBufferParameters CalculateReservoirBufferParameters(
            uint renderWidth, uint renderHeight, CheckerboardMode checkerboardMode)
        {
            renderWidth = (checkerboardMode == CheckerboardMode.Off)
                ? renderWidth
                : (renderWidth + 1) / 2;

            uint renderWidthBlocks = (renderWidth + RtxdiConstants.RTXDI_RESERVOIR_BLOCK_SIZE - 1)
                                     / RtxdiConstants.RTXDI_RESERVOIR_BLOCK_SIZE;
            uint renderHeightBlocks = (renderHeight + RtxdiConstants.RTXDI_RESERVOIR_BLOCK_SIZE - 1)
                                      / RtxdiConstants.RTXDI_RESERVOIR_BLOCK_SIZE;

            RTXDI_ReservoirBufferParameters p;
            p.reservoirBlockRowPitch = renderWidthBlocks
                * (RtxdiConstants.RTXDI_RESERVOIR_BLOCK_SIZE * RtxdiConstants.RTXDI_RESERVOIR_BLOCK_SIZE);
            p.reservoirArrayPitch = p.reservoirBlockRowPitch * renderHeightBlocks;
            p.pad1 = 0;
            p.pad2 = 0;
            return p;
        }

        public static void ComputePdfTextureSize(uint maxItems,
            out uint outWidth, out uint outHeight, out uint outMipLevels)
        {
            double textureWidth = Math.Max(1.0, Math.Ceiling(Math.Sqrt((double)maxItems)));
            textureWidth = Math.Pow(2.0, Math.Ceiling(Math.Log(textureWidth, 2.0)));
            double textureHeight = Math.Max(1.0, Math.Ceiling(maxItems / textureWidth));
            textureHeight = Math.Pow(2.0, Math.Ceiling(Math.Log(textureHeight, 2.0)));
            double textureMips = Math.Max(1.0, Math.Log(Math.Max(textureWidth, textureHeight), 2.0) + 1.0);

            outWidth    = (uint)textureWidth;
            outHeight   = (uint)textureHeight;
            outMipLevels = (uint)textureMips;
        }

        public static void FillNeighborOffsetBuffer(byte[] buffer, uint neighborOffsetCount)
        {
            // Create a sequence of low-discrepancy samples within a unit radius around the origin
            // for "randomly" sampling neighbors during spatial resampling

            int R = 250;
            const float phi2 = 1.0f / 1.3247179572447f;
            uint num = 0;
            float u = 0.5f;
            float v = 0.5f;
            while (num < neighborOffsetCount * 2)
            {
                u += phi2;
                v += phi2 * phi2;
                if (u >= 1.0f) u -= 1.0f;
                if (v >= 1.0f) v -= 1.0f;

                float rSq = (u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f);
                if (rSq > 0.25f)
                    continue;

                buffer[num++] = (byte)((u - 0.5f) * R);
                buffer[num++] = (byte)((v - 0.5f) * R);
            }
        }

        // 32 bit Jenkins hash
        public static uint JenkinsHash(uint a)
        {
            // http://burtleburtle.net/bob/hash/integer.html
            a = (a + 0x7ed55d16) + (a << 12);
            a = (a ^ 0xc761c23c) ^ (a >> 19);
            a = (a + 0x165667b1) + (a << 5);
            a = (a + 0xd3a2646c) ^ (a << 9);
            a = (a + 0xfd7046c5) + (a << 3);
            a = (a ^ 0xb55a4f09) ^ (a >> 16);
            return a;
        }
    }
}
