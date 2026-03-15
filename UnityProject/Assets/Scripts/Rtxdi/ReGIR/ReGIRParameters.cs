// Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

using System.Runtime.InteropServices;

namespace Rtxdi.ReGIR
{
    public static class ReGIRConstants
    {
        public const int RTXDI_ONION_MAX_LAYER_GROUPS = 8;
        public const int RTXDI_ONION_MAX_RINGS        = 52;

        public const uint RTXDI_REGIR_DISABLED = 0;
        public const uint RTXDI_REGIR_GRID     = 1;
        public const uint RTXDI_REGIR_ONION    = 2;

        public const uint REGIR_LOCAL_LIGHT_PRESAMPLING_MODE_UNIFORM   = 0;
        public const uint REGIR_LOCAL_LIGHT_PRESAMPLING_MODE_POWER_RIS = 1;

        public const uint REGIR_LOCAL_LIGHT_FALLBACK_MODE_UNIFORM   = 0;
        public const uint REGIR_LOCAL_LIGHT_FALLBACK_MODE_POWER_RIS = 1;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReGIR_OnionLayerGroup
    {
        public float innerRadius;
        public float outerRadius;
        public float invLogLayerScale;
        public int layerCount;

        public float invEquatorialCellAngle;
        public int cellsPerLayer;
        public int ringOffset;
        public int ringCount;

        public float equatorialCellAngle;
        public float layerScale;
        public int layerCellOffset;
        public int pad1;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReGIR_OnionRing
    {
        public float cellAngle;
        public float invCellAngle;
        public int cellOffset;
        public int cellCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReGIR_CommonParameters
    {
        public uint localLightSamplingFallbackMode;
        public float centerX;
        public float centerY;
        public float centerZ;

        public uint risBufferOffset;
        public uint lightsPerCell;
        public float cellSize;
        public float samplingJitter;

        public uint localLightPresamplingMode;
        public uint numRegirBuildSamples;
        public uint pad1;
        public uint pad2;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReGIR_GridParameters
    {
        public uint cellsX;
        public uint cellsY;
        public uint cellsZ;
        public uint pad1;
    }

    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct ReGIR_OnionParameters
    {
        public fixed byte _layers[ReGIRConstants.RTXDI_ONION_MAX_LAYER_GROUPS * 48]; // sizeof(ReGIR_OnionLayerGroup) = 48
        public fixed byte _rings[ReGIRConstants.RTXDI_ONION_MAX_RINGS * 16];         // sizeof(ReGIR_OnionRing) = 16

        public uint numLayerGroups;
        public float cubicRootFactor;
        public float linearFactor;
        public float pad1;

        public ReGIR_OnionLayerGroup GetLayer(int index)
        {
            fixed (byte* ptr = _layers)
            {
                return ((ReGIR_OnionLayerGroup*)ptr)[index];
            }
        }

        public void SetLayer(int index, ReGIR_OnionLayerGroup value)
        {
            fixed (byte* ptr = _layers)
            {
                ((ReGIR_OnionLayerGroup*)ptr)[index] = value;
            }
        }

        public ReGIR_OnionRing GetRing(int index)
        {
            fixed (byte* ptr = _rings)
            {
                return ((ReGIR_OnionRing*)ptr)[index];
            }
        }

        public void SetRing(int index, ReGIR_OnionRing value)
        {
            fixed (byte* ptr = _rings)
            {
                ((ReGIR_OnionRing*)ptr)[index] = value;
            }
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReGIR_Parameters
    {
        public ReGIR_CommonParameters commonParams;
        public ReGIR_GridParameters gridParams;
        public ReGIR_OnionParameters onionParams;
    }
}
