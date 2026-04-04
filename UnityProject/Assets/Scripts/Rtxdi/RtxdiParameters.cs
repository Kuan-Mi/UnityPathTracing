// Copyright (c) 2020-2026, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

using System.Runtime.InteropServices;

namespace Rtxdi
{
    /// <summary>
    /// Constants from RtxdiParameters.h and related headers.
    /// </summary>
    public static class RtxdiConstants
    {
        public const uint RTXDI_LIGHT_COMPACT_BIT                           = 0x80000000u;
        public const uint RTXDI_LIGHT_INDEX_MASK                            = 0x7fffffff;
        public const uint RTXDI_RESERVOIR_BLOCK_SIZE                        = 16;

        public const uint RTXDI_BIAS_CORRECTION_OFF                         = 0;
        public const uint RTXDI_BIAS_CORRECTION_BASIC                       = 1;
        public const uint RTXDI_BIAS_CORRECTION_PAIRWISE                    = 2;
        public const uint RTXDI_BIAS_CORRECTION_RAY_TRACED                  = 3;

        public const uint ReSTIRDI_LocalLightSamplingMode_UNIFORM           = 0;
        public const uint ReSTIRDI_LocalLightSamplingMode_POWER_RIS         = 1;
        public const uint ReSTIRDI_LocalLightSamplingMode_REGIR_RIS         = 2;

        public const uint RTXDI_RESTIRPT_RECONNECTION_MODE_FIXED_THRESHOLD  = 0;
        public const uint RTXDI_RESTIRPT_RECONNECTION_MODE_FOOTPRINT        = 1;

        public const uint RTXDI_NAIVE_SAMPLING_M_THRESHOLD                  = 2;
        public const uint RTXDI_ENABLE_PRESAMPLING                          = 1;
        public const uint RTXDI_INVALID_LIGHT_INDEX                         = 0xffffffffu;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_RISBufferSegmentParameters
    {
        public uint bufferOffset;
        public uint tileSize;
        public uint tileCount;
        public uint pad1;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_LightBufferRegion
    {
        public uint firstLightIndex;
        public uint numLights;
        public uint pad1;
        public uint pad2;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_EnvironmentLightBufferParameters
    {
        public uint lightPresent;
        public uint lightIndex;
        public uint pad1;
        public uint pad2;
    }

    /// <summary>
    /// Per-frame runtime parameters passed to the shader.
    /// frameIndex is used to drive checkerboard field and permutation sampling.
    /// </summary>
    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_RuntimeParameters
    {
        public uint neighborOffsetMask;      // spatial neighbor lookup mask
        public uint activeCheckerboardField; // 0 = off, 1 = odd pixels, 2 = even pixels
        public uint frameIndex;
        public uint pad2;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_LightBufferParameters
    {
        public RTXDI_LightBufferRegion localLightBufferRegion;
        public RTXDI_LightBufferRegion infiniteLightBufferRegion;
        public RTXDI_EnvironmentLightBufferParameters environmentLightParams;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_ReservoirBufferParameters
    {
        public uint reservoirBlockRowPitch;
        public uint reservoirArrayPitch;
        public uint pad1;
        public uint pad2;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_BoilingFilterParameters
    {
        public uint  enableBoilingFilter;
        public float boilingFilterStrength;
        public uint  pad1;
        public uint  pad2;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_PackedDIReservoir
    {
        public uint  lightData;
        public uint  uvData;
        public uint  mVisibility;
        public uint  distanceAge;
        public float targetPdf;
        public float weight;
    }
}
