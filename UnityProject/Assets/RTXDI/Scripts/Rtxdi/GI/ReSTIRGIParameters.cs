// Copyright (c) 2020-2026, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;

namespace Rtxdi.GI
{
    // -------------------------------------------------------------------------
    // Enum
    // -------------------------------------------------------------------------

    /// <summary>
    /// Pairwise mode is NOT supported for GI bias correction.
    /// </summary>
    public enum RTXDI_GIBiasCorrectionMode : uint
    {
        Off = RtxdiConstants.RTXDI_BIAS_CORRECTION_OFF,
        Basic = RtxdiConstants.RTXDI_BIAS_CORRECTION_BASIC,
        Raytraced = RtxdiConstants.RTXDI_BIAS_CORRECTION_RAY_TRACED,
    }

    // -------------------------------------------------------------------------
    // Packed reservoir
    // -------------------------------------------------------------------------

    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_PackedGIReservoir
    {
        public float3 position;
        public uint packed_miscData_age_M;

        public uint packed_radiance; // 32-bit LogLUV format
        public float weight;
        public uint packed_normal; // 2x 16-bit snorms in octahedral mapping
        public float unused;
    }

    // -------------------------------------------------------------------------
    // Structs  (layout matches RTXDI-Library/Include/Rtxdi/GI/ReSTIRGIParameters.h)
    // -------------------------------------------------------------------------

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_GIBufferIndices
    {
        public uint secondarySurfaceReSTIRDIOutputBufferIndex;
        public uint temporalResamplingInputBufferIndex;
        public uint temporalResamplingOutputBufferIndex;
        public uint spatialResamplingInputBufferIndex;

        public uint spatialResamplingOutputBufferIndex;
        public uint finalShadingInputBufferIndex;
        public uint pad1;
        public uint pad2;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_GITemporalResamplingParameters
    {
        [Range(0f, 1f)]
        public float depthThreshold;

        [Range(0f, 1f)]
        public float normalThreshold;

        [Range(0, 40)]
        public uint maxHistoryLength;

        // Resample from a region around current pixel when motion vector finds no match.
        [Toggle]
        public uint enableFallbackSampling;

        public RTXDI_GIBiasCorrectionMode biasCorrectionMode;

        // Discard reservoirs older than this age.
        [Range(0, 60)]
        public uint maxReservoirAge;

        // Permutation sampling for denoiser-friendly output.
        [Toggle]
        public uint enablePermutationSampling;

        // Per-frame uniform random number (set by SetFrameIndex).
        [HideInInspector]
        public uint uniformRandomNumber;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_GISpatialResamplingParameters
    {
        [Range(0f, 1f)]
        public float depthThreshold;

        [Range(0f, 1f)]
        public float normalThreshold;

        [Range(0, 32)]
        public uint numSamples;

        [Range(0, 64f)]
        public float samplingRadius;

        public RTXDI_GIBiasCorrectionMode biasCorrectionMode;

        [HideInInspector]
        public uint pad1;

        [HideInInspector]
        public uint pad2;

        [HideInInspector]
        public uint pad3;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_GISpatioTemporalResamplingParameters
    {
        [Range(0f, 1f)]
        public float depthThreshold;

        [Range(0f, 1f)]
        public float normalThreshold;

        public RTXDI_GIBiasCorrectionMode biasCorrectionMode;

        [Range(0, 32)]
        public uint numSamples;

        [Range(0f, 64f)]
        public float samplingRadius;

        [Range(0, 40)]
        public uint maxHistoryLength;

        public uint enableFallbackSampling;
        public uint maxReservoirAge;
        public uint enablePermutationSampling;

        [HideInInspector]
        public uint uniformRandomNumber;

        [HideInInspector]
        public uint pad1;

        [HideInInspector]
        public uint pad2;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_GIFinalShadingParameters
    {
        [Toggle]
        public uint enableFinalVisibility;

        [Toggle]
        public uint enableFinalMIS;

        [HideInInspector]
        public uint pad1;

        [HideInInspector]
        public uint pad2;
    }

    /// <summary>
    /// Full GI parameter block passed to shaders each frame.
    /// </summary>
    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_GIParameters
    {
        public RTXDI_ReservoirBufferParameters reservoirBufferParams;
        public RTXDI_GIBufferIndices bufferIndices;
        public RTXDI_GITemporalResamplingParameters temporalResamplingParams;
        public RTXDI_BoilingFilterParameters boilingFilterParams;
        public RTXDI_GISpatialResamplingParameters spatialResamplingParams;
        public RTXDI_GISpatioTemporalResamplingParameters spatioTemporalResamplingParams;
        public RTXDI_GIFinalShadingParameters finalShadingParams;
    }
}