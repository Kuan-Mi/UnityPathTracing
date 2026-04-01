// Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
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
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_PackedGIReservoir
    {
        public float3 position;
        public uint packed_miscData_age_M;

        public uint packed_radiance; // 32bit LogLUV format
        public float weight;
        public uint packed_normal; // 2x 16-bit snorms in octahedral mapping
        public float unused;
    }

    public enum ResTIRGI_TemporalBiasCorrectionMode : uint
    {
        Off = RtxdiConstants.RTXDI_BIAS_CORRECTION_OFF,
        Basic = RtxdiConstants.RTXDI_BIAS_CORRECTION_BASIC,

        // Pairwise is not supported
        Raytraced = RtxdiConstants.RTXDI_BIAS_CORRECTION_RAY_TRACED,
    }

    public enum ResTIRGI_SpatialBiasCorrectionMode : uint
    {
        Off = RtxdiConstants.RTXDI_BIAS_CORRECTION_OFF,
        Basic = RtxdiConstants.RTXDI_BIAS_CORRECTION_BASIC,

        // Pairwise is not supported
        Raytraced = RtxdiConstants.RTXDI_BIAS_CORRECTION_RAY_TRACED,
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct ReSTIRGI_TemporalResamplingParameters
    {
        [Range(0.0f, 1)]
        public float depthThreshold;

        [Range(0.0f, 1)]
        public float normalThreshold;

        [Range(0, 1)]
        public uint enablePermutationSampling;

        [Range(0, 40)]
        public uint maxHistoryLength;

        [Range(0, 40)]
        public uint maxReservoirAge;

        [Range(0, 1)]
        public uint enableBoilingFilter;

        [Range(0.0f, 1)]
        public float boilingFilterStrength;

        [Range(0, 1)]
        public uint enableFallbackSampling;

        public ResTIRGI_TemporalBiasCorrectionMode temporalBiasCorrectionMode;
        public uint uniformRandomNumber;
        public uint pad2;
        public uint pad3;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct ReSTIRGI_SpatialResamplingParameters
    {
        [Range(0.0f, 1)]
        public float spatialDepthThreshold;
        [Range(0.0f, 1)]
        public float spatialNormalThreshold;
        [Range(0, 40)]
        public uint numSpatialSamples;
        [Range(0, 64)]
        public float spatialSamplingRadius;

        public ResTIRGI_SpatialBiasCorrectionMode spatialBiasCorrectionMode;
        public uint pad1;
        public uint pad2;
        public uint pad3;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct ReSTIRGI_FinalShadingParameters
    {
        public uint enableFinalVisibility;
        public uint enableFinalMIS;
        public uint pad1;
        public uint pad2;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReSTIRGI_BufferIndices
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

    [StructLayout(LayoutKind.Sequential)]
    public struct ReSTIRGI_Parameters
    {
        public RTXDI_ReservoirBufferParameters reservoirBufferParams;
        public ReSTIRGI_BufferIndices bufferIndices;
        public ReSTIRGI_TemporalResamplingParameters temporalResamplingParams;
        public ReSTIRGI_SpatialResamplingParameters spatialResamplingParams;
        public ReSTIRGI_FinalShadingParameters finalShadingParams;
    }
}