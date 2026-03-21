// Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

using System.Runtime.InteropServices;

namespace Rtxdi.DI
{
    public enum ReSTIRDI_LocalLightSamplingMode : uint
    {
        Uniform   = RtxdiConstants.ReSTIRDI_LocalLightSamplingMode_UNIFORM,
        Power_RIS = RtxdiConstants.ReSTIRDI_LocalLightSamplingMode_POWER_RIS,
        ReGIR_RIS = RtxdiConstants.ReSTIRDI_LocalLightSamplingMode_REGIR_RIS,
    }

    public enum ReSTIRDI_TemporalBiasCorrectionMode : uint
    {
        Off       = RtxdiConstants.RTXDI_BIAS_CORRECTION_OFF,
        Basic     = RtxdiConstants.RTXDI_BIAS_CORRECTION_BASIC,
        Pairwise  = RtxdiConstants.RTXDI_BIAS_CORRECTION_PAIRWISE,
        Raytraced = RtxdiConstants.RTXDI_BIAS_CORRECTION_RAY_TRACED,
    }

    public enum ReSTIRDI_SpatialBiasCorrectionMode : uint
    {
        Off       = RtxdiConstants.RTXDI_BIAS_CORRECTION_OFF,
        Basic     = RtxdiConstants.RTXDI_BIAS_CORRECTION_BASIC,
        Pairwise  = RtxdiConstants.RTXDI_BIAS_CORRECTION_PAIRWISE,
        Raytraced = RtxdiConstants.RTXDI_BIAS_CORRECTION_RAY_TRACED,
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct ReSTIRDI_BufferIndices
    {
        public uint initialSamplingOutputBufferIndex;
        public uint temporalResamplingInputBufferIndex;
        public uint temporalResamplingOutputBufferIndex;
        public uint spatialResamplingInputBufferIndex;

        public uint spatialResamplingOutputBufferIndex;
        public uint shadingInputBufferIndex;
        public uint pad1;
        public uint pad2;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct ReSTIRDI_InitialSamplingParameters
    {
        public uint numPrimaryLocalLightSamples;
        public uint numPrimaryInfiniteLightSamples;
        public uint numPrimaryEnvironmentSamples;
        public uint numPrimaryBrdfSamples;

        public float brdfCutoff;
        public uint enableInitialVisibility;
        public uint environmentMapImportanceSampling;
        public ReSTIRDI_LocalLightSamplingMode localLightSamplingMode;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct ReSTIRDI_TemporalResamplingParameters
    {
        public float temporalDepthThreshold;
        public float temporalNormalThreshold;
        public uint maxHistoryLength;
        public ReSTIRDI_TemporalBiasCorrectionMode temporalBiasCorrection;

        public uint enablePermutationSampling;
        public float permutationSamplingThreshold;
        public uint enableBoilingFilter;
        public float boilingFilterStrength;

        public uint discardInvisibleSamples;
        public uint uniformRandomNumber;
        public uint pad2;
        public uint pad3;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct ReSTIRDI_SpatialResamplingParameters
    {
        public float spatialDepthThreshold;
        public float spatialNormalThreshold;
        public ReSTIRDI_SpatialBiasCorrectionMode spatialBiasCorrection;
        public uint numSpatialSamples;

        public uint numDisocclusionBoostSamples;
        public float spatialSamplingRadius;
        public uint neighborOffsetMask;
        public uint discountNaiveSamples;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct ReSTIRDI_ShadingParameters
    {
        public uint enableFinalVisibility;
        public uint reuseFinalVisibility;
        public uint finalVisibilityMaxAge;
        public float finalVisibilityMaxDistance;

        public uint enableDenoiserInputPacking;
        public uint pad1;
        public uint pad2;
        public uint pad3;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct ReSTIRDI_Parameters
    {
        public RTXDI_ReservoirBufferParameters reservoirBufferParams;
        public ReSTIRDI_BufferIndices bufferIndices;
        public ReSTIRDI_InitialSamplingParameters initialSamplingParams;
        public ReSTIRDI_TemporalResamplingParameters temporalResamplingParams;
        public ReSTIRDI_SpatialResamplingParameters spatialResamplingParams;
        public ReSTIRDI_ShadingParameters shadingParams;
    }
}
