// Copyright (c) 2020-2026, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

using System.Runtime.InteropServices;
using UnityEngine;

namespace Rtxdi.DI
{
    // -------------------------------------------------------------------------
    // Enums
    // -------------------------------------------------------------------------

    public enum ReSTIRDI_LocalLightSamplingMode : uint
    {
        Uniform = RtxdiConstants.ReSTIRDI_LocalLightSamplingMode_UNIFORM,
        Power_RIS = RtxdiConstants.ReSTIRDI_LocalLightSamplingMode_POWER_RIS,
        ReGIR_RIS = RtxdiConstants.ReSTIRDI_LocalLightSamplingMode_REGIR_RIS,
    }

    public enum ReSTIRDI_TemporalBiasCorrectionMode : uint
    {
        Off = RtxdiConstants.RTXDI_BIAS_CORRECTION_OFF,
        Basic = RtxdiConstants.RTXDI_BIAS_CORRECTION_BASIC,
        Raytraced = RtxdiConstants.RTXDI_BIAS_CORRECTION_RAY_TRACED,
    }

    public enum ReSTIRDI_SpatialBiasCorrectionMode : uint
    {
        Off = RtxdiConstants.RTXDI_BIAS_CORRECTION_OFF,
        Basic = RtxdiConstants.RTXDI_BIAS_CORRECTION_BASIC,
        Pairwise = RtxdiConstants.RTXDI_BIAS_CORRECTION_PAIRWISE,
        Raytraced = RtxdiConstants.RTXDI_BIAS_CORRECTION_RAY_TRACED,
    }

    public enum ReSTIRDI_SpatioTemporalBiasCorrectionMode : uint
    {
        Off = RtxdiConstants.RTXDI_BIAS_CORRECTION_OFF,
        Basic = RtxdiConstants.RTXDI_BIAS_CORRECTION_BASIC,
        Pairwise = RtxdiConstants.RTXDI_BIAS_CORRECTION_PAIRWISE,
        Raytraced = RtxdiConstants.RTXDI_BIAS_CORRECTION_RAY_TRACED,
    }

    // -------------------------------------------------------------------------
    // Structs  (layout matches RTXDI-Library/Include/Rtxdi/DI/ReSTIRDIParameters.h)
    // -------------------------------------------------------------------------

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_DIBufferIndices
    {
        public uint initialSamplingOutputBufferIndex;
        public uint temporalResamplingInputBufferIndex;
        public uint temporalResamplingOutputBufferIndex;
        public uint spatialResamplingInputBufferIndex;

        public uint spatialResamplingOutputBufferIndex;
        public uint shadingInputBufferIndex;

        [HideInInspector]
        public uint pad1;

        [HideInInspector]
        public uint pad2;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_DIInitialSamplingParameters
    {
        [Range(0, 16)]
        public uint numLocalLightSamples;

        [Range(0, 16)]
        public uint numInfiniteLightSamples;

        [Range(0, 16)]
        public uint numEnvironmentSamples;

        [Range(0, 16)]
        public uint numBrdfSamples;

        [Range(0, 0.001f)]
        public float brdfCutoff;

        [Range(0, 0.01f)]
        public float brdfRayMinT;

        public ReSTIRDI_LocalLightSamplingMode localLightSamplingMode;

        [Toggle]
        public uint enableInitialVisibility;

        [Toggle]
        public uint environmentMapImportanceSampling;

        [HideInInspector]
        public uint pad1;

        [HideInInspector]
        public uint pad2;

        [HideInInspector]
        public uint pad3;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_DITemporalResamplingParameters
    {
        // Maximum history length for temporal reuse, measured in frames.
        [Range(0, 40)]
        public uint maxHistoryLength;

        // Bias correction mode for temporal reuse.
        public ReSTIRDI_TemporalBiasCorrectionMode biasCorrectionMode;

        // Surface depth similarity threshold (relative). 0.1 = 10% of current depth.
        [Range(0f, 1f)]
        public float depthThreshold;

        // Surface normal similarity threshold (dot product).
        [Range(0f, 1f)]
        public float normalThreshold;

        // Skip bias correction ray trace when invisible samples are discarded.
        [Toggle]
        public uint enableVisibilityShortcut;

        // Permutation sampling for denoiser-friendly temporal variation.
        [Toggle]
        public uint enablePermutationSampling;

        // Per-frame uniform random number for permutation sampling (set by SetFrameIndex).
        [HideInInspector]
        public uint uniformRandomNumber;

        // Not used inside TemporalResampling.hlsl directly, but stored here for completeness.
        [Range(0f, 1f)]
        public float permutationSamplingThreshold;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_DISpatialResamplingParameters
    {
        // Number of spatial neighbor samples (1-32).
        [Range(0, 32)]
        public uint numSamples;

        [Range(0, 32)]
        // Neighbor samples used when history is insufficient (disocclusion boost).
        public uint numDisocclusionBoostSamples;

        [Range(0, 64)]
        // Screen-space sampling radius in pixels.
        public float samplingRadius;

        // Bias correction mode for spatial reuse.
        public ReSTIRDI_SpatialBiasCorrectionMode biasCorrectionMode;

        // Surface depth similarity threshold (relative).
        [Range(0f, 1f)]
        public float depthThreshold;

        // Surface normal similarity threshold.
        [Range(0f, 1f)]
        public float normalThreshold;

        [Range(0f, 30f)]
        // Disocclusion boost activated when current reservoir M < targetHistoryLength.
        public uint targetHistoryLength;

        [Toggle]
        // Compare surface materials before accepting a spatial sample.
        public uint enableMaterialSimilarityTest;

        // Do not spread current-frame or low-history samples to neighbors.
        [Toggle]
        public uint discountNaiveSamples;

        [HideInInspector]
        public uint pad1;

        [HideInInspector]
        public uint pad2;

        [HideInInspector]
        public uint pad3;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_DISpatioTemporalResamplingParameters
    {
        // Common surface similarity thresholds
        [Range(0f, 1f)]
        public float depthThreshold;

        [Range(0f, 1f)]
        public float normalThreshold;

        public ReSTIRDI_SpatioTemporalBiasCorrectionMode biasCorrectionMode;

        // Temporal parameters
        [Range(0, 40)]
        public uint maxHistoryLength;

        [Toggle]
        public uint enablePermutationSampling;

        [HideInInspector]
        public uint uniformRandomNumber;

        [Toggle]
        public uint enableVisibilityShortcut;

        // Spatial parameters
        [Range(0, 32)]
        public uint numSamples;

        [Range(0, 32)]
        public uint numDisocclusionBoostSamples;

        [Range(0, 64)]
        public float samplingRadius;

        [Toggle]
        public uint enableMaterialSimilarityTest;

        [Toggle]
        public uint discountNaiveSamples;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_ShadingParameters
    {
        [Toggle]
        public uint enableFinalVisibility;

        [Toggle]
        public uint reuseFinalVisibility;

        [Range(0, 8)]
        public uint finalVisibilityMaxAge;

        [Range(0, 32f)]
        public float finalVisibilityMaxDistance;

        [Toggle]
        public uint enableDenoiserInputPacking;

        [HideInInspector]
        public uint pad1;

        [HideInInspector]
        public uint pad2;

        [HideInInspector]
        public uint pad3;
    }

    /// <summary>
    /// Full DI parameter block passed to shaders each frame.
    /// </summary>
    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_Parameters
    {
        public RTXDI_ReservoirBufferParameters reservoirBufferParams;
        public RTXDI_DIBufferIndices bufferIndices;
        public RTXDI_DIInitialSamplingParameters initialSamplingParams;
        public RTXDI_DITemporalResamplingParameters temporalResamplingParams;
        public RTXDI_BoilingFilterParameters boilingFilterParams;
        public RTXDI_DISpatialResamplingParameters spatialResamplingParams;
        public RTXDI_DISpatioTemporalResamplingParameters spatioTemporalResamplingParams;
        public RTXDI_ShadingParameters shadingParams;
    }
}