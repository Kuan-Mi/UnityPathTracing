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
        Uniform   = RtxdiConstants.ReSTIRDI_LocalLightSamplingMode_UNIFORM,
        Power_RIS = RtxdiConstants.ReSTIRDI_LocalLightSamplingMode_POWER_RIS,
        ReGIR_RIS = RtxdiConstants.ReSTIRDI_LocalLightSamplingMode_REGIR_RIS,
    }

    /// <summary>
    /// Note: Pairwise mode is not supported for temporal resampling in the new API.
    /// </summary>
    public enum ReSTIRDI_TemporalBiasCorrectionMode : uint
    {
        Off       = RtxdiConstants.RTXDI_BIAS_CORRECTION_OFF,
        Basic     = RtxdiConstants.RTXDI_BIAS_CORRECTION_BASIC,
        Raytraced = RtxdiConstants.RTXDI_BIAS_CORRECTION_RAY_TRACED,
    }

    public enum ReSTIRDI_SpatialBiasCorrectionMode : uint
    {
        Off       = RtxdiConstants.RTXDI_BIAS_CORRECTION_OFF,
        Basic     = RtxdiConstants.RTXDI_BIAS_CORRECTION_BASIC,
        Pairwise  = RtxdiConstants.RTXDI_BIAS_CORRECTION_PAIRWISE,
        Raytraced = RtxdiConstants.RTXDI_BIAS_CORRECTION_RAY_TRACED,
    }

    public enum ReSTIRDI_SpatioTemporalBiasCorrectionMode : uint
    {
        Off       = RtxdiConstants.RTXDI_BIAS_CORRECTION_OFF,
        Basic     = RtxdiConstants.RTXDI_BIAS_CORRECTION_BASIC,
        Pairwise  = RtxdiConstants.RTXDI_BIAS_CORRECTION_PAIRWISE,
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
        public uint pad1;
        public uint pad2;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_DIInitialSamplingParameters
    {
        [Range(0, 16)] public uint numLocalLightSamples;
        [Range(0, 16)] public uint numInfiniteLightSamples;
        [Range(0, 16)] public uint numEnvironmentSamples;
        [Range(0, 16)] public uint numBrdfSamples;

        public float brdfCutoff;
        public float brdfRayMinT;
        public ReSTIRDI_LocalLightSamplingMode localLightSamplingMode;
        public uint  enableInitialVisibility;

        public uint  environmentMapImportanceSampling;
        public uint  pad1;
        public uint  pad2;
        public uint  pad3;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_DITemporalResamplingParameters
    {
        // Maximum history length for temporal reuse, measured in frames.
        [Range(0, 40)] public uint maxHistoryLength;

        // Bias correction mode for temporal reuse.
        public ReSTIRDI_TemporalBiasCorrectionMode biasCorrectionMode;

        // Surface depth similarity threshold (relative). 0.1 = 10% of current depth.
        [Range(0f, 1f)] public float depthThreshold;

        // Surface normal similarity threshold (dot product).
        [Range(0f, 1f)] public float normalThreshold;

        // Skip bias correction ray trace when invisible samples are discarded.
        public uint enableVisibilityShortcut;

        // Permutation sampling for denoiser-friendly temporal variation.
        public uint enablePermutationSampling;

        // Per-frame uniform random number for permutation sampling (set by SetFrameIndex).
        [HideInInspector] public uint uniformRandomNumber;

        // Not used inside TemporalResampling.hlsli directly, but stored here for completeness.
        public float permutationSamplingThreshold;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_DISpatialResamplingParameters
    {
        // Number of spatial neighbor samples (1-32).
        [Range(0, 32)] public uint numSamples;

        // Neighbor samples used when history is insufficient (disocclusion boost).
        public uint numDisocclusionBoostSamples;

        // Screen-space sampling radius in pixels.
        public float samplingRadius;

        // Bias correction mode for spatial reuse.
        public ReSTIRDI_SpatialBiasCorrectionMode biasCorrectionMode;

        // Surface depth similarity threshold (relative).
        [Range(0f, 1f)] public float depthThreshold;

        // Surface normal similarity threshold.
        [Range(0f, 1f)] public float normalThreshold;

        // Disocclusion boost activated when current reservoir M < targetHistoryLength.
        public uint targetHistoryLength;

        // Compare surface materials before accepting a spatial sample.
        public uint enableMaterialSimilarityTest;

        // Do not spread current-frame or low-history samples to neighbors.
        public uint discountNaiveSamples;

        [HideInInspector] public uint pad1;
        [HideInInspector] public uint pad2;
        [HideInInspector] public uint pad3;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_DISpatioTemporalResamplingParameters
    {
        // Common surface similarity thresholds
        [Range(0f, 1f)] public float depthThreshold;
        [Range(0f, 1f)] public float normalThreshold;

        public ReSTIRDI_SpatioTemporalBiasCorrectionMode biasCorrectionMode;

        // Temporal parameters
        [Range(0, 40)] public uint maxHistoryLength;

        public uint enablePermutationSampling;
        [HideInInspector] public uint uniformRandomNumber;
        public uint enableVisibilityShortcut;

        // Spatial parameters
        [Range(0, 32)] public uint numSamples;
        public uint  numDisocclusionBoostSamples;
        public float samplingRadius;
        public uint  enableMaterialSimilarityTest;
        public uint  discountNaiveSamples;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_ShadingParameters
    {
        public uint  enableFinalVisibility;
        public uint  reuseFinalVisibility;
        [Range(0, 8)]   public uint  finalVisibilityMaxAge;
        [Range(0, 32f)] public float finalVisibilityMaxDistance;

        public uint  enableDenoiserInputPacking;
        [HideInInspector] public uint pad1;
        [HideInInspector] public uint pad2;
        [HideInInspector] public uint pad3;
    }

    /// <summary>
    /// Full DI parameter block passed to shaders each frame.
    /// </summary>
    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_Parameters
    {
        public RTXDI_ReservoirBufferParameters          reservoirBufferParams;
        public RTXDI_DIBufferIndices                    bufferIndices;
        public RTXDI_DIInitialSamplingParameters        initialSamplingParams;
        public RTXDI_DITemporalResamplingParameters     temporalResamplingParams;
        public RTXDI_BoilingFilterParameters            boilingFilterParams;
        public RTXDI_DISpatialResamplingParameters      spatialResamplingParams;
        public RTXDI_DISpatioTemporalResamplingParameters spatioTemporalResamplingParams;
        public RTXDI_ShadingParameters                  shadingParams;
    }
}
