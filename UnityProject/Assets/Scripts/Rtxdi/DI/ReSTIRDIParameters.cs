// Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
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
        Pairwise = RtxdiConstants.RTXDI_BIAS_CORRECTION_PAIRWISE,
        Raytraced = RtxdiConstants.RTXDI_BIAS_CORRECTION_RAY_TRACED,
    }

    public enum ReSTIRDI_SpatialBiasCorrectionMode : uint
    {
        Off = RtxdiConstants.RTXDI_BIAS_CORRECTION_OFF,
        Basic = RtxdiConstants.RTXDI_BIAS_CORRECTION_BASIC,
        Pairwise = RtxdiConstants.RTXDI_BIAS_CORRECTION_PAIRWISE,
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

        public override string ToString()
        {
            return $"BufferIndices: " +
                   $"initialSamplingOutputBufferIndex={initialSamplingOutputBufferIndex}, " +
                   $"temporalResamplingInputBufferIndex={temporalResamplingInputBufferIndex}, " +
                   $"temporalResamplingOutputBufferIndex={temporalResamplingOutputBufferIndex}, " +
                   $"spatialResamplingInputBufferIndex={spatialResamplingInputBufferIndex}, " +
                   $"spatialResamplingOutputBufferIndex={spatialResamplingOutputBufferIndex}, " +
                   $"shadingInputBufferIndex={shadingInputBufferIndex}";
        }
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct ReSTIRDI_InitialSamplingParameters
    {
        [Range(0, 16)]
        public uint numPrimaryLocalLightSamples;

        [Range(0, 16)]
        public uint numPrimaryInfiniteLightSamples;

        [Range(0, 16)]
        public uint numPrimaryEnvironmentSamples;

        [Range(0, 16)]
        public uint numPrimaryBrdfSamples;

        public float brdfCutoff;

        [Toggle]
        public uint enableInitialVisibility;

        [Range(0, 16)]
        public uint environmentMapImportanceSampling;

        public ReSTIRDI_LocalLightSamplingMode localLightSamplingMode;

        public override string ToString()
        {
            return $"InitialSamplingParameters: " +
                   $"numPrimaryLocalLightSamples={numPrimaryLocalLightSamples}, " +
                   $"numPrimaryInfiniteLightSamples={numPrimaryInfiniteLightSamples}, " +
                   $"numPrimaryEnvironmentSamples={numPrimaryEnvironmentSamples}, " +
                   $"numPrimaryBrdfSamples={numPrimaryBrdfSamples}, " +
                   $"brdfCutoff={brdfCutoff}, " +
                   $"enableInitialVisibility={enableInitialVisibility}, " +
                   $"environmentMapImportanceSampling={environmentMapImportanceSampling}, " +
                   $"localLightSamplingMode={localLightSamplingMode}";
        }
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct ReSTIRDI_TemporalResamplingParameters
    {
        [Range(0.0f, 1)]
        public float temporalDepthThreshold;

        [Range(0.0f, 1)]
        public float temporalNormalThreshold;

        [Range(0, 40)]
        public uint maxHistoryLength;

        public ReSTIRDI_TemporalBiasCorrectionMode temporalBiasCorrection;

        [Toggle]
        public uint enablePermutationSampling;

        [Range(0, 1.0f)]
        public float permutationSamplingThreshold;

        [Toggle]
        public uint enableBoilingFilter;

        [Range(0, 1.0f)]
        public float boilingFilterStrength;

        [Range(0, 4)]
        public uint discardInvisibleSamples;

        [HideInInspector]
        public uint uniformRandomNumber;

        [HideInInspector]
        public uint pad2;

        [HideInInspector]
        public uint pad3;

        public override string ToString()
        {
            return $"TemporalResamplingParameters: " +
                   $"temporalDepthThreshold={temporalDepthThreshold}, " +
                   $"temporalNormalThreshold={temporalNormalThreshold}, " +
                   $"maxHistoryLength={maxHistoryLength}, " +
                   $"temporalBiasCorrection={temporalBiasCorrection}, " +
                   $"enablePermutationSampling={enablePermutationSampling}, " +
                   $"permutationSamplingThreshold={permutationSamplingThreshold}, " +
                   $"enableBoilingFilter={enableBoilingFilter}, " +
                   $"boilingFilterStrength={boilingFilterStrength}, " +
                   $"discardInvisibleSamples={discardInvisibleSamples}, " +
                   $"uniformRandomNumber={uniformRandomNumber}";
        }
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct ReSTIRDI_SpatialResamplingParameters
    {
        [Range(0.0f, 1)]
        public float spatialDepthThreshold;
        [Range(0.0f, 1)]
        public float spatialNormalThreshold;
        public ReSTIRDI_SpatialBiasCorrectionMode spatialBiasCorrection;
        
        [Range(0, 40)]
        public uint numSpatialSamples;

        public uint numDisocclusionBoostSamples;
        public float spatialSamplingRadius;
        public uint neighborOffsetMask;
        public uint discountNaiveSamples;

        public override string ToString()
        {
            return $"SpatialResamplingParameters: " +
                   $"spatialDepthThreshold={spatialDepthThreshold}, " +
                   $"spatialNormalThreshold={spatialNormalThreshold}, " +
                   $"spatialBiasCorrection={spatialBiasCorrection}, " +
                   $"numSpatialSamples={numSpatialSamples}, " +
                   $"numDisocclusionBoostSamples={numDisocclusionBoostSamples}, " +
                   $"spatialSamplingRadius={spatialSamplingRadius}, " +
                   $"neighborOffsetMask={neighborOffsetMask}, " +
                   $"discountNaiveSamples={discountNaiveSamples}";
        }
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct ReSTIRDI_ShadingParameters
    {
        [Toggle]
        public uint enableFinalVisibility;

        [Toggle]
        public uint reuseFinalVisibility;

        [Range(0, 8)]
        public uint finalVisibilityMaxAge;

        [Range(0, 32)]
        public float finalVisibilityMaxDistance;
        
        [Toggle]
        public uint enableDenoiserInputPacking;

        [HideInInspector]
        public uint pad1;

        [HideInInspector]
        public uint pad2;

        [HideInInspector]
        public uint pad3;

        public override string ToString()
        {
            return $"ShadingParameters: " +
                   $"enableFinalVisibility={enableFinalVisibility}, " +
                   $"reuseFinalVisibility={reuseFinalVisibility}, " +
                   $"finalVisibilityMaxAge={finalVisibilityMaxAge}, " +
                   $"finalVisibilityMaxDistance={finalVisibilityMaxDistance}, " +
                   $"enableDenoiserInputPacking={enableDenoiserInputPacking}";
        }
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

        public override string ToString()
        {
            return $"ReSTIRDI_Parameters: " +
                   $"\n{reservoirBufferParams}" +
                   $"\n{bufferIndices}" +
                   $"\n{initialSamplingParams}" +
                   $"\n{temporalResamplingParams}" +
                   $"\n{spatialResamplingParams}" +
                   $"\n{shadingParams}";
        }
    }
}