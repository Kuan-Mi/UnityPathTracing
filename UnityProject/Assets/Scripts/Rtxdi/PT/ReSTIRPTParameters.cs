// Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

using System.Runtime.InteropServices;
using UnityEngine;

namespace Rtxdi.PT
{
    // -------------------------------------------------------------------------
    // Constants
    // -------------------------------------------------------------------------

    public static class ReSTIRPTConstants
    {
        public const uint PTPathTraceInvocationType_Initial             = 1;
        public const uint PTPathTraceInvocationType_Temporal            = 2;
        public const uint PTPathTraceInvocationType_TemporalInverse     = 3;
        public const uint PTPathTraceInvocationType_Spatial             = 4;
        public const uint PTPathTraceInvocationType_SpatialInverse      = 5;
        public const uint PTPathTraceInvocationType_DebugTemporalRetrace = 11;
        public const uint PTPathTraceInvocationType_DebugSpatialRetrace  = 12;
    }

    // -------------------------------------------------------------------------
    // Enums
    // -------------------------------------------------------------------------

    public enum RTXDI_PTReconnectionMode : ushort
    {
        FixedThreshold = 0, // RTXDI_RESTIRPT_RECONNECTION_MODE_FIXED_THRESHOLD
        Footprint      = 1, // RTXDI_RESTIRPT_RECONNECTION_MODE_FOOTPRINT
    }

    /// <summary>
    /// Invocation type set by RTXDI before each call to RAB_PathTrace.
    /// </summary>
    public enum RTXDI_PTPathTraceInvocationType : ushort
    {
        Initial              = 1,
        Temporal             = 2,
        TemporalInverse      = 3,
        Spatial              = 4,
        SpatialInverse       = 5,
        DebugTemporalRetrace = 11,
        DebugSpatialRetrace  = 12,
    }

    // -------------------------------------------------------------------------
    // Structs  (layout matches RTXDI-Library/Include/Rtxdi/PT/ReSTIRPTParameters.h)
    // -------------------------------------------------------------------------

    /// <summary>
    /// Packed path-tracing reservoir: 4 x uint4 = 64 bytes.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct RTXDI_PackedPTReservoir
    {
        public fixed uint Data0[4];
        public fixed uint Data1[4];
        public fixed uint Data2[4];
        public fixed uint Data3[4];
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_PTBufferIndices
    {
        public uint initialPathTracerOutputBufferIndex;
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
    public struct RTXDI_PTInitialSamplingParameters
    {
        public uint numInitialSamples;
        public uint maxBounceDepth;
        public uint maxRcVertexLength;
        public uint pad;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_PTReconnectionParameters
    {
        // Footprint method parameters
        public float minConnectionFootprint;
        public float minConnectionFootprintSigma;
        public float minPdfRoughness;
        public float minPdfRoughnessSigma;

        // Threshold method parameters
        public float roughnessThreshold;
        public float distanceThreshold;

        // Mode selection (ushort to match C++ uint16_t)
        public RTXDI_PTReconnectionMode reconnectionMode;
        [HideInInspector] public uint pad1;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_PTHybridShiftPerFrameParameters
    {
        public uint maxBounceDepth;
        public uint maxRcVertexLength;
        [HideInInspector] public uint pad1;
        [HideInInspector] public uint pad2;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_PTTemporalResamplingParameters
    {
        [Range(0f, 1f)] public float depthThreshold;
        [Range(0f, 1f)] public float normalThreshold;
        public uint enablePermutationSampling;
        [Range(0, 40)]  public uint  maxHistoryLength;

        public uint maxReservoirAge;
        public uint enableFallbackSampling;
        public uint enableVisibilityBeforeCombine;

        // Per-frame uniform random number (set by SetFrameIndex).
        [HideInInspector] public uint uniformRandomNumber;

        public uint  duplicationBasedHistoryReduction;
        public float historyReductionStrength;
        [HideInInspector] public uint pad1;
        [HideInInspector] public uint pad2;
    }

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_PTSpatialResamplingParameters
    {
        [Range(0, 32)] public uint  numSpatialSamples;
        public uint  numDisocclusionBoostSamples;
        public uint  maxTemporalHistory;
        public uint  duplicationBasedHistoryReduction;

        public float samplingRadius;
        [Range(0f, 1f)] public float normalThreshold;
        [Range(0f, 1f)] public float depthThreshold;
        [HideInInspector] public uint pad1;
    }

    /// <summary>
    /// Full path-tracing parameter block passed to shaders each frame.
    /// </summary>
    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RTXDI_PTParameters
    {
        public RTXDI_ReservoirBufferParameters      reservoirBuffer;
        public RTXDI_PTBufferIndices                bufferIndices;
        public RTXDI_PTInitialSamplingParameters    initialSampling;
        public RTXDI_PTReconnectionParameters       reconnection;
        public RTXDI_PTTemporalResamplingParameters temporalResampling;
        public RTXDI_PTHybridShiftPerFrameParameters hybridShift;
        public RTXDI_BoilingFilterParameters        boilingFilter;
        public RTXDI_PTSpatialResamplingParameters  spatialResampling;
    }
}
