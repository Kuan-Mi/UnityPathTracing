// Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

using Rtxdi.DI;
using UnityEngine;

namespace Rtxdi.DI
{
    public static class ReSTIRDIDefaults
    {
        public const uint NumReservoirBuffers = 3;

        public static ReSTIRDI_BufferIndices GetDefaultBufferIndices()
        {
            return new ReSTIRDI_BufferIndices
            {
                initialSamplingOutputBufferIndex  = 0,
                temporalResamplingInputBufferIndex  = 0,
                temporalResamplingOutputBufferIndex = 0,
                spatialResamplingInputBufferIndex   = 0,
                spatialResamplingOutputBufferIndex  = 0,
                shadingInputBufferIndex            = 0,
            };
        }

        public static ReSTIRDI_InitialSamplingParameters GetDefaultInitialSamplingParams()
        {
            return new ReSTIRDI_InitialSamplingParameters
            {
                brdfCutoff = 0.0001f,
                enableInitialVisibility = 1,
                environmentMapImportanceSampling = 1,
                localLightSamplingMode = ReSTIRDI_LocalLightSamplingMode.Uniform,
                numPrimaryBrdfSamples = 1,
                numPrimaryEnvironmentSamples = 1,
                numPrimaryInfiniteLightSamples = 1,
                numPrimaryLocalLightSamples = 8,
            };
        }

        public static ReSTIRDI_TemporalResamplingParameters GetDefaultTemporalResamplingParams()
        {
            return new ReSTIRDI_TemporalResamplingParameters
            {
                boilingFilterStrength = 0.2f,
                discardInvisibleSamples = 0,
                enableBoilingFilter = 1,
                enablePermutationSampling = 1,
                maxHistoryLength = 20,
                permutationSamplingThreshold = 0.9f,
                temporalBiasCorrection = ReSTIRDI_TemporalBiasCorrectionMode.Basic,
                temporalDepthThreshold = 0.1f,
                temporalNormalThreshold = 0.5f,
            };
        }

        public static ReSTIRDI_SpatialResamplingParameters GetDefaultSpatialResamplingParams()
        {
            return new ReSTIRDI_SpatialResamplingParameters
            {
                numDisocclusionBoostSamples = 8,
                numSpatialSamples = 1,
                spatialBiasCorrection = ReSTIRDI_SpatialBiasCorrectionMode.Basic,
                spatialDepthThreshold = 0.1f,
                spatialNormalThreshold = 0.5f,
                spatialSamplingRadius = 32.0f,
            };
        }

        public static ReSTIRDI_ShadingParameters GetDefaultShadingParams()
        {
            return new ReSTIRDI_ShadingParameters
            {
                enableDenoiserInputPacking = 0,
                enableFinalVisibility = 1,
                finalVisibilityMaxAge = 4,
                finalVisibilityMaxDistance = 16f,
                reuseFinalVisibility = 1,
            };
        }
    }

    public enum ReSTIRDI_ResamplingMode : uint
    {
        None = 0,
        Temporal,
        Spatial,
        TemporalAndSpatial,
        FusedSpatiotemporal,
    }

    public struct RISBufferSegmentParameters
    {
        public uint tileSize;
        public uint tileCount;
    }

    public struct ReSTIRDIStaticParameters
    {
        public uint NeighborOffsetCount;
        public uint RenderWidth;
        public uint RenderHeight;
        public CheckerboardMode CheckerboardSamplingMode;

        public static ReSTIRDIStaticParameters Default()
        {
            return new ReSTIRDIStaticParameters
            {
                NeighborOffsetCount = 8192,
                RenderWidth = 0,
                RenderHeight = 0,
                CheckerboardSamplingMode = CheckerboardMode.Off,
            };
        }
    }

    public class ReSTIRDIContext
    {
        public const uint NumReservoirBuffers = 3;

        private uint m_lastFrameOutputReservoir;
        private uint m_currentFrameOutputReservoir;
        private uint m_frameIndex;

        private ReSTIRDIStaticParameters m_staticParams;
        private ReSTIRDI_ResamplingMode m_resamplingMode;
        private RTXDI_ReservoirBufferParameters m_reservoirBufferParams;
        private RTXDI_RuntimeParameters m_runtimeParams;
        private ReSTIRDI_BufferIndices m_bufferIndices;
        private ReSTIRDI_InitialSamplingParameters m_initialSamplingParams;
        private ReSTIRDI_TemporalResamplingParameters m_temporalResamplingParams;
        private ReSTIRDI_SpatialResamplingParameters m_spatialResamplingParams;
        private ReSTIRDI_ShadingParameters m_shadingParams;

        public ReSTIRDIContext(ReSTIRDIStaticParameters staticParams)
        {
            Debug.Assert(staticParams.RenderWidth > 0);
            Debug.Assert(staticParams.RenderHeight > 0);

            m_lastFrameOutputReservoir = 0;
            m_currentFrameOutputReservoir = 0;
            m_frameIndex = 0;
            m_staticParams = staticParams;
            m_resamplingMode = ReSTIRDI_ResamplingMode.TemporalAndSpatial;
            m_reservoirBufferParams = RtxdiUtils.CalculateReservoirBufferParameters(
                staticParams.RenderWidth, staticParams.RenderHeight, staticParams.CheckerboardSamplingMode);
            m_bufferIndices = ReSTIRDIDefaults.GetDefaultBufferIndices();
            m_initialSamplingParams = ReSTIRDIDefaults.GetDefaultInitialSamplingParams();
            m_temporalResamplingParams = ReSTIRDIDefaults.GetDefaultTemporalResamplingParams();
            m_spatialResamplingParams = ReSTIRDIDefaults.GetDefaultSpatialResamplingParams();
            m_shadingParams = ReSTIRDIDefaults.GetDefaultShadingParams();

            m_runtimeParams = new RTXDI_RuntimeParameters();
            UpdateCheckerboardField();
            m_runtimeParams.neighborOffsetMask = m_staticParams.NeighborOffsetCount - 1;
            UpdateBufferIndices();
        }

        // Getters
        public ReSTIRDI_ResamplingMode GetResamplingMode()               => m_resamplingMode;
        public RTXDI_RuntimeParameters GetRuntimeParams()                => m_runtimeParams;
        public RTXDI_ReservoirBufferParameters GetReservoirBufferParameters() => m_reservoirBufferParams;
        public ReSTIRDI_BufferIndices GetBufferIndices()                 => m_bufferIndices;
        public ReSTIRDI_InitialSamplingParameters GetInitialSamplingParameters() => m_initialSamplingParams;
        public ReSTIRDI_TemporalResamplingParameters GetTemporalResamplingParameters() => m_temporalResamplingParams;
        public ReSTIRDI_SpatialResamplingParameters GetSpatialResamplingParameters() => m_spatialResamplingParams;
        public ReSTIRDI_ShadingParameters GetShadingParameters()         => m_shadingParams;
        public uint GetFrameIndex()                                       => m_frameIndex;
        public ref readonly ReSTIRDIStaticParameters GetStaticParameters() => ref m_staticParams;

        // Setters
        public void SetFrameIndex(uint frameIndex)
        {
            m_frameIndex = frameIndex;
            m_temporalResamplingParams.uniformRandomNumber = RtxdiUtils.JenkinsHash(m_frameIndex);
            m_lastFrameOutputReservoir = m_currentFrameOutputReservoir;
            UpdateBufferIndices();
            UpdateCheckerboardField();
        }

        public void SetResamplingMode(ReSTIRDI_ResamplingMode resamplingMode)
        {
            m_resamplingMode = resamplingMode;
            UpdateBufferIndices();
        }

        public void SetInitialSamplingParameters(ReSTIRDI_InitialSamplingParameters initialSamplingParams)
        {
            m_initialSamplingParams = initialSamplingParams;
        }

        public void SetTemporalResamplingParameters(ReSTIRDI_TemporalResamplingParameters temporalResamplingParams)
        {
            m_temporalResamplingParams = temporalResamplingParams;
            m_temporalResamplingParams.uniformRandomNumber = RtxdiUtils.JenkinsHash(m_frameIndex);
        }

        public void SetSpatialResamplingParameters(ReSTIRDI_SpatialResamplingParameters spatialResamplingParams)
        {
            var srp = spatialResamplingParams;
            srp.neighborOffsetMask = m_spatialResamplingParams.neighborOffsetMask;
            m_spatialResamplingParams = srp;
        }

        public void SetShadingParameters(ReSTIRDI_ShadingParameters shadingParams)
        {
            m_shadingParams = shadingParams;
        }

        // Private helpers
        private void UpdateBufferIndices()
        {
            bool useTemporalResampling =
                m_resamplingMode == ReSTIRDI_ResamplingMode.Temporal ||
                m_resamplingMode == ReSTIRDI_ResamplingMode.TemporalAndSpatial ||
                m_resamplingMode == ReSTIRDI_ResamplingMode.FusedSpatiotemporal;

            bool useSpatialResampling =
                m_resamplingMode == ReSTIRDI_ResamplingMode.Spatial ||
                m_resamplingMode == ReSTIRDI_ResamplingMode.TemporalAndSpatial ||
                m_resamplingMode == ReSTIRDI_ResamplingMode.FusedSpatiotemporal;

            if (m_resamplingMode == ReSTIRDI_ResamplingMode.FusedSpatiotemporal)
            {
                m_bufferIndices.initialSamplingOutputBufferIndex = (m_lastFrameOutputReservoir + 1) % NumReservoirBuffers;
                m_bufferIndices.temporalResamplingInputBufferIndex = m_lastFrameOutputReservoir;
                m_bufferIndices.shadingInputBufferIndex = m_bufferIndices.initialSamplingOutputBufferIndex;
            }
            else
            {
                m_bufferIndices.initialSamplingOutputBufferIndex = (m_lastFrameOutputReservoir + 1) % NumReservoirBuffers;
                m_bufferIndices.temporalResamplingInputBufferIndex = m_lastFrameOutputReservoir;
                m_bufferIndices.temporalResamplingOutputBufferIndex = (m_bufferIndices.temporalResamplingInputBufferIndex + 1) % NumReservoirBuffers;
                m_bufferIndices.spatialResamplingInputBufferIndex = useTemporalResampling
                    ? m_bufferIndices.temporalResamplingOutputBufferIndex
                    : m_bufferIndices.initialSamplingOutputBufferIndex;
                m_bufferIndices.spatialResamplingOutputBufferIndex = (m_bufferIndices.spatialResamplingInputBufferIndex + 1) % NumReservoirBuffers;
                m_bufferIndices.shadingInputBufferIndex = useSpatialResampling
                    ? m_bufferIndices.spatialResamplingOutputBufferIndex
                    : m_bufferIndices.temporalResamplingOutputBufferIndex;
            }
            m_currentFrameOutputReservoir = m_bufferIndices.shadingInputBufferIndex;
        }

        private void UpdateCheckerboardField()
        {
            switch (m_staticParams.CheckerboardSamplingMode)
            {
                case CheckerboardMode.Black:
                    m_runtimeParams.activeCheckerboardField = ((m_frameIndex & 1u) != 0) ? 1u : 2u;
                    break;
                case CheckerboardMode.White:
                    m_runtimeParams.activeCheckerboardField = ((m_frameIndex & 1u) != 0) ? 2u : 1u;
                    break;
                case CheckerboardMode.Off:
                default:
                    m_runtimeParams.activeCheckerboardField = 0;
                    break;
            }
        }
    }
}
