// Copyright (c) 2020-2026, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

using UnityEngine;

namespace Rtxdi.DI
{
    public static class ReSTIRDIDefaults
    {
        public const uint NumReservoirBuffers = 3;

        public static RTXDI_DIBufferIndices GetDefaultBufferIndices()
        {
            return new RTXDI_DIBufferIndices
            {
                initialSamplingOutputBufferIndex    = 0,
                temporalResamplingInputBufferIndex   = 0,
                temporalResamplingOutputBufferIndex  = 0,
                spatialResamplingInputBufferIndex    = 0,
                spatialResamplingOutputBufferIndex   = 0,
                shadingInputBufferIndex              = 0,
            };
        }

        public static RTXDI_DIInitialSamplingParameters GetDefaultInitialSamplingParams()
        {
            return new RTXDI_DIInitialSamplingParameters
            {
                numLocalLightSamples          = 8,
                numInfiniteLightSamples        = 1,
                numEnvironmentSamples          = 1,
                numBrdfSamples                 = 1,
                brdfCutoff                     = 0.0001f,
                brdfRayMinT                    = 0.001f,
                localLightSamplingMode         = ReSTIRDI_LocalLightSamplingMode.Uniform,
                enableInitialVisibility        = 1,
                environmentMapImportanceSampling = 1,
            };
        }

        public static RTXDI_DITemporalResamplingParameters GetDefaultTemporalResamplingParams()
        {
            return new RTXDI_DITemporalResamplingParameters
            {
                maxHistoryLength          = 20,
                biasCorrectionMode        = ReSTIRDI_TemporalBiasCorrectionMode.Basic,
                depthThreshold            = 0.1f,
                normalThreshold           = 0.5f,
                enableVisibilityShortcut  = 0,
                enablePermutationSampling = 1,
                uniformRandomNumber       = 0,
                permutationSamplingThreshold = 0.9f,
            };
        }

        public static RTXDI_BoilingFilterParameters GetDefaultBoilingFilterParams()
        {
            return new RTXDI_BoilingFilterParameters
            {
                enableBoilingFilter    = 1,
                boilingFilterStrength  = 0.2f,
            };
        }

        public static RTXDI_DISpatialResamplingParameters GetDefaultSpatialResamplingParams()
        {
            return new RTXDI_DISpatialResamplingParameters
            {
                numSamples                    = 1,
                numDisocclusionBoostSamples   = 8,
                samplingRadius                = 32.0f,
                biasCorrectionMode            = ReSTIRDI_SpatialBiasCorrectionMode.Basic,
                depthThreshold                = 0.1f,
                normalThreshold               = 0.5f,
                targetHistoryLength           = 0,
                enableMaterialSimilarityTest  = 1,
                discountNaiveSamples          = 1,
            };
        }

        public static RTXDI_DISpatioTemporalResamplingParameters GetDefaultSpatioTemporalResamplingParams()
        {
            return new RTXDI_DISpatioTemporalResamplingParameters
            {
                depthThreshold                = 0.1f,
                normalThreshold               = 0.5f,
                biasCorrectionMode            = ReSTIRDI_SpatioTemporalBiasCorrectionMode.Basic,
                maxHistoryLength              = 20,
                enablePermutationSampling     = 1,
                uniformRandomNumber           = 0,
                enableVisibilityShortcut      = 0,
                numSamples                    = 1,
                numDisocclusionBoostSamples   = 8,
                samplingRadius                = 32.0f,
                enableMaterialSimilarityTest  = 1,
                discountNaiveSamples          = 1,
            };
        }

        public static RTXDI_ShadingParameters GetDefaultShadingParams()
        {
            return new RTXDI_ShadingParameters
            {
                enableFinalVisibility      = 1,
                reuseFinalVisibility       = 1,
                finalVisibilityMaxAge      = 4,
                finalVisibilityMaxDistance = 16f,
                enableDenoiserInputPacking = 0,
            };
        }
    }

    // -------------------------------------------------------------------------

    public enum ReSTIRDI_ResamplingMode : uint
    {
        None                 = 0,
        Temporal             = 1,
        Spatial              = 2,
        TemporalAndSpatial   = 3,
        FusedSpatiotemporal  = 4,
    }

    public struct RISBufferSegmentParameters
    {
        public uint tileSize;
        public uint tileCount;
    }

    public struct ReSTIRDIStaticParameters
    {
        public uint             NeighborOffsetCount;
        public uint             RenderWidth;
        public uint             RenderHeight;
        public CheckerboardMode CheckerboardSamplingMode;

        public static ReSTIRDIStaticParameters Default()
        {
            return new ReSTIRDIStaticParameters
            {
                NeighborOffsetCount      = 8192,
                RenderWidth              = 0,
                RenderHeight             = 0,
                CheckerboardSamplingMode = CheckerboardMode.Off,
            };
        }
    }

    // -------------------------------------------------------------------------

    public class ReSTIRDIContext
    {
        public const uint NumReservoirBuffers = 3;

        private uint m_lastFrameOutputReservoir;
        private uint m_currentFrameOutputReservoir;

        private ReSTIRDIStaticParameters                  m_staticParams;
        private ReSTIRDI_ResamplingMode                   m_resamplingMode;
        private RTXDI_ReservoirBufferParameters           m_reservoirBufferParams;
        private RTXDI_RuntimeParameters                   m_runtimeParams;
        private RTXDI_DIBufferIndices                     m_bufferIndices;
        private RTXDI_DIInitialSamplingParameters         m_initialSamplingParams;
        private RTXDI_DITemporalResamplingParameters      m_temporalResamplingParams;
        private RTXDI_BoilingFilterParameters             m_boilingFilterParams;
        private RTXDI_DISpatialResamplingParameters       m_spatialResamplingParams;
        private RTXDI_DISpatioTemporalResamplingParameters m_spatioTemporalResamplingParams;
        private RTXDI_ShadingParameters                   m_shadingParams;

        public ReSTIRDIContext(ReSTIRDIStaticParameters staticParams)
        {
            Debug.Assert(staticParams.RenderWidth  > 0);
            Debug.Assert(staticParams.RenderHeight > 0);

            m_lastFrameOutputReservoir    = 0;
            m_currentFrameOutputReservoir = 0;
            m_staticParams      = staticParams;
            m_resamplingMode    = ReSTIRDI_ResamplingMode.TemporalAndSpatial;
            m_reservoirBufferParams = RtxdiUtils.CalculateReservoirBufferParameters(
                staticParams.RenderWidth, staticParams.RenderHeight, staticParams.CheckerboardSamplingMode);

            m_runtimeParams     = new RTXDI_RuntimeParameters();
            m_bufferIndices     = ReSTIRDIDefaults.GetDefaultBufferIndices();
            m_initialSamplingParams        = ReSTIRDIDefaults.GetDefaultInitialSamplingParams();
            m_temporalResamplingParams     = ReSTIRDIDefaults.GetDefaultTemporalResamplingParams();
            m_boilingFilterParams          = ReSTIRDIDefaults.GetDefaultBoilingFilterParams();
            m_spatialResamplingParams      = ReSTIRDIDefaults.GetDefaultSpatialResamplingParams();
            m_spatioTemporalResamplingParams = ReSTIRDIDefaults.GetDefaultSpatioTemporalResamplingParams();
            m_shadingParams                = ReSTIRDIDefaults.GetDefaultShadingParams();

            UpdateCheckerboardField();
            m_runtimeParams.neighborOffsetMask = m_staticParams.NeighborOffsetCount - 1;
            UpdateBufferIndices();
        }

        // --- Getters ---

        public ReSTIRDI_ResamplingMode           GetResamplingMode()                      => m_resamplingMode;
        public RTXDI_RuntimeParameters           GetRuntimeParams()                       => m_runtimeParams;
        public RTXDI_ReservoirBufferParameters   GetReservoirBufferParameters()           => m_reservoirBufferParams;
        public RTXDI_DIBufferIndices             GetBufferIndices()                       => m_bufferIndices;
        public RTXDI_DIInitialSamplingParameters GetInitialSamplingParameters()           => m_initialSamplingParams;
        public RTXDI_DITemporalResamplingParameters GetTemporalResamplingParameters()     => m_temporalResamplingParams;
        public RTXDI_BoilingFilterParameters     GetBoilingFilterParameters()             => m_boilingFilterParams;
        public RTXDI_DISpatialResamplingParameters GetSpatialResamplingParameters()       => m_spatialResamplingParams;
        public RTXDI_DISpatioTemporalResamplingParameters GetSpatioTemporalResamplingParameters() => m_spatioTemporalResamplingParams;
        public RTXDI_ShadingParameters           GetShadingParameters()                  => m_shadingParams;
        public uint                              GetFrameIndex()                          => m_runtimeParams.frameIndex;
        public ref readonly ReSTIRDIStaticParameters GetStaticParameters()               => ref m_staticParams;

        // --- Setters ---

        public void SetFrameIndex(uint frameIndex)
        {
            m_runtimeParams.frameIndex = frameIndex;
            m_temporalResamplingParams.uniformRandomNumber        = RtxdiUtils.JenkinsHash(frameIndex);
            m_spatioTemporalResamplingParams.uniformRandomNumber  = RtxdiUtils.JenkinsHash(frameIndex);
            m_lastFrameOutputReservoir = m_currentFrameOutputReservoir;
            UpdateBufferIndices();
            UpdateCheckerboardField();
        }

        public void SetResamplingMode(ReSTIRDI_ResamplingMode resamplingMode)
        {
            m_resamplingMode = resamplingMode;
            UpdateBufferIndices();
        }

        public void SetInitialSamplingParameters(RTXDI_DIInitialSamplingParameters initialSamplingParams)
        {
            m_initialSamplingParams = initialSamplingParams;
        }

        public void SetTemporalResamplingParameters(RTXDI_DITemporalResamplingParameters temporalResamplingParams)
        {
            m_temporalResamplingParams = temporalResamplingParams;
            m_temporalResamplingParams.uniformRandomNumber = RtxdiUtils.JenkinsHash(m_runtimeParams.frameIndex);
        }

        public void SetBoilingFilterParameters(RTXDI_BoilingFilterParameters boilingFilterParams)
        {
            m_boilingFilterParams = boilingFilterParams;
        }

        public void SetSpatialResamplingParameters(RTXDI_DISpatialResamplingParameters spatialResamplingParams)
        {
            m_spatialResamplingParams = spatialResamplingParams;
        }

        public void SetSpatioTemporalResamplingParameters(RTXDI_DISpatioTemporalResamplingParameters spatioTemporalResamplingParams)
        {
            m_spatioTemporalResamplingParams = spatioTemporalResamplingParams;
            m_spatioTemporalResamplingParams.uniformRandomNumber = RtxdiUtils.JenkinsHash(m_runtimeParams.frameIndex);
        }

        public void SetShadingParameters(RTXDI_ShadingParameters shadingParams)
        {
            m_shadingParams = shadingParams;
        }

        // --- Private helpers ---

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
                m_bufferIndices.initialSamplingOutputBufferIndex   = (m_lastFrameOutputReservoir + 1) % NumReservoirBuffers;
                m_bufferIndices.temporalResamplingInputBufferIndex  = m_lastFrameOutputReservoir;
                m_bufferIndices.shadingInputBufferIndex             = m_bufferIndices.initialSamplingOutputBufferIndex;
            }
            else
            {
                m_bufferIndices.initialSamplingOutputBufferIndex    = (m_lastFrameOutputReservoir + 1) % NumReservoirBuffers;
                m_bufferIndices.temporalResamplingInputBufferIndex   = m_lastFrameOutputReservoir;
                m_bufferIndices.temporalResamplingOutputBufferIndex  = (m_bufferIndices.temporalResamplingInputBufferIndex + 1) % NumReservoirBuffers;
                m_bufferIndices.spatialResamplingInputBufferIndex    = useTemporalResampling
                    ? m_bufferIndices.temporalResamplingOutputBufferIndex
                    : m_bufferIndices.initialSamplingOutputBufferIndex;
                m_bufferIndices.spatialResamplingOutputBufferIndex   = (m_bufferIndices.spatialResamplingInputBufferIndex + 1) % NumReservoirBuffers;
                m_bufferIndices.shadingInputBufferIndex              = useSpatialResampling
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
                    m_runtimeParams.activeCheckerboardField = ((m_runtimeParams.frameIndex & 1u) != 0) ? 1u : 2u;
                    break;
                case CheckerboardMode.White:
                    m_runtimeParams.activeCheckerboardField = ((m_runtimeParams.frameIndex & 1u) != 0) ? 2u : 1u;
                    break;
                case CheckerboardMode.Off:
                default:
                    m_runtimeParams.activeCheckerboardField = 0;
                    break;
            }
        }
    }
}
