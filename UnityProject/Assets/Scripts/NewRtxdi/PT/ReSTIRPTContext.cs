// Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

namespace Rtxdi.PT
{
    public static class ReSTIRPTDefaults
    {
        public const uint NumReservoirBuffers = 2;

        public static RTXDI_PTBufferIndices GetDefaultBufferIndices()
        {
            return new RTXDI_PTBufferIndices
            {
                initialPathTracerOutputBufferIndex  = 0,
                temporalResamplingInputBufferIndex   = 0,
                temporalResamplingOutputBufferIndex  = 0,
                spatialResamplingInputBufferIndex    = 0,
                spatialResamplingOutputBufferIndex   = 0,
                finalShadingInputBufferIndex         = 0,
            };
        }

        public static RTXDI_PTInitialSamplingParameters GetDefaultInitialSamplingParams()
        {
            return new RTXDI_PTInitialSamplingParameters
            {
                numInitialSamples  = 1,
                maxBounceDepth     = 3,
                maxRcVertexLength  = 5,
            };
        }

        public static RTXDI_PTReconnectionParameters GetDefaultReconnectionParameters()
        {
            return new RTXDI_PTReconnectionParameters
            {
                minConnectionFootprint      = 0.02f,
                minConnectionFootprintSigma = 0.2f,
                minPdfRoughness             = 0.1f,
                minPdfRoughnessSigma        = 0.01f,
                roughnessThreshold          = 0.1f,
                distanceThreshold           = 0.0f,
                reconnectionMode            = RTXDI_PTReconnectionMode.Footprint,
            };
        }

        public static RTXDI_PTHybridShiftPerFrameParameters GetDefaultHybridShiftParams()
        {
            var ip = GetDefaultInitialSamplingParams();
            return new RTXDI_PTHybridShiftPerFrameParameters
            {
                maxBounceDepth    = ip.maxBounceDepth,
                maxRcVertexLength = ip.maxRcVertexLength,
            };
        }

        public static RTXDI_PTTemporalResamplingParameters GetDefaultTemporalResamplingParams()
        {
            return new RTXDI_PTTemporalResamplingParameters
            {
                depthThreshold                    = 0.1f,
                normalThreshold                   = 0.6f,
                enablePermutationSampling         = 0,
                maxHistoryLength                  = 8,
                maxReservoirAge                   = 30,
                enableFallbackSampling            = 1,
                enableVisibilityBeforeCombine     = 0,
                uniformRandomNumber               = 0,
                duplicationBasedHistoryReduction  = 0,
                historyReductionStrength          = 0.8f,
            };
        }

        public static RTXDI_BoilingFilterParameters GetDefaultBoilingFilterParams()
        {
            return new RTXDI_BoilingFilterParameters
            {
                enableBoilingFilter   = 1,
                boilingFilterStrength = 0.2f,
            };
        }

        public static RTXDI_PTSpatialResamplingParameters GetDefaultSpatialResamplingParams()
        {
            return new RTXDI_PTSpatialResamplingParameters
            {
                numSpatialSamples                = 1,
                numDisocclusionBoostSamples      = 8,
                maxTemporalHistory               = 8,
                duplicationBasedHistoryReduction = 0,
                samplingRadius                   = 32.0f,
                normalThreshold                  = 0.6f,
                depthThreshold                   = 0.1f,
            };
        }
    }

    // -------------------------------------------------------------------------

    public enum ReSTIRPT_ResamplingMode : uint
    {
        None               = 0,
        Temporal           = 1,
        Spatial            = 2,
        TemporalAndSpatial = 3,
    }

    public struct ReSTIRPTStaticParameters
    {
        public uint             RenderWidth;
        public uint             RenderHeight;
        public CheckerboardMode CheckerboardSamplingMode;

        public static ReSTIRPTStaticParameters Default()
        {
            return new ReSTIRPTStaticParameters
            {
                RenderWidth              = 0,
                RenderHeight             = 0,
                CheckerboardSamplingMode = CheckerboardMode.Off,
            };
        }
    }

    // -------------------------------------------------------------------------

    public class ReSTIRPTContext
    {
        public const uint NumReservoirBuffers = 2;

        private ReSTIRPTStaticParameters              m_staticParams;
        private uint                                   m_frameIndex;
        private RTXDI_ReservoirBufferParameters        m_reservoirBufferParams;
        private ReSTIRPT_ResamplingMode                m_resamplingMode;
        private RTXDI_PTBufferIndices                  m_bufferIndices;
        private RTXDI_PTInitialSamplingParameters      m_initialSamplingParams;
        private RTXDI_PTHybridShiftPerFrameParameters  m_hybridShiftParams;
        private RTXDI_PTReconnectionParameters         m_reconnectionParams;
        private RTXDI_PTTemporalResamplingParameters   m_temporalResamplingParams;
        private RTXDI_BoilingFilterParameters          m_boilingFilterParameters;
        private RTXDI_PTSpatialResamplingParameters    m_spatialResamplingParams;

        public ReSTIRPTContext(ReSTIRPTStaticParameters staticParams)
        {
            m_staticParams          = staticParams;
            m_frameIndex            = 0;
            m_reservoirBufferParams = RtxdiUtils.CalculateReservoirBufferParameters(
                staticParams.RenderWidth, staticParams.RenderHeight, staticParams.CheckerboardSamplingMode);
            m_resamplingMode            = ReSTIRPT_ResamplingMode.None;
            m_bufferIndices             = ReSTIRPTDefaults.GetDefaultBufferIndices();
            m_initialSamplingParams     = ReSTIRPTDefaults.GetDefaultInitialSamplingParams();
            m_hybridShiftParams         = ReSTIRPTDefaults.GetDefaultHybridShiftParams();
            m_reconnectionParams        = ReSTIRPTDefaults.GetDefaultReconnectionParameters();
            m_temporalResamplingParams  = ReSTIRPTDefaults.GetDefaultTemporalResamplingParams();
            m_boilingFilterParameters   = ReSTIRPTDefaults.GetDefaultBoilingFilterParams();
            m_spatialResamplingParams   = ReSTIRPTDefaults.GetDefaultSpatialResamplingParams();

            UpdateBufferIndices();
        }

        // --- Getters ---

        public ReSTIRPTStaticParameters             GetStaticParams()                      => m_staticParams;
        public uint                                  GetFrameIndex()                        => m_frameIndex;
        public RTXDI_ReservoirBufferParameters       GetReservoirBufferParameters()         => m_reservoirBufferParams;
        public ReSTIRPT_ResamplingMode               GetResamplingMode()                    => m_resamplingMode;
        public RTXDI_PTBufferIndices                 GetBufferIndices()                     => m_bufferIndices;
        public RTXDI_PTInitialSamplingParameters     GetInitialSamplingParameters()         => m_initialSamplingParams;
        public RTXDI_PTHybridShiftPerFrameParameters GetHybridShiftParameters()             => m_hybridShiftParams;
        public RTXDI_PTReconnectionParameters        GetReconnectionParameters()            => m_reconnectionParams;
        public RTXDI_PTTemporalResamplingParameters  GetTemporalResamplingParameters()      => m_temporalResamplingParams;
        public RTXDI_BoilingFilterParameters         GetBoilingFilterParameters()           => m_boilingFilterParameters;
        public RTXDI_PTSpatialResamplingParameters   GetSpatialResamplingParameters()       => m_spatialResamplingParams;

        // --- Setters ---

        public void SetFrameIndex(uint frameIndex)
        {
            m_frameIndex = frameIndex;
            m_temporalResamplingParams.uniformRandomNumber = RtxdiUtils.JenkinsHash(frameIndex);
            UpdateBufferIndices();
        }

        public void SetResamplingMode(ReSTIRPT_ResamplingMode resamplingMode)
        {
            m_resamplingMode = resamplingMode;
            UpdateBufferIndices();
        }

        public void SetInitialSamplingParameters(RTXDI_PTInitialSamplingParameters initialSamplingParams)
        {
            m_initialSamplingParams = initialSamplingParams;
        }

        public void SetHybridShiftParameters(RTXDI_PTHybridShiftPerFrameParameters hybridShiftParams)
        {
            m_hybridShiftParams = hybridShiftParams;
        }

        public void SetReconnectionParameters(RTXDI_PTReconnectionParameters reconnectionParams)
        {
            m_reconnectionParams = reconnectionParams;
        }

        public void SetTemporalResamplingParameters(RTXDI_PTTemporalResamplingParameters temporalResamplingParams)
        {
            m_temporalResamplingParams = temporalResamplingParams;
            m_temporalResamplingParams.uniformRandomNumber = RtxdiUtils.JenkinsHash(m_frameIndex);
        }

        public void SetBoilingFilterParameters(RTXDI_BoilingFilterParameters parameters)
        {
            m_boilingFilterParameters = parameters;
        }

        public void SetSpatialResamplingParameters(RTXDI_PTSpatialResamplingParameters spatialResamplingParams)
        {
            m_spatialResamplingParams = spatialResamplingParams;
        }

        // --- Private helpers ---

        private void UpdateBufferIndices()
        {
            switch (m_resamplingMode)
            {
                case ReSTIRPT_ResamplingMode.None:
                    m_bufferIndices.initialPathTracerOutputBufferIndex = 0;
                    m_bufferIndices.finalShadingInputBufferIndex        = 0;
                    break;

                case ReSTIRPT_ResamplingMode.Temporal:
                    m_bufferIndices.initialPathTracerOutputBufferIndex  = m_frameIndex & 1;
                    m_bufferIndices.temporalResamplingInputBufferIndex   = 1 - m_bufferIndices.initialPathTracerOutputBufferIndex;
                    m_bufferIndices.temporalResamplingOutputBufferIndex  = m_bufferIndices.initialPathTracerOutputBufferIndex;
                    m_bufferIndices.finalShadingInputBufferIndex         = m_bufferIndices.temporalResamplingOutputBufferIndex;
                    break;

                case ReSTIRPT_ResamplingMode.Spatial:
                    m_bufferIndices.initialPathTracerOutputBufferIndex  = 0;
                    m_bufferIndices.spatialResamplingInputBufferIndex    = 0;
                    m_bufferIndices.spatialResamplingOutputBufferIndex   = 1;
                    m_bufferIndices.finalShadingInputBufferIndex         = 1;
                    break;

                case ReSTIRPT_ResamplingMode.TemporalAndSpatial:
                    m_bufferIndices.initialPathTracerOutputBufferIndex  = 0;
                    m_bufferIndices.temporalResamplingInputBufferIndex   = 1;
                    m_bufferIndices.temporalResamplingOutputBufferIndex  = 0;
                    m_bufferIndices.spatialResamplingInputBufferIndex    = 0;
                    m_bufferIndices.spatialResamplingOutputBufferIndex   = 1;
                    m_bufferIndices.finalShadingInputBufferIndex         = 1;
                    break;
            }
        }
    }
}
