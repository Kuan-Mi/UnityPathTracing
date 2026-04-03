// Copyright (c) 2020-2026, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

namespace Rtxdi.GI
{
    public static class ReSTIRGIDefaults
    {
        public const uint NumReservoirBuffers = 2;

        public static RTXDI_GIBufferIndices GetDefaultBufferIndices()
        {
            return new RTXDI_GIBufferIndices
            {
                secondarySurfaceReSTIRDIOutputBufferIndex = 0,
                temporalResamplingInputBufferIndex         = 0,
                temporalResamplingOutputBufferIndex        = 0,
                spatialResamplingInputBufferIndex          = 0,
                spatialResamplingOutputBufferIndex         = 0,
                finalShadingInputBufferIndex               = 0,
            };
        }

        public static RTXDI_GITemporalResamplingParameters GetDefaultTemporalResamplingParams()
        {
            return new RTXDI_GITemporalResamplingParameters
            {
                depthThreshold           = 0.1f,
                normalThreshold          = 0.6f,
                maxHistoryLength         = 8,
                enableFallbackSampling   = 1,
                biasCorrectionMode       = RTXDI_GIBiasCorrectionMode.Basic,
                maxReservoirAge          = 30,
                enablePermutationSampling = 0,
                uniformRandomNumber      = 0,
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

        public static RTXDI_GISpatialResamplingParameters GetDefaultSpatialResamplingParams()
        {
            return new RTXDI_GISpatialResamplingParameters
            {
                depthThreshold    = 0.1f,
                normalThreshold   = 0.6f,
                numSamples        = 2,
                samplingRadius    = 32.0f,
                biasCorrectionMode = RTXDI_GIBiasCorrectionMode.Basic,
            };
        }

        public static RTXDI_GISpatioTemporalResamplingParameters GetDefaultSpatioTemporalResamplingParams()
        {
            return new RTXDI_GISpatioTemporalResamplingParameters
            {
                depthThreshold           = 0.1f,
                normalThreshold          = 0.6f,
                biasCorrectionMode       = RTXDI_GIBiasCorrectionMode.Basic,
                numSamples               = 2,
                samplingRadius           = 32.0f,
                maxHistoryLength         = 8,
                enableFallbackSampling   = 1,
                maxReservoirAge          = 30,
                enablePermutationSampling = 0,
                uniformRandomNumber      = 0,
            };
        }

        public static RTXDI_GIFinalShadingParameters GetDefaultFinalShadingParams()
        {
            return new RTXDI_GIFinalShadingParameters
            {
                enableFinalVisibility = 1,
                enableFinalMIS        = 1,
            };
        }
    }

    // -------------------------------------------------------------------------

    public enum ReSTIRGI_ResamplingMode : uint
    {
        None                = 0,
        Temporal            = 1,
        Spatial             = 2,
        TemporalAndSpatial  = 3,
        FusedSpatiotemporal = 4,
    }

    public struct ReSTIRGIStaticParameters
    {
        public uint             RenderWidth;
        public uint             RenderHeight;
        public CheckerboardMode CheckerboardSamplingMode;

        public static ReSTIRGIStaticParameters Default()
        {
            return new ReSTIRGIStaticParameters
            {
                RenderWidth              = 0,
                RenderHeight             = 0,
                CheckerboardSamplingMode = CheckerboardMode.Off,
            };
        }
    }

    // -------------------------------------------------------------------------

    public class ReSTIRGIContext
    {
        public const uint NumReservoirBuffers = 2;

        private ReSTIRGIStaticParameters                  m_staticParams;
        private uint                                       m_frameIndex;
        private RTXDI_ReservoirBufferParameters            m_reservoirBufferParams;
        private ReSTIRGI_ResamplingMode                    m_resamplingMode;
        private RTXDI_GIBufferIndices                      m_bufferIndices;
        private RTXDI_GITemporalResamplingParameters       m_temporalResamplingParams;
        private RTXDI_BoilingFilterParameters              m_boilingFilterParams;
        private RTXDI_GISpatialResamplingParameters        m_spatialResamplingParams;
        private RTXDI_GISpatioTemporalResamplingParameters m_spatioTemporalResamplingParams;
        private RTXDI_GIFinalShadingParameters             m_finalShadingParams;

        public ReSTIRGIContext(ReSTIRGIStaticParameters staticParams)
        {
            m_staticParams          = staticParams;
            m_frameIndex            = 0;
            m_reservoirBufferParams = RtxdiUtils.CalculateReservoirBufferParameters(
                staticParams.RenderWidth, staticParams.RenderHeight, staticParams.CheckerboardSamplingMode);
            m_resamplingMode               = ReSTIRGI_ResamplingMode.None;
            m_bufferIndices                = ReSTIRGIDefaults.GetDefaultBufferIndices();
            m_temporalResamplingParams     = ReSTIRGIDefaults.GetDefaultTemporalResamplingParams();
            m_boilingFilterParams          = ReSTIRGIDefaults.GetDefaultBoilingFilterParams();
            m_spatialResamplingParams      = ReSTIRGIDefaults.GetDefaultSpatialResamplingParams();
            m_spatioTemporalResamplingParams = ReSTIRGIDefaults.GetDefaultSpatioTemporalResamplingParams();
            m_finalShadingParams           = ReSTIRGIDefaults.GetDefaultFinalShadingParams();
        }

        // --- Getters ---

        public ReSTIRGIStaticParameters                  GetStaticParams()                         => m_staticParams;
        public uint                                       GetFrameIndex()                           => m_frameIndex;
        public RTXDI_ReservoirBufferParameters            GetReservoirBufferParameters()            => m_reservoirBufferParams;
        public ReSTIRGI_ResamplingMode                    GetResamplingMode()                       => m_resamplingMode;
        public RTXDI_GIBufferIndices                      GetBufferIndices()                        => m_bufferIndices;
        public RTXDI_GITemporalResamplingParameters       GetTemporalResamplingParameters()         => m_temporalResamplingParams;
        public RTXDI_BoilingFilterParameters              GetBoilingFilterParameters()              => m_boilingFilterParams;
        public RTXDI_GISpatialResamplingParameters        GetSpatialResamplingParameters()          => m_spatialResamplingParams;
        public RTXDI_GISpatioTemporalResamplingParameters GetSpatioTemporalResamplingParameters()   => m_spatioTemporalResamplingParams;
        public RTXDI_GIFinalShadingParameters             GetFinalShadingParameters()               => m_finalShadingParams;

        // --- Setters ---

        public void SetFrameIndex(uint frameIndex)
        {
            m_frameIndex = frameIndex;
            m_temporalResamplingParams.uniformRandomNumber        = RtxdiUtils.JenkinsHash(frameIndex);
            m_spatioTemporalResamplingParams.uniformRandomNumber  = RtxdiUtils.JenkinsHash(frameIndex);
            UpdateBufferIndices();
        }

        public void SetResamplingMode(ReSTIRGI_ResamplingMode resamplingMode)
        {
            m_resamplingMode = resamplingMode;
            UpdateBufferIndices();
        }

        public void SetTemporalResamplingParameters(RTXDI_GITemporalResamplingParameters temporalResamplingParams)
        {
            m_temporalResamplingParams = temporalResamplingParams;
            m_temporalResamplingParams.uniformRandomNumber = RtxdiUtils.JenkinsHash(m_frameIndex);
        }

        public void SetBoilingFilterParameters(RTXDI_BoilingFilterParameters boilingFilterParams)
        {
            m_boilingFilterParams = boilingFilterParams;
        }

        public void SetSpatialResamplingParameters(RTXDI_GISpatialResamplingParameters spatialResamplingParams)
        {
            m_spatialResamplingParams = spatialResamplingParams;
        }

        public void SetSpatioTemporalResamplingParameters(RTXDI_GISpatioTemporalResamplingParameters spatioTemporalParams)
        {
            m_spatioTemporalResamplingParams = spatioTemporalParams;
        }

        public void SetFinalShadingParameters(RTXDI_GIFinalShadingParameters finalShadingParams)
        {
            m_finalShadingParams = finalShadingParams;
        }

        // --- Private helpers ---

        private void UpdateBufferIndices()
        {
            switch (m_resamplingMode)
            {
                case ReSTIRGI_ResamplingMode.None:
                    m_bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex = 0;
                    m_bufferIndices.finalShadingInputBufferIndex = 0;
                    break;

                case ReSTIRGI_ResamplingMode.Temporal:
                    m_bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex = m_frameIndex & 1;
                    m_bufferIndices.temporalResamplingInputBufferIndex =
                        (m_bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex != 0) ? 0u : 1u;
                    m_bufferIndices.temporalResamplingOutputBufferIndex =
                        m_bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex;
                    m_bufferIndices.finalShadingInputBufferIndex =
                        m_bufferIndices.temporalResamplingOutputBufferIndex;
                    break;

                case ReSTIRGI_ResamplingMode.Spatial:
                    m_bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex = 0;
                    m_bufferIndices.spatialResamplingInputBufferIndex  = 0;
                    m_bufferIndices.spatialResamplingOutputBufferIndex = 1;
                    m_bufferIndices.finalShadingInputBufferIndex       = 1;
                    break;

                case ReSTIRGI_ResamplingMode.TemporalAndSpatial:
                    m_bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex = 0;
                    m_bufferIndices.temporalResamplingInputBufferIndex   = 1;
                    m_bufferIndices.temporalResamplingOutputBufferIndex  = 0;
                    m_bufferIndices.spatialResamplingInputBufferIndex    = 0;
                    m_bufferIndices.spatialResamplingOutputBufferIndex   = 1;
                    m_bufferIndices.finalShadingInputBufferIndex         = 1;
                    break;

                case ReSTIRGI_ResamplingMode.FusedSpatiotemporal:
                    m_bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex = m_frameIndex & 1;
                    m_bufferIndices.temporalResamplingInputBufferIndex =
                        (m_bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex != 0) ? 0u : 1u;
                    m_bufferIndices.spatialResamplingOutputBufferIndex =
                        m_bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex;
                    m_bufferIndices.finalShadingInputBufferIndex =
                        m_bufferIndices.spatialResamplingOutputBufferIndex;
                    break;
            }
        }
    }
}
