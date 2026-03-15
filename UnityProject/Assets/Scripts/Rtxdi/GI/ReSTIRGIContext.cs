// Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

using Rtxdi.GI;

namespace Rtxdi.GI
{
    public static class ReSTIRGIDefaults
    {
        public const uint NumReservoirBuffers = 2;

        public static ReSTIRGI_BufferIndices GetDefaultBufferIndices()
        {
            return new ReSTIRGI_BufferIndices
            {
                secondarySurfaceReSTIRDIOutputBufferIndex = 0,
                temporalResamplingInputBufferIndex = 0,
                temporalResamplingOutputBufferIndex = 0,
                spatialResamplingInputBufferIndex = 0,
                spatialResamplingOutputBufferIndex = 0,
                finalShadingInputBufferIndex = 0,
            };
        }

        public static ReSTIRGI_TemporalResamplingParameters GetDefaultTemporalResamplingParams()
        {
            return new ReSTIRGI_TemporalResamplingParameters
            {
                boilingFilterStrength = 0.2f,
                depthThreshold = 0.1f,
                enableBoilingFilter = 1,
                enableFallbackSampling = 1,
                enablePermutationSampling = 0,
                maxHistoryLength = 8,
                maxReservoirAge = 30,
                normalThreshold = 0.6f,
                temporalBiasCorrectionMode = ResTIRGI_TemporalBiasCorrectionMode.Basic,
            };
        }

        public static ReSTIRGI_SpatialResamplingParameters GetDefaultSpatialResamplingParams()
        {
            return new ReSTIRGI_SpatialResamplingParameters
            {
                numSpatialSamples = 2,
                spatialBiasCorrectionMode = ResTIRGI_SpatialBiasCorrectionMode.Basic,
                spatialDepthThreshold = 0.1f,
                spatialNormalThreshold = 0.6f,
                spatialSamplingRadius = 32.0f,
            };
        }

        public static ReSTIRGI_FinalShadingParameters GetDefaultFinalShadingParams()
        {
            return new ReSTIRGI_FinalShadingParameters
            {
                enableFinalMIS = 1,
                enableFinalVisibility = 1,
            };
        }
    }

    public enum ReSTIRGI_ResamplingMode : uint
    {
        None = 0,
        Temporal = 1,
        Spatial = 2,
        TemporalAndSpatial = 3,
        FusedSpatiotemporal = 4,
    }

    public struct ReSTIRGIStaticParameters
    {
        public uint RenderWidth;
        public uint RenderHeight;
        public CheckerboardMode CheckerboardSamplingMode;

        public static ReSTIRGIStaticParameters Default()
        {
            return new ReSTIRGIStaticParameters
            {
                RenderWidth = 0,
                RenderHeight = 0,
                CheckerboardSamplingMode = CheckerboardMode.Off,
            };
        }
    }

    public class ReSTIRGIContext
    {
        public static uint numReservoirBuffers = ReSTIRGIDefaults.NumReservoirBuffers;

        private ReSTIRGIStaticParameters m_staticParams;
        private uint m_frameIndex;
        private RTXDI_ReservoirBufferParameters m_reservoirBufferParams;
        private ReSTIRGI_ResamplingMode m_resamplingMode;
        private ReSTIRGI_BufferIndices m_bufferIndices;
        private ReSTIRGI_TemporalResamplingParameters m_temporalResamplingParams;
        private ReSTIRGI_SpatialResamplingParameters m_spatialResamplingParams;
        private ReSTIRGI_FinalShadingParameters m_finalShadingParams;

        public ReSTIRGIContext(ReSTIRGIStaticParameters staticParams)
        {
            m_staticParams = staticParams;
            m_frameIndex = 0;
            m_reservoirBufferParams = RtxdiUtils.CalculateReservoirBufferParameters(
                staticParams.RenderWidth, staticParams.RenderHeight, staticParams.CheckerboardSamplingMode);
            m_resamplingMode = ReSTIRGI_ResamplingMode.None;
            m_bufferIndices = ReSTIRGIDefaults.GetDefaultBufferIndices();
            m_temporalResamplingParams = ReSTIRGIDefaults.GetDefaultTemporalResamplingParams();
            m_spatialResamplingParams = ReSTIRGIDefaults.GetDefaultSpatialResamplingParams();
            m_finalShadingParams = ReSTIRGIDefaults.GetDefaultFinalShadingParams();
        }

        // Getters
        public ReSTIRGIStaticParameters GetStaticParams()                         => m_staticParams;
        public uint GetFrameIndex()                                                => m_frameIndex;
        public RTXDI_ReservoirBufferParameters GetReservoirBufferParameters()     => m_reservoirBufferParams;
        public ReSTIRGI_ResamplingMode GetResamplingMode()                        => m_resamplingMode;
        public ReSTIRGI_BufferIndices GetBufferIndices()                          => m_bufferIndices;
        public ReSTIRGI_TemporalResamplingParameters GetTemporalResamplingParameters() => m_temporalResamplingParams;
        public ReSTIRGI_SpatialResamplingParameters GetSpatialResamplingParameters()   => m_spatialResamplingParams;
        public ReSTIRGI_FinalShadingParameters GetFinalShadingParameters()        => m_finalShadingParams;

        // Setters
        public void SetFrameIndex(uint frameIndex)
        {
            m_frameIndex = frameIndex;
            m_temporalResamplingParams.uniformRandomNumber = RtxdiUtils.JenkinsHash(m_frameIndex);
            UpdateBufferIndices();
        }

        public void SetResamplingMode(ReSTIRGI_ResamplingMode resamplingMode)
        {
            m_resamplingMode = resamplingMode;
            UpdateBufferIndices();
        }

        public void SetTemporalResamplingParameters(ReSTIRGI_TemporalResamplingParameters temporalResamplingParams)
        {
            m_temporalResamplingParams = temporalResamplingParams;
            m_temporalResamplingParams.uniformRandomNumber = RtxdiUtils.JenkinsHash(m_frameIndex);
        }

        public void SetSpatialResamplingParameters(ReSTIRGI_SpatialResamplingParameters spatialResamplingParams)
        {
            m_spatialResamplingParams = spatialResamplingParams;
        }

        public void SetFinalShadingParameters(ReSTIRGI_FinalShadingParameters finalShadingParams)
        {
            m_finalShadingParams = finalShadingParams;
        }

        // Private
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
                    m_bufferIndices.spatialResamplingInputBufferIndex = 0;
                    m_bufferIndices.spatialResamplingOutputBufferIndex = 1;
                    m_bufferIndices.finalShadingInputBufferIndex = 1;
                    break;

                case ReSTIRGI_ResamplingMode.TemporalAndSpatial:
                    m_bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex = 0;
                    m_bufferIndices.temporalResamplingInputBufferIndex = 1;
                    m_bufferIndices.temporalResamplingOutputBufferIndex = 0;
                    m_bufferIndices.spatialResamplingInputBufferIndex = 0;
                    m_bufferIndices.spatialResamplingOutputBufferIndex = 1;
                    m_bufferIndices.finalShadingInputBufferIndex = 1;
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
