/***************************************************************************
 # Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

#include "Rtxdi/GI/ReSTIRGI.h"

namespace rtxdi
{

ReSTIRGI_BufferIndices GetDefaultReSTIRGIBufferIndices()
{
    ReSTIRGI_BufferIndices bufferIndices = {};
    bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex = 0;
    bufferIndices.temporalResamplingInputBufferIndex = 0;
    bufferIndices.temporalResamplingOutputBufferIndex = 0;
    bufferIndices.spatialResamplingInputBufferIndex = 0;
    bufferIndices.spatialResamplingOutputBufferIndex = 0;
    return bufferIndices;
}

ReSTIRGI_TemporalResamplingParameters GetDefaultReSTIRGITemporalResamplingParams()
{
    ReSTIRGI_TemporalResamplingParameters params = {};
    params.boilingFilterStrength = 0.2f;
    params.depthThreshold = 0.1f;
    params.enableBoilingFilter = true;
    params.enableFallbackSampling = true;
    params.enablePermutationSampling = false;
    params.maxHistoryLength = 8;
    params.maxReservoirAge = 30;
    params.normalThreshold = 0.6f;
    params.temporalBiasCorrectionMode = ResTIRGI_TemporalBiasCorrectionMode::Basic;
    return params;
}

ReSTIRGI_SpatialResamplingParameters GetDefaultReSTIRGISpatialResamplingParams()
{
    ReSTIRGI_SpatialResamplingParameters params = {};
    params.numSpatialSamples = 2;
    params.spatialBiasCorrectionMode = ResTIRGI_SpatialBiasCorrectionMode::Basic;
    params.spatialDepthThreshold = 0.1f;
    params.spatialNormalThreshold = 0.6f;
    params.spatialSamplingRadius = 32.0f;
    return params;
}

ReSTIRGI_FinalShadingParameters GetDefaultReSTIRGIFinalShadingParams()
{
    ReSTIRGI_FinalShadingParameters params = {};
    params.enableFinalMIS = true;
    params.enableFinalVisibility = true;
    return params;
}

ReSTIRGIContext::ReSTIRGIContext(const ReSTIRGIStaticParameters& staticParams) :
    m_staticParams(staticParams),
    m_frameIndex(0),
    m_reservoirBufferParams(CalculateReservoirBufferParameters(staticParams.RenderWidth, staticParams.RenderHeight, staticParams.CheckerboardSamplingMode)),
    m_resamplingMode(rtxdi::ReSTIRGI_ResamplingMode::None),
    m_bufferIndices(GetDefaultReSTIRGIBufferIndices()),
    m_temporalResamplingParams(GetDefaultReSTIRGITemporalResamplingParams()),
    m_spatialResamplingParams(GetDefaultReSTIRGISpatialResamplingParams()),
    m_finalShadingParams(GetDefaultReSTIRGIFinalShadingParams())
{
}

ReSTIRGIStaticParameters ReSTIRGIContext::GetStaticParams() const
{
    return m_staticParams;
}

uint32_t ReSTIRGIContext::GetFrameIndex() const
{
    return m_frameIndex;
}

RTXDI_ReservoirBufferParameters ReSTIRGIContext::GetReservoirBufferParameters() const
{
    return m_reservoirBufferParams;
}

ReSTIRGI_ResamplingMode ReSTIRGIContext::GetResamplingMode() const
{
    return m_resamplingMode;
}

ReSTIRGI_BufferIndices ReSTIRGIContext::GetBufferIndices() const
{
    return m_bufferIndices;
}

ReSTIRGI_TemporalResamplingParameters ReSTIRGIContext::GetTemporalResamplingParameters() const
{
    return m_temporalResamplingParams;
}

ReSTIRGI_SpatialResamplingParameters ReSTIRGIContext::GetSpatialResamplingParameters() const
{
    return m_spatialResamplingParams;
}

ReSTIRGI_FinalShadingParameters ReSTIRGIContext::GetFinalShadingParameters() const
{
    return m_finalShadingParams;
}

void ReSTIRGIContext::SetFrameIndex(uint32_t frameIndex)
{
    m_frameIndex = frameIndex;
    m_temporalResamplingParams.uniformRandomNumber = JenkinsHash(m_frameIndex);
    UpdateBufferIndices();
}

void ReSTIRGIContext::SetResamplingMode(ReSTIRGI_ResamplingMode resamplingMode)
{
    m_resamplingMode = resamplingMode;
    UpdateBufferIndices();
}

void ReSTIRGIContext::SetTemporalResamplingParameters(const ReSTIRGI_TemporalResamplingParameters& temporalResamplingParams)
{
    m_temporalResamplingParams = temporalResamplingParams;
    m_temporalResamplingParams.uniformRandomNumber = JenkinsHash(m_frameIndex);
}

void ReSTIRGIContext::SetSpatialResamplingParameters(const ReSTIRGI_SpatialResamplingParameters& spatialResamplingParams)
{
    m_spatialResamplingParams = spatialResamplingParams;
}

void ReSTIRGIContext::SetFinalShadingParameters(const ReSTIRGI_FinalShadingParameters& finalShadingParams)
{
    m_finalShadingParams = finalShadingParams;
}

void ReSTIRGIContext::UpdateBufferIndices()
{
    switch (m_resamplingMode)
    {
    case rtxdi::ReSTIRGI_ResamplingMode::None:
        m_bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex = 0;
        m_bufferIndices.finalShadingInputBufferIndex = 0;
        break;
    case rtxdi::ReSTIRGI_ResamplingMode::Temporal:
        m_bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex = m_frameIndex & 1;
        m_bufferIndices.temporalResamplingInputBufferIndex = !m_bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex;
        m_bufferIndices.temporalResamplingOutputBufferIndex = m_bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex;
        m_bufferIndices.finalShadingInputBufferIndex = m_bufferIndices.temporalResamplingOutputBufferIndex;
        break;
    case rtxdi::ReSTIRGI_ResamplingMode::Spatial:
        m_bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex = 0;
        m_bufferIndices.spatialResamplingInputBufferIndex = 0;
        m_bufferIndices.spatialResamplingOutputBufferIndex = 1;
        m_bufferIndices.finalShadingInputBufferIndex = 1;
        break;
    case rtxdi::ReSTIRGI_ResamplingMode::TemporalAndSpatial:
        m_bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex = 0;
        m_bufferIndices.temporalResamplingInputBufferIndex = 1;
        m_bufferIndices.temporalResamplingOutputBufferIndex = 0;
        m_bufferIndices.spatialResamplingInputBufferIndex = 0;
        m_bufferIndices.spatialResamplingOutputBufferIndex = 1;
        m_bufferIndices.finalShadingInputBufferIndex = 1;
        break;
    case rtxdi::ReSTIRGI_ResamplingMode::FusedSpatiotemporal:
        m_bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex = m_frameIndex & 1;
        m_bufferIndices.temporalResamplingInputBufferIndex = !m_bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex;
        m_bufferIndices.spatialResamplingOutputBufferIndex = m_bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex;
        m_bufferIndices.finalShadingInputBufferIndex = m_bufferIndices.spatialResamplingOutputBufferIndex;
        break;
    }
}

}
