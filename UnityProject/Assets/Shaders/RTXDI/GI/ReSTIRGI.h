/***************************************************************************
 # Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

#pragma once

#include <stdint.h>
#include "Rtxdi/RtxdiUtils.h"
#include "Rtxdi/GI/ReSTIRGIParameters.h"

namespace rtxdi
{

static constexpr uint32_t c_NumReSTIRGIReservoirBuffers = 2;

struct ReSTIRGIStaticParameters
{
    uint32_t RenderWidth = 0;
    uint32_t RenderHeight = 0;
    CheckerboardMode CheckerboardSamplingMode = CheckerboardMode::Off;
};

enum class ReSTIRGI_ResamplingMode : uint32_t
{
    None = 0,
    Temporal = 1,
    Spatial = 2,
    TemporalAndSpatial = 3,
    FusedSpatiotemporal = 4,
};

ReSTIRGI_BufferIndices GetDefaultReSTIRGIBufferIndices();
ReSTIRGI_TemporalResamplingParameters GetDefaultReSTIRGITemporalResamplingParams();
ReSTIRGI_SpatialResamplingParameters GetDefaultReSTIRGISpatialResamplingParams();
ReSTIRGI_FinalShadingParameters GetDefaultReSTIRGIFinalShadingParams();

class ReSTIRGIContext
{
public:
    ReSTIRGIContext(const ReSTIRGIStaticParameters& params);

    ReSTIRGIStaticParameters GetStaticParams() const;

    uint32_t GetFrameIndex() const;
    RTXDI_ReservoirBufferParameters GetReservoirBufferParameters() const;
    ReSTIRGI_ResamplingMode GetResamplingMode() const;
    ReSTIRGI_BufferIndices GetBufferIndices() const;
    ReSTIRGI_TemporalResamplingParameters GetTemporalResamplingParameters() const;
    ReSTIRGI_SpatialResamplingParameters GetSpatialResamplingParameters() const;
    ReSTIRGI_FinalShadingParameters GetFinalShadingParameters() const;

    void SetFrameIndex(uint32_t frameIndex);
    void SetResamplingMode(ReSTIRGI_ResamplingMode resamplingMode);
    void SetTemporalResamplingParameters(const ReSTIRGI_TemporalResamplingParameters& temporalResamplingParams);
    void SetSpatialResamplingParameters(const ReSTIRGI_SpatialResamplingParameters& spatialResamplingParams);
    void SetFinalShadingParameters(const ReSTIRGI_FinalShadingParameters& finalShadingParams);

    static uint32_t numReservoirBuffers;

private:
    ReSTIRGIStaticParameters m_staticParams;

    uint32_t m_frameIndex;
    RTXDI_ReservoirBufferParameters m_reservoirBufferParams;
    ReSTIRGI_ResamplingMode m_resamplingMode;
    ReSTIRGI_BufferIndices m_bufferIndices;
    ReSTIRGI_TemporalResamplingParameters m_temporalResamplingParams;
    ReSTIRGI_SpatialResamplingParameters m_spatialResamplingParams;
    ReSTIRGI_FinalShadingParameters m_finalShadingParams;

    void UpdateBufferIndices();
};

}
