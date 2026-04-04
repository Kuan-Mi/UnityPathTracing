/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <stdint.h>
#include "Assets/Shaders/Rtxdi/RtxdiUtils.h"
#include "Assets/Shaders/Rtxdi/GI/ReSTIRGIParameters.h"

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

RTXDI_GIBufferIndices GetDefaultReSTIRGIBufferIndices();
RTXDI_GITemporalResamplingParameters GetDefaultReSTIRGITemporalResamplingParams();
RTXDI_BoilingFilterParameters GetDefaultReSTIRGIBoilingFilterParams();
RTXDI_GISpatialResamplingParameters GetDefaultReSTIRGISpatialResamplingParams();
RTXDI_GISpatioTemporalResamplingParameters GetDefaultReSTIRGISpatioTemporalResamplingParams();
RTXDI_GIFinalShadingParameters GetDefaultReSTIRGIFinalShadingParams();

class ReSTIRGIContext
{
public:
    ReSTIRGIContext(const ReSTIRGIStaticParameters& params);

    ReSTIRGIStaticParameters GetStaticParams() const;

    uint32_t GetFrameIndex() const;
    RTXDI_ReservoirBufferParameters GetReservoirBufferParameters() const;
    ReSTIRGI_ResamplingMode GetResamplingMode() const;
    RTXDI_GIBufferIndices GetBufferIndices() const;
    RTXDI_GITemporalResamplingParameters GetTemporalResamplingParameters() const;
    RTXDI_BoilingFilterParameters GetBoilingFilterParameters() const;
    RTXDI_GISpatialResamplingParameters GetSpatialResamplingParameters() const;
    RTXDI_GISpatioTemporalResamplingParameters GetSpatioTemporalResamplingParameters() const;
    RTXDI_GIFinalShadingParameters GetFinalShadingParameters() const;

    void SetFrameIndex(uint32_t frameIndex);
    void SetResamplingMode(ReSTIRGI_ResamplingMode resamplingMode);
    void SetTemporalResamplingParameters(const RTXDI_GITemporalResamplingParameters& temporalResamplingParams);
    void SetBoilingFilterParameters(const RTXDI_BoilingFilterParameters& boilingFilterParams);
    void SetSpatialResamplingParameters(const RTXDI_GISpatialResamplingParameters& spatialResamplingParams);
    void SetSpatioTemporalResamplingParameters(const RTXDI_GISpatioTemporalResamplingParameters& spatioTemporalParams);
    void SetFinalShadingParameters(const RTXDI_GIFinalShadingParameters& finalShadingParams);

private:
    ReSTIRGIStaticParameters m_staticParams;

    uint32_t m_frameIndex;
    RTXDI_ReservoirBufferParameters m_reservoirBufferParams;
    ReSTIRGI_ResamplingMode m_resamplingMode;
    RTXDI_GIBufferIndices m_bufferIndices;
    RTXDI_GITemporalResamplingParameters m_temporalResamplingParams;
    RTXDI_BoilingFilterParameters m_boilingFilterParams;
    RTXDI_GISpatialResamplingParameters m_spatialResamplingParams;
    RTXDI_GISpatioTemporalResamplingParameters m_spatioTemporalResamplingParams;
    RTXDI_GIFinalShadingParameters m_finalShadingParams;

    void UpdateBufferIndices();
};

}
