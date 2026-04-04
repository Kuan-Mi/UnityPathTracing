/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "Rtxdi/RtxdiUtils.h"
#include "Rtxdi/PT/ReSTIRPTParameters.h"

namespace rtxdi
{

static constexpr uint32_t c_NumReSTIRPTReservoirBuffers = 2;

struct ReSTIRPTStaticParameters
{
    uint32_t RenderWidth = 0;
    uint32_t RenderHeight = 0;
    CheckerboardMode CheckerboardSamplingMode = CheckerboardMode::Off;
};

enum class ReSTIRPT_ResamplingMode : uint32_t
{
    None = 0,
    Temporal = 1,
    Spatial = 2,
    TemporalAndSpatial = 3,
};

RTXDI_PTBufferIndices GetDefaultReSTIRPTBufferIndices();
RTXDI_PTInitialSamplingParameters GetDefaultReSTIRPTInitialSamplingParams();
RTXDI_PTTemporalResamplingParameters GetDefaultReSTIRPTTemporalResamplingParams();
RTXDI_PTReconnectionParameters GetDefaultReSTIRPTReconnectionParameters();
RTXDI_PTHybridShiftPerFrameParameters GetDefaultReSTIRPTHybridShiftParams();
RTXDI_BoilingFilterParameters GetDefaultReSTIRPTBoilingFilterParams();
RTXDI_PTSpatialResamplingParameters GetDefaultReSTIRPTSpatialResamplingParams();

class ReSTIRPTContext
{
public:
    ReSTIRPTContext(const ReSTIRPTStaticParameters& staticParams);

    ReSTIRPTStaticParameters GetStaticParams() const;

    uint32_t GetFrameIndex() const;
    RTXDI_ReservoirBufferParameters GetReservoirBufferParameters() const;
    ReSTIRPT_ResamplingMode GetResamplingMode() const;
    RTXDI_PTBufferIndices GetBufferIndices() const;
    RTXDI_PTInitialSamplingParameters GetInitialSamplingParameters() const;
    RTXDI_PTHybridShiftPerFrameParameters GetHybridShiftParameters() const;
    RTXDI_PTReconnectionParameters GetReconnectionParameters() const;
    RTXDI_PTTemporalResamplingParameters GetTemporalResamplingParameters() const;
    RTXDI_BoilingFilterParameters GetBoilingFilterParameters() const;
    RTXDI_PTSpatialResamplingParameters GetSpatialResamplingParameters() const;

    void SetFrameIndex(uint32_t frameIndex);
    void SetResamplingMode(ReSTIRPT_ResamplingMode resamplingMode);
    void SetInitialSamplingParameters(const RTXDI_PTInitialSamplingParameters& initialSamplingParams);
    void SetHybridShiftParameters(const RTXDI_PTHybridShiftPerFrameParameters& hybridShiftParams);
    void SetReconnectionParameters(const RTXDI_PTReconnectionParameters& reconnectionParams);
    void SetTemporalResamplingParameters(const RTXDI_PTTemporalResamplingParameters& temporalResamplingParams);
    void SetBoilingFilterParameters(const RTXDI_BoilingFilterParameters& parameters);
    void SetSpatialResamplingParameters(const RTXDI_PTSpatialResamplingParameters& spatialResamplingParams);

private:
    ReSTIRPTStaticParameters m_staticParams;    

    uint32_t m_frameIndex;
    RTXDI_ReservoirBufferParameters m_reservoirBufferParams;
    ReSTIRPT_ResamplingMode m_resamplingMode;
    RTXDI_PTBufferIndices m_bufferIndices;
    RTXDI_PTInitialSamplingParameters m_initialSamplingParams;
    RTXDI_PTHybridShiftPerFrameParameters m_hybridShiftParams;
    RTXDI_PTReconnectionParameters m_reconnectionParams;
    RTXDI_PTTemporalResamplingParameters m_temporalResamplingParams;
    RTXDI_BoilingFilterParameters m_boilingFilterParameters;
    RTXDI_PTSpatialResamplingParameters m_spatialResamplingParams;

    void UpdateBufferIndices();
};

}