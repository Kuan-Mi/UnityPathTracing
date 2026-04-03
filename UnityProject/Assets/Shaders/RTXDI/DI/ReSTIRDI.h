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
#include <memory>
#include <vector>

#include "Rtxdi/RtxdiUtils.h"
#include "Rtxdi/DI/ReSTIRDIParameters.h"

namespace rtxdi
{
    static constexpr uint32_t c_NumReSTIRDIReservoirBuffers = 3;

    enum class ReSTIRDI_ResamplingMode : uint32_t
    {
        None,
        Temporal,
        Spatial,
        TemporalAndSpatial,
        FusedSpatiotemporal
    };

    struct RISBufferSegmentParameters
    {
        uint32_t tileSize;
        uint32_t tileCount;
    };

    // Parameters used to initialize the ReSTIRDIContext
    // Changing any of these requires recreating the context.
    struct ReSTIRDIStaticParameters
    {
        uint32_t NeighborOffsetCount = 8192;
        uint32_t RenderWidth = 0;
        uint32_t RenderHeight = 0;

        CheckerboardMode CheckerboardSamplingMode = CheckerboardMode::Off;
    };

    RTXDI_DIBufferIndices GetDefaultReSTIRDIBufferIndices();
    RTXDI_DIInitialSamplingParameters GetDefaultReSTIRDIInitialSamplingParams();
    RTXDI_DITemporalResamplingParameters GetDefaultReSTIRDITemporalResamplingParams();
    RTXDI_BoilingFilterParameters GetDefaultReSTIRDIBoilingFilterParams();
    RTXDI_DISpatialResamplingParameters GetDefaultReSTIRDISpatialResamplingParams();
    RTXDI_DISpatioTemporalResamplingParameters GetDefaultReSTIRDISpatioTemporalResamplingParams();
    RTXDI_ShadingParameters GetDefaultReSTIRDIShadingParams();

    // Make this constructor take static RTXDI params, update its dynamic ones
    class ReSTIRDIContext
    {
    public:
        ReSTIRDIContext(const ReSTIRDIStaticParameters& params);

        RTXDI_ReservoirBufferParameters GetReservoirBufferParameters() const;
        ReSTIRDI_ResamplingMode GetResamplingMode() const;
        RTXDI_RuntimeParameters GetRuntimeParams() const;
        RTXDI_DIBufferIndices GetBufferIndices() const;
        RTXDI_DIInitialSamplingParameters GetInitialSamplingParameters() const;
        RTXDI_DITemporalResamplingParameters GetTemporalResamplingParameters() const;
        RTXDI_BoilingFilterParameters GetBoilingFilterParameters() const;
        RTXDI_DISpatialResamplingParameters GetSpatialResamplingParameters() const;
        RTXDI_DISpatioTemporalResamplingParameters GetSpatioTemporalResamplingParameters() const;
        RTXDI_ShadingParameters GetShadingParameters() const;

        uint32_t GetFrameIndex() const;
        const ReSTIRDIStaticParameters& GetStaticParameters() const;

        void SetFrameIndex(uint32_t frameIndex);
        void SetResamplingMode(ReSTIRDI_ResamplingMode resamplingMode);
        void SetInitialSamplingParameters(const RTXDI_DIInitialSamplingParameters& initialSamplingParams);
        void SetTemporalResamplingParameters(const RTXDI_DITemporalResamplingParameters& temporalResamplingParams);
        void SetBoilingFilterParameters(const RTXDI_BoilingFilterParameters& boilingFilterParams);
        void SetSpatialResamplingParameters(const RTXDI_DISpatialResamplingParameters& spatialResamplingParams);
        void SetSpatioTemporalResamplingParameters(const RTXDI_DISpatioTemporalResamplingParameters& spatioTemporalResamplingParams);
        void SetShadingParameters(const RTXDI_ShadingParameters& shadingParams);

    private:
        uint32_t m_lastFrameOutputReservoir;
        uint32_t m_currentFrameOutputReservoir;

        ReSTIRDIStaticParameters m_staticParams;

        ReSTIRDI_ResamplingMode m_resamplingMode;
        RTXDI_ReservoirBufferParameters m_reservoirBufferParams;
        RTXDI_RuntimeParameters m_runtimeParams;
        RTXDI_DIBufferIndices m_bufferIndices;
        
        RTXDI_DIInitialSamplingParameters m_initialSamplingParams;
        RTXDI_DITemporalResamplingParameters m_temporalResamplingParams;
        RTXDI_BoilingFilterParameters m_boilingFilterParams;
        RTXDI_DISpatialResamplingParameters m_spatialResamplingParams;
        RTXDI_DISpatioTemporalResamplingParameters m_spatioTemporalResamplingParams;
        RTXDI_ShadingParameters m_shadingParams;

        void UpdateBufferIndices();
        void UpdateCheckerboardField();
    };
}
