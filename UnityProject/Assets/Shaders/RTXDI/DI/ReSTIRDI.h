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

    ReSTIRDI_BufferIndices GetDefaultReSTIRDIBufferIndices();
    ReSTIRDI_InitialSamplingParameters GetDefaultReSTIRDIInitialSamplingParams();
    ReSTIRDI_TemporalResamplingParameters GetDefaultReSTIRDITemporalResamplingParams();
    ReSTIRDI_SpatialResamplingParameters GetDefaultReSTIRDISpatialResamplingParams();
    ReSTIRDI_ShadingParameters GetDefaultReSTIRDIShadingParams();

    // Make this constructor take static RTXDI params, update its dynamic ones
    class ReSTIRDIContext
    {
    public:
        ReSTIRDIContext(const ReSTIRDIStaticParameters& params);

        RTXDI_ReservoirBufferParameters GetReservoirBufferParameters() const;
        ReSTIRDI_ResamplingMode GetResamplingMode() const;
        RTXDI_RuntimeParameters GetRuntimeParams() const;
        ReSTIRDI_BufferIndices GetBufferIndices() const;
        ReSTIRDI_InitialSamplingParameters GetInitialSamplingParameters() const;
        ReSTIRDI_TemporalResamplingParameters GetTemporalResamplingParameters() const;
        ReSTIRDI_SpatialResamplingParameters GetSpatialResamplingParameters() const;
        ReSTIRDI_ShadingParameters GetShadingParameters() const;

        uint32_t GetFrameIndex() const;
        const ReSTIRDIStaticParameters& GetStaticParameters() const;

        void SetFrameIndex(uint32_t frameIndex);
        void SetResamplingMode(ReSTIRDI_ResamplingMode resamplingMode);
        void SetInitialSamplingParameters(const ReSTIRDI_InitialSamplingParameters& initialSamplingParams);
        void SetTemporalResamplingParameters(const ReSTIRDI_TemporalResamplingParameters& temporalResamplingParams);
        void SetSpatialResamplingParameters(const ReSTIRDI_SpatialResamplingParameters& spatialResamplingParams);
        void SetShadingParameters(const ReSTIRDI_ShadingParameters& shadingParams);

        static const uint32_t NumReservoirBuffers;

    private:
        uint32_t m_lastFrameOutputReservoir;
        uint32_t m_currentFrameOutputReservoir;

        uint32_t m_frameIndex;

        ReSTIRDIStaticParameters m_staticParams;

        ReSTIRDI_ResamplingMode m_resamplingMode;
        RTXDI_ReservoirBufferParameters m_reservoirBufferParams;
        RTXDI_RuntimeParameters m_runtimeParams;
        ReSTIRDI_BufferIndices m_bufferIndices;
        
        ReSTIRDI_InitialSamplingParameters m_initialSamplingParams;
        ReSTIRDI_TemporalResamplingParameters m_temporalResamplingParams;
        ReSTIRDI_SpatialResamplingParameters m_spatialResamplingParams;
        ReSTIRDI_ShadingParameters m_shadingParams;

        void UpdateBufferIndices();
        void UpdateCheckerboardField();
    };
}
