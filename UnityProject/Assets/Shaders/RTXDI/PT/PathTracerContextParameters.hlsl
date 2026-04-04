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

#ifndef RTXDI_PATH_TRACING_CONTEXT_PARAMETERS_HLSLI
#define RTXDI_PATH_TRACING_CONTEXT_PARAMETERS_HLSLI

#include "PathReconnectibility.hlsl"

struct RTXDI_PathTracerContextParameters
{
    // Initial sampling params
    uint16_t maxBounces;
    uint16_t maxRcVertexLength;

    // Hybrid shift params
    uint16_t RcVertexLength;
    uint16_t SelectedPathLength;

    RTXDI_PTReconnectionParameters rParams;
    RTXDI_PTReconnectionRuntimeParameters rrParams;
};

#endif // RTXDI_PATH_TRACING_CONTEXT_PARAMETERS_HLSLI