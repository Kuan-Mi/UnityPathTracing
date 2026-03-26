/***************************************************************************
 # Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

/*
This header file is the bridge between the RTXDI resampling functions
and the application resources and parts of shader functionality.

The RTXDI SDK provides the resampling logic, and the application provides
other necessary aspects:
    - Material BRDF evaluation;
    - Ray tracing and transparent/alpha-tested material processing;
    - Light sampling functions and emission profiles.

The structures and functions that are necessary for SDK operation
start with the RAB_ prefix (for RTXDI-Application Bridge).

All structures defined here are opaque for the SDK, meaning that
it makes no assumptions about their contents, they are just passed
between the bridge functions.
*/

#ifndef RTXDI_APPLICATION_BRIDGE_HLSLI
#define RTXDI_APPLICATION_BRIDGE_HLSLI

// See RtxdiApplicationBridge.hlsli in the full sample app for more information.
// This is a minimal viable implementation.

#include "../ShaderParameters.hlsl"
//#include "../SceneGeometry.hlsli"

#include "RAB_Buffers.hlsl"

#include "RAB_LightInfo.hlsl"
#include "RAB_LightSample.hlsl"
#include "RAB_LightSampling.hlsl"
#include "RAB_Material.hlsl"
#include "RAB_RandomSamplerState.hlsl"
#include "RAB_RayPayload.hlsl"
#include "RAB_RTShaders.hlsl"
#include "RAB_SpatialHelpers.hlsl"
#include "RAB_Surface.hlsl"
#include "RAB_VisibilityTest.hlsl"

#endif // RTXDI_APPLICATION_BRIDGE_HLSLI
