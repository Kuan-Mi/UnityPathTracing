#ifndef SHADER_PARAMETERS_H
#define SHADER_PARAMETERS_H

#include <Assets/Shaders/donut/brdf.hlsli>
#include <Assets/Shaders/donut/packing.hlsli>
#include <Assets/Shaders/donut/utils.hlsli>

#include <Assets/Shaders/Include/Shared.hlsl>

#ifdef USE_FULL_RAY
#include <Assets/Shaders/Include/RayTracingShared.hlsl>
#else
#include <Assets/Shaders/Include/LightRayTracingShared.hlsl>
#endif

#include <Assets/Shaders/Rtxdi/DI/ReSTIRDIParameters.h>
#include <Assets/Shaders/Rtxdi/GI/ReSTIRGIParameters.h>
#include <Assets/Shaders/Rtxdi/ReGIR/ReGIRParameters.h>

#include "BRDFPTParameters.h"

#define BACKGROUND_DEPTH INF
#define RTXDI_SCREEN_SPACE_GROUP_SIZE 8


struct ResamplingConstants
{
    RTXDI_RuntimeParameters runtimeParams;

    uint frameIndex;
    uint enablePreviousTLAS;
    uint denoiserMode;
    uint discountNaiveSamples;

    uint enableBrdfIndirect;
    uint enableBrdfAdditiveBlend;
    uint enableAccumulation; // StoreShadingOutput
    uint pad1;

    RTXDI_LightBufferParameters lightBufferParams;
    RTXDI_RISBufferSegmentParameters localLightsRISBufferSegmentParams;
    RTXDI_RISBufferSegmentParameters environmentLightRISBufferSegmentParams;

    ReSTIRDI_Parameters restirDI;
    ReGIR_Parameters regir;
    ReSTIRGI_Parameters restirGI;
    BRDFPathTracing_Parameters brdfPT;

    uint visualizeRegirCells;
    uint3 pad2;

    uint2 environmentPdfTextureSize;
    uint2 localLightPdfTextureSize;
};

struct SecondaryGBufferData
{
    float3 worldPos;
    uint normal;

    uint2 throughputAndFlags; // .x = throughput.rg as float16, .y = throughput.b as float16, flags << 16
    uint diffuseAlbedo; // R11G11B10_UFLOAT
    uint specularAndRoughness; // R8G8B8A8_Gamma_UFLOAT

    float3 emission;
    float pdf;
};

static const uint kSecondaryGBuffer_IsSpecularRay = 1;
static const uint kSecondaryGBuffer_IsDeltaSurface = 2;
static const uint kSecondaryGBuffer_IsEnvironmentMap = 4;

static const uint kPolymorphicLightTypeShift = 24;
static const uint kPolymorphicLightTypeMask = 0xf;
static const uint kPolymorphicLightShapingEnableBit = 1 << 28;
static const uint kPolymorphicLightIesProfileEnableBit = 1 << 29;
static const float kPolymorphicLightMinLog2Radiance = -8.f;
static const float kPolymorphicLightMaxLog2Radiance = 40.f;

enum PolymorphicLightType
{
    kSphere = 0,
    kCylinder,
    kDisk,
    kRect,
    kTriangle,
    kDirectional,
    kEnvironment,
    kPoint
};


// Stores shared light information (type) and specific light information
// See PolymorphicLight.hlsli for encoding format
struct PolymorphicLightInfo
{
    // uint4[0]
    float3 center;
    uint colorTypeAndFlags; // RGB8 + uint8 (see the kPolymorphicLight... constants above)

    // uint4[1]
    uint direction1; // oct-encoded
    uint direction2; // oct-encoded
    uint scalars; // 2x float16
    uint logRadiance; // uint16

    // uint4[2] -- optional, contains only shaping data
    uint iesProfileIndex;
    uint primaryAxis; // oct-encoded
    uint cosConeAngleAndSoftness; // 2x float16
    uint padding;
};


#endif // SHADER_PARAMETERS_H
