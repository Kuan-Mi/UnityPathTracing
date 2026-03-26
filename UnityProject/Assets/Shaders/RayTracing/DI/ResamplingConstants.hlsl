#include "Assets/Shaders/Rtxdi/DI/ReSTIRDIParameters.h"



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

// RTXDI resources
StructuredBuffer<PolymorphicLightInfo> t_LightDataBuffer;
Buffer<float2> t_NeighborOffsets;

RWStructuredBuffer<RTXDI_PackedDIReservoir> u_LightReservoirs;

StructuredBuffer<uint> t_GeometryInstanceToLight;

#define RTXDI_LIGHT_RESERVOIR_BUFFER u_LightReservoirs
#define RTXDI_NEIGHBOR_OFFSETS_BUFFER t_NeighborOffsets

#define BACKGROUND_DEPTH 65504.f

#define RTXDI_ENABLE_PRESAMPLING 1
#define RTXDI_REGIR_MODE RTXDI_REGIR_ONION

static const uint kSecondaryGBuffer_IsSpecularRay = 1;
static const uint kSecondaryGBuffer_IsDeltaSurface = 2;
static const uint kSecondaryGBuffer_IsEnvironmentMap = 4;

static const uint kPolymorphicLightTypeShift = 24;
static const uint kPolymorphicLightTypeMask = 0xf;
static const uint kPolymorphicLightShapingEnableBit = 1 << 28;
static const uint kPolymorphicLightIesProfileEnableBit = 1 << 29;
static const float kPolymorphicLightMinLog2Radiance = -8.f;
static const float kPolymorphicLightMaxLog2Radiance = 40.f;

