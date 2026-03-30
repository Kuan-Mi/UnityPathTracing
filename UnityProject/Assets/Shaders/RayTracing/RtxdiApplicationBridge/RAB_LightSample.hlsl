#ifndef RTXDI_RAB_LIGHT_INFO_HLSLI
#define RTXDI_RAB_LIGHT_INFO_HLSLI

struct RAB_LightSample
{
    float3 position;
    float3 normal;
    float3 radiance;
    float solidAnglePdf;
    PolymorphicLightType lightType;
};

RAB_LightSample RAB_EmptyLightSample()
{
    return (RAB_LightSample)0;
}

bool RAB_IsAnalyticLightSample(RAB_LightSample lightSample)
{
    return lightSample.lightType != PolymorphicLightType::kTriangle && 
        lightSample.lightType != PolymorphicLightType::kEnvironment;
}

float RAB_LightSampleSolidAnglePdf(RAB_LightSample lightSample)
{
    return lightSample.solidAnglePdf;
}

#endif // RTXDI_RAB_LIGHT_INFO_HLSLI