#ifndef RAB_LIGHT_INFO_HLSLI
#define RAB_LIGHT_INFO_HLSLI

//#include "../ShaderParameters.h"
#include "../TriangleLight.hlsl"
#include "../PolymorphicLight.hlsl"
#include "RAB_Surface.hlsl"
#include "RAB_LightSample.hlsl"

typedef PolymorphicLightInfo RAB_LightInfo;

// 返回一个无效的光源实例
RAB_LightInfo RAB_EmptyLightInfo()
{
    return (RAB_LightInfo)0;
}

// Load the packed light information from the buffer.
// Ignore the previousFrame parameter as our lights are static in this sample.
// 无视 previousFrame 参数，因为我们在这个示例中使用的是静态光源。

// 根据索引，从当前帧或上一帧加载多态光源的信息。有关所需信息的说明，请参阅 RAB_LightInfo 。
// 传递给此函数的索引将位于 RTXDI_LightBufferParameters 提供的三个范围之一内。

// 这些范围不必连续地打包在一个缓冲区中，也不必从零开始。应用程序可以选择使用光索引中的一些较高位来存储信息。光索引的低 31 位可用；最高位保留供内部使用。
RAB_LightInfo RAB_LoadLightInfo(uint index, bool previousFrame)
{
    return t_LightDataBuffer[index];
}

// 不实现
RAB_LightInfo RAB_LoadCompactLightInfo(uint linearIndex)
{
    return RAB_EmptyLightInfo();
}

// 不实现
bool RAB_StoreCompactLightInfo(uint linearIndex, RAB_LightInfo lightInfo)
{
    return false;
}


// 计算给定光照在指定体积内任意表面上的权重。用于世界空间光照网格构建（ReGIR）。
float RAB_GetLightTargetPdfForVolume(RAB_LightInfo light, float3 volumeCenter, float volumeRadius)
{
    return PolymorphicLight::getWeightForVolume(light, volumeCenter, volumeRadius);
}

// Compute the position on a triangle light given a pair of random numbers
// 对相对于给定接收表面的多态光进行采样。对于大多数光照类型，“uv”参数只是一对均匀分布的随机数，最初由 RAB_GetNextRandom 函数生成并存储在光照库中。
// 对于重要性采样的环境光，“uv”参数具有 PDF 纹理中的纹理坐标，并归一化到 (0..1) 范围内。
RAB_LightSample RAB_SamplePolymorphicLight(RAB_LightInfo lightInfo, RAB_Surface surface, float2 uv)
{
    PolymorphicLightSample pls = PolymorphicLight::calcSample(lightInfo, uv, surface.worldPos);

    RAB_LightSample lightSample;
    lightSample.position = pls.position;
    lightSample.normal = pls.normal;
    lightSample.radiance = pls.radiance;
    lightSample.solidAnglePdf = pls.solidAnglePdf;
    lightSample.lightType = getLightType(lightInfo);
    return lightSample;
}

#endif // RAB_LIGHT_INFO_HLSLI
