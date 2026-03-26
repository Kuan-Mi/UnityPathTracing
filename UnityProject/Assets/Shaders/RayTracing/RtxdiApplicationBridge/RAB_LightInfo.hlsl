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

// 不实现
// 计算给定光照在指定体积内任意表面上的权重。用于世界空间光照网格构建（ReGIR）。
float RAB_GetLightTargetPdfForVolume(RAB_LightInfo light, float3 volumeCenter, float volumeRadius)
{
    return PolymorphicLight::getWeightForVolume(light, volumeCenter, volumeRadius);
}

// // 不是RAB必要函数，只是为了方便将TriangleLight存储到RAB_LightInfo中，供后续加载和使用
// RAB_LightInfo Store(TriangleLight triLight)
// {
//     RAB_LightInfo lightInfo = (RAB_LightInfo)0;
//
//     lightInfo.radiance = Pack_R16G16B16A16_FLOAT(float4(triLight.radiance, 0));
//     lightInfo.center = triLight.base + (triLight.edge1 + triLight.edge2) / 3.0;
//     lightInfo.direction1 = ndirToOctUnorm32(normalize(triLight.edge1));
//     lightInfo.direction2 = ndirToOctUnorm32(normalize(triLight.edge2));
//     lightInfo.scalars = f32tof16(length(triLight.edge1)) | (f32tof16(length(triLight.edge2)) << 16);
//         
//     return lightInfo;
// }

#endif // RAB_LIGHT_INFO_HLSLI
