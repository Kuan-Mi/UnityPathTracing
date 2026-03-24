#ifndef RAB_LIGHT_INFO_HLSLI
#define RAB_LIGHT_INFO_HLSLI

//#include "../ShaderParameters.h"
#include "../TriangleLight.hlsl"

// 返回一个无效的光源实例
RAB_LightInfo RAB_EmptyLightInfo()
{
    return (RAB_LightInfo)0;
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
    return 0.0;
}

// 不是RAB必要函数，只是为了方便将TriangleLight存储到RAB_LightInfo中，供后续加载和使用
RAB_LightInfo Store(TriangleLight triLight)
{
    RAB_LightInfo lightInfo = (RAB_LightInfo)0;

    lightInfo.radiance = Pack_R16G16B16A16_FLOAT(float4(triLight.radiance, 0));
    lightInfo.center = triLight.base + (triLight.edge1 + triLight.edge2) / 3.0;
    lightInfo.direction1 = ndirToOctUnorm32(normalize(triLight.edge1));
    lightInfo.direction2 = ndirToOctUnorm32(normalize(triLight.edge2));
    lightInfo.scalars = f32tof16(length(triLight.edge1)) | (f32tof16(length(triLight.edge2)) << 16);
        
    return lightInfo;
}

#endif // RAB_LIGHT_INFO_HLSLI
