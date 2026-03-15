#ifndef RTXDI_RAB_LIGHT_INFO_HLSLI
#define RTXDI_RAB_LIGHT_INFO_HLSLI

#include "../TriangleLight.hlsli"

// 表示光源上的一个点及其辐射度，该辐射度相对于用于生成样本的表面进行加权。
// 光源样本由 RAB_SamplePolymorphicLight 函数生成，该函数接受一个 RAB_LightInfo 、一个 RAB_Surface 和一对随机数。在内部， RAB_LightSample 实例仅用于计算目标 PDF（通过 RAB_GetLightSampleTargetPdfForSurface ），不会存储在任何地方。
// ReSTIR 存储和重用的光源样本以样本引用的形式存储，即 RTXDI_LightSampleRef 结构的实例，该结构仅存储光源索引和随机数。然后，对于每个用于加权的曲面，重新计算光源在光源上的实际位置。
struct RAB_LightSample
{
    float3 position;
    float3 normal;
    float3 radiance;
    float solidAnglePdf;
};

// 返回一个无效的光源样本实例
RAB_LightSample RAB_EmptyLightSample()
{
    return (RAB_LightSample)0;
}

// 返回是否是解析光，这里我们不实现解析光，所以直接返回false
// 如果光样本来自解析光（例如球体或矩形图元），而解析光不能被 BRDF 光线采样，则返回 true 。
bool RAB_IsAnalyticLightSample(RAB_LightSample lightSample)
{
    return false;
}

// 返回solid angle pdf
float RAB_LightSampleSolidAnglePdf(RAB_LightSample lightSample)
{
    return lightSample.solidAnglePdf;
}

// 根据给定的随机数和观察者位置，从三角形光源上采样一个点，并计算该点的辐射度和solid angle pdf。
RAB_LightSample CalcSample(TriangleLight triLight, in const float2 random, in const float3 viewerPosition)
{
    RAB_LightSample result;

    float3 bary = sampleTriangle(random);
    result.position = triLight.base + triLight.edge1 * bary.y + triLight.edge2 * bary.z;
    result.normal = triLight.normal;

    result.solidAnglePdf = triLight.calcSolidAnglePdf(viewerPosition, result.position, result.normal);

    result.radiance = triLight.radiance;

    return result;   
}

#endif // RTXDI_RAB_LIGHT_INFO_HLSLI
