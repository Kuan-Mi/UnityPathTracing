#ifndef RAB_LIGHT_SAMPLING_HLSLI
#define RAB_LIGHT_SAMPLING_HLSLI

#include "RAB_Material.hlsl"
#include "RAB_RayPayload.hlsl"
#include "RAB_Surface.hlsl"

// 将世界空间方向转换为一对数字，当将这对数字传递给 RAB_SamplePolymorphicLight 作为环境光时，将在同一方向上进行采样。
float2 RAB_GetEnvironmentMapRandXYFromDir(float3 worldDir)
{
    return float2(0.0, 0.0);
}

// 根据环境贴图 PDF 纹理，计算从环境贴图中采样特定方向相对于所有其他可能方向的概率。
float RAB_EvaluateEnvironmentMapSamplingPdf(float3 L)
{
    // No Environment sampling
    return 0;
}

// 基于局部光 PDF 纹理，使用重要性采样计算从局部光池中采样特定光的概率。
float RAB_EvaluateLocalLightSourcePdf(uint lightIndex)
{
    // Uniform pdf
    // return 1.0 / gNumLights;
    return 1.0 / g_Const.lightBufferParams.localLightBufferRegion.numLights;
}

float3 RAB_GetReflectedRadianceForSurface(float3 incomingRadianceLocation, float3 incomingRadiance, RAB_Surface surface)
{
    return float3(0.0, 0.0, 0.0);
}

float RAB_GetReflectedLuminanceForSurface(float3 incomingRadianceLocation, float3 incomingRadiance, RAB_Surface surface)
{
    return 0.0;
}

// Evaluate the surface BRDF and compute the weighted reflected radiance for the given light sample
// 评估表面双向反射分布函数 (BRDF) 并计算给定光样本的加权反射辐射度
float3 ShadeSurfaceWithLightSample(RAB_LightSample lightSample, RAB_Surface surface)
{
    // Ignore invalid light samples
    if (lightSample.solidAnglePdf <= 0)
        return 0;

    float3 L = normalize(lightSample.position - surface.worldPos);

    // Ignore light samples that are below the geometric surface (but above the normal mapped surface)
    if (dot(L, surface.geoNormal) <= 0)
        return 0;


    float3 V = surface.viewDir;
    
    // Evaluate the BRDF
    float diffuse = Lambert(surface.normal, -L);
    float3 specular = GGX_times_NdotL(V, L, surface.normal, max(RAB_GetMaterial(surface).roughness, kMinRoughness), RAB_GetMaterial(surface).specularF0);

    float3 reflectedRadiance = lightSample.radiance * (diffuse * surface.material.diffuseAlbedo + specular);

    return reflectedRadiance / lightSample.solidAnglePdf;
}

// Compute the target PDF (p-hat) for the given light sample relative to a surface
// 计算给定表面使用该光照样本进行着色时，每个光照样本的权重。
// 可以使用精确或近似的双向反射分布函数 (BRDF) 计算来计算权重。
// 即使所有样本的权重都固定为 1.0，ReSTIR 也能收敛到正确的光照结果，但结果会非常嘈杂。
// 权重的缩放比例可以任意，只要在所有光照和表面上保持一致即可。
float RAB_GetLightSampleTargetPdfForSurface(RAB_LightSample lightSample, RAB_Surface surface)
{
    // Second-best implementation: the PDF is proportional to the reflected radiance.
    // The best implementation would be taking visibility into account,
    // but that would be prohibitively expensive.
    
    // 次优实现方案：PDF 与反射辐射亮度成正比。
    // 最佳实现方案应考虑可见性，
    // 但这将导致计算成本过高。
    return calcLuminance(ShadeSurfaceWithLightSample(lightSample, surface));
}

float RAB_GetGISampleTargetPdfForSurface(float3 samplePosition, float3 sampleRadiance, RAB_Surface surface)
{
    return 0.0;
}

// 返回表面到光源样本的方向和距离。
void RAB_GetLightDirDistance(RAB_Surface surface, RAB_LightSample lightSample,
    out float3 o_lightDir,
    out float o_lightDistance)
{
    float3 toLight = lightSample.position - surface.worldPos;
    o_lightDistance = length(toLight);
    o_lightDir = toLight / o_lightDistance;
}

bool RTXDI_CompareRelativeDifference(float reference, float candidate, float threshold);

float3 GetEnvironmentRadiance(float3 direction)
{
    return float3(0.0, 0.0, 0.0);
}

bool IsComplexSurface(int2 pixelPosition, RAB_Surface surface)
{
    return true;
}

uint getLightIndex(uint instanceID, uint geometryIndex, uint primitiveIndex)
{
    if (primitiveIndex == INF)
    {
        return RTXDI_InvalidLightIndex;
    }
    
    return primitiveIndex;
    
    // uint lightIndex = RTXDI_InvalidLightIndex;
    // InstanceData hitInstance = t_InstanceData[instanceID];
    // uint geometryInstanceIndex = hitInstance.firstGeometryInstanceIndex + geometryIndex;
    // lightIndex = t_GeometryInstanceToLight[geometryInstanceIndex];
    // if (lightIndex != RTXDI_InvalidLightIndex)
    //   lightIndex += primitiveIndex;
    // return lightIndex;
}

// Return true if anything was hit. If false, RTXDI will do environment map sampling
// o_lightIndex: If hit, must be a valid light index for RAB_LoadLightInfo, if no local light was hit, must be RTXDI_InvalidLightIndex
// randXY: The randXY that corresponds to the hit location and is the same used for RAB_SamplePolymorphicLight

// 如果击中目标，则返回 true。如果为 false，RTXDI 将进行环境贴图采样。
// o_lightIndex：如果击中目标，则必须是 RAB_LoadLightInfo 的有效光源索引；如果没有击中本地光源，则必须是 RTXDI_InvalidLightIndex。
// randXY：与击中位置对应的 randXY 值，与 RAB_SamplePolymorphicLight 使用的 randXY 值相同。

// 使用给定参数追踪光线，寻找光源。
// 如果找到局部光源，则返回 true 并将光源采样信息填充到输出参数中。
// 如果击中非光源场景对象，则返回 true 并将 o_lightIndex 设置为 RTXDI_InvalidLightIndex 。
// 如果未击中任何对象，则返回 false ，RTXDI 将尝试进行环境贴图采样。
bool RAB_TraceRayForLocalLight(float3 origin, float3 direction, float tMin, float tMax,
    out uint o_lightIndex, out float2 o_randXY)
{
    o_lightIndex = RTXDI_InvalidLightIndex;
    o_randXY = 0;

    
    
    GeometryProps geometryProps0;
    MaterialProps materialProps0;
    CastRay(origin, direction, tMin, tMax, GetConeAngleFromRoughness(0.0, 0.0),  FLAG_NON_TRANSPARENT, geometryProps0, materialProps0);

    bool hitAnything = !geometryProps0.IsMiss();
    if (hitAnything)
    {
        o_lightIndex = getLightIndex(geometryProps0.instanceIndex, 0, geometryProps0.primitiveIndex);
        if (o_lightIndex != RTXDI_InvalidLightIndex)
        {
            float2 hitUV =  geometryProps0.barycentrics;
            o_randXY = randomFromBarycentric(hitUVToBarycentric(hitUV));
        }
    }
    
    return hitAnything;
    
    // RayDesc ray;
    // ray.Origin = origin;
    // ray.Direction = direction;
    // ray.TMin = tMin;
    // ray.TMax = tMax;
    //
    // RayQuery<RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> rayQuery;
    // rayQuery.TraceRayInline(SceneBVH, RAY_FLAG_NONE, INSTANCE_MASK_OPAQUE, ray);
    // rayQuery.Proceed();
    //
    // bool hitAnything = rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    // if (hitAnything)
    // {
    //     o_lightIndex = getLightIndex(rayQuery.CommittedInstanceID(), rayQuery.CommittedGeometryIndex(), rayQuery.CommittedPrimitiveIndex());
    //     if (o_lightIndex != RTXDI_InvalidLightIndex)
    //     {
    //         float2 hitUV = rayQuery.CommittedTriangleBarycentrics();
    //         o_randXY = randomFromBarycentric(hitUVToBarycentric(hitUV));
    //     }
    // }
    //
    // return hitAnything;
}

// Compute the position on a triangle light given a pair of random numbers
// 对相对于给定接收表面的多态光进行采样。对于大多数光照类型，“uv”参数只是一对均匀分布的随机数，最初由 RAB_GetNextRandom 函数生成并存储在光照库中。
// 对于重要性采样的环境光，“uv”参数具有 PDF 纹理中的纹理坐标，并归一化到 (0..1) 范围内。
RAB_LightSample RAB_SamplePolymorphicLight(RAB_LightInfo lightInfo, RAB_Surface surface, float2 uv)
{
    TriangleLight triLight = TriangleLight::Create(lightInfo);
    return CalcSample(triLight, uv, surface.worldPos);
}

#endif // RAB_LIGHT_SAMPLING_HLSLI
