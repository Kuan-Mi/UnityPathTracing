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
    uint2 pdfTextureSize = g_Const.localLightPdfTextureSize.xy;
    uint2 texelPosition = RTXDI_LinearIndexToZCurve(lightIndex);
    float texelValue = t_LocalLightPdfTexture[texelPosition].r;

    int lastMipLevel = max(0, int(floor(log2(max(pdfTextureSize.x, pdfTextureSize.y)))));
    float averageValue = t_LocalLightPdfTexture.mips[lastMipLevel][uint2(0, 0)].x;

    // See the comment at 'sum' in RAB_EvaluateEnvironmentMapSamplingPdf.
    // The same texture shape considerations apply to local lights.
    float sum = averageValue * square(1u << lastMipLevel);

    return texelValue / sum;
}

float3 RAB_GetReflectedRadianceForSurface(float3 incomingRadianceLocation, float3 incomingRadiance, RAB_Surface surface)
{
    float3 L = normalize(incomingRadianceLocation - surface.worldPos);
    float3 N = surface.normal;
    float3 V = surface.viewDir;

    if (dot(L, surface.geoNormal) <= 0)
        return 0;

    float d = Lambert(N, -L);
    float3 s;
    if (surface.material.roughness == 0)
        s = 0;
    else
        s = GGX_times_NdotL(V, L, N, max(surface.material.roughness, kMinRoughness), surface.material.specularF0);

    return incomingRadiance * (d * surface.material.diffuseAlbedo + s);
}

float RAB_GetReflectedLuminanceForSurface(float3 incomingRadianceLocation, float3 incomingRadiance, RAB_Surface surface)
{
    return RTXDI_Luminance(RAB_GetReflectedRadianceForSurface(incomingRadianceLocation, incomingRadiance, surface));
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
    if (lightSample.solidAnglePdf <= 0)
        return 0;

    return RAB_GetReflectedLuminanceForSurface(lightSample.position, lightSample.radiance, surface) / lightSample.solidAnglePdf;
}

float RAB_GetGISampleTargetPdfForSurface(float3 samplePosition, float3 sampleRadiance, RAB_Surface surface)
{
    float3 reflectedRadiance = RAB_GetReflectedRadianceForSurface(samplePosition, sampleRadiance, surface);

    return RTXDI_Luminance(reflectedRadiance);
}


void RAB_GetLightDirDistance(RAB_Surface surface, RAB_LightSample lightSample,
    out float3 o_lightDir,
    out float o_lightDistance)
{
    if (lightSample.lightType == PolymorphicLightType::kEnvironment)
    {
        o_lightDir = -lightSample.normal;
        o_lightDistance = DISTANT_LIGHT_DISTANCE;
    }
    else
    {
        float3 toLight = lightSample.position - surface.worldPos;
        o_lightDistance = length(toLight);
        o_lightDir = toLight / o_lightDistance;
    }
}

bool RTXDI_CompareRelativeDifference(float reference, float candidate, float threshold);

float3 GetEnvironmentRadiance(float3 direction)
{
    return float3(0.0, 0.0, 0.0);
}

bool IsComplexSurface(int2 pixelPosition, RAB_Surface surface)
{
    float originalRoughness = gOut_Normal_Roughness[pixelPosition].a;
    return originalRoughness < (surface.material.roughness * g_Const.restirDI.temporalResamplingParams.permutationSamplingThreshold);
}

uint getLightIndex(uint instanceID, uint geometryIndex, uint primitiveIndex)
{
    if (primitiveIndex == INF)
    {
        return RTXDI_InvalidLightIndex;
    }

    uint start = t_GeometryInstanceToLight[instanceID];

    if (start == RTXDI_InvalidLightIndex)
    {
        return RTXDI_InvalidLightIndex;
    }


    // return instanceID;
    return start + primitiveIndex;
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
    CastRay(origin, direction, tMin, tMax, GetConeAngleFromRoughness(0.0, 0.0), FLAG_NON_TRANSPARENT, geometryProps0, materialProps0);

    bool hitAnything = !geometryProps0.IsMiss();
    if (hitAnything)
    {
        o_lightIndex = getLightIndex(geometryProps0.instanceIndex, 0, geometryProps0.primitiveIndex);
        if (o_lightIndex != RTXDI_InvalidLightIndex)
        {
            float2 hitUV = geometryProps0.barycentrics;
            o_randXY = randomFromBarycentric(hitUVToBarycentric(hitUV));
        }
    }

    return hitAnything;
}



#endif // RAB_LIGHT_SAMPLING_HLSLI
