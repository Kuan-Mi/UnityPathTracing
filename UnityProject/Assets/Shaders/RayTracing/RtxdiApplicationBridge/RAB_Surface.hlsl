#ifndef RTXDI_RAB_SURFACE_HLSLI
#define RTXDI_RAB_SURFACE_HLSLI

// #include "../GBufferHelpers.hlsli"

#include "Assets/Shaders/Rtxdi/Utils/RandomSamplerstate.hlsl"
#include "Assets/Shaders/Rtxdi/Utils/Color.hlsl"
#include "Assets/Shaders/Rtxdi/Utils/BrdfRaySample.hlsl"

#include "RAB_Material.hlsl"

// A surface with enough information to evaluate BRDFs
// 存储表面信息，包括其位置、方向和材质参数，
// 其中材质参数必须可通过 RAB_GetMaterial 函数以 RAB_Material 形式访问。
// 此外， RAB_Surface 结构还应包含视图方向。
// 该结构必须包含使用 RAB_GetLightSampleTargetPdfForSurface 函数计算材质 BRDF 所需的一切信息。

// RAB_Surface 的实例在 RAB_GetGBufferSurface 函数中构建，该函数根据给定的像素位置从当前或之前的 G 缓冲区加载表面信息，并由时间和空间重采样函数调用。
// 或者，这些实例也可以直接在主光线着色器中生成，例如，并将其作为当前着色表面传递给重采样函数。
struct RAB_Surface
{
    float3 worldPos;
    float3 viewDir;
    float3 normal;
    float3 geoNormal;
    float viewDepth;
    float diffuseProbability;
    RAB_Material material;
};

// 返回一个无效的表面实例，表示没有有效的表面信息可用。对于无效表面，RAB_IsSurfaceValid 应返回 false。
RAB_Surface RAB_EmptySurface()
{
    RAB_Surface surface = (RAB_Surface)0;
    surface.viewDepth = -BACKGROUND_DEPTH;
    return surface;
}

// 返回 true 如果表面包含有效信息（例如，来自 G-buffer 的有效深度值），否则返回 false。
// 测试所提供的表面是否包含有效的几何图形。如果表面是从超出边界的像素加载的，或者是从包含天空或其他背景的像素加载的，则此函数应返回 false 。
bool RAB_IsSurfaceValid(RAB_Surface surface)
{
    return surface.viewDepth != -BACKGROUND_DEPTH;
}

// 获取表面的世界空间位置。
float3 RAB_GetSurfaceWorldPos(RAB_Surface surface)
{
    return surface.worldPos;
}

void RAB_SetSurfaceWorldPos(inout RAB_Surface surface, float3 worldPos)
{
    surface.worldPos = worldPos;
}

// 获取表面的材质信息。
RAB_Material RAB_GetMaterial(RAB_Surface surface)
{
    return surface.material;
}

// 获取表面的法线方向。（材质法线）
float3 RAB_GetSurfaceNormal(RAB_Surface surface)
{
    return surface.normal;
}

// World space normal of the geometric primitive (i.e. without normal map)
float3 RAB_GetSurfaceGeoNormal(RAB_Surface surface)
{
    return surface.geoNormal;
}

// Vector from position to camera eye or previous bounce
float3 RAB_GetSurfaceViewDir(RAB_Surface surface)
{
    return surface.viewDir;
}

// 获取表面的深度信息。
// 它不必是严格意义上的线性深度（例如 viewPosition.z ），也可以是到摄像机的距离或主路径长度。
// 提供给 RTXDI_TemporalResampling 或 RTXDI_SpatioTemporalResampling 的运动矢量，其 .z 分量必须计算为同一表面在前一帧和当前帧的线性深度之差。
float RAB_GetSurfaceLinearDepth(RAB_Surface surface)
{
    return -surface.viewDepth;
}

float RAB_GetSurfaceRoughness(RAB_Surface surface)
{
    return surface.material.roughness;
}

void RAB_SetSurfaceNormal(inout RAB_Surface surface, float3 normal)
{
    surface.normal = normal;
}

float getSurfaceDiffuseProbability(RAB_Surface surface)
{
    float diffuseWeight = calcLuminance(surface.material.diffuseAlbedo);
    float specularWeight = calcLuminance(Schlick_Fresnel(surface.material.specularF0, dot(surface.viewDir, surface.normal)));
    float sumWeights = diffuseWeight + specularWeight;
    return sumWeights < 1e-7f ? 1.f : (diffuseWeight / sumWeights);
}

RAB_Surface GetGBufferSurface(int2 pixelPosition,
                              bool previousFrame,
                              float4x4 ViewToWorld,
                              float3 cameraGlobalPos,
                              Texture2D<float> depthTexture,
                              Texture2D<uint> normalsTexture,
                              Texture2D<uint> geoNormalsTexture)
{
    RAB_Surface surface = RAB_EmptySurface();

    if (any(pixelPosition >= gRectSize))
        return surface;

    surface.viewDepth = depthTexture[pixelPosition];

    if (surface.viewDepth == -BACKGROUND_DEPTH)
        return surface;

    surface.material = RAB_GetGBufferMaterial(pixelPosition, previousFrame);
    surface.geoNormal = octToNdirUnorm32(geoNormalsTexture[pixelPosition]);
    surface.normal = octToNdirUnorm32(normalsTexture[pixelPosition]);

    float2 pixelUv = float2(pixelPosition + 0.5) / gRectSize;
    
    float2 sampleUv = pixelUv + gJitter;

    float3 Xv = Geometry::ReconstructViewPosition(sampleUv, gCameraFrustum, surface.viewDepth, gOrthoMode);
    float3 X = Geometry::AffineTransform(ViewToWorld, Xv);
    surface.worldPos = X;

    surface.viewDir = normalize(cameraGlobalPos - surface.worldPos);
    surface.diffuseProbability = getSurfaceDiffuseProbability(surface);

    return surface;
}

// Load a sample from the previous G-buffer.
// 从当前或之前的 G-buffer 加载表面信息以构建 RAB_Surface 实例。
// 像素位置可能超出边界或为负值，在这种情况下，函数应返回无效表面。
// 应调用 RAB_GetGBufferMaterial 来填充其 RAB_Material 数据。
RAB_Surface RAB_GetGBufferSurface(int2 pixelPosition, bool previousFrame)
{
    if (previousFrame)
    {
        return GetGBufferSurface(
            pixelPosition,
            previousFrame,
            gViewToWorldPrev,
            gCameraGlobalPosPrev.xyz,
            t_PrevGBufferDepth,
            t_PrevGBufferNormals,
            t_PrevGBufferGeoNormals);
    }
    else
    {
        return GetGBufferSurface(
            pixelPosition,
            previousFrame,
            gViewToWorld,
            gCameraGlobalPos.xyz,
            t_GBufferDepth,
            t_GBufferNormals,
            t_GBufferGeoNormals);
    }
}

// 将世界空间方向转换为切线空间方向
float3 worldToTangent(RAB_Surface surface, float3 w)
{
    // reconstruct tangent frame based off worldspace normal
    // this is ok for isotropic BRDFs
    // for anisotropic BRDFs, we need a user defined tangent

    // 基于世界空间法线重建切线坐标系
    // 这对于各向同性BRDF是可以接受的
    // 对于各向异性BRDF，我们需要用户定义的切线坐标系
    float3 tangent;
    float3 bitangent;
    ConstructONB(surface.normal, tangent, bitangent);

    return float3(dot(bitangent, w), dot(tangent, w), dot(surface.normal, w));
}

// 将切线空间方向转换为世界空间方向
float3 tangentToWorld(RAB_Surface surface, float3 h)
{
    // reconstruct tangent frame based off worldspace normal
    // this is ok for isotropic BRDFs
    // for anisotropic BRDFs, we need a user defined tangent

    // 基于世界空间法线重建切线坐标系
    // 这对于各向同性BRDF是可以接受的
    // 对于各向异性BRDF，我们需要用户定义的切线坐标系
    float3 tangent;
    float3 bitangent;
    ConstructONB(surface.normal, tangent, bitangent);

    return bitangent * h.x + tangent * h.y + surface.normal * h.z;
}

float3 RAB_SurfaceEvaluateDeltaReflectionThroughput(RAB_Surface surface, float3 L)
{
    float3 N = surface.normal;
    float NoL = saturate(dot(L, N));
    float3 F0 = surface.material.specularF0;
    float3 F = Schlick_Fresnel(F0, NoL);
    return F / (1.f - surface.diffuseProbability);
}


/*
 * The following interface functions are split between BRDF and BSDF versions
 *
 * The BRDF versions
 *   - DO NOT handle delta bounces (mirrors & refraction)
 *   - Are used by ReSTIR DI
 *
 * The BSDF versions
 *   - DO handle delta bounces (mirrors & refraction)
 *   - Are used by ReSTIR PT
 *
 * This split allows ReSTIR DI, which does not support delta bounces,
 *   to be integrated and maintained separately from ReSTIR PT, which
 *   does support delta bounces.
 *
 */

/*
 *  BRDF surface functions used by ReSTIR DI
 */

/*
 * Evaluate the BRDF for the surface given the outgoing direction
 * Used by ReSTIR DI
 */
float3 RAB_SurfaceEvaluateBrdfTimesNoL(RAB_Surface surface, float3 L)
{
    float3 N = surface.normal;
    float3 V = surface.viewDir;

    if (dot(L, surface.geoNormal) <= 0)
        return 0;

    float d = Lambert(N, -L);
    float3 s = float3(0.0f, 0.0f, 0.0f);
    if (surface.material.roughness >= kMinRoughness)
        s = GGX_times_NdotL(V, L, N, max(surface.material.roughness, kMinRoughness), surface.material.specularF0);

    return d * surface.material.diffuseAlbedo + s;
}


// Output an importanced sampled reflection direction from the BRDF given the view
// Return true if the returned direction is above the surface

// 根据给定的视图，从 BRDF 中输出一个重要的采样反射方向
// 如果返回的方向在表面上方，则返回 true
// 对表面的双向反射分布函数进行重要性采样，并返回采样方向。
bool RAB_SurfaceImportanceSampleBrdf(RAB_Surface surface, inout RTXDI_RandomSamplerState rng, out float3 dir)
{
    float3 rand;
    rand.x = RTXDI_GetNextRandom(rng);
    rand.y = RTXDI_GetNextRandom(rng);
    rand.z = RTXDI_GetNextRandom(rng);
    if (rand.x < surface.diffuseProbability)
    {
        float pdf;
        float3 h = SampleCosHemisphere(rand.yz, pdf);
        dir = tangentToWorld(surface, h);
    }
    else
    {
        float3 Ve = normalize(worldToTangent(surface, surface.viewDir));
        float3 h = ImportanceSampleGGX_VNDF(rand.yz, max(surface.material.roughness, kMinRoughness), Ve, 1.0);
        h = normalize(h);
        dir = reflect(-surface.viewDir, tangentToWorld(surface, h));
    }

    return dot(surface.normal, dir) > 0.f;
}

// Return PDF wrt solid angle for the BRDF in the given dir
// 返回给定方向上 BRDF 的关于立体角的 PDF
float RAB_SurfaceEvaluateBrdfPdf(RAB_Surface surface, float3 dir)
{
    float cosTheta = saturate(dot(surface.normal, dir));
    float diffusePdf = cosTheta / M_PI;
    float specularPdf = ImportanceSampleGGX_VNDF_PDF(max(surface.material.roughness, kMinRoughness), surface.normal, surface.viewDir, dir);
    float pdf = cosTheta > 0.f ? lerp(specularPdf, diffusePdf, surface.diffuseProbability) : 0.f;
    return pdf;
}


float3 RAB_GetReflectedBrdfRadianceForSurface(float3 incomingRadianceLocation, float3 incomingRadiance, RAB_Surface surface)
{
    float3 L = normalize(incomingRadianceLocation - surface.worldPos);
    float3 brdfTimesNoL = RAB_SurfaceEvaluateBrdfTimesNoL(surface, L);
    return incomingRadiance * brdfTimesNoL;
}


float RAB_GetReflectedBrdfLuminanceForSurface(float3 incomingRadianceLocation, float3 incomingRadiance, RAB_Surface surface)
{
    return RTXDI_Luminance(RAB_GetReflectedBrdfRadianceForSurface(incomingRadianceLocation, incomingRadiance, surface));
}


/*
 * Evaluate the BSDF for the surface given the outgoing direction
 * isDelta assumes L is the mirror reflection of -surface.viewDir about its normal.
 * Used by ReSTIR PT
 */
float3 RAB_SurfaceEvaluateBsdfTimesNoL(RAB_Surface surface, float3 L, bool isDelta)
{
	float3 N = surface.normal;
    float3 V = surface.viewDir;

    if (dot(L, surface.geoNormal) <= 0)
        return 0;

    float d = 0;
    float3 s = float3(0.0f, 0.0f, 0.0f);
    if(isDelta)
    {
        float3 N = surface.normal;
        float NoL = saturate(dot(L, N));
        float3 F0 = surface.material.specularF0;
        float3 F = Schlick_Fresnel(F0, NoL);
        s = F / (1.f - surface.diffuseProbability);
    }
    else
    {
        d = Lambert(N, -L);
        if (surface.material.roughness >= kMinRoughness)
        s = GGX_times_NdotL(V, L, N, surface.material.roughness, surface.material.specularF0);
    }

	return d * surface.material.diffuseAlbedo + s;
}

/*
 * Importance sample the BSDF for the surface
 * Used by ReSTIR PT
 */
bool RAB_SurfaceImportanceSampleBsdf(RAB_Surface surface, inout RTXDI_RandomSamplerState rng, out float3 dir, out RTXDI_BrdfRaySampleProperties brsp)
{
    float3 rand;
    rand.x = RTXDI_GetNextRandom(rng);
    rand.y = RTXDI_GetNextRandom(rng);
    rand.z = RTXDI_GetNextRandom(rng);

    brsp.SetReflection();

    if (rand.x < surface.diffuseProbability)
    {
        float pdf;
        float3 h = SampleCosHemisphere(rand.yz, pdf);
        dir = tangentToWorld(surface, h);
        brsp.SetDiffuse();
        brsp.SetContinuous();
    }
    else
    {
        brsp.SetSpecular();
        // Perfect mirror reflection
        if(surface.material.roughness < kMinRoughness)
        {
            dir = reflect(-surface.viewDir, surface.normal);
            brsp.SetDelta();
        }
        // Glossy reflection
        else
        {
            float3 Ve = normalize(worldToTangent(surface, surface.viewDir));
            float3 h = ImportanceSampleGGX_VNDF(rand.yz, max(surface.material.roughness, kMinRoughness), Ve, 1.0);
            h = normalize(h);
            dir = reflect(-surface.viewDir, tangentToWorld(surface, h));
            brsp.SetContinuous();
        }
    }

    return dot(surface.normal, dir) > 0.f;
}

/*
 * Evaluate the PDF of the BSDF for the surface given the outgoing direction
 * Used by ReSTIR PT
 */
float RAB_SurfaceEvaluateBsdfPdf(RAB_Surface surface, float3 dir, RTXDI_BrdfRaySampleProperties brsp)
{
    float cosTheta = saturate(dot(surface.normal, dir));
    float diffusePdf = cosTheta / M_PI;
    float specularPdf = 0.0;
    if(surface.material.roughness >= kMinRoughness)
    {
        specularPdf = ImportanceSampleGGX_VNDF_PDF(max(surface.material.roughness, kMinRoughness), surface.normal, surface.viewDir, dir);
    }
    else if(brsp.IsDelta())
    {
        specularPdf = 1.0;
    }
    float pdf = cosTheta > 0.f ? lerp(specularPdf, diffusePdf, surface.diffuseProbability) : 0.f;
    return pdf;
}

/*
 * Multiply incoming radiance by the BSDF and cosine term
 * Used by ReSTIR PT
 */
float3 RAB_GetReflectedBsdfRadianceForSurface(float3 incomingRadianceLocation, float3 incomingRadiance, RAB_Surface surface)
{
    float3 L = normalize(incomingRadianceLocation - surface.worldPos);
	float3 brdfTimesNoL = RAB_SurfaceEvaluateBsdfTimesNoL(surface, L, false);
	return incomingRadiance * brdfTimesNoL;
}



#endif // RTXDI_RAB_SURFACE_HLSLI
