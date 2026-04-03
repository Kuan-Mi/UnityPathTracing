// #include "Assets/Shaders/Include/LightRayTracingShared.hlsl"
#define SECONDARY_SURFACE_PAYLOAD 

#include "../ShaderParameters.hlsl" 

#include "Assets/Shaders/RayTracing/RtxdiApplicationBridge/RAB_Buffers.hlsl"

#include "Assets/Shaders/RayTracing/RtxdiApplicationBridge/RAB_LightInfo.hlsl"
#include "Assets/Shaders/RayTracing/RtxdiApplicationBridge/RAB_LightSample.hlsl"
#include "Assets/Shaders/RayTracing/RtxdiApplicationBridge/RAB_Material.hlsl"
#include "Assets/Shaders/RayTracing/RtxdiApplicationBridge/RAB_RayPayload.hlsl"
#include "Assets/Shaders/RayTracing/RtxdiApplicationBridge/RAB_RTShaders.hlsl"
#include "Assets/Shaders/RayTracing/RtxdiApplicationBridge/RAB_SpatialHelpers.hlsl"
#include "Assets/Shaders/RayTracing/RtxdiApplicationBridge/RAB_Surface.hlsl"

#include <Assets/Shaders/RTXDI/DI/Reservoir.hlsl>
#include "Assets/Shaders/RayTracing/ShadingHelpers.hlsl"
#include "Assets/Shaders/Include/Payload.hlsl"
#include "Assets/Shaders/RTXDI/DI/ReservoirStorage.hlsl"

#pragma max_recursion_depth 1

RaytracingAccelerationStructure gWorldTlas;


uint ToRayFlag2(uint flag)
{
    if (flag == FLAG_TRANSPARENT)
        return RAY_FLAG_CULL_OPAQUE;
    else
        return RAY_FLAG_NONE;
}

[shader("miss")]
void SECONDARY_SURFACE_PAYLOADShader(inout SecondarySurfacePayload payload : SV_RayPayload)
{
    payload.hitT = INF;
}

SecondarySurfacePayload GetSec(float3 origin, float3 direction, float Tmin, float Tmax, uint mask)
{
    RayDesc rayDesc;
    rayDesc.Origin = origin;
    rayDesc.Direction = direction;
    rayDesc.TMin = Tmin;
    rayDesc.TMax = Tmax;

    SecondarySurfacePayload payload = (SecondarySurfacePayload)0;

    TraceRay(gWorldTlas, ToRayFlag2(mask), mask, 0, 1, 0, rayDesc, payload);

    return payload;
}


//========================================================================================
// MAIN
//========================================================================================
[shader("raygeneration")]
void MainRayGenShader()
{
    uint2 GlobalIndex = DispatchRaysIndex().xy;


    uint2 pixelPosition = RTXDI_ReservoirPosToPixelPos(GlobalIndex, g_Const.runtimeParams.activeCheckerboardField);

    RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);

    if (!RAB_IsSurfaceValid(surface))
        return;

    RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(GlobalIndex, g_Const.runtimeParams.frameIndex, RTXDI_GI_GENERATE_INITIAL_SAMPLES_RANDOM_SEED);

    float3 tangent, bitangent;
    branchlessONB(surface.normal, tangent, bitangent);


    float distance = max(1, 0.1 * length(surface.worldPos - gCameraGlobalPos));

    RayDesc ray;
    ray.TMin = 0.001f * distance;
    ray.TMax = 1000;


    float2 Rand;
    Rand.x = RTXDI_GetNextRandom(rng);
    Rand.y = RTXDI_GetNextRandom(rng);

    float3 V = normalize(gCameraGlobalPos.xyz - surface.worldPos);


    bool isSpecularRay = false;
    bool isDeltaSurface = surface.material.roughness < kMinRoughness;
    float specular_PDF;
    float3 BRDF_over_PDF;
    float overall_PDF;

    {
        float3 specularDirection;
        float3 specular_BRDF_over_PDF;
        {
            float3 Ve = float3(dot(V, tangent), dot(V, bitangent), dot(V, surface.normal));
            float3 He = sampleGGX_VNDF(Ve, surface.material.roughness, Rand);
            float3 H = isDeltaSurface ? surface.normal : normalize(He.x * tangent + He.y * bitangent + He.z * surface.normal);
            specularDirection = reflect(-V, H);

            float HoV = saturate(dot(H, V));
            float NoV = saturate(dot(surface.normal, V));
            float3 F = Schlick_Fresnel(surface.material.specularF0, HoV);
            float G1 = isDeltaSurface ? 1.0 : (NoV > 0) ? G1_Smith(surface.material.roughness, NoV) : 0;
            specular_BRDF_over_PDF = F * G1;
        }

        float3 diffuseDirection;
        float diffuse_BRDF_over_PDF;
        {
            float solidAnglePdf;
            float3 localDirection = sampleCosHemisphere(Rand, solidAnglePdf);
            diffuseDirection = tangent * localDirection.x + bitangent * localDirection.y + surface.normal * localDirection.z;
            diffuse_BRDF_over_PDF = 1.0;
        }

        // Ignores PDF of specular or diffuse
        // Chooses PDF based on relative luminance
        specular_PDF = saturate(calcLuminance(specular_BRDF_over_PDF) /
            calcLuminance(specular_BRDF_over_PDF + diffuse_BRDF_over_PDF * surface.material.diffuseAlbedo));

        isSpecularRay = RTXDI_GetNextRandom(rng) < specular_PDF;

        if (isSpecularRay)
        {
            ray.Direction = specularDirection;
            BRDF_over_PDF = specular_BRDF_over_PDF / specular_PDF;
        }
        else
        {
            ray.Direction = diffuseDirection;
            BRDF_over_PDF = diffuse_BRDF_over_PDF / (1.0 - specular_PDF);
        }

        const float specularLobe_PDF = ImportanceSampleGGX_VNDF_PDF(surface.material.roughness, surface.normal, V, ray.Direction);
        const float diffuseLobe_PDF = saturate(dot(ray.Direction, surface.normal)) / c_pi;

        // For delta surfaces, we only pass the diffuse lobe to ReSTIR GI, and this pdf is for that.
        overall_PDF = isDeltaSurface ? diffuseLobe_PDF : lerp(diffuseLobe_PDF, specularLobe_PDF, specular_PDF);
    }


    if (dot(surface.geoNormal, ray.Direction) <= 0.0)
    {
        BRDF_over_PDF = 0.0;
        ray.TMax = 0;
    }

    ray.Origin = surface.worldPos;


    float3 radiance = 0;

    // RayPayload payload = (RayPayload)0;
    // payload.instanceID = ~0u;
    // payload.throughput = 1.0;


    float3 throughput = 1.0;


    SecondarySurfacePayload payload = GetSec(ray.Origin, ray.Direction, ray.TMin, ray.TMax, (gOnScreen == SHOW_INSTANCE_INDEX || gOnScreen == SHOW_NORMAL) ? GEOMETRY_ALL : FLAG_NON_TRANSPARENT);


    // gOut_DirectLighting[pixelPosition] = materialProps0.Lemi;
    // gOut_DirectLighting[pixelPosition] = surface.normal;

    uint gbufferIndex = RTXDI_ReservoirPositionToPointer(g_Const.restirGI.reservoirBufferParams, GlobalIndex, 0);

    struct
    {
        float3 position;
        float3 normal;
        float3 diffuseAlbedo;
        float3 specularF0;
        float roughness;
        bool isEnvironmentMap;
    } secondarySurface;

    const bool includeEmissiveComponent = g_Const.brdfPT.enableIndirectEmissiveSurfaces || (isSpecularRay && isDeltaSurface);


    if (!payload.IsMiss())
    {
        if (includeEmissiveComponent)
            radiance += payload.Lemi;

        float3 pos = ray.Origin + ray.Direction * payload.hitT;

        secondarySurface.position = pos;
        secondarySurface.normal = Packing::DecodeUnitVector(payload.normal);

        float2 rAm = Packing::UintToRg16f(payload.roughnessAndMetalness);

        float roughness = rAm.x;
        float metalness = rAm.y;
        float3 baseColor = Packing::UintToRgba(payload.baseColor, 8, 8, 8, 8).xyz;

        BRDF::ConvertBaseColorMetalnessToAlbedoRf0(baseColor, metalness, secondarySurface.diffuseAlbedo, secondarySurface.specularF0);
        secondarySurface.roughness = roughness;
        secondarySurface.isEnvironmentMap = false;
    }
    else
    {
        // if (g_Const.sceneConstants.enableEnvironmentMap && includeEmissiveComponent)
        if (includeEmissiveComponent)
        {
            // float3 environmentRadiance = GetEnvironmentRadiance(ray.Direction);
            float3 environmentRadiance = GetSkyIntensity(ray.Direction);
            radiance += environmentRadiance;
        }

        secondarySurface.position = ray.Origin + ray.Direction * DISTANT_LIGHT_DISTANCE;
        secondarySurface.normal = -ray.Direction;
        secondarySurface.diffuseAlbedo = 0;
        secondarySurface.specularF0 = 0;
        secondarySurface.roughness = 0;
        secondarySurface.isEnvironmentMap = true;
    }

    if (g_Const.enableBrdfIndirect)
    {
        SecondaryGBufferData secondaryGBufferData = (SecondaryGBufferData)0;
        secondaryGBufferData.worldPos = secondarySurface.position;
        secondaryGBufferData.normal = ndirToOctUnorm32(secondarySurface.normal);
        secondaryGBufferData.throughputAndFlags = Pack_R16G16B16A16_FLOAT(float4(throughput * BRDF_over_PDF, 0));
        secondaryGBufferData.diffuseAlbedo = Pack_R11G11B10_UFLOAT(secondarySurface.diffuseAlbedo);
        secondaryGBufferData.specularAndRoughness = Pack_R8G8B8A8_Gamma_UFLOAT(float4(secondarySurface.specularF0, secondarySurface.roughness));

        if (g_Const.brdfPT.enableReSTIRGI)
        {
            if (isSpecularRay && isDeltaSurface)
            {
                // Special case for specular rays on delta surfaces: they bypass ReSTIR GI and are shaded
                // entirely in the ShadeSecondarySurfaces pass, so they need the right throughput here.
            }
            else
            {
                // BRDF_over_PDF will be multiplied after resampling GI reservoirs.
                secondaryGBufferData.throughputAndFlags = Pack_R16G16B16A16_FLOAT(float4(throughput, 0));
            }

            // The emission from the secondary surface needs to be added when creating the initial
            // GI reservoir sample in ShadeSecondarySurface.hlsl. It need to be stored separately.
            secondaryGBufferData.emission = radiance;
            radiance = 0;

            secondaryGBufferData.pdf = overall_PDF;
        }

        uint flags = 0;
        if (isSpecularRay) flags |= kSecondaryGBuffer_IsSpecularRay;
        if (isDeltaSurface) flags |= kSecondaryGBuffer_IsDeltaSurface;
        if (secondarySurface.isEnvironmentMap) flags |= kSecondaryGBuffer_IsEnvironmentMap;
        secondaryGBufferData.throughputAndFlags.y |= flags << 16;

        u_SecondaryGBuffer[gbufferIndex] = secondaryGBufferData;
    }


    if (any(radiance > 0) || !g_Const.enableBrdfAdditiveBlend)
    {
        radiance *= throughput;

        float3 diffuse = isSpecularRay ? 0.0 : radiance * BRDF_over_PDF;
        float3 specular = isSpecularRay ? radiance * BRDF_over_PDF : 0.0;
        float diffuseHitT = payload.hitT;
        float specularHitT = payload.hitT;

        specular = DemodulateSpecular(surface.material.specularF0, specular);

        float3 finalColor = (diffuse * surface.material.diffuseAlbedo) + specular;

        // finalColor += gIn_EmissiveLighting[pixelPosition];
        finalColor *= gExposure;

        gOut_DirectLighting[pixelPosition] += finalColor;

        StoreShadingOutput(finalColor, pixelPosition, !g_Const.enableBrdfAdditiveBlend, !g_Const.enableBrdfIndirect);
    }
}
