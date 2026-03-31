// #include "Assets/Shaders/Include/LightRayTracingShared.hlsl"
# define USE_FULL_RAY 

#include "Assets/Shaders/RayTracing/RtxdiApplicationBridge/RtxdiApplicationBridge.hlsl"
#include <Assets/Shaders/RTXDI/DI/Reservoir.hlsl>


#pragma max_recursion_depth 1


float3 DemodulateSpecular(float3 surfaceSpecularF0, float3 specular)
{
    return specular / max(0.01, surfaceSpecularF0);
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
    
    RAB_RandomSamplerState rng = RAB_InitRandomSampler(GlobalIndex, 5);
    
    float3 tangent, bitangent;
    branchlessONB(surface.normal, tangent, bitangent);
    
    
    float distance = max(1, 0.1 * length(surface.worldPos - gCameraGlobalPos));

    RayDesc ray;
    ray.TMin = 0.001f * distance;
    ray.TMax = 1000;

    
    float2 Rand;
    Rand.x = RAB_GetNextRandom(rng);
    Rand.y = RAB_GetNextRandom(rng);

    float3 V = normalize(gCameraGlobalPos - surface.worldPos);
    
    
    bool isSpecularRay = false;
    bool isDeltaSurface = surface.material.roughness == 0;
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

        specular_PDF = saturate(calcLuminance(specular_BRDF_over_PDF) /
            calcLuminance(specular_BRDF_over_PDF + diffuse_BRDF_over_PDF * surface.material.diffuseAlbedo));

        isSpecularRay = RAB_GetNextRandom(rng) < specular_PDF;

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
    
    GeometryProps geometryProps0;
    MaterialProps materialProps0;
    CastRay(ray.Origin, ray.Direction, ray.TMin, ray.TMax, GetConeAngleFromRoughness(0.0, 0.0), (gOnScreen == SHOW_INSTANCE_INDEX || gOnScreen == SHOW_NORMAL) ? GEOMETRY_ALL : FLAG_NON_TRANSPARENT, geometryProps0, materialProps0);

    
    
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

    
    if (!geometryProps0.IsMiss())
    {
        if (includeEmissiveComponent)
            radiance += materialProps0.Lemi;
        
        secondarySurface.position = geometryProps0.X;
        secondarySurface.normal = geometryProps0.N;
        BRDF::ConvertBaseColorMetalnessToAlbedoRf0(materialProps0.baseColor, materialProps0.metalness, secondarySurface.diffuseAlbedo, secondarySurface.specularF0);
        secondarySurface.roughness = materialProps0.roughness;
        secondarySurface.isEnvironmentMap = false;
        
    }else
    {
        // if (g_Const.sceneConstants.enableEnvironmentMap && includeEmissiveComponent)
        // {
        //     float3 environmentRadiance = GetEnvironmentRadiance(ray.Direction);
        //     radiance += environmentRadiance;
        // }
        
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
        float diffuseHitT = geometryProps0.hitT;
        float specularHitT = geometryProps0.hitT;

        specular = DemodulateSpecular(surface.material.specularF0, specular);

        float3 finalColor = (diffuse * surface.material.diffuseAlbedo) + specular;

        // finalColor += gIn_EmissiveLighting[pixelPosition];
        finalColor *= gExposure;

        gOut_DirectLighting[pixelPosition] = finalColor;
        
        
        // StoreShadingOutput(GlobalIndex, pixelPosition,
        //     surface.viewDepth, surface.material.roughness, diffuse, specular, geometryProps0.hitT, !g_Const.enableBrdfAdditiveBlend, !g_Const.enableBrdfIndirect);
    }
}
