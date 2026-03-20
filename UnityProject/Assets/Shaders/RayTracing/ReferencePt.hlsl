#include "Assets/Shaders/Include/Shared.hlsl"
#include "Assets/Shaders/Include/RayTracingShared.hlsl"

#include "Assets/Shaders/NRD/NRD.hlsli"

#pragma max_recursion_depth 1

// Output
RWTexture2D<float3> g_Output;

#define K_TWO_PI                6.283185307f

float3 RandomUnitVector()
{
    float z = Rng::Hash::GetFloat() * 2.0f - 1.0f;
    float a = Rng::Hash::GetFloat() * K_TWO_PI;
    float r = sqrt(1.0f - z * z);
    float x = r * cos(a);
    float y = r * sin(a);
    return float3(x, y, z);
}

float FresnelReflectAmountOpaque(float n1, float n2, float3 incident, float3 normal)
{
    // Schlick's aproximation
    float r0 = (n1 - n2) / (n1 + n2);
    r0 *= r0;
    float cosX = -dot(normal, incident);
    float x = 1.0 - cosX;
    float xx = x*x;
    return r0 + (1.0 - r0)*xx*xx*x;
}

[shader("raygeneration")]
void MainRayGenShader()
{
    uint2 pixelPos = DispatchRaysIndex().xy;

    float2 pixelUv = float2(pixelPos + 0.5) / gRectSize;
    float2 sampleUv = pixelUv + gJitter;

    if (pixelUv.x > 1.0 || pixelUv.y > 1.0)
    {
        #if( USE_DRS_STRESS_TEST == 1 )
        WriteResult(pixelPos, GARBAGE, GARBAGE, GARBAGE, GARBAGE);
        #endif

        return;
    }

    // Initialize RNG
    Rng::Hash::Initialize(pixelPos, gFrameIndex);

    //================================================================================================================================================================================
    // Primary ray
    //================================================================================================================================================================================

    float3 cameraRayOrigin = 0;
    float3 cameraRayDirection = 0;
    GetCameraRay(cameraRayOrigin, cameraRayDirection, sampleUv);

    GeometryProps geometryProps0;
    MaterialProps materialProps0;
    
    uint safeNet = 0;


    float3 radiance = float3(0, 0, 0);
    float3 throughput = float3(1, 1, 1);
    
    float3 rayOrigin = cameraRayOrigin;
    float3 rayDirection = cameraRayDirection;
    
    uint bounceIndexOpaque = 0;
    uint bounceIndexTransparent = 0;
    
    do
    {
        CastRay(rayOrigin, rayDirection, 0.0, 1000.0, GetConeAngleFromRoughness(0.0, 0.0),  GEOMETRY_ALL , geometryProps0, materialProps0);
        float3 reflectionRayDir =  reflect(rayDirection, materialProps0.N);
        float3 diffuseRayDir = normalize( materialProps0.N + RandomUnitVector() );
        float3 specularRayDir = lerp(reflectionRayDir, diffuseRayDir, materialProps0.roughness);
        
        float fresnelFactor =   FresnelReflectAmountOpaque(1.0, 1.7, rayDirection, materialProps0.N);
        float specularChance = lerp(materialProps0.metalness, 1, fresnelFactor * (1 - materialProps0.roughness));
        float doSpecular = Rng::Hash::GetFloat() < specularChance;
        float3 reflectedRayDir = lerp(diffuseRayDir, specularRayDir, doSpecular);
        float k = (doSpecular == 1) ? specularChance : 1 - specularChance;
        
        radiance += throughput * materialProps0.Lemi;
        throughput *= materialProps0.baseColor / max(0.001, k);
        
        rayOrigin = geometryProps0.GetXoffset(geometryProps0.N);
        rayDirection = reflectedRayDir;
    
    }
    while (bounceIndexOpaque <= 10  && ++safeNet < 100);
    
    // float3 prevRadiance = g_Output[pixelPos];
    //
    // float3 result = lerp(prevRadiance, radiance, 1.0f / float(g_ConvergenceStep + 1));
    //
    // g_Output[pixelPos] = result;
    
    
    g_Output[pixelPos] = radiance;
}
