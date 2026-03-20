#include "Assets/Shaders/Include/Shared.hlsl"
#include "Assets/Shaders/Include/RayTracingShared.hlsl"

#include "Assets/Shaders/NRD/NRD.hlsli"

#pragma max_recursion_depth 1

// Output
RWTexture2D<float4> g_Output;

uint _ReferenceBounceNum;
uint g_ConvergenceStep;
float g_split;

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
    float xx = x * x;
    return r0 + (1.0 - r0) * xx * xx * x;
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

    GeometryProps geometryProps;
    MaterialProps materialProps;

    uint safeNet = 0;

    float3 radiance = float3(0, 0, 0);
    float3 throughput = float3(1, 1, 1);

    float3 rayOrigin = cameraRayOrigin;
    float3 rayDirection = cameraRayDirection;

    uint bounceIndexOpaque = 0;
    uint bounceIndexTransparent = 0;

    do
    {
        CastRay(rayOrigin, rayDirection, 0.0, 1000.0, GetConeAngleFromRoughness(0.0, 0.0), GEOMETRY_ALL, geometryProps, materialProps);
        float3 reflectionRayDir = reflect(rayDirection, materialProps.N);
        float3 diffuseRayDir = normalize(materialProps.N + RandomUnitVector());
        float3 specularRayDir = lerp(reflectionRayDir, diffuseRayDir, materialProps.roughness);

        float diffuseProbability = EstimateDiffuseProbability(geometryProps, materialProps);
        bool isDiffuse = Rng::Hash::GetFloat() < diffuseProbability;
        
        float doSpecular = !isDiffuse;
        float3 reflectedRayDir = lerp(diffuseRayDir, specularRayDir, doSpecular);
        float k = (doSpecular == 1) ? 1 - diffuseProbability : diffuseProbability;
        
        float3 albedo, Rf0;
        BRDF::ConvertBaseColorMetalnessToAlbedoRf0(materialProps.baseColor, materialProps.metalness, albedo, Rf0);

        albedo = lerp(albedo, Rf0, doSpecular);
        
        radiance += throughput * materialProps.Lemi;
        throughput *= albedo / max(0.001, k);

        if (!geometryProps.IsMiss())
        {
            bounceIndexOpaque++;
        }
        else
        {
            
            break;
        }

        rayOrigin = geometryProps.GetXoffset(reflectedRayDir,10);
        rayDirection = reflectedRayDir;
    }
    while (bounceIndexOpaque <= _ReferenceBounceNum && ++safeNet < 100);

    float3 prevRadiance = g_Output[pixelPos].xyz;
    
    radiance = ApplyExposure(radiance);
    
    radiance = Color::HdrToLinear_Uncharted(radiance);

    float3 result = lerp(prevRadiance, radiance, 1.0f / float(g_ConvergenceStep + 1));
    
    float alpha = pixelUv.x < g_split ? 1.0 : 0.0;

    g_Output[pixelPos] = float4(result, alpha);
}
