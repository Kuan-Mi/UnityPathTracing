#include "Assets/Shaders/Include/Shared.hlsl"
#include "Assets/Shaders/Include/RayTracingShared.hlsl"
#include "Assets/Shaders/NRD/NRD.hlsli"

#pragma max_recursion_depth 1

// Output
RWTexture2D<float4> g_Output;

uint _ReferenceBounceNum;
uint g_ConvergenceStep;
float g_split;

[shader("raygeneration")]
void MainRayGenShader()
{
    uint2 pixelPos = DispatchRaysIndex().xy;

    float2 pixelUv = float2(pixelPos + 0.5) / gRectSize;
    float2 sampleUv = pixelUv + gJitter;

    if (pixelUv.x > 1.0 || pixelUv.y > 1.0)
    {
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

    float3 radiance = float3(0, 0, 0);
    float3 throughput = float3(1, 1, 1);

    float3 rayOrigin = cameraRayOrigin;
    float3 rayDirection = cameraRayDirection;

    float2 coneAngle = GetConeAngleFromRoughness(0.0, 0.0);
    uint bounceIndexOpaque = 0;

    do
    {
        CastRay(rayOrigin, rayDirection, 0.0, 1000.0, coneAngle, GEOMETRY_ALL, geometryProps, materialProps);

        radiance += throughput * materialProps.Lemi;

        if (geometryProps.IsMiss())
        {
            // sky color 已经计入了 Lemi 中，这里不用额外添加了
            break;
        }

        float diffuseProbability = EstimateDiffuseProbability(geometryProps, materialProps);

        bool isDiffuse = Rng::Hash::GetFloat() < diffuseProbability;

        rayDirection = GenerateRayAndUpdateThroughput(geometryProps, materialProps, throughput, isDiffuse, Rng::Hash::GetFloat2());
        throughput /= isDiffuse ? diffuseProbability : (1.0 - diffuseProbability);

        rayOrigin = geometryProps.X + rayDirection * 0.01;
        bounceIndexOpaque += 1;
    }
    while (bounceIndexOpaque <= _ReferenceBounceNum);

    float3 prev_radiance = g_Output[pixelPos].xyz;

    radiance = ApplyExposure(radiance);

    radiance = Color::HdrToLinear_Uncharted(radiance);

    float3 result = lerp(prev_radiance, radiance, 1.0f / float(g_ConvergenceStep + 1));

    float alpha = pixelUv.x < g_split ? 1.0 : 0.0;

    g_Output[pixelPos] = float4(result, alpha);
}
