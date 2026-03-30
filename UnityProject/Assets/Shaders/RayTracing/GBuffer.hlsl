#include "Assets/Shaders/Include/Shared.hlsl"
#include "Assets/Shaders/Include/RayTracingShared.hlsl"

#include "Assets/Shaders/NRD/NRD.hlsli"
#include "Assets/Shaders/donut/utils.hlsli"
#include "Assets/Shaders/donut/packing.hlsli"
#pragma max_recursion_depth 1

RWTexture2D<float> u_ViewDepth;
RWTexture2D<uint> u_DiffuseAlbedo;
RWTexture2D<uint> u_SpecularRough;
RWTexture2D<uint> u_Normals;
RWTexture2D<uint> u_GeoNormals;
RWTexture2D<float4> u_Emissive;
RWTexture2D<float4> u_MotionVectors;

float GetMaterialID(GeometryProps geometryProps, MaterialProps materialProps)
{
    bool isHair = geometryProps.Has(FLAG_HAIR);
    bool isMetal = materialProps.metalness > 0.5;

    return isHair ? MATERIAL_ID_HAIR : (isMetal ? MATERIAL_ID_METAL : MATERIAL_ID_DEFAULT);
}

//========================================================================================
// MAIN
//========================================================================================
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
    CastRay(cameraRayOrigin, cameraRayDirection, 0.0, 1000.0, GetConeAngleFromRoughness(0.0, 0.0), (gOnScreen == SHOW_INSTANCE_INDEX || gOnScreen == SHOW_NORMAL) ? GEOMETRY_ALL : FLAG_NON_TRANSPARENT, geometryProps0, materialProps0);

    //================================================================================================================================================================================
    // Primary surface replacement ( aka jump through mirrors )
    //================================================================================================================================================================================

    float3 X0 = geometryProps0.X;

    float viewZ0 = Geometry::AffineTransform(gWorldToView, geometryProps0.X).z;

    bool isTaa5x5 = geometryProps0.Has(FLAG_HAIR | FLAG_SKIN) || geometryProps0.IsMiss(); // switched TAA to "higher quality & slower response" mode
    float viewZAndTaaMask0 = abs(viewZ0) * FP16_VIEWZ_SCALE * (isTaa5x5 ? -1.0 : 1.0);

    //================================================================================================================================================================================
    // G-buffer ( guides )
    //================================================================================================================================================================================

    // Motion
    float3 Xvirtual = X0;
    float3 XvirtualPrev = Xvirtual + geometryProps0.Xprev - geometryProps0.X;
    float3 motion = GetMotion(Xvirtual, XvirtualPrev);

    u_MotionVectors[pixelPos] = float4(motion, viewZAndTaaMask0); // IMPORTANT: keep viewZ before PSR ( needed for glass )

    // ViewZ
    float viewZ = Geometry::AffineTransform(gWorldToView, Xvirtual).z;
    viewZ = geometryProps0.IsMiss() ? Math::Sign(viewZ) * INF : viewZ;

    u_ViewDepth[pixelPos] = viewZ;

    // Early out
    if (geometryProps0.IsMiss())
    {
        #if( USE_INF_STRESS_TEST == 1 )
        WriteResult(pixelPos, GARBAGE, GARBAGE, GARBAGE, GARBAGE);
        #endif

        return;
    }

    // Normal, roughness and material ID
    float3 N = materialProps0.N;
    float materialID = GetMaterialID(geometryProps0, materialProps0);
    
    
    float3 albedo, Rf0;
    BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps0.baseColor, materialProps0.metalness, albedo, Rf0 );
    
    u_DiffuseAlbedo[pixelPos] = Pack_R11G11B10_UFLOAT (albedo);
    u_SpecularRough[pixelPos] = Pack_R8G8B8A8_Gamma_UFLOAT(float4(Rf0, materialProps0.roughness));
    u_Normals[pixelPos] = ndirToOctUnorm32(materialProps0.N);
    u_GeoNormals[pixelPos] = ndirToOctUnorm32(geometryProps0.N);
    u_Emissive[pixelPos] = float4(materialProps0.Lemi, viewZAndTaaMask0); // viewZ is needed for glass ray tracing
}
