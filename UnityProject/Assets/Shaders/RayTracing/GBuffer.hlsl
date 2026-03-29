#include "Assets/Shaders/Include/Shared.hlsl"
#include "Assets/Shaders/Include/RayTracingShared.hlsl"

#include "Assets/Shaders/NRD/NRD.hlsli"
#include "Assets/Shaders/donut/utils.hlsli"
#pragma max_recursion_depth 1

// 运动矢量（Motion Vector），用于描述像素在当前帧与上一帧之间的运动，以及视深（ViewZ）和TAA遮罩信息。
RWTexture2D<float4> gOut_Mv;
// 视空间深度（ViewZ），即像素在视空间中的Z值。
RWTexture2D<float> gOut_ViewZ;
// 法线、粗糙度和材质ID的打包信息。用于后续的去噪和材质区分。
RWTexture2D<float4> gOut_Normal_Roughness;
// 基础色（BaseColor，已转为sRGB）和金属度（Metalness）。
RWTexture2D<float4> gOut_BaseColor_Metalness;

RWTexture2D<uint> gOut_GeoNormal;

// 直接自发光（Direct Emission），即材质的自发光分量。
RWTexture2D<float3> gOut_DirectEmission;

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

    gOut_Mv[pixelPos] = float4(motion, viewZAndTaaMask0); // IMPORTANT: keep viewZ before PSR ( needed for glass )

    // ViewZ
    float viewZ = Geometry::AffineTransform(gWorldToView, Xvirtual).z;
    viewZ = geometryProps0.IsMiss() ? Math::Sign(viewZ) * INF : viewZ;

    gOut_ViewZ[pixelPos] = viewZ;

    // Emission
    gOut_DirectEmission[pixelPos] = materialProps0.Lemi;

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

    gOut_Normal_Roughness[pixelPos] = NRD_FrontEnd_PackNormalAndRoughness(N, materialProps0.roughness, materialID);
    gOut_GeoNormal[pixelPos] = ndirToOctUnorm32(geometryProps0.N);
    gOut_BaseColor_Metalness[pixelPos] = float4(Color::ToSrgb(materialProps0.baseColor), materialProps0.metalness);
}
