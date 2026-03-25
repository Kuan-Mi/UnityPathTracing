#include "Assets/Shaders/Include/Shared.hlsl"
#include "Assets/Shaders/Include/RayTracingShared.hlsl"

#include "Assets/Shaders/NRD/NRD.hlsli"

#pragma max_recursion_depth 1
Texture2D<float4> gOut_Mv;
Texture2D<float> gOut_ViewZ;
Texture2D<float4> gOut_Normal_Roughness;
Texture2D<float4> gOut_BaseColor_Metalness;
Texture2D<uint> gOut_GeoNormal;


// RTXDI：上一帧 GBuffer
Texture2D<float> gIn_PrevViewZ;
Texture2D<float4> gIn_PrevNormalRoughness;
Texture2D<float4> gIn_PrevBaseColorMetalness;
Texture2D<uint> gIn_PrevGeoNormal;

Texture2D<float3> gIn_EmissiveLighting;

RWTexture2D<float3> gOut_DirectLighting;

RWTexture2D<int2> u_TemporalSamplePositions;

Texture2D t_LocalLightPdfTexture;


#include "Assets/Shaders/Rtxdi/RtxdiParameters.h"
#include "Assets/Shaders/Rtxdi/DI/ReSTIRDIParameters.h"
#include "Assets/Shaders/donut/packing.hlsli"
#include "Assets/Shaders/donut/brdf.hlsli"

#include "ResamplingConstants.hlsl"

#include "../RtxdiApplicationBridge/RtxdiApplicationBridge.hlsl"
#include <Assets/Shaders/RTXDI/DI/SpatialResampling.hlsl>


struct SplitBrdf
{
    float demodulatedDiffuse;
    float3 specular;
};

SplitBrdf EvaluateBrdf(RAB_Surface surface, float3 samplePosition)
{
    float3 N = surface.normal;
    float3 V = surface.viewDir;
    float3 L = normalize(samplePosition - surface.worldPos);

    SplitBrdf brdf;
    brdf.demodulatedDiffuse = Lambert(surface.normal, -L);
    if (surface.material.roughness == 0)
        brdf.specular = 0;
    else
        brdf.specular = GGX_times_NdotL(V, L, surface.normal, max(surface.material.roughness, kMinRoughness), surface.material.specularF0);
    return brdf;
}

bool ShadeSurfaceWithLightSample(
    inout RTXDI_DIReservoir reservoir,
    RAB_Surface surface,
    RAB_LightSample lightSample,
    bool previousFrameTLAS,
    bool enableVisibilityReuse,
    out float3 diffuse,
    out float3 specular,
    out float lightDistance)
{
    diffuse = 0;
    specular = 0;
    lightDistance = 0;

    if (lightSample.solidAnglePdf <= 0)
        return false;

    bool needToStore = false;
    if (g_Const.restirDI.shadingParams.enableFinalVisibility)
    {
        float3 visibility = 0;
        bool visibilityReused = false;

        if (g_Const.restirDI.shadingParams.reuseFinalVisibility && enableVisibilityReuse)
        {
            RTXDI_VisibilityReuseParameters rparams;
            rparams.maxAge = g_Const.restirDI.shadingParams.finalVisibilityMaxAge;
            rparams.maxDistance = g_Const.restirDI.shadingParams.finalVisibilityMaxDistance;

            visibilityReused = RTXDI_GetDIReservoirVisibility(reservoir, rparams, visibility);
        }

        if (!visibilityReused)
        {
            visibility = GetFinalVisibility(surface, lightSample.position);
            // visibility = GetFinalVisibility(SceneBVH, surface, lightSample.position);
            RTXDI_StoreVisibilityInDIReservoir(reservoir, visibility, g_Const.restirDI.temporalResamplingParams.discardInvisibleSamples);
            needToStore = true;
        }

        lightSample.radiance *= visibility;
    }

    lightSample.radiance *= RTXDI_GetDIReservoirInvPdf(reservoir) / lightSample.solidAnglePdf;

    if (any(lightSample.radiance > 0))
    {
        SplitBrdf brdf = EvaluateBrdf(surface, lightSample.position);

        diffuse = brdf.demodulatedDiffuse * lightSample.radiance;
        specular = brdf.specular * lightSample.radiance;

        lightDistance = length(lightSample.position - surface.worldPos);
    }

    return needToStore;
}


float3 DemodulateSpecular(float3 surfaceSpecularF0, float3 specular)
{
    return specular / max(0.01, surfaceSpecularF0);
}

RWBuffer<uint2> u_RisBuffer;

#define RTXDI_RIS_BUFFER u_RisBuffer

[shader("raygeneration")]
void MainRayGenShader()
{
    uint2 GlobalIndex = DispatchRaysIndex().xy;

    const RTXDI_RuntimeParameters params = g_Const.runtimeParams;

    uint2 pixelPosition = RTXDI_ReservoirPosToPixelPos(GlobalIndex, params.activeCheckerboardField);


    RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);

    RTXDI_DIReservoir reservoir = RTXDI_LoadDIReservoir(g_Const.restirDI.reservoirBufferParams, GlobalIndex, g_Const.restirDI.bufferIndices.shadingInputBufferIndex);

    float3 diffuse = 0;
    float3 specular = 0;
    float lightDistance = 0;
    float2 currLuminance = 0;


    if (RTXDI_IsValidDIReservoir(reservoir))
    {
        RAB_LightInfo lightInfo = RAB_LoadLightInfo(RTXDI_GetDIReservoirLightIndex(reservoir), false);

        RAB_LightSample lightSample = RAB_SamplePolymorphicLight(lightInfo, surface, RTXDI_GetDIReservoirSampleUV(reservoir));

        bool needToStore = ShadeSurfaceWithLightSample(reservoir, surface, lightSample,
                                                       /* previousFrameTLAS = */ false, /* enableVisibilityReuse = */ true, diffuse, specular, lightDistance);

        // currLuminance = float2(calcLuminance(diffuse * surface.material.diffuseAlbedo), calcLuminance(specular));

        specular = DemodulateSpecular(surface.material.specularF0, specular);

        float3 finalColor = ShadeSurfaceWithLightSample(lightSample, surface)  * RTXDI_GetDIReservoirInvPdf(reservoir);
        gOut_DirectLighting[pixelPosition] = finalColor + gIn_EmissiveLighting[pixelPosition];

        // gOut_DirectLighting[pixelPosition] = diffuse + specular;

        if (needToStore)
        {
            RTXDI_StoreDIReservoir(reservoir, g_Const.restirDI.reservoirBufferParams, pixelPosition, g_Const.restirDI.bufferIndices.shadingInputBufferIndex);
        }
    }
    else
    {
        gOut_DirectLighting[pixelPosition] = float3(0,0,0) + gIn_EmissiveLighting[pixelPosition];
    }
    
    // uint tileSize = g_Const.localLightsRISBufferSegmentParams.tileSize; // 通常是 128 或 256
    // uint tileCount = g_Const.localLightsRISBufferSegmentParams.tileCount;
    //
    // // 2. 确定每个 Tile 在屏幕上显示的尺寸 (假设显示为正方形)
    // // 如果 tileSize 是 256，则每个块是 16x16；如果是 128，则大约是 11x11
    // uint side = (uint)sqrt((float)tileSize); 
    //
    // // 3. 计算当前像素属于第几个 Tile，以及是该 Tile 里的第几个采样点
    // uint2 tileGridPos = pixelPosition / side;  // 屏幕上 Tile 的行列坐标
    // uint2 inTilePos = pixelPosition % side;    // 在当前 Tile 方块内的像素偏移
    //
    // // 计算 tileIndex：假设横向平铺
    // // 我们需要知道屏幕宽度方向能放多少个 Tile 块
    // // 注意：RTXDI_GetScreenSize() 需要替换为你引擎中获取分辨率的函数或常量
    // uint tilesPerRow = 785 / side; 
    //
    // uint tileIndex = tileGridPos.y * tilesPerRow + tileGridPos.x;
    // uint sampleInTile = inTilePos.y * side + inTilePos.x;
    //
    //
    //
    // // gOut_DirectLighting[pixelPosition] = float3(tileGridPos / 16.0f, 0);
    // // gOut_DirectLighting[pixelPosition] = tileSize;
    //
    //
    // // 4. 边界检查：确保不越界
    // if (tileIndex < tileCount && sampleInTile < tileSize)
    // {
    //     // 计算 Buffer 指针
    //     uint risBufferPtr = sampleInTile + tileIndex * tileSize;
    //
    //     // 读取数据
    //     uint2 risData = RTXDI_RIS_BUFFER[risBufferPtr];
    //     uint lightIndex = risData.x & ~RTXDI_LIGHT_COMPACT_BIT;
    //     float invSourcePdf = asfloat(risData.y);
    //
    //     // 5. 可视化处理
    //     if (lightIndex == 0 && invSourcePdf == 0)
    //     {
    //         // 这里的像素可能是空的（没抽中灯）
    //         gOut_DirectLighting[pixelPosition] = float3(0.05, 0.05, 0.05); 
    //     }
    //     else
    //     {
    //         // 使用哈希函数将 lightIndex 转换为鲜艳的颜色，以便区分不同的灯
    //         float3 color;
    //         color.r = frac(sin(float(lightIndex) * 12.9898) * 43758.5453);
    //         color.g = frac(sin(float(lightIndex) * 78.233) * 43758.5453);
    //         color.b = frac(sin(float(lightIndex) * 45.164) * 43758.5453);
    //     
    //         // 为了区分 Tile 边界，给每个块加个微小的边框感
    //         if (inTilePos.x == 0 || inTilePos.y == 0) color *= 0.5;
    //
    //         gOut_DirectLighting[pixelPosition] = color;
    //     }
    // }
    // else
    // {
    //     // 超出 Tile 总数或屏幕范围的部分显示为黑色
    //     gOut_DirectLighting[pixelPosition] = float3(0, 0, 0);
    // }
    
    
    
    
    
    // float3 origin = gCameraGlobalPos;
    // float3 dir = normalize(surface.worldPos - origin);
    // uint o_lightIndex;
    // float2 o_randXY;
    //
    // bool hit = RAB_TraceRayForLocalLight(origin, dir, 0, 1000, o_lightIndex, o_randXY);
    //
    //
    // RAB_LightInfo lightInfo = RAB_LoadLightInfo(0, false);
    // RAB_LightSample lightSample = RAB_SamplePolymorphicLight(lightInfo, surface, o_randXY);
    //
    // float3 cc = lightInfo.center;
    // cc = lightSample.normal;
    //
    //
    // float3 finalColor = ShadeSurfaceWithLightSample(lightSample, surface) ;
    //
    // gOut_DirectLighting[pixelPosition] = cc;
    
    
    //
    //
    // // gOut_DirectLighting[pixelPosition] = float3(o_randXY,0);
    // //
    // if (o_lightIndex == RTXDI_InvalidLightIndex)
    // {
    //     gOut_DirectLighting[pixelPosition] = 0;
    // }
    // else
    // {
    //
    //     uint2 pdfTexturePosition = RTXDI_LinearIndexToZCurve(o_lightIndex);
    //     float power = t_LocalLightPdfTexture.Load(int3(pdfTexturePosition >> 1 , 1));
    //     
    //     float3 color ; 
    //     color = power;
    //     
    //     gOut_DirectLighting[pixelPosition] = color;
    //     
    //     // gOut_DirectLighting[pixelPosition] = o_lightIndex / 12.0;
    //     
    //     
    //     // RAB_LightInfo lightInfo = RAB_LoadLightInfo(o_lightIndex, false);
    //     // RAB_LightSample lightSample = RAB_SamplePolymorphicLight(lightInfo,
    //     //                                                          surface, o_randXY);
    //     //
    //     // gOut_DirectLighting[pixelPosition] = lightSample.radiance;
    // }
}
