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
Texture2D<uint>   gIn_PrevGeoNormal;

RWTexture2D<float3> gOut_DirectLighting;


#include "Assets/Shaders/Rtxdi/RtxdiParameters.h"
#include "Assets/Shaders/donut/packing.hlsli"
#include "Assets/Shaders/donut/brdf.hlsli"


struct ResamplingConstants
{
    RTXDI_RuntimeParameters runtimeParams;
    RTXDI_LightBufferParameters lightBufferParams;
    RTXDI_ReservoirBufferParameters restirDIReservoirBufferParams;

    uint frameIndex;
    uint numInitialSamples;
    uint numSpatialSamples;
    uint useAccurateGBufferNormal;

    uint numInitialBRDFSamples;
    float brdfCutoff;
    uint2 pad2;

    uint enableResampling;
    uint unbiasedMode;
    uint inputBufferIndex;
    uint outputBufferIndex;
};

RWStructuredBuffer<ResamplingConstants> ResampleConstants;
#define g_Const ResampleConstants[0]


// RTXDI resources
StructuredBuffer<RAB_LightInfo> t_LightDataBuffer;
Buffer<float2> t_NeighborOffsets;

RWStructuredBuffer<RTXDI_PackedDIReservoir> u_LightReservoirs;

#define RTXDI_LIGHT_RESERVOIR_BUFFER u_LightReservoirs
#define RTXDI_NEIGHBOR_OFFSETS_BUFFER t_NeighborOffsets

#define BACKGROUND_DEPTH 65504.f

#define RTXDI_ENABLE_PRESAMPLING 0

#include "RtxdiApplicationBridge/RtxdiApplicationBridge.hlsl"
#include "Assets/Shaders/RTXDI/DI/InitialSampling.hlsl"
#include <Assets/Shaders/RTXDI/DI/SpatioTemporalResampling.hlsl>

[shader("raygeneration")]
void MainRayGenShader()
{
    uint2 pixelPos = DispatchRaysIndex().xy;
    
    // Test RTXDI

    const RTXDI_LightBufferParameters lightBufferParams = g_Const.lightBufferParams;

    RAB_Surface primarySurface =  RAB_GetGBufferSurface(pixelPos,false);

    RTXDI_DIReservoir reservoir = RTXDI_EmptyDIReservoir();

    RAB_RandomSamplerState rng = RAB_InitRandomSampler(pixelPos, 1);

    RTXDI_SampleParameters sampleParams = RTXDI_InitSampleParameters(
        g_Const.numInitialSamples, // local light samples 
        // 局部光源采样数
        0, // infinite light samples
        // 无限光源采样数
        0, // environment map samples
        // 环境贴图采样数
        g_Const.numInitialBRDFSamples,
        g_Const.brdfCutoff,
        0.001f);

    // Generate the initial sample
    RAB_LightSample lightSample = RAB_EmptyLightSample();
    RTXDI_DIReservoir localReservoir = RTXDI_SampleLocalLights(rng, rng, primarySurface, sampleParams, ReSTIRDI_LocalLightSamplingMode_UNIFORM, lightBufferParams.localLightBufferRegion, lightSample);
    RTXDI_CombineDIReservoirs(reservoir, localReservoir, 0.5, localReservoir.targetPdf);


    // Resample BRDF samples.
    RAB_LightSample brdfSample = RAB_EmptyLightSample();
    RTXDI_DIReservoir brdfReservoir = RTXDI_SampleBrdf(rng, primarySurface, sampleParams, lightBufferParams, brdfSample);
    bool selectBrdf = RTXDI_CombineDIReservoirs(reservoir, brdfReservoir, RAB_GetNextRandom(rng), brdfReservoir.targetPdf);
    if (selectBrdf)
    {
        lightSample = brdfSample;
    }

    RTXDI_FinalizeResampling(reservoir, 1.0, 1.0);
    reservoir.M = 1;

    // BRDF was generated with a trace so no need to trace visibility again
    // BRDF 是通过追踪生成的，因此无需再次追踪可见性
    if (RTXDI_IsValidDIReservoir(reservoir) && !selectBrdf)
    // if (RTXDI_IsValidDIReservoir(reservoir))
    {
        // See if the initial sample is visible from the surface
        // 查看初始样本对于表面是否可见
        if (!RAB_GetConservativeVisibility(primarySurface, lightSample))
        {
            // If not visible, discard the sample (but keep the M)
            // 如果不可见，则丢弃样本（但保留 M 值）
            RTXDI_StoreVisibilityInDIReservoir(reservoir, 0, true);
        }
    }

    bool foundTemporalSurface = true;
    float3 debugColor = 0;
    if (g_Const.enableResampling)
    {
        debugColor = 1;
        float3 motion = gOut_Mv[pixelPos].xyz;
        
        RTXDI_DISpatioTemporalResamplingParameters stparams;
        stparams.screenSpaceMotion = motion;
        stparams.sourceBufferIndex = g_Const.inputBufferIndex;
        stparams.maxHistoryLength = 20;
        stparams.biasCorrectionMode = g_Const.unbiasedMode ? RTXDI_BIAS_CORRECTION_RAY_TRACED : RTXDI_BIAS_CORRECTION_BASIC;
        stparams.depthThreshold = 0.1;
        stparams.normalThreshold = 0.5;
        stparams.numSamples = g_Const.numSpatialSamples + 1;
        stparams.numDisocclusionBoostSamples = 0;
        stparams.samplingRadius = 32;
        stparams.enableVisibilityShortcut = true;
        stparams.enablePermutationSampling = true;
        stparams.discountNaiveSamples = false;


        // This variable will receive the position of the sample reused from the previous frame.
        // It's only needed for gradient evaluation, ignore it here.
        int2 temporalSamplePixelPos = -1;


        // Call the resampling function, update the reservoir and lightSample variables
        reservoir = RTXDI_DISpatioTemporalResampling(pixelPos, primarySurface, reservoir,
                                                     rng, g_Const.runtimeParams, g_Const.restirDIReservoirBufferParams, stparams, temporalSamplePixelPos, lightSample, foundTemporalSurface, debugColor);
    }

    float3 shadingOutput = 0;

    // Shade the surface with the selected light sample
    // 使用选定的光照样本对表面进行着色
    if (RTXDI_IsValidDIReservoir(reservoir))
    {
        // Compute the correctly weighted reflected radiance
        // 计算正确加权的反射辐射亮度
        shadingOutput = ShadeSurfaceWithLightSample(lightSample, primarySurface)
            * RTXDI_GetDIReservoirInvPdf(reservoir);

        // Test if the selected light is visible from the surface
        // 测试选定的光源对于表面是否可见
        bool visibility = RAB_GetConservativeVisibility(primarySurface, lightSample);

        // If not visible, discard the shading output and the light sample
        // 如果不可见，则丢弃着色输出和光照样本
        if (!visibility)
        {
            shadingOutput = 0;
            RTXDI_StoreVisibilityInDIReservoir(reservoir, 0, true);
        }
    }
    else
    {
        // debugColor = float3(1,0,0);
    }
    
    float3 x = gOut_DirectLighting[pixelPos];
    gOut_DirectLighting[pixelPos] = float4(shadingOutput, 1);
    
    
    RAB_Surface preSurface =  RAB_GetGBufferSurface(pixelPos,true);
    RAB_Surface currentSurface =  RAB_GetGBufferSurface(pixelPos,false);
    
    
    // gOut_DirectLighting[pixelPos] = float4(currentSurface.material.specularF0  - preSurface.material.specularF0, 1);
    // debugColor = foundTemporalSurface;
    // gOut_DirectLighting[pixelPos] = float4(debugColor, 1);
    
    // RAB_Surface rab_get_g_buffer_surface = RAB_GetGBufferSurface(pixelPos,true);
    // gOut_DirectLighting[pixelPos] = float4(rab_get_g_buffer_surface.normal, 1.0);
    // uint pointer = RTXDI_ReservoirPositionToPointer(g_Const.restirDIReservoirBufferParams, pixelPos, 0);

    // if (gShowLight)
    // {
    //     RAB_LightInfo rab_load_light_info = RAB_LoadLightInfo(geometryProps0.primitiveIndex, false);
    //     float3 lightRadiance = Unpack_R16G16B16A16_FLOAT(rab_load_light_info.radiance);
    //     gOut_DirectLighting[pixelPos] = float4(lightRadiance, 1);
    // }

    // debugTest = -theirDepth;
    // gOut_DirectLighting[pixelPos] = float4(debugColor, 1.0);

    RTXDI_StoreDIReservoir(reservoir, g_Const.restirDIReservoirBufferParams, pixelPos, g_Const.outputBufferIndex);


    
}