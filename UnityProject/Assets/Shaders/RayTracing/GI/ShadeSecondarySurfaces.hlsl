#pragma max_recursion_depth 1

Texture2D<float3> gIn_EmissiveLighting;
RWTexture2D<int2> u_TemporalSamplePositions;

#include "../RtxdiApplicationBridge/RtxdiApplicationBridge.hlsl"

#include <Assets/Shaders/RTXDI/DI/SpatialResampling.hlsl>
#include <Assets/Shaders/RTXDI/LightSampling/PresamplingFunctions.hlsl>
#include <Assets/Shaders/RTXDI/DI/InitialSampling.hlsl>
#include <Assets/Shaders/RTXDI/ReGIR/ReGIRSampling.hlsl>
#include "Assets/Shaders/RayTracing/ShadingHelpers.hlsl"

#include "Assets/Shaders/RayTracing/ShadingHelpers.hlsl"

#include <Assets/Shaders/Rtxdi/GI/Reservoir.hlsl>

static const float c_MaxIndirectRadiance = 10;

#ifdef USE_RAY_QUERY
[numthreads(RTXDI_SCREEN_SPACE_GROUP_SIZE, RTXDI_SCREEN_SPACE_GROUP_SIZE, 1)]
void main(uint2 GlobalIndex : SV_DispatchThreadID)
#else
[shader("raygeneration")]
void MainRayGenShader()
#endif
{
    #ifndef USE_RAY_QUERY
    uint2 GlobalIndex = DispatchRaysIndex().xy;
    #endif

    uint2 pixelPosition = RTXDI_ReservoirPosToPixelPos(GlobalIndex, g_Const.runtimeParams.activeCheckerboardField);


    if (any(pixelPosition > int2(gRectSize)))
        return;

    RTXDI_RandomSamplerState rng = RTXDI_InitRandomSampler(GlobalIndex, g_Const.runtimeParams.frameIndex, RTXDI_SECONDARY_DI_GENERATE_INITIAL_SAMPLES_RANDOM_SEED);
    RTXDI_RandomSamplerState tileRng = RTXDI_InitRandomSampler(GlobalIndex / RTXDI_TILE_SIZE_IN_PIXELS, g_Const.runtimeParams.frameIndex, RTXDI_SECONDARY_DI_GENERATE_INITIAL_SAMPLES_RANDOM_SEED);

    const RTXDI_RuntimeParameters params = g_Const.runtimeParams;
    const uint gbufferIndex = RTXDI_ReservoirPositionToPointer(g_Const.restirDI.reservoirBufferParams, GlobalIndex, 0);


    RAB_Surface primarySurface = RAB_GetGBufferSurface(pixelPosition, false);

    SecondaryGBufferData secondaryGBufferData = u_SecondaryGBuffer[gbufferIndex];

    const float3 throughput = Unpack_R16G16B16A16_FLOAT(secondaryGBufferData.throughputAndFlags).rgb;
    const uint secondaryFlags = secondaryGBufferData.throughputAndFlags.y >> 16;
    const bool isValidSecondarySurface = any(throughput != 0);
    const bool isSpecularRay = (secondaryFlags & kSecondaryGBuffer_IsSpecularRay) != 0;
    const bool isDeltaSurface = (secondaryFlags & kSecondaryGBuffer_IsDeltaSurface) != 0;
    const bool isEnvironmentMap = (secondaryFlags & kSecondaryGBuffer_IsEnvironmentMap) != 0;


    RAB_Surface secondarySurface;
    float3 radiance = secondaryGBufferData.emission;


    // Unpack the G-buffer data
    secondarySurface.worldPos = secondaryGBufferData.worldPos;
    secondarySurface.viewDepth = 1.0; // doesn't matter
    secondarySurface.normal = octToNdirUnorm32(secondaryGBufferData.normal);
    secondarySurface.geoNormal = secondarySurface.normal;
    secondarySurface.material.diffuseAlbedo = Unpack_R11G11B10_UFLOAT(secondaryGBufferData.diffuseAlbedo);
    float4 specularRough = Unpack_R8G8B8A8_Gamma_UFLOAT(secondaryGBufferData.specularAndRoughness);
    secondarySurface.material.specularF0 = specularRough.rgb;
    secondarySurface.material.roughness = specularRough.a;
    secondarySurface.diffuseProbability = getSurfaceDiffuseProbability(secondarySurface);
    secondarySurface.viewDir = normalize(primarySurface.worldPos - secondarySurface.worldPos);


    // gOut_DirectLighting[pixelPosition] = isValidSecondarySurface;
    
    
    // Shade the secondary surface.
    if (isValidSecondarySurface && !isEnvironmentMap)
    {

        RAB_LightSample lightSample;
        RTXDI_DIReservoir reservoir = RTXDI_SampleLightsForSurface(rng, tileRng, secondarySurface,
            g_Const.brdfPT.secondarySurfaceReSTIRDIParams.initialSamplingParams, g_Const.lightBufferParams,
#if RTXDI_ENABLE_PRESAMPLING
        g_Const.localLightsRISBufferSegmentParams, g_Const.environmentLightRISBufferSegmentParams,
#if RTXDI_REGIR_MODE != RTXDI_REGIR_MODE_DISABLED
        g_Const.regir,
#endif
#endif
        lightSample);
        
        if (g_Const.brdfPT.enableSecondaryResampling)
        {
            // Try to find this secondary surface in the G-buffer. If found, resample the lights
            // from that G-buffer surface into the reservoir using the spatial resampling function.
        
            // float4 secondaryClipPos = mul(float4(secondaryGBufferData.worldPos, 1.0), gWorldToClip);
            float4 secondaryClipPos = Geometry::ProjectiveTransform( gWorldToClip, secondaryGBufferData.worldPos );
            
            secondaryClipPos.xyz /= secondaryClipPos.w;
        
            if (all(abs(secondaryClipPos.xy) < 1.0) && secondaryClipPos.w > 0)
            {
                float2 uv = secondaryClipPos.xy * float2(0.5, -0.5) + 0.5;
                int2 secondaryPixelPos = int2(uv * gRectSize);
                
                uint sourceBufferIndex = g_Const.restirDI.bufferIndices.shadingInputBufferIndex;
                secondarySurface.viewDepth = -secondaryClipPos.w;
                reservoir = RTXDI_DISpatialResampling(secondaryPixelPos, secondarySurface, reservoir,
                          rng, params, g_Const.restirDI.reservoirBufferParams, sourceBufferIndex, g_Const.brdfPT.secondarySurfaceReSTIRDIParams.spatialResamplingParams, lightSample);

                
                
            }
        }

        float3 indirectDiffuse = 0;
        float3 indirectSpecular = 0;
        float lightDistance = 0;
        ShadeSurfaceWithLightSample(reservoir, secondarySurface, lightSample, /* previousFrameTLAS = */ false,
            /* enableVisibilityReuse = */ false, /* enableVisibilityShortcut */ false, indirectDiffuse, indirectSpecular, lightDistance);
        
        
        radiance += indirectDiffuse * secondarySurface.material.diffuseAlbedo + indirectSpecular;

        
        // Firefly suppression
        float indirectLuminance = calcLuminance(radiance);
        if (indirectLuminance > c_MaxIndirectRadiance)
            radiance *= c_MaxIndirectRadiance / indirectLuminance;
    }
    
    bool outputShadingResult = true;
    
    if (g_Const.brdfPT.enableReSTIRGI)
    {
        RTXDI_GIReservoir reservoir = RTXDI_EmptyGIReservoir();

        // For delta reflection rays, just output the shading result in this shader
        // and don't include it into ReSTIR GI reservoirs.
        outputShadingResult = isSpecularRay && isDeltaSurface;

        if (isValidSecondarySurface && !outputShadingResult)
        {
            // This pixel has a valid indirect sample so it stores information as an initial GI reservoir.
            reservoir = RTXDI_MakeGIReservoir(secondarySurface.worldPos,
                secondarySurface.normal, radiance, secondaryGBufferData.pdf);
        }
        uint2 reservoirPosition = RTXDI_PixelPosToReservoirPos(pixelPosition, g_Const.runtimeParams.activeCheckerboardField);
        RTXDI_StoreGIReservoir(reservoir, g_Const.restirGI.reservoirBufferParams, reservoirPosition, g_Const.restirGI.bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex);

        // Save the initial sample radiance for MIS in the final shading pass
        secondaryGBufferData.emission = outputShadingResult ? 0 : radiance;
        u_SecondaryGBuffer[gbufferIndex] = secondaryGBufferData;
    }
    
    if (outputShadingResult)
    {
        float3 diffuse = 0;
        float3 specular = 0;
        if (!isDeltaSurface)
        {
            diffuse = isSpecularRay ? 0.0 : radiance * throughput.rgb;
            specular = isSpecularRay ? radiance * throughput.rgb : 0.0;
            // specular = DemodulateSpecular(primarySurface.material.specularF0, specular);
        }
        float3 finalColor = (diffuse * secondarySurface.material.diffuseAlbedo) + specular;

        // finalColor += gIn_EmissiveLighting[pixelPosition];
        finalColor *= gExposure; 
        
        
        StoreShadingOutput(finalColor,pixelPosition,false, true);
        // gOut_DirectLighting[pixelPosition] += finalColor;
    }
}
