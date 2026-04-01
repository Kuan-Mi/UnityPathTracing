#pragma max_recursion_depth 1

#include "../RtxdiApplicationBridge/RtxdiApplicationBridge.hlsl"


#include <Assets/Shaders/RTXDI/DI/Reservoir.hlsl>
#include <Assets/Shaders/RTXDI/GI/Reservoir.hlsl>
#include <Assets/Shaders/RTXDI/Utils/ReservoirAddressing.hlsl>
#include "../ShadingHelpers.hlsl"

static const float kMaxBrdfValue = 1e4;
static const float kMISRoughness = 0.3;

float GetGIMISWeight(const SplitBrdf roughBrdf, const SplitBrdf trueBrdf, const float3 diffuseAlbedo)
{
    float3 combinedRoughBrdf = roughBrdf.demodulatedDiffuse * diffuseAlbedo + roughBrdf.specular;
    float3 combinedTrueBrdf  = trueBrdf.demodulatedDiffuse  * diffuseAlbedo + trueBrdf.specular;

    combinedRoughBrdf = clamp(combinedRoughBrdf, 1e-4, kMaxBrdfValue);
    combinedTrueBrdf  = clamp(combinedTrueBrdf,  0,    kMaxBrdfValue);

    const float initWeight = saturate(calcLuminance(combinedTrueBrdf) / calcLuminance(combinedTrueBrdf + combinedRoughBrdf));
    return initWeight * initWeight * initWeight;
}

RTXDI_GIReservoir LoadGIInitialSampleReservoir(int2 reservoirPosition, RAB_Surface primarySurface)
{
    const uint gbufferIndex = RTXDI_ReservoirPositionToPointer(g_Const.restirGI.reservoirBufferParams, reservoirPosition, 0);
    const SecondaryGBufferData secondaryGBufferData = u_SecondaryGBuffer[gbufferIndex];

    const float3 normal     = octToNdirUnorm32(secondaryGBufferData.normal);
    const float3 throughput = Unpack_R16G16B16A16_FLOAT(secondaryGBufferData.throughputAndFlags).rgb;

    return RTXDI_MakeGIReservoir(secondaryGBufferData.worldPos,
        normal, secondaryGBufferData.emission * throughput, secondaryGBufferData.pdf);
}

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

    const RAB_Surface primarySurface = RAB_GetGBufferSurface(pixelPosition, false);

    const uint2 reservoirPosition = RTXDI_PixelPosToReservoirPos(pixelPosition, g_Const.runtimeParams.activeCheckerboardField);
    const RTXDI_GIReservoir reservoir = RTXDI_LoadGIReservoir(g_Const.restirGI.reservoirBufferParams, reservoirPosition,
        g_Const.restirGI.bufferIndices.secondarySurfaceReSTIRDIOutputBufferIndex);

    float3 diffuse  = 0;
    float3 specular = 0;

    if (RTXDI_IsValidGIReservoir(reservoir))
    {
        float3 radiance = reservoir.radiance * reservoir.weightSum;

        float3 visibility = 1.0;
        if (g_Const.restirGI.finalShadingParams.enableFinalVisibility)
        {
            visibility = GetFinalVisibility(primarySurface, reservoir.position);
        }

        radiance *= visibility;

        const SplitBrdf brdf = EvaluateBrdf(primarySurface, reservoir.position);

        if (g_Const.restirGI.finalShadingParams.enableFinalMIS)
        {
            const RTXDI_GIReservoir initialReservoir = LoadGIInitialSampleReservoir(reservoirPosition, primarySurface);
            const SplitBrdf brdf0 = EvaluateBrdf(primarySurface, initialReservoir.position);

            RAB_Surface roughenedSurface = primarySurface;
            roughenedSurface.material.roughness = max(roughenedSurface.material.roughness, kMISRoughness);

            const SplitBrdf roughBrdf  = EvaluateBrdf(roughenedSurface, reservoir.position);
            const SplitBrdf roughBrdf0 = EvaluateBrdf(roughenedSurface, initialReservoir.position);

            const float finalWeight   = 1.0 - GetGIMISWeight(roughBrdf,  brdf,  primarySurface.material.diffuseAlbedo);
            const float initialWeight =       GetGIMISWeight(roughBrdf0, brdf0, primarySurface.material.diffuseAlbedo);

            const float3 initialRadiance = initialReservoir.radiance * initialReservoir.weightSum;

            diffuse  = brdf.demodulatedDiffuse  * radiance         * finalWeight
                     + brdf0.demodulatedDiffuse * initialRadiance  * initialWeight;
            specular = brdf.specular            * radiance         * finalWeight
                     + brdf0.specular           * initialRadiance  * initialWeight;
        }
        else
        {
            diffuse  = brdf.demodulatedDiffuse * radiance;
            specular = brdf.specular           * radiance;
        }

        // specular = DemodulateSpecular(primarySurface.material.specularF0, specular);
        
        float3 finalColor = (diffuse * primarySurface.material.diffuseAlbedo) + specular;
        
        gOut_DirectLighting[pixelPosition] = finalColor;

        
    }

    // StoreShadingOutput(GlobalIndex, pixelPosition,
    //     primarySurface.viewDepth, primarySurface.material.roughness, diffuse, specular, 0, false, true);
}
