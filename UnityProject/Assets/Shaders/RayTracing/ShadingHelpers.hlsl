#ifndef SHADING_HELPERS_HLSLI
#define SHADING_HELPERS_HLSLI


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

#ifndef SECONDARY_SURFACE_PAYLOAD
bool ShadeSurfaceWithLightSample(
    inout RTXDI_DIReservoir reservoir,
    RAB_Surface surface,
    RAB_LightSample lightSample,
    bool previousFrameTLAS,
    bool enableVisibilityReuse,
    bool enableVisibilityShortcut,
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
            RTXDI_StoreVisibilityInDIReservoir(reservoir, visibility, enableVisibilityShortcut);

            // RTXDI_StoreVisibilityInDIReservoir(reservoir, visibility, g_Const.restirDI.temporalResamplingParams.discardInvisibleSamples);
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

#endif


float3 DemodulateSpecular(float3 surfaceSpecularF0, float3 specular)
{
    return specular / max(0.01, surfaceSpecularF0);
}


void StoreShadingOutput(
    float3 finalColor,
    uint2 pixelPosition,
    bool isFirstPass,
    bool isLastPass)
{
    if (!isFirstPass)
    {
        float3 priorLight = gOut_DirectLighting[pixelPosition];

        finalColor += priorLight;
    }

    if (isLastPass)
    {
        RAB_Surface surface = RAB_GetGBufferSurface(pixelPosition, false);
        if (!RAB_IsSurfaceValid(surface))
        {

            float2 pixelUv = float2(pixelPosition + 0.5) / gRectSize;
            float2 sampleUv = pixelUv + gJitter;


            float3 cameraRayOrigin = 0;
            float3 cameraRayDirection = 0;
            
            
            GetCameraRay(cameraRayOrigin, cameraRayDirection, sampleUv);
            
            
            finalColor = GetSkyIntensity(cameraRayDirection);
        }
    }

    //
    // finalColor = RAB_IsSurfaceValid(surface);
    // // finalColor = -surface.viewDepth /INF ;
    gOut_DirectLighting[pixelPosition] = finalColor;
}

#endif // SHADING_HELPERS_HLSLI
