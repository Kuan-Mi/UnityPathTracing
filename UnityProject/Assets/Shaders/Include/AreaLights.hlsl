// AreaLights.hlsl
// Rectangular and disc area lights with single stochastic shadow rays.
// Mirrors the sun path in GetLighting(): NoL+SmoothStep first, then BRDF,
// then SSS Burley override of Cdiff + shadow origin, then shadow ray.
// lightType: 0 = rectangle, 1 = disc
#ifndef AREA_LIGHTS_HLSL
#define AREA_LIGHTS_HLSL


#define AREA_LIGHT_RECT 0.0
#define AREA_LIGHT_DISC 1.0

struct AreaLight
{
    float3 position;    // World-space centre
    float  halfWidth;   // Rect: half-extent along right.  Disc: radius.
    float3 right;       // Unit right vector (world-space)
    float  halfHeight;  // Rect: half-extent along up.     Disc: unused.
    float3 up;          // Unit up vector (world-space)
    float  lightType;   // 0 = rectangle, 1 = disc
    float3 color;       // Pre-multiplied color * intensity
    float  pad2;
};

StructuredBuffer<AreaLight> gIn_AreaLights;

// Evaluate the direct lighting contribution of all area lights (rect + disc) at a surface point.
// A single stochastic shadow ray is cast per light per frame; soft shadows emerge via accumulation.
// When isSSS is true, also adds a Burley-sampled subsurface scattering contribution.
float3 EvaluateAreaLights(GeometryProps geo, MaterialProps mat, bool isSSS)
{
    float3 result = 0.0;

    float3 albedo, Rf0;
    BRDF::ConvertBaseColorMetalnessToAlbedoRf0(mat.baseColor, mat.metalness, albedo, Rf0);

    [loop]
    for (uint i = 0; i < gAreaLightCount; i++)
    {
        AreaLight light = gIn_AreaLights[i];

        float3 lightNormal = normalize(cross(light.right, light.up));

        // Stochastic sample point on the light surface
        float2 xi = Rng::Hash::GetFloat2();
        float3 samplePos;
        float  lightArea;

        if (light.lightType > 0.5) // disc
        {
            float r     = light.halfWidth * sqrt(xi.x);
            float theta = 6.28318530718 * xi.y;
            samplePos = light.position
                      + light.right * (r * cos(theta))
                      + light.up    * (r * sin(theta));
            lightArea = 3.14159265359 * light.halfWidth * light.halfWidth;
        }
        else // rectangle
        {
            float2 uv = xi * 2.0 - 1.0;
            samplePos = light.position
                      + light.right * (uv.x * light.halfWidth)
                      + light.up    * (uv.y * light.halfHeight);
            lightArea = 4.0 * light.halfWidth * light.halfHeight;
        }

        float3 toLight = samplePos - geo.X;
        float  dist    = length(toLight);
        if (dist < 0.0001) continue;

        float3 L = toLight / dist;

        // One-sided: reject back face
        float cosLight = dot(lightNormal, -L);
        if (cosLight <= 0.0) continue;

        float solidAngle = cosLight * lightArea / max(dist * dist, 0.0001);

        // ---------------------------------------------------------------
        // NoL + SSS-aware shadow factor (matches GetLighting sun path)
        // ---------------------------------------------------------------
        float NoL_geom  = dot(geo.N, L);
        float minThresh = isSSS ? gSssMinThreshold : 0.03;
        float shadow    = Math::SmoothStep(minThresh, 0.1, NoL_geom);
        if (shadow == 0.0) continue;

        // ---------------------------------------------------------------
        // BRDF
        // ---------------------------------------------------------------
        float3 V   = geo.V;
        float3 H   = normalize(L + V);
        float  NoL = saturate(dot(mat.N, L));
        float  NoH = saturate(dot(mat.N, H));
        float  VoH = saturate(dot(V, H));
        float  NoV = abs(dot(mat.N, V));

        float  D  = BRDF::DistributionTerm(mat.roughness, NoH);
        float  G  = BRDF::GeometryTermMod(mat.roughness, NoL, NoV, VoH, NoH);
        float3 F  = BRDF::FresnelTerm(Rf0, VoH);
        float  Kd = BRDF::DiffuseTerm(mat.roughness, NoL, NoV, VoH);

        float3 Clinc = light.color * solidAngle;
        float3 Cspec = F * D * G * NoL;           // Clinc excluded — multiplied at end
        float3 Cdiff = Kd * albedo * NoL * Clinc; // Clinc included

        // ---------------------------------------------------------------
        // SSS: Burley sample -> exit point -> override Cdiff + shadow origin
        // ---------------------------------------------------------------
        GeometryProps shadowGeo   = geo;
        float3        L_shadow    = L;
        float         dist_shadow = dist;

#if( RTXCR_INTEGRATION == 1 )
        if (isSSS)
        {
            RTXCR_SubsurfaceMaterialData sssMat = (RTXCR_SubsurfaceMaterialData)0;
            sssMat.transmissionColor = albedo;
            sssMat.scatteringColor   = mat.scatteringColor;
            sssMat.scale             = gSssScale / gUnitToMetersMultiplier;
            sssMat.g                 = 0.0;

            float3 Xoff = geo.GetXoffset(geo.N, PT_SHADOW_RAY_OFFSET);
            float3x3 basis = Geometry::GetBasis(geo.N);
            RTXCR_SubsurfaceInteraction sssInteraction =
                RTXCR_CreateSubsurfaceInteraction(Xoff, basis[2], basis[0], basis[1]);

            RTXCR_SubsurfaceSample sssSample = (RTXCR_SubsurfaceSample)0;
            RTXCR_EvalBurleyDiffusionProfile(sssMat, sssInteraction,
                gSssMaxSampleRadius / gUnitToMetersMultiplier, false, Rng::Hash::GetFloat2(), sssSample);

            GeometryProps sssProps;
            MaterialProps sssMaterialProps;
            CastRay(sssSample.samplePosition, -sssInteraction.normal,
                    0.0, INF, float2(geo.mip, 0.0), FLAG_NON_TRANSPARENT, sssProps, sssMaterialProps);

            if (!sssProps.IsMiss() && sssProps.Has(FLAG_SKIN))
            {
                shadowGeo = sssProps;
                // Shadow targets the original sampled point on the area light surface
                float3 toLightExit = samplePos - sssProps.X;
                dist_shadow = length(toLightExit);
                L_shadow = dist_shadow > 0.0001 ? toLightExit / dist_shadow : L;

                float NoL_sss = saturate(dot(sssMaterialProps.N, L_shadow));
                Cdiff = RTXCR_EvalBssrdf(sssSample, Clinc, NoL_sss);
            }
        }
#endif

        // ---------------------------------------------------------------
        // Shadow ray from exit point (SSS) or entry point (non-SSS)
        // ---------------------------------------------------------------
        float3 Xoffset = shadowGeo.GetXoffset(L_shadow, PT_BOUNCE_RAY_OFFSET);
        float  shadowHitT = CastVisibilityRay_AnyHit(
            Xoffset, L_shadow, 0.0, dist_shadow,
            float2(shadowGeo.mip, 0.0),
            gWorldTlas, FLAG_NON_TRANSPARENT, 0);
        if (shadowHitT != INF) continue;

        result += (Clinc * Cspec + Cdiff * (1.0 - F)) * shadow;
    }

    return result;
}

#endif