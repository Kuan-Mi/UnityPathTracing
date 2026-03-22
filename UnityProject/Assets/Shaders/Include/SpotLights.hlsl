// SpotLights.hlsl
// Point-like spot lights with cone falloff and shadow rays.
// Mirrors the sun path in GetLighting(): NoL+SmoothStep first, then BRDF,
// then SSS Burley override of Cdiff + shadow origin, then shadow ray.

struct SpotLight
{
    float3 position;       // World-space position
    float  range;          // Maximum range (hard cutoff)
    float3 direction;      // Normalized forward direction (from light toward scene)
    float  cosOuterAngle;  // cos(outer half-angle)
    float3 color;          // Pre-multiplied color * intensity (cd/m^2)
    float  cosInnerAngle;  // cos(inner half-angle), for smooth penumbra edge
};

StructuredBuffer<SpotLight> gIn_SpotLights;

// Evaluate the direct lighting contribution of all spot lights at a surface point.
// Shadow rays are hard (single ray, no angular jitter) since these are ideal point lights.
// When isSSS is true, also adds a Burley-sampled subsurface scattering contribution.
float3 EvaluateSpotLights(GeometryProps geo, MaterialProps mat, bool isSSS)
{
    float3 result = 0.0;

    float3 albedo, Rf0;
    BRDF::ConvertBaseColorMetalnessToAlbedoRf0(mat.baseColor, mat.metalness, albedo, Rf0);

    [loop]
    for (uint i = 0; i < gSpotLightCount; i++)
    {
        SpotLight light = gIn_SpotLights[i];

        // ---------------------------------------------------------------
        // Geometry
        // ---------------------------------------------------------------
        float3 toLight = light.position - geo.X;
        float  dist    = length(toLight);
        if (dist >= light.range) continue;

        float3 L = toLight / dist;

        // ---------------------------------------------------------------
        // Cone falloff
        // ---------------------------------------------------------------
        float cosAngle = dot(-L, light.direction);
        if (cosAngle <= light.cosOuterAngle) continue;
        float cone = Math::SmoothStep(light.cosOuterAngle, light.cosInnerAngle, cosAngle);

        // ---------------------------------------------------------------
        // Distance attenuation
        // ---------------------------------------------------------------
        float atten     = 1.0 / max(dist * dist, 0.0001);
        float rangeFade = Math::SmoothStep(light.range, light.range * 0.75, dist);
        atten *= rangeFade;

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

        // Clinc baked into Cdiff so the SSS override (which also bakes Clinc) is uniform.
        float3 Clinc = light.color * atten * cone;
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
                float3 toLightExit = light.position - sssProps.X;
                dist_shadow = length(toLightExit);
                L_shadow = dist_shadow > 0.0001 ? toLightExit / dist_shadow : L;

                float NoL_sss = saturate(dot(sssMaterialProps.N, L_shadow));
                // sun pattern: Cdiff = RTXCR_EvalBssrdf(sssSample, Csun,  NoL_sss)
                // spot:        Cdiff = RTXCR_EvalBssrdf(sssSample, Clinc, NoL_sss)
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

        // Cspec still needs Clinc; Cdiff already contains Clinc (both paths)
        result += (Clinc * Cspec + Cdiff * (1.0 - F)) * shadow;
    }

    return result;
}
