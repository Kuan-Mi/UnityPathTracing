// PointLights.hlsl
// Omnidirectional point lights, optionally with a sphere radius for soft shadows.
// Mirrors the sun path in GetLighting(): NoL+SmoothStep first, then BRDF,
// then SSS Burley override of Cdiff + shadow origin, then shadow ray.

struct PointLight
{
    float3 position;    // World-space position
    float  range;       // Maximum range (hard cutoff)
    float3 color;       // Pre-multiplied color * intensity
    float  radius;      // Sphere radius (0 = hard point light)
};

StructuredBuffer<PointLight> gIn_PointLights;

// Evaluate the direct lighting contribution of all point lights at a surface point.
// When radius > 0, a single stochastic shadow ray is cast per light per frame;
// soft shadows emerge via temporal accumulation.
// When isSSS is true, also adds a Burley-sampled subsurface scattering contribution.
float3 EvaluatePointLights(GeometryProps geo, MaterialProps mat, bool isSSS)
{
    float3 result = 0.0;

    float3 albedo, Rf0;
    BRDF::ConvertBaseColorMetalnessToAlbedoRf0(mat.baseColor, mat.metalness, albedo, Rf0);

    [loop]
    for (uint i = 0; i < gPointLightCount; i++)
    {
        PointLight light = gIn_PointLights[i];

        float3 L;
        float  dist;
        float3 Clinc;   // light irradiance factor, NoL NOT included

        if (light.radius > 0.0001)
        {
            // -----------------------------------------------------------
            // Sphere light: stochastic sample on hemisphere facing geo.X
            // -----------------------------------------------------------
            float3 toSurface = normalize(geo.X - light.position);

            float3 T, B;
            float3 up = abs(toSurface.x) < 0.9 ? float3(1, 0, 0) : float3(0, 1, 0);
            T = normalize(cross(toSurface, up));
            B = cross(toSurface, T);

            float2 xi   = Rng::Hash::GetFloat2();
            float  cosT = xi.x;
            float  sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
            float  phi  = 6.28318530718 * xi.y;

            float3 sampleDir = T * (sinT * cos(phi))
                             + B * (sinT * sin(phi))
                             + toSurface * cosT;

            float3 samplePos = light.position + sampleDir * light.radius;

            float3 toLight = samplePos - geo.X;
            dist = length(toLight);
            if (dist < 0.0001) continue;
            L = toLight / dist;

            float centDist = length(light.position - geo.X);
            if (centDist >= light.range) continue;

            float cosLight = dot(sampleDir, -L);
            if (cosLight <= 0.0) continue;

            float rangeFade      = Math::SmoothStep(light.range, light.range * 0.75, centDist);
            float solidAngleWeight = cosLight * 2.0 / max(dist * dist, 0.0001) * rangeFade;
            Clinc = light.color * solidAngleWeight;
        }
        else
        {
            // -----------------------------------------------------------
            // Ideal point light: hard shadow ray, inverse-square falloff
            // -----------------------------------------------------------
            float3 toLight = light.position - geo.X;
            dist = length(toLight);
            if (dist >= light.range) continue;
            L = toLight / dist;

            float atten     = 1.0 / max(dist * dist, 0.0001);
            float rangeFade = Math::SmoothStep(light.range, light.range * 0.75, dist);
            Clinc = light.color * atten * rangeFade;
        }

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
            sssMat.scatteringColor   = gSssScatteringColor;
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

                // For sphere lights, re-sample the hemisphere facing the SSS exit point
                // so the soft-shadow distribution matches the entry-point sample.
                float3 shadowTarget;
                if (light.radius > 0.0001)
                {
                    float3 toSurface_sss = normalize(sssProps.X - light.position);
                    float3 up_sss  = abs(toSurface_sss.x) < 0.9 ? float3(1, 0, 0) : float3(0, 1, 0);
                    float3 T_sss   = normalize(cross(toSurface_sss, up_sss));
                    float3 B_sss   = cross(toSurface_sss, T_sss);

                    float2 xi_sss   = Rng::Hash::GetFloat2();
                    float  cosT_sss = xi_sss.x;
                    float  sinT_sss = sqrt(max(0.0, 1.0 - cosT_sss * cosT_sss));
                    float  phi_sss  = 6.28318530718 * xi_sss.y;

                    float3 sampleDir_sss = T_sss * (sinT_sss * cos(phi_sss))
                                         + B_sss * (sinT_sss * sin(phi_sss))
                                         + toSurface_sss * cosT_sss;
                    shadowTarget = light.position + sampleDir_sss * light.radius;
                }
                else
                {
                    shadowTarget = light.position;
                }

                float3 toLightExit = shadowTarget - sssProps.X;
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
