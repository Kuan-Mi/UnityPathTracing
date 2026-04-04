// PointLights.hlsl
// Omnidirectional point lights, optionally with a sphere radius for soft shadows.
// Mirrors the sun path in GetLighting(): NoL+SmoothStep first, then BRDF,
// then SSS Burley override of Cdiff + shadow origin, then shadow ray.

struct PointLightInfo
{
    float3 position; // World-space position
    float range; // Maximum range (hard cutoff)
    float3 color; // Pre-multiplied color * intensity
    float radius; // Sphere radius (0 = hard point light)
};

StructuredBuffer<PointLightInfo> gIn_PointLights;

// Evaluate the direct lighting contribution of all point lights at a surface point.
// When radius > 0, a single stochastic shadow ray is cast per light per frame;
// soft shadows emerge via temporal accumulation.
// When isSSS is true, also adds a Burley-sampled subsurface scattering contribution.
float3 EvaluatePointLights(GeometryProps geo, MaterialProps mat, bool isSSS)
{
    float3 result = 0.0;

    float3 albedo, Rf0;
    BRDF::ConvertBaseColorMetalnessToAlbedoRf0(mat.baseColor, mat.metalness, albedo, Rf0);

    // ---------------------------------------------------------------
    // SSS: find the subsurface entry point once, outside the light loop.
    // The entry point depends only on geometry/material/camera, not on
    // the number of lights; computing it per-light would cost O(N) rays.
    // ---------------------------------------------------------------
#if( RTXCR_INTEGRATION == 1 )
    GeometryProps sssProps;
    MaterialProps sssMaterialProps;
    RTXCR_SubsurfaceSample sssSample = (RTXCR_SubsurfaceSample)0;
    bool sssHit = false;

    if (isSSS)
    {
        RTXCR_SubsurfaceMaterialData sssMat = (RTXCR_SubsurfaceMaterialData)0;
        sssMat.transmissionColor = albedo;
        sssMat.scatteringColor = mat.scatteringColor;
        sssMat.scale = gSssScale / gUnitToMetersMultiplier;
        sssMat.g = 0.0;

        float3 Xoff = geo.GetXoffset(geo.N, PT_SHADOW_RAY_OFFSET);
        float3x3 basis = Geometry::GetBasis(geo.N);
        RTXCR_SubsurfaceInteraction sssInteraction =
            RTXCR_CreateSubsurfaceInteraction(Xoff, basis[2], basis[0], basis[1]);

        RTXCR_EvalBurleyDiffusionProfile(sssMat, sssInteraction,
                                         gSssMaxSampleRadius / gUnitToMetersMultiplier, false, Rng::Hash::GetFloat2(), sssSample);

        CastRay(sssSample.samplePosition, -sssInteraction.normal,
                0.0, INF, float2(geo.mip, 0.0), FLAG_NON_TRANSPARENT, sssProps, sssMaterialProps);

        sssHit = !sssProps.IsMiss() && sssProps.Has(FLAG_SKIN);
    }
#endif

    [loop]
    for (uint i = 0; i < gPointLightCount; i++)
    {
        PointLightInfo light = gIn_PointLights[i];

        float3 L;
        float dist;
        float3 Clinc; // light irradiance factor, NoL NOT included

        // Early range cull before consuming any RNG budget.
        float centDist = length(light.position - geo.X);
        if (centDist >= light.range) continue;

        float rangeFade = Math::SmoothStep(light.range, light.range * 0.75, centDist);

        float3 samplePos;
        // if (light.radius > 0.0)
        {
            // -----------------------------------------------------------
            // Sphere light: stochastic hemisphere sample for soft shadows.
            // -----------------------------------------------------------
            float3 toSurface = normalize(geo.X - light.position);

            float3 T, B;
            float3 up = abs(toSurface.x) < 0.9 ? float3(1, 0, 0) : float3(0, 1, 0);
            T = normalize(cross(toSurface, up));
            B = cross(toSurface, T);

            float2 xi = Rng::Hash::GetFloat2();
            float cosT = xi.x;
            float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
            float phi = 6.28318530718 * xi.y;

            float3 sampleDir = T * (sinT * cos(phi))
                + B * (sinT * sin(phi))
                + toSurface * cosT;

            samplePos = light.position + sampleDir * light.radius;

            float3 toLight = samplePos - geo.X;
            dist = length(toLight);
            if (dist < 0.0001) continue;
            L = toLight / dist;

            float cosLight = dot(sampleDir, -L);
            if (cosLight <= 0.0) continue;

            Clinc = light.color * (cosLight * 2.0 / max(dist * dist, 0.0001) * rangeFade);
        }
        // else
        // {
        //     // -----------------------------------------------------------
        //     // Hard point light: no area, plain inverse-square falloff.
        //     // Hemisphere cosine weighting must NOT be used here — it would
        //     // multiply Clinc by xi.x (a per-frame random), causing severe
        //     // frame-to-frame flicker on stationary lights.
        //     // -----------------------------------------------------------
        //     samplePos = light.position;
        //     dist = centDist;
        //     L = (light.position - geo.X) / dist;
        //
        //     Clinc = light.color * (rangeFade / max(dist * dist, 0.0001));
        // }

        // ---------------------------------------------------------------
        // NoL + SSS-aware shadow factor (matches GetLighting sun path)
        // ---------------------------------------------------------------
        float NoL_geom = dot(geo.N, L);
        float minThresh = isSSS ? gSssMinThreshold : 0.03;
        float shadow = Math::SmoothStep(minThresh, 0.1, NoL_geom);

        if (shadow == 0.0) continue;

        // ---------------------------------------------------------------
        // Common BRDF
        // ---------------------------------------------------------------
        float3 V = geo.V;
        float3 H = normalize(L + V);

        float NoL = saturate(dot(mat.N, L));
        float NoH = saturate(dot(mat.N, H));
        float VoH = saturate(dot(V, H));
        float NoV = abs(dot(mat.N, V));

        float D = BRDF::DistributionTerm(mat.roughness, NoH);
        float G = BRDF::GeometryTermMod(mat.roughness, NoL, NoV, VoH, NoH);
        float3 F = BRDF::FresnelTerm(Rf0, VoH);
        float Kdiff = BRDF::DiffuseTerm(mat.roughness, NoL, NoV, VoH);

        float3 Cspec = F * D * G * NoL; // Clinc excluded — multiplied at end
        float3 Cdiff = Kdiff * Clinc * albedo * NoL; // Clinc included

        float3 lighting = Cspec * Clinc;

        // ---------------------------------------------------------------
        // SSS: override Cdiff + redirect shadow ray to pre-computed entry
        // point. Clinc_sss is re-evaluated at the entry point to correctly
        // apply inverse-square falloff from the point of light penetration,
        // not from the (potentially distant) exit point on geo.X.
        // ---------------------------------------------------------------
        GeometryProps shadowGeo = geo;
        float3 L_shadow = L;
        float dist_shadow = dist;

#if( RTXCR_INTEGRATION == 1 )
        if (isSSS && sssHit)
        {
            float centDist_entry = length(light.position - sssProps.X);
            float rangeFade_entry = Math::SmoothStep(light.range, light.range * 0.75, centDist_entry);

            float3 shadowTarget;
            float3 Clinc_sss = 0.0;
            bool validSss = false;

            if (light.radius > 0.0001)
            {
                // Re-sample the hemisphere facing the SSS entry point.
                // xi_sss is coupled exclusively to Clinc_sss so the shadow
                // direction and its energy estimate come from the same Monte
                // Carlo sample, fixing the decoupled-variance bug of the
                // original code where xi and xi_sss were independent.
                float3 toSurface_sss = normalize(sssProps.X - light.position);
                float3 up_sss = abs(toSurface_sss.x) < 0.9 ? float3(1, 0, 0) : float3(0, 1, 0);
                float3 T_sss = normalize(cross(toSurface_sss, up_sss));
                float3 B_sss = cross(toSurface_sss, T_sss);

                float2 xi_sss = Rng::Hash::GetFloat2();
                float cosT_sss = xi_sss.x;
                float sinT_sss = sqrt(max(0.0, 1.0 - cosT_sss * cosT_sss));
                float phi_sss = 6.28318530718 * xi_sss.y;

                float3 sampleDir_sss = T_sss * (sinT_sss * cos(phi_sss))
                    + B_sss * (sinT_sss * sin(phi_sss))
                    + toSurface_sss * cosT_sss;

                shadowTarget = light.position + sampleDir_sss * light.radius;

                float3 toEntry = shadowTarget - sssProps.X;
                float dist_entry = length(toEntry);
                if (dist_entry >= 0.0001)
                {
                    float cosLight_sss = dot(sampleDir_sss, -toEntry / dist_entry);
                    if (cosLight_sss > 0.0)
                    {
                        Clinc_sss = light.color * (cosLight_sss * 2.0 / max(dist_entry * dist_entry, 0.0001) * rangeFade_entry);
                        validSss = true;
                    }
                }
            }
            else
            {
                // Hard point light: inverse-square falloff at entry point.
                shadowTarget = light.position;
                float3 toEntry = shadowTarget - sssProps.X;
                float dist_entry = length(toEntry);
                if (dist_entry >= 0.0001)
                {
                    Clinc_sss = light.color * (rangeFade_entry / max(dist_entry * dist_entry, 0.0001));
                    validSss = true;
                }
            }

            if (validSss)
            {
                shadowGeo = sssProps;
                float3 toEntry = shadowTarget - sssProps.X;
                dist_shadow = length(toEntry);
                L_shadow = dist_shadow > 0.0001 ? toEntry / dist_shadow : L;

                float NoL_sss = saturate(dot(sssMaterialProps.N, L_shadow));
                Cdiff = RTXCR_EvalBssrdf(sssSample, Clinc_sss, NoL_sss);
            }
        }
#endif

        lighting += Cdiff * (1.0 - F);
        lighting *= shadow;

        // ---------------------------------------------------------------
        // Shadow ray from exit point (SSS) or entry point (non-SSS)
        // ---------------------------------------------------------------

        if (Color::Luminance(lighting) != 0)
        {
            float3 Xshadow = shadowGeo.GetXoffset(L_shadow, PT_BOUNCE_RAY_OFFSET);
            float shadowHitT = CastVisibilityRay_AnyHit(Xshadow, L_shadow, 0.0, dist_shadow, float2(shadowGeo.mip, 0.0), gWorldTlas, FLAG_NON_TRANSPARENT, 0);
            lighting *= float(shadowHitT == INF);
        }

        result += lighting;
    }

    return result;
}
