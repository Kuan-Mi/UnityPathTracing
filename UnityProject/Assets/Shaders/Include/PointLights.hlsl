// PointLights.hlsl
// Omnidirectional point lights, optionally with a sphere radius for soft shadows.
// When radius == 0: hard shadow ray (ideal point light).
// When radius  > 0: stochastic sample on the hemisphere facing the surface,
//                   producing soft shadows that converge via temporal accumulation.
// All point lights are accumulated into a single float3 per pixel.
// No NRD denoising is applied — the result is composited directly.

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
// soft shadows emerge via temporal accumulation (same strategy as area lights).
float3 EvaluatePointLights(GeometryProps geo, MaterialProps mat)
{
    float3 result = 0.0;

    [loop]
    for (uint i = 0; i < gPointLightCount; i++)
    {
        PointLight light = gIn_PointLights[i];

        // ---------------------------------------------------------------
        // Determine sample position on the light
        // ---------------------------------------------------------------
        float3 samplePos;
        float  solidAngleWeight;   // cosLight * lightArea / dist^2  (or 1/dist^2 for a point)
        float  shadowDist;

        if (light.radius > 0.0001)
        {
            // -----------------------------------------------------------
            // Sphere light: uniformly sample the hemisphere facing geo.X.
            // light.color = luminous intensity I (W/sr), same as hard point light.
            // Surface radiance: L_e = I / (pi*r^2).
            // PDF = 1 / (2*pi*r^2).  Estimator = L_e * cosLight * 2*pi*r^2 / dist^2
            //                                   = I * cosLight * 2 / dist^2   (r^2 cancels).
            // -----------------------------------------------------------
            float3 toSurface = normalize(geo.X - light.position);

            // Build an orthonormal basis aligned to toSurface
            float3 T, B;
            float3 up = abs(toSurface.x) < 0.9 ? float3(1, 0, 0) : float3(0, 1, 0);
            T = normalize(cross(toSurface, up));
            B = cross(toSurface, T);

            // Uniform hemisphere sampling
            float2 xi    = Rng::Hash::GetFloat2();
            float  cosT  = xi.x;                            // cosine from hemisphere pole
            float  sinT  = sqrt(max(0.0, 1.0 - cosT * cosT));
            float  phi   = 6.28318530718 * xi.y;

            float3 sampleDir = T * (sinT * cos(phi))
                             + B * (sinT * sin(phi))
                             + toSurface * cosT;            // unit vector toward surface side

            samplePos = light.position + sampleDir * light.radius;

            float3 toLight = samplePos - geo.X;
            float  dist    = length(toLight);
            if (dist < 0.0001) continue;

            float3 L   = toLight / dist;
            float  NoL = saturate(dot(mat.N, L));
            if (NoL == 0.0) continue;

            // Range hard cutoff (use sphere center distance for consistency)
            float centDist = length(light.position - geo.X);
            if (centDist >= light.range) continue;

            float cosLight = dot(sampleDir, -L);   // > 0 by hemisphere construction
            if (cosLight <= 0.0) continue;

            // light.color is luminous intensity I (W/sr), same convention as the hard point light.
            // Surface radiance of the sphere: L_e = I / (pi * r^2).
            // Monte Carlo estimator: L_e * 2*pi*r^2 * cosLight / dist^2
            //                      = I * 2 * cosLight / dist^2   (r^2 cancels).
            // This makes brightness independent of radius, matching the point light at r->0.
            float rangeFade    = Math::SmoothStep(light.range, light.range * 0.75, centDist);
            solidAngleWeight   = cosLight * 2.0 / max(dist * dist, 0.0001) * rangeFade;
            shadowDist         = dist;

            // ---------------------------------------------------------------
            // Stochastic shadow ray
            // ---------------------------------------------------------------
            float3 Xoffset    = geo.GetXoffset(L, PT_BOUNCE_RAY_OFFSET);
            float2 mipAndCone = float2(geo.mip, 0.0);

            float shadowHitT = CastVisibilityRay_AnyHit(
                Xoffset, L,
                0.0, shadowDist,
                mipAndCone,
                gWorldTlas,
                FLAG_NON_TRANSPARENT, 0);

            if (shadowHitT != INF)
                continue;   // Occluded

            // ---------------------------------------------------------------
            // PBR BRDF
            // ---------------------------------------------------------------
            float3 albedo, Rf0;
            BRDF::ConvertBaseColorMetalnessToAlbedoRf0(mat.baseColor, mat.metalness, albedo, Rf0);

            float3 V   = geo.V;
            float3 H   = normalize(L + V);
            float  NoH = saturate(dot(mat.N, H));
            float  VoH = saturate(dot(V, H));
            float  NoV = abs(dot(mat.N, V));

            float  D    = BRDF::DistributionTerm(mat.roughness, NoH);
            float  G    = BRDF::GeometryTermMod(mat.roughness, NoL, NoV, VoH, NoH);
            float3 F    = BRDF::FresnelTerm(Rf0, VoH);
            float  Kd   = BRDF::DiffuseTerm(mat.roughness, NoL, NoV, VoH);

            float3 Cspec = F * D * G * NoL;
            float3 Cdiff = Kd * albedo * NoL;
            float3 brdf  = Cspec + Cdiff * (1.0 - F);

            result += light.color * brdf * solidAngleWeight;
        }
        else
        {
            // -----------------------------------------------------------
            // Ideal point light: hard shadow ray, inverse-square falloff.
            // -----------------------------------------------------------
            float3 toLight = light.position - geo.X;
            float  dist    = length(toLight);

            // Range hard cutoff
            if (dist >= light.range)
                continue;

            float3 L   = toLight / dist;
            float  NoL = saturate(dot(mat.N, L));

            if (NoL == 0.0)
                continue;

            // ---------------------------------------------------------------
            // Distance attenuation: inverse-square with smooth range rolloff
            // ---------------------------------------------------------------
            float atten     = 1.0 / max(dist * dist, 0.0001);
            float rangeFade = Math::SmoothStep(light.range, light.range * 0.75, dist);
            atten *= rangeFade;

            // ---------------------------------------------------------------
            // Hard shadow ray (single ray — point light has no angular radius)
            // ---------------------------------------------------------------
            float3 Xoffset    = geo.GetXoffset(L, PT_BOUNCE_RAY_OFFSET);
            float2 mipAndCone = float2(geo.mip, 0.0);

            float shadowHitT = CastVisibilityRay_AnyHit(
                Xoffset, L,
                0.0, dist,
                mipAndCone,
                gWorldTlas,
                FLAG_NON_TRANSPARENT, 0);

            if (shadowHitT != INF)
                continue;   // Occluded

            // ---------------------------------------------------------------
            // PBR BRDF (same D·G·F path as spot lights)
            // ---------------------------------------------------------------
            float3 albedo, Rf0;
            BRDF::ConvertBaseColorMetalnessToAlbedoRf0(mat.baseColor, mat.metalness, albedo, Rf0);

            float3 V   = geo.V;
            float3 H   = normalize(L + V);
            float  NoH = saturate(dot(mat.N, H));
            float  VoH = saturate(dot(V, H));
            float  NoV = abs(dot(mat.N, V));

            float  D    = BRDF::DistributionTerm(mat.roughness, NoH);
            float  G    = BRDF::GeometryTermMod(mat.roughness, NoL, NoV, VoH, NoH);
            float3 F    = BRDF::FresnelTerm(Rf0, VoH);
            float  Kd   = BRDF::DiffuseTerm(mat.roughness, NoL, NoV, VoH);

            float3 Cspec = F * D * G * NoL;
            float3 Cdiff = Kd * albedo * NoL;
            float3 brdf  = Cspec + Cdiff * (1.0 - F);

            result += light.color * brdf * atten;
        }
    }

    return result;
}
