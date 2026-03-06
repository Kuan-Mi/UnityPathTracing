// PointLights.hlsl
// Omnidirectional point lights with hard shadow rays.
// All point lights are accumulated into a single float3 per pixel.
// No NRD denoising is applied — the result is composited directly.

struct PointLight
{
    float3 position;    // World-space position
    float  range;       // Maximum range (hard cutoff)
    float3 color;       // Pre-multiplied color * intensity
    float  pad;
};

StructuredBuffer<PointLight> gIn_PointLights;

// Evaluate the direct lighting contribution of all point lights at a surface point.
// Shadow rays are hard (single ray, no angular jitter) since these are ideal point lights.
float3 EvaluatePointLights(GeometryProps geo, MaterialProps mat)
{
    float3 result = 0.0;

    [loop]
    for (uint i = 0; i < gPointLightCount; i++)
    {
        PointLight light = gIn_PointLights[i];

        // ---------------------------------------------------------------
        // Geometry term
        // ---------------------------------------------------------------
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

    return result;
}
