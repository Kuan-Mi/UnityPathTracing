// AreaLights.hlsl
// Rectangular area lights with single stochastic shadow rays.
// All area lights are accumulated into a single float3 per pixel.
// No NRD denoising is applied — the result is composited directly.

struct AreaLight
{
    float3 position;    // World-space centre of the rectangle
    float  halfWidth;   // Half-extent along the right axis
    float3 right;       // Unit right vector (world-space)
    float  halfHeight;  // Half-extent along the up axis
    float3 up;          // Unit up vector (world-space)
    float  pad;
    float3 color;       // Pre-multiplied color * intensity
    float  pad2;
};

StructuredBuffer<AreaLight> gIn_AreaLights;

// Evaluate the direct lighting contribution of all rectangular area lights at a surface point.
// A single stochastic shadow ray is cast per light per frame; soft shadows emerge via accumulation.
float3 EvaluateAreaLights(GeometryProps geo, MaterialProps mat)
{
    float3 result = 0.0;

    [loop]
    for (uint i = 0; i < gAreaLightCount; i++)
    {
        AreaLight light = gIn_AreaLights[i];

        // ---------------------------------------------------------------
        // Derive light normal (one-sided: front face only)
        // The normal points in the direction the light emits.
        // ---------------------------------------------------------------
        float3 lightNormal = normalize(cross(light.right, light.up));

        // ---------------------------------------------------------------
        // Stochastic sample point on the light surface
        // ---------------------------------------------------------------
        float2 xi = Rng::Hash::GetFloat2() * 2.0 - 1.0; // [-1, 1]^2
        float3 samplePos = light.position
                         + light.right  * (xi.x * light.halfWidth)
                         + light.up     * (xi.y * light.halfHeight);

        // ---------------------------------------------------------------
        // Geometry term
        // ---------------------------------------------------------------
        float3 toLight = samplePos - geo.X;
        float  dist    = length(toLight);

        if (dist < 0.0001)
            continue;

        float3 L   = toLight / dist;
        float  NoL = saturate(dot(mat.N, L));

        if (NoL == 0.0)
            continue;

        // One-sided: reject rays hitting the back face of the light
        float cosLight = dot(lightNormal, -L); // > 0 when surface is on the emitting side
        if (cosLight <= 0.0)
            continue;

        // ---------------------------------------------------------------
        // Solid-angle weight
        // Solid angle contribution ≈ A * cos(theta_light) / dist^2
        // Rectangle area = (2 * halfWidth) * (2 * halfHeight) = 4 * halfWidth * halfHeight
        // ---------------------------------------------------------------
        float solidAngle = cosLight * (4.0 * light.halfWidth * light.halfHeight)
                         / max(dist * dist, 0.0001);

        // ---------------------------------------------------------------
        // Shadow ray (single stochastic sample — soft shadows via accumulation)
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

        result += light.color * brdf * solidAngle;
    }

    return result;
}
