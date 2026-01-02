#include "Sharc/SharcCommon.h"
RaytracingAccelerationStructure gWorldTlas : register(t0, space1);

RWStructuredBuffer<uint64_t> gInOut_SharcHashEntriesBuffer;
RWStructuredBuffer<SharcAccumulationData> gInOut_SharcAccumulated;
RWStructuredBuffer<SharcPackedData> gInOut_SharcResolved;

#define RTXCR_INTEGRATION 0

#if( USE_STOCHASTIC_SAMPLING == 1 )
#define TEX_SAMPLER gNearestMipmapNearestSampler
#else
#define TEX_SAMPLER gLinearMipmapLinearSampler
#endif

#if( USE_LOAD == 1 )
#define SAMPLE( coords ) Load( int3( coords ) )
#else
#define SAMPLE( coords ) SampleLevel( TEX_SAMPLER, coords.xy, coords.z )
#endif

float3x3 Hair_GetBasis(float3 N, float3 T)
{
    float3 B = cross(N, T);

    return float3x3(T, B, N);
}

#include "Payload.hlsl"

struct GeometryProps
{
    float3 X; // 命中点的世界空间坐标
    float3 Xprev; // 命中点在上一帧的世界空间坐标（用于时序去噪/运动矢量）
    float3 V; // 视线方向（通常为 -ray 方向）
    float4 T; // 切线向量（xyz）和副切线符号（w）
    float3 N; // 法线向量（世界空间）
    float mip;
    float hitT; // 光线命中的距离（t值），INF表示未命中
    float curvature; // 曲率估算值（用于材质、去噪等）
    uint textureOffsetAndFlags;
    uint instanceIndex; // 命中的实例索引（用于查找InstanceData）

    float3 GetXoffset(float3 offsetDir, float amount = PT_BOUNCE_RAY_OFFSET)
    {
        float viewZ = Geometry::AffineTransform(gWorldToView, X).z;
        amount *= gUnproject * abs(viewZ);

        return X + offsetDir * max(amount, 0.00001);
    }

    bool Has(uint flag)
    {
        return (textureOffsetAndFlags & (flag << FLAG_FIRST_BIT)) != 0;
    }

    void SetFlag(uint flag)
    {
        textureOffsetAndFlags |= (flag << FLAG_FIRST_BIT);
    }

    bool IsMiss()
    {
        return hitT == INF;
    }
};

struct MaterialProps
{
    float3 Lemi;
    float3 N;
    float3 T;
    float3 baseColor;
    float roughness;
    float metalness;
    float curvature;
};

float2 GetConeAngleFromAngularRadius(float mip, float tanConeAngle)
{
    // In any case, we are limited by the output resolution
    tanConeAngle = max(tanConeAngle, gTanPixelAngularRadius);

    return float2(mip, tanConeAngle);
}

float2 GetConeAngleFromRoughness(float mip, float roughness)
{
    float tanConeAngle = roughness * roughness * 0.05; // TODO: tweaked to be accurate and give perf boost

    return GetConeAngleFromAngularRadius(mip, tanConeAngle);
}

[shader("miss")]
void MainMissShader(inout MainRayPayload payload : SV_RayPayload)
{
    payload.hitT = INF;
    float3 ray = WorldRayDirection();
    payload.X = WorldRayOrigin() + ray * payload.hitT;
    payload.Xprev = payload.X;

    payload.Lemi = GetSkyIntensity(ray);
}

[shader("miss")]
void MissShadow(inout MainRayPayload payload : SV_RayPayload)
{
    payload.hitT = INF;
}

void CastRay(float3 origin, float3 direction, float Tmin, float Tmax, float2 mipAndCone, uint flag, out GeometryProps props, out MaterialProps matProps)
{
    RayDesc rayDesc;
    rayDesc.Origin = origin;
    rayDesc.Direction = direction;
    rayDesc.TMin = Tmin;
    rayDesc.TMax = Tmax;

    MainRayPayload payload = (MainRayPayload)0;
    payload.mipAndCone = mipAndCone;

    TraceRay(gWorldTlas, flag, 0xFF, 0, 1, 0, rayDesc, payload);

    props = (GeometryProps)0;
    props.mip = mipAndCone.x;
    props.hitT = payload.hitT;
    props.instanceIndex = payload.instanceIndex;
    props.N = payload.N;
    props.curvature = payload.curvature;


    props.mip = payload.mipAndCone.x;

    props.T = payload.T;
    props.X = payload.X;
    // 全静止物体
    props.Xprev = payload.X;
    props.V = -direction;
    props.textureOffsetAndFlags = payload.textureOffsetAndFlags;

    matProps = (MaterialProps)0;
    matProps.baseColor = payload.baseColor;
    matProps.roughness = payload.roughness;
    matProps.metalness = payload.metalness;
    matProps.Lemi = payload.Lemi;
    // 这三个应该从贴图再计算一次
    matProps.curvature = payload.curvature;
    matProps.N = payload.matN;
    matProps.T = payload.T.xyz;
}

// Compile-time flags for "GetLighting"
#define LIGHTING    0x01
#define SHADOW      0x02
#define SSS         0x04

float3 GetLighting(GeometryProps geometryProps, MaterialProps materialProps, uint flags, out float3 Xshadow)
{
    float3 lighting = 0.0;

    // Lighting
    Xshadow = geometryProps.X;

    if ((flags & LIGHTING) != 0)
    {
        float3 Csun = GetSunIntensity(gSunDirection.xyz);
        float3 Csky = GetSkyIntensity(-geometryProps.V);
        float NoL = saturate(dot(geometryProps.N, gSunDirection.xyz));
        bool isSSS = (flags & SSS) != 0 && geometryProps.Has(FLAG_SKIN);
        float minThreshold = isSSS ? -0.2 : 0.03; // TODO: hand-tuned for SSS, a helper in RTXCR SDK is needed
        float shadow = Math::SmoothStep(minThreshold, 0.1, NoL);

        // COMMON MATERIAL
        if (shadow != 0.0)
        {
            // Extract materials
            float3 albedo, Rf0;
            BRDF::ConvertBaseColorMetalnessToAlbedoRf0(materialProps.baseColor.xyz, materialProps.metalness, albedo, Rf0);

            // Pseudo sky importance sampling
            float3 Cimp = lerp(Csky, Csun, Math::SmoothStep(0.0, 0.2, materialProps.roughness));
            Cimp *= Math::SmoothStep(-0.01, 0.05, gSunDirection.z);

            // Common BRDF
            float3 N = materialProps.N;
            float3 L = gSunDirection.xyz;
            float3 V = geometryProps.V;
            float3 H = normalize(L + V);

            float NoL = saturate(dot(N, L));
            float NoH = saturate(dot(N, H));
            float VoH = saturate(dot(V, H));
            float NoV = abs(dot(N, V));

            float D = BRDF::DistributionTerm(materialProps.roughness, NoH);
            float G = BRDF::GeometryTermMod(materialProps.roughness, NoL, NoV, VoH, NoH);
            float3 F = BRDF::FresnelTerm(Rf0, VoH);
            float Kdiff = BRDF::DiffuseTerm(materialProps.roughness, NoL, NoV, VoH);

            float3 Cspec = saturate(F * D * G * NoL);
            float3 Cdiff = Kdiff * Csun * albedo * NoL;

            lighting = Cspec * Cimp;

            lighting += Cdiff * (1.0 - F);
            lighting *= shadow;
        }
    }
    else
        lighting = 1.0;

    // Shadow
    const uint instanceInclusionMask = FLAG_NON_TRANSPARENT; // Default shadow rays must ignore transparency // TODO: what about translucency?
    const uint rayFlags = 0;

    if ((flags & SHADOW) != 0 && Color::Luminance(lighting) != 0 && !gDisableShadowsAndEnableImportanceSampling)
    {
        float2 rnd = Rng::Hash::GetFloat2();
        rnd = ImportanceSampling::Cosine::GetRay(rnd).xy;
        rnd *= gTanSunAngularRadius;

        float3 sunDirection = normalize(gSunBasisX.xyz * rnd.x + gSunBasisY.xyz * rnd.y + gSunDirection.xyz);

        // float2 mipAndCone = GetConeAngleFromAngularRadius(geometryProps.mip, gTanSunAngularRadius);
        // float hitT = CastVisibilityRay_AnyHit(Xshadow, sunDirection, 0.0, INF, mipAndCone, gWorldTlas, instanceInclusionMask, rayFlags);

        RayDesc rayDesc;
        rayDesc.Origin = Xshadow;
        rayDesc.Direction = sunDirection;
        rayDesc.TMin = 0;
        rayDesc.TMax = 1000;

        MainRayPayload shadowPayload = (MainRayPayload)0;
        TraceRay(gWorldTlas, RAY_FLAG_NONE | RAY_FLAG_CULL_NON_OPAQUE, 0xFF, 0, 1, 1, rayDesc, shadowPayload);
        float hitT = shadowPayload.hitT;

        lighting *= float(hitT == INF);
    }

    return lighting;
}

float3 GetLighting(GeometryProps geometryProps, MaterialProps materialProps, uint flags)
{
    float3 unused;
    return GetLighting(geometryProps, materialProps, flags, unused);
}


// Compile-time flags for "GenerateRayAndUpdateThroughput"
#define HAIR 0x1

float3 GenerateRayAndUpdateThroughput(inout GeometryProps geometryProps, inout MaterialProps materialProps, inout float3 throughput, uint sampleMaxNum, bool isDiffuse, float2 rnd, uint flags)
{
    // bool isHair = ( flags & HAIR ) != 0 && RTXCR_INTEGRATION == 1 && geometryProps.Has( FLAG_HAIR );

    bool isHair = false;
    float3x3 mLocalBasis = isHair ? Hair_GetBasis(materialProps.N, materialProps.T) : Geometry::GetBasis(materialProps.N);
    float3 Vlocal = Geometry::RotateVector(mLocalBasis, geometryProps.V);

    // Importance sampling
    float3 rayLocal = 0;
    uint emissiveHitNum = 0;

    for (uint sampleIndex = 0; sampleIndex < sampleMaxNum; sampleIndex++)
    {
        // Generate a ray in local space
        float3 candidateRayLocal;
        #if( RTXCR_INTEGRATION == 1 )
        if (isHair)
        {
            float2 rand[2] = {Rng::Hash::GetFloat2(), Rng::Hash::GetFloat2()};

            float3 specular = 0.0;
            float3 diffuse = 0.0;
            float pdf = 0.0;

            RTXCR_HairInteractionSurface hairSurface = Hair_GetSurface(Vlocal);
            RTXCR_HairMaterialInteractionBcsdf hairMaterial = Hair_GetMaterial();
            RTXCR_SampleFarFieldBcsdf(hairSurface, hairMaterial, Vlocal, 2.0 * rnd.x - 1.0, rnd.y, rand, candidateRayLocal, specular, diffuse, pdf);
        }
        else
        #endif
        if (isDiffuse)
            candidateRayLocal = ImportanceSampling::Cosine::GetRay(rnd);
        else
        {
            float3 Hlocal = ImportanceSampling::VNDF::GetRay(rnd, materialProps.roughness, Vlocal, PT_SPEC_LOBE_ENERGY);
            candidateRayLocal = reflect(-Vlocal, Hlocal);
        }

        // If IS enabled, check the candidate in LightBVH
        bool isEmissiveHit = false;
        // if( gDisableShadowsAndEnableImportanceSampling && sampleMaxNum != 1 )
        // {
        //     float3 candidateRay = Geometry::RotateVectorInverse( mLocalBasis, candidateRayLocal );
        //     float2 mipAndCone = GetConeAngleFromRoughness( geometryProps.mip, isDiffuse ? 1.0 : materialProps.roughness );
        //     float3 Xoffset = geometryProps.GetXoffset( geometryProps.N );
        //
        //     float distanceToLight = CastVisibilityRay_AnyHit( Xoffset, candidateRay, 0.0, INF, mipAndCone, gLightTlas, FLAG_NON_TRANSPARENT, PT_RAY_FLAGS );
        //     isEmissiveHit = distanceToLight != INF;
        //
        // #if( USE_BIAS_FIX == 1 )
        //     // Checking the candidate ray in "gWorldTlas" to get occlusion information eliminates negligible specular and hair bias
        //     if( isEmissiveHit && !isDiffuse )
        //     {
        //         float distanceToOccluder = CastVisibilityRay_AnyHit( Xoffset, candidateRay, 0.0, distanceToLight, mipAndCone, gWorldTlas, FLAG_NON_TRANSPARENT, PT_RAY_FLAGS );
        //         isEmissiveHit = distanceToOccluder >= distanceToLight;
        //     }
        // #endif
        // }

        // Count rays hitting emissive surfaces
        if (isEmissiveHit)
            emissiveHitNum++;

        // Save either the first ray or the last ray hitting an emissive
        if (isEmissiveHit || sampleIndex == 0)
            rayLocal = candidateRayLocal;

        rnd = Rng::Hash::GetFloat2();
    }

    // Adjust throughput by percentage of rays hitting any emissive surface
    // IMPORTANT: do not modify throughput if there is no an emissive hit, it's needed for a non-IS ray
    if (emissiveHitNum != 0)
        throughput *= float(emissiveHitNum) / float(sampleMaxNum);

    // Update throughput
    #if( NRD_MODE < OCCLUSION )
    float3 albedo, Rf0;
    BRDF::ConvertBaseColorMetalnessToAlbedoRf0(materialProps.baseColor, materialProps.metalness, albedo, Rf0);

    float3 Nlocal = float3(0, 0, 1);
    float3 Hlocal = normalize(Vlocal + rayLocal);

    float NoL = saturate(dot(Nlocal, rayLocal));
    float VoH = abs(dot(Vlocal, Hlocal));

    #if( RTXCR_INTEGRATION == 1 )
    if (isHair)
    {
        float3 specular = 0.0;
        float3 diffuse = 0.0;
        float pdf = 0.0;

        RTXCR_HairInteractionSurface hairGeometry = Hair_GetSurface(Vlocal);
        RTXCR_HairMaterialInteractionBcsdf hairMaterial = Hair_GetMaterial();
        RTXCR_HairFarFieldBcsdfEval(hairGeometry, hairMaterial, rayLocal, Vlocal, specular, diffuse, pdf);

        throughput *= pdf > 0.0 ? (specular + diffuse) / pdf : 0.0;
    }
    else
    #endif
    if (isDiffuse)
    {
        float NoV = abs(dot(Nlocal, Vlocal));

        // NoL is canceled by "Cosine::GetPDF"
        throughput *= albedo;
        throughput *= Math::Pi(1.0) * BRDF::DiffuseTerm_Burley(materialProps.roughness, NoL, NoV, VoH); // PI / PI
    }
    else
    {
        // See paragraph "Usage in Monte Carlo renderer" from http://jcgt.org/published/0007/04/01/paper.pdf
        float3 F = BRDF::FresnelTerm_Schlick(Rf0, VoH);

        throughput *= F;
        throughput *= BRDF::GeometryTerm_Smith(materialProps.roughness, NoL);
    }

    // Translucency
    if (USE_TRANSLUCENCY && geometryProps.Has(FLAG_LEAF) && isDiffuse)
    {
        if (Rng::Hash::GetFloat() < LEAF_TRANSLUCENCY)
        {
            rayLocal = -rayLocal;
            geometryProps.X -= LEAF_THICKNESS * geometryProps.N;
            throughput /= LEAF_TRANSLUCENCY;
        }
        else
            throughput /= 1.0 - LEAF_TRANSLUCENCY;
    }
    #endif

    // Transform to world space
    float3 ray = Geometry::RotateVectorInverse(mLocalBasis, rayLocal);

    // ( Optional ) Helpful insignificant fixes
    float NoLgeom = dot(geometryProps.N, ray);
    if (!isHair && NoLgeom < 0.0)
    {
        if (isDiffuse)
        {
            // Terminate diffuse paths pointing inside the surface
            throughput = 0.0;
        }
        else
        {
            // Patch ray direction and shading normal to avoid self-intersections ( https://arxiv.org/pdf/1705.01263.pdf, Appendix 3 )
            float b = abs(dot(geometryProps.N, materialProps.N)) * 0.99;

            ray = normalize(ray + geometryProps.N * abs(NoLgeom) * Math::PositiveRcp(b));
            materialProps.N = normalize(geometryProps.V + ray);
        }
    }

    return ray;
}

float3 GetMaterialDemodulation(GeometryProps geometryProps, MaterialProps materialProps)
{
    float3 albedo, Rf0;
    BRDF::ConvertBaseColorMetalnessToAlbedoRf0(materialProps.baseColor, materialProps.metalness, albedo, Rf0);

    float NoV = abs(dot(geometryProps.N, geometryProps.V));
    float3 Fenv = _NRD_EnvironmentTerm_Rtg(Rf0, NoV, materialProps.roughness);

    return (albedo + Fenv) * 0.95 + 0.05;
}

float GetDeltaEventRay(GeometryProps geometryProps, bool isReflection, float eta, out float3 Xoffset, out float3 ray)
{
    if (isReflection)
        ray = reflect(-geometryProps.V, geometryProps.N);
    else
    {
        float3 I = -geometryProps.V;
        float NoI = dot(geometryProps.N, I);
        float k = max(1.0 - eta * eta * (1.0 - NoI * NoI), 0.0);

        ray = normalize(eta * I - (eta * NoI + sqrt(k)) * geometryProps.N);
        eta = 1.0 / eta;
    }

    float amount = geometryProps.Has(FLAG_TRANSPARENT) ? PT_GLASS_RAY_OFFSET : PT_BOUNCE_RAY_OFFSET;
    float s = Math::Sign(dot(ray, geometryProps.N));

    Xoffset = geometryProps.GetXoffset(geometryProps.N * s, amount);

    return eta;
}

bool IsDelta(MaterialProps materialProps)
{
    return materialProps.roughness < 0.041 // TODO: tweaked for kitchen
        && (materialProps.metalness > 0.941 || Color::Luminance(materialProps.baseColor) < 0.005)
        && sqrt(abs(materialProps.curvature)) < 2.5;
}

float EstimateDiffuseProbability(GeometryProps geometryProps, MaterialProps materialProps, bool useMagicBoost = false)
{
    // IMPORTANT: can't be used for hair tracing, but applicable in other hair related calculations
    float3 albedo, Rf0;
    BRDF::ConvertBaseColorMetalnessToAlbedoRf0(materialProps.baseColor, materialProps.metalness, albedo, Rf0);

    float NoV = abs(dot(materialProps.N, geometryProps.V));
    float3 Fenv = BRDF::EnvironmentTerm_Rtg(Rf0, NoV, materialProps.roughness);

    float lumSpec = Color::Luminance(Fenv);
    float lumDiff = Color::Luminance(albedo * (1.0 - Fenv));

    float diffProb = lumDiff / max(lumDiff + lumSpec, NRD_EPS);

    // Boost diffussiness ( aka diffuse-like behavior ) if roughness is high
    if (useMagicBoost)
        diffProb = lerp(diffProb, 1.0, GetSpecMagicCurve(materialProps.roughness));

    // Clamp probability to a sane range. High energy fireflies are very undesired. They can be get rid of only
    // if the number of accumulated samples exeeds 100-500. NRD accumulates for not more than 30 frames only
    float diffProbClamped = clamp(diffProb, 1.0 / PT_MAX_FIREFLY_RELATIVE_INTENSITY, 1.0 - 1.0 / PT_MAX_FIREFLY_RELATIVE_INTENSITY);

    [flatten]
    if (diffProb < PT_EVIL_TWIN_LOBE_TOLERANCE)
        return 0.0; // no diffuse materials are common ( metals )
    else if (diffProb > 1.0 - PT_EVIL_TWIN_LOBE_TOLERANCE)
        return 1.0; // no specular materials are uncommon ( broken material model? )
    else
        return diffProbClamped;
}

float ReprojectIrradiance(bool isPrevFrame, bool isRefraction, Texture2D<float3> texDiff, Texture2D<float4> texSpecViewZ, GeometryProps geometryProps, uint2 pixelPos, out float3 Ldiff, out float3 Lspec)
{
    // Get UV and ignore back projection
    float2 uv = Geometry::GetScreenUv(isPrevFrame ? gWorldToClipPrev : gWorldToClip, geometryProps.X, true) - gJitter;

    float2 rescale = (isPrevFrame ? gRectSizePrev : gRectSize) * gInvRenderSize;
    float4 data = texSpecViewZ.SampleLevel(gNearestSampler, uv * rescale, 0);
    float prevViewZ = abs(data.w) / FP16_VIEWZ_SCALE;

    // Initial state
    float weight = 1.0;
    float2 pixelUv = float2(pixelPos + 0.5) * gInvRectSize;

    // Relaxed checks for refractions
    float viewZ = abs(Geometry::AffineTransform(isPrevFrame ? gWorldToViewPrev : gWorldToView, geometryProps.X).z);
    float err = (viewZ - prevViewZ) * Math::PositiveRcp(max(viewZ, prevViewZ));

    if (isRefraction)
    {
        // Confidence - viewZ ( PSR makes prevViewZ further than the original primary surface )
        weight *= Math::LinearStep(0.01, 0.005, saturate(err));

        // Fade-out on screen edges ( hard )
        weight *= all(saturate(uv) == uv);
    }
    else
    {
        // Confidence - viewZ
        weight *= Math::LinearStep(0.01, 0.005, abs(err));

        // Fade-out on screen edges ( soft )
        float2 f = Math::LinearStep(0.0, 0.1, uv) * Math::LinearStep(1.0, 0.9, uv);
        weight *= f.x * f.y;

        // Confidence - ignore back-facing
        // Instead of storing previous normal we can store previous NoL, if signs do not match we hit the surface from the opposite side
        float NoL = dot(geometryProps.N, gSunDirection.xyz);
        weight *= float(NoL * Math::Sign(data.w) > 0.0);

        // Confidence - ignore too short rays
        float2 uv = Geometry::GetScreenUv(gWorldToClip, geometryProps.X, true) - gJitter;
        float d = length((uv - pixelUv) * gRectSize);
        weight *= Math::LinearStep(1.0, 3.0, d);
    }

    // Ignore sky
    weight *= float(!geometryProps.IsMiss());

    // Use only if radiance is on the screen
    weight *= float(gOnScreen < SHOW_AMBIENT_OCCLUSION);

    // Add global confidence
    if (isPrevFrame)
        weight *= gPrevFrameConfidence; // see C++ code for details

    // Read data
    Ldiff = texDiff.SampleLevel(gNearestSampler, uv * rescale, 0);
    Lspec = data.xyz;

    // Avoid NANs
    [flatten]
    if (any(isnan(Ldiff) | isinf(Ldiff) | isnan(Lspec) | isinf(Lspec)) || NRD_MODE >= OCCLUSION) // TODO: needed?
    {
        Ldiff = 0;
        Lspec = 0;
        weight = 0;
    }

    // Avoid really bad reprojection
    float f = saturate(weight / 0.001);
    Ldiff *= f;
    Lspec *= f;

    return weight;
}
