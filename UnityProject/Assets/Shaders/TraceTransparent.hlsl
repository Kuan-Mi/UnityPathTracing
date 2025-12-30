#include "ml.hlsli"
#include "NRDInclude/NRD.hlsli"
#include "RayPayload.hlsl"
#include "GlobalResource.hlsl"
#include "Shared.hlsl"

Texture2D<float3> gIn_ComposedDiff;
Texture2D<float4> gIn_ComposedSpec_ViewZ;

RWTexture2D<float3> gOut_Composed;
RWTexture2D<float4> gInOut_Mv;


[shader("miss")]
void MainMissShader(inout MainRayPayload payload : SV_RayPayload)
{
    payload.hitT = INF;
    float3 ray = WorldRayDirection();
    payload.X = WorldRayOrigin() + ray * payload.hitT;
    payload.Xprev = payload.X;

    payload.Lemi = GetSkyIntensity(ray);

    // payload.emission = g_EnvTex.SampleLevel(sampler_g_EnvTex, WorldRayDirection(), 0).xyz;
    //
    // // payload.emission = float3(0.5,0.5,0.5); // 固定背景色，便于调试
    // payload.bounceIndexOpaque = -1;
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

    TraceRay(g_AccelStruct, flag, 0xFF, 0, 1, 0, rayDesc, payload);


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


void GetCameraRay(out float3 origin, out float3 direction, float2 sampleUv)
{
    // https://www.slideshare.net/TiagoAlexSousa/graphics-gems-from-cryengine-3-siggraph-2013 ( slides 23+ )

    // Pinhole ray
    float3 Xv = Geometry::ReconstructViewPosition(sampleUv, gCameraFrustum, gNearZ);
    direction = normalize(Xv);

    // Distorted ray
    float2 rnd = Rng::Hash::GetFloat2();
    rnd = ImportanceSampling::Cosine::GetRay(rnd).xy;
    Xv.xy += rnd * gAperture;

    float3 Fv = direction * gFocalDistance; // z-plane
    #if 0
    Fv /= dot(vForward, direction); // radius
    #endif

    origin = Geometry::AffineTransform(gViewToWorld, Xv);
    direction = normalize(Geometry::RotateVector(gViewToWorld, Fv - Xv));
}

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

float3 GetMotion(float3 X, float3 Xprev)
{
    float3 motion = Xprev - X;

    float viewZ = -Geometry::AffineTransform(gWorldToView, X).z;
    float2 sampleUv = Geometry::GetScreenUv(gWorldToClip, X);

    float viewZprev = -Geometry::AffineTransform(gWorldToViewPrev, Xprev).z;
    float2 sampleUvPrev = Geometry::GetScreenUv(gWorldToClipPrev, Xprev);

    // IMPORTANT: scaling to "pixel" unit significantly improves utilization of FP16
    motion.xy = (sampleUvPrev - sampleUv) * gRectSize;

    // IMPORTANT: 2.5D motion is preferred over 3D motion due to imprecision issues caused by FP16 rounding negative effects
    motion.z = viewZprev - viewZ;
    // return 0;
    return motion;
}


struct TraceTransparentDesc
{
    // Geometry properties
    GeometryProps geometryProps;

    // Pixel position
    uint2 pixelPos;

    // Is reflection or refraction in first segment?
    bool isReflection;
};


#define PT_DELTA_BOUNCES_NUM                8

#define PT_GLASS_MIN_F                      0.05 // adds a bit of stability and bias


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


SamplerState sampler_point_clamp;

float ReprojectIrradiance(bool isPrevFrame, bool isRefraction, Texture2D<float3> texDiff, Texture2D<float4> texSpecViewZ, GeometryProps geometryProps, uint2 pixelPos, out float3 Ldiff, out float3 Lspec)
{
    // Get UV and ignore back projection
    float2 uv = Geometry::GetScreenUv(isPrevFrame ? gWorldToClipPrev : gWorldToClip, geometryProps.X, true) - gJitter;

    float2 rescale = (isPrevFrame ? gRectSizePrev : gRectSize) * gInvRenderSize;
    float4 data = texSpecViewZ.SampleLevel(sampler_point_clamp, uv * rescale, 0);
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
    // weight *= float( gOnScreen < SHOW_AMBIENT_OCCLUSION );
    // weight *= float( gOnScreen < SHOW_AMBIENT_OCCLUSION );

    // Add global confidence
    if (isPrevFrame)
        weight *= gPrevFrameConfidence; // see C++ code for details

    // Read data
    Ldiff = texDiff.SampleLevel(sampler_point_clamp, uv * rescale, 0);
    Lspec = data.xyz;

    // Avoid NANs
    [flatten]
    if (any(isnan(Ldiff) | isinf(Ldiff) | isnan(Lspec) | isinf(Lspec))) // TODO: needed?
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


#define LIGHTING    0x01
#define SHADOW      0x02
#define SSS         0x04


float3 GetLighting(GeometryProps geometryProps, inout MaterialProps materialProps, uint flags)
{
    float3 lighting = 0.0;

    // Lighting
    float3 Xshadow = geometryProps.X;

    float3 Csun = GetSunIntensity(gSunDirection.xyz);
    float3 Csky = GetSkyIntensity(-geometryProps.V);
    float NoL = saturate(dot(geometryProps.N, gSunDirection.xyz));
    bool isSSS = false;
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
        Cimp *= Math::SmoothStep(-0.01, 0.05, gSunDirection.y);

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

    // Shadow
    if ((flags & SHADOW) != 0 && Color::Luminance(lighting) != 0)
    {
        float2 rnd = Rng::Hash::GetFloat2();
        rnd = ImportanceSampling::Cosine::GetRay(rnd).xy;
        rnd *= gTanSunAngularRadius;

        float3 sunDirection = normalize(gSunBasisX.xyz * rnd.x + gSunBasisY.xyz * rnd.y + gSunDirection.xyz);
        float2 mipAndCone = GetConeAngleFromAngularRadius(geometryProps.mip, gTanSunAngularRadius);

        // GeometryProps shadowRayGeometryProps;
        // MaterialProps shadowRayMaterialProps;
        // CastRay(Xshadow, sunDirection, 0.0,INF, mipAndCone, shadowRayGeometryProps, shadowRayMaterialProps);
        // float hitT = shadowRayGeometryProps.hitT;

        RayDesc rayDesc;
        rayDesc.Origin = Xshadow;
        rayDesc.Direction = sunDirection;
        rayDesc.TMin = 0;
        rayDesc.TMax = 1000;

        MainRayPayload shadowPayload = (MainRayPayload)0;
        TraceRay(g_AccelStruct, RAY_FLAG_NONE | RAY_FLAG_CULL_NON_OPAQUE, 0xFF, 0, 1, 1, rayDesc, shadowPayload);
        float hitT = shadowPayload.hitT;

        lighting *= float(hitT == INF);
    }

    return lighting;
}

float3 TraceTransparent(TraceTransparentDesc desc)
{
    float eta = BRDF::IOR::Air / BRDF::IOR::Glass;

    GeometryProps geometryProps = desc.geometryProps;
    float pathThroughput = 1.0;
    bool isReflection = desc.isReflection;
    float bayer = Sequence::Bayer4x4(desc.pixelPos, gFrameIndex);


    MaterialProps materialProps;

    [loop]
    for (uint bounce = 1; bounce <= PT_DELTA_BOUNCES_NUM; bounce++)
    {
        // Reflection or refraction?
        float NoV = abs(dot(geometryProps.N, geometryProps.V));
        float F = BRDF::FresnelTerm_Dielectric(eta, NoV);

        if (bounce == 1)
            pathThroughput *= isReflection ? F : 1.0 - F;
        else
        {
            // float rnd = frac(bayer + Sequence::Halton(bounce, 3)); // "Halton( bounce, 2 )" works worse than others

            float rnd = Rng::Hash::GetFloat();
            
            F = clamp(F, PT_GLASS_MIN_F, 1.0 - PT_GLASS_MIN_F); // TODO: needed?

            isReflection = rnd < F; // TODO: if "F" is clamped, "pathThroughput" should be adjusted too
        }


        // uint flags = bounce == PT_DELTA_BOUNCES_NUM ? FLAG_NON_TRANSPARENT : GEOMETRY_ALL;
        // uint flags = bounce == PT_DELTA_BOUNCES_NUM ? FLAG_NON_TRANSPARENT : GEOMETRY_ALL;
        uint flags = bounce == PT_DELTA_BOUNCES_NUM ? RAY_FLAG_CULL_NON_OPAQUE : RAY_FLAG_NONE;


        float3 Xoffset, ray;
        eta = GetDeltaEventRay(geometryProps, isReflection, eta, Xoffset, ray);

        CastRay(Xoffset, ray, 0.0, INF, GetConeAngleFromRoughness(geometryProps.mip, 0.0), flags, geometryProps, materialProps);


        bool isAir = eta < 1.0;

        float extinction = isAir ? 0.0 : 1.0; // TODO: tint color?
        if (!geometryProps.IsMiss()) // TODO: fix for non-convex geometry
            pathThroughput *= exp(-extinction * geometryProps.hitT * 1);


        // Is opaque hit found?
        if (!geometryProps.Has(FLAG_TRANSPARENT)) // TODO: stop if pathThroughput is low
            break;
    }


    float4 Lcached = float4(materialProps.Lemi, 0.0);
    if (!geometryProps.IsMiss())
    {
        // L1 cache - reproject previous frame, carefully treating specular
        float3 prevLdiff, prevLspec;
        float reprojectionWeight = ReprojectIrradiance(false, !isReflection, gIn_ComposedDiff, gIn_ComposedSpec_ViewZ, geometryProps, desc.pixelPos, prevLdiff, prevLspec);
        Lcached = float4(prevLdiff + prevLspec, reprojectionWeight);


        if (Rng::Hash::GetFloat() > Lcached.w)
        {
            float3 L = GetLighting(geometryProps, materialProps, LIGHTING | SHADOW) + materialProps.Lemi;
            Lcached.xyz = max(Lcached.xyz, L);
        }
    }

    return Lcached.xyz * pathThroughput;
}

[shader("raygeneration")]
void MainRayGenShader()
{
    uint2 pixelPos = DispatchRaysIndex().xy;

    float2 pixelUv = float2(pixelPos + 0.5) / gRectSize;
    float2 sampleUv = pixelUv + gJitter;

    if (pixelUv.x > 1.0 || pixelUv.y > 1.0)
    {
        return;
    }

    Rng::Hash::Initialize(pixelPos, gFrameIndex);

    float3 diff = gIn_ComposedDiff[pixelPos];
    float3 spec = gIn_ComposedSpec_ViewZ[pixelPos].xyz;
    float3 Lsum = diff + spec;

    // Primary ray for transparent geometry only
    float3 cameraRayOrigin = (float3)0;
    float3 cameraRayDirection = (float3)0;
    GetCameraRay(cameraRayOrigin, cameraRayDirection, sampleUv);

    float viewZAndTaaMask = gInOut_Mv[pixelPos].w;
    float viewZ = Math::Sign(gNearZ) * abs(viewZAndTaaMask) / FP16_VIEWZ_SCALE; // viewZ before PSR
    float3 Xv = Geometry::ReconstructViewPosition(sampleUv, gCameraFrustum, viewZ, 0);
    float tmin0 = 0 == 0 ? length(Xv) : abs(Xv.z);


    GeometryProps geometryPropsT;
    MaterialProps materialPropsT;

    CastRay(cameraRayOrigin, cameraRayDirection, 0.0, tmin0, GetConeAngleFromRoughness(0.0, 0.0), RAY_FLAG_CULL_OPAQUE, geometryPropsT, materialPropsT);

    if (!geometryPropsT.IsMiss() && geometryPropsT.hitT < tmin0)
    {
        viewZAndTaaMask = -abs(viewZAndTaaMask);

        float3 mvT = GetMotion(geometryPropsT.X, geometryPropsT.Xprev);
        gInOut_Mv[pixelPos] = float4(mvT, viewZAndTaaMask);


        TraceTransparentDesc desc;
        desc.geometryProps = geometryPropsT;
        desc.pixelPos = pixelPos;

        desc.isReflection = true;
        float3 reflection = TraceTransparent(desc);
        Lsum = reflection;

        desc.isReflection = false;
        float3 refraction = TraceTransparent(desc);
        Lsum += refraction;
    }

    // Apply exposure
    Lsum = ApplyExposure(Lsum);

    // Output
    gOut_Composed[pixelPos] = Lsum;
}
