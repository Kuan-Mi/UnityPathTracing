#include "UnityShaderVariables.cginc"
#include "ml.hlsli"
#include "NRDInclude/NRD.hlsli"

// #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
#include "_Utils.hlsl"
#include "RayPayload.hlsl"
#include "GlobalResource.hlsl"

#pragma max_recursion_depth 1

// Input


TextureCube<float4> g_EnvTex; // 环境贴图，用于在 Miss 时返回背景光照
SamplerState sampler_g_EnvTex;

StructuredBuffer<uint4> gIn_ScramblingRanking;
StructuredBuffer<uint4> gIn_Sobol;

// Output
RWTexture2D<float3> g_Output;

// 运动矢量（Motion Vector），用于描述像素在当前帧与上一帧之间的运动，以及视深（ViewZ）和TAA遮罩信息。
RWTexture2D<float4> gOut_Mv;
// 视空间深度（ViewZ），即像素在视空间中的Z值。
RWTexture2D<float> gOut_ViewZ;
// 法线、粗糙度和材质ID的打包信息。用于后续的去噪和材质区分。
RWTexture2D<float4> gOut_Normal_Roughness;
// 基础色（BaseColor，已转为sRGB）和金属度（Metalness）。
RWTexture2D<float4> gOut_BaseColor_Metalness;

// 直接光照（Direct Lighting），即主光线命中点的直接光照结果。
RWTexture2D<float3> gOut_DirectLighting;
// 直接自发光（Direct Emission），即材质的自发光分量。
RWTexture2D<float3> gOut_DirectEmission;

// 阴影数据（Shadow Data），如半影宽度等，用于软阴影和去噪。
RWTexture2D<float> gOut_ShadowData;
// 漫反射光照结果（Diffuse Radiance），包含噪声和打包后的信息。
RWTexture2D<float4> gOut_Diff;
// 高光反射光照结果（Specular Radiance），包含噪声和打包后的信息。
RWTexture2D<float4> gOut_Spec;

struct TraceOpaqueResult
{
    float3 diffRadiance;
    float diffHitDist;

    float3 specRadiance;
    float specHitDist;

    float3 debug;
};


[shader("miss")]
void MainMissShader(inout MainRayPayload payload : SV_RayPayload)
{
    payload.hitT = INF; 
    float3 ray = WorldRayDirection();
    payload.X = WorldRayOrigin() +ray * payload.hitT;
    payload.Xprev = payload.X;
    
    payload.Lemi = GetSkyIntensity( ray );
    
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

float3 EvalFromClip(float4 clip)
{
    float4 viewPos = mul(unity_CameraInvProjection, clip);
    // 输出未经透视除法时查看 w 的符号/大小也有用，调试时可写到其它 buffer
    viewPos /= viewPos.w;
    return normalize(viewPos.xyz);
}

#define BLUE_NOISE_SPATIAL_DIM              128 // see StaticTexture::ScramblingRanking
#define BLUE_NOISE_TEMPORAL_DIM             4 // good values: 4-8 for shadows, 8-16 for occlusion, 8-32 for lighting

#define USE_SANITIZATION                    0 // NRD sample is NAN/INF free


uint Bayer4x4ui(uint2 samplePos, uint frameIndex)
{
    uint2 samplePosWrap = samplePos & 3;
    uint a = 2068378560 * (1 - (samplePosWrap.x >> 1)) + 1500172770 * (samplePosWrap.x >> 1);
    uint b = (samplePosWrap.y + ((samplePosWrap.x & 1) << 2)) << 2;

    uint sampleOffset = frameIndex;

    return ((a >> b) + sampleOffset) & 0xF;
}

float2 GetBlueNoise(uint2 pixelPos, uint seed = 0)
{
    // https://eheitzresearch.wordpress.com/772-2/
    // https://belcour.github.io/blog/research/publication/2019/06/17/sampling-bluenoise.html

    // Sample index
    uint sampleIndex = (g_FrameIndex + seed) & (BLUE_NOISE_TEMPORAL_DIM - 1);

    sampleIndex = 3;

    // pixelPos /= 8;

    uint2 uv = pixelPos & (BLUE_NOISE_SPATIAL_DIM - 1);
    uint index = uv.x + uv.y * BLUE_NOISE_SPATIAL_DIM;
    uint3 A = gIn_ScramblingRanking[index].xyz;

    // return float2(A.x/256.0 , A.y / 256.0);
    uint rankedSampleIndex = sampleIndex ^ A.z;


    // return float2(rankedSampleIndex / float(BLUE_NOISE_TEMPORAL_DIM), 0);


    uint4 B = gIn_Sobol[rankedSampleIndex & 255];
    float4 blue = (float4(B ^ A.xyxy) + 0.5) * (1.0 / 256.0);

    // ( Optional ) Randomize in [ 0; 1 / 256 ] area to get rid of possible banding
    uint d = Bayer4x4ui(pixelPos, g_FrameIndex);
    float2 dither = (float2(d & 3, d >> 2) + 0.5) * (1.0 / 4.0);
    blue += (dither.xyxy - 0.5) * (1.0 / 256.0);

    return saturate(blue.xy);
}

#define PT_SHADOW_RAY_OFFSET                1.0 // pixels


float2 GetConeAngleFromAngularRadius(float mip, float tanConeAngle)
{
    // In any case, we are limited by the output resolution
    tanConeAngle = max(tanConeAngle, gTanPixelAngularRadius);

    return float2(mip, tanConeAngle);
}

float3 GetMotion(float3 X, float3 Xprev)
{
    float3 motion = Xprev - X;

    float viewZ = Geometry::AffineTransform(gWorldToView, X).z;
    float2 sampleUv = Geometry::GetScreenUv(gWorldToClip, X);

    float viewZprev = Geometry::AffineTransform(gWorldToViewPrev, Xprev).z;
    float2 sampleUvPrev = Geometry::GetScreenUv(gWorldToClipPrev, Xprev);

    // IMPORTANT: scaling to "pixel" unit significantly improves utilization of FP16
    motion.xy = (sampleUvPrev - sampleUv) * gRectSize;

    // IMPORTANT: 2.5D motion is preferred over 3D motion due to imprecision issues caused by FP16 rounding negative effects
    motion.z = viewZprev - viewZ;
    // motion.z = 0;
    motion.z = -motion.z;
    // return 0;
    return motion;
}

#define NRD_MODE                            NORMAL

#define NORMAL                              0
#define SH                                  1 // NORMAL + SH (SG) resolve
#define OCCLUSION                           2
#define DIRECTIONAL_OCCLUSION               3 // diffuse OCCLUSION + SH (SG) resolve
#define gBounceNum               1 // diffuse OCCLUSION + SH (SG) resolve


#define PT_MAX_FIREFLY_RELATIVE_INTENSITY   20.0 // no more than 20x energy increase in case of probabilistic sampling
#define PT_EVIL_TWIN_LOBE_TOLERANCE         0.005 // normalized %


// Resolution
#define RESOLUTION_FULL                     0
#define RESOLUTION_FULL_PROBABILISTIC       1
#define RESOLUTION_HALF                     2


#define gTracingMode                      RESOLUTION_FULL
#define gSampleNum                      1
#define gRR                      0
#define gFrameIndex                      g_FrameIndex
#define gMinProbability                      0

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
    // if( useMagicBoost )
    //     diffProb = lerp( diffProb, 1.0, GetSpecMagicCurve( materialProps.roughness ) );

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

#define PT_SPEC_LOBE_ENERGY                 0.95 // trimmed to 95%


float3x3 GetBasisUnity(float3 N)
{
    // 注意：这里的 sy 对应原代码中的 sz
    // 原代码是基于 Z 轴构造，现在改为基于 Y 轴（因为你的映射中 N 对应 Y）
    float sy = Math::Sign(N.y);
    float a = 1.0f / (sy + N.y);
    float xa = N.x * a;
    float b = N.z * xa;
    float c = N.z * sy;

    // 重新排列分量以匹配旋转逻辑
    // 原 T = ( c * N.x * a - 1.0f, sz * b, c ) -> 映射到 Y-up 空间
    float3 T = float3(sy * b, c, c * N.z * a - 1.0f);

    // 原 B = ( b, N.y * ya - sz, N.y ) -> 映射到 Y-up 空间
    float3 B = float3(N.x * xa - sy, N.x, b);

    // 返回矩阵。在 Unity HLSL 中，这会将 T, B, N 作为矩阵的行。
    // RotateVector(m, V) 通常等价于 float3(dot(T,V), dot(B,V), dot(N,V))
    return float3x3(T, B, N);
}

float3x3 GetBasisUnity2(float3 N)
{
    float sy = Math::Sign(N.y);
    float a = 1.0f / (sy + N.y);

    float ya = N.y * a;
    float b = N.x * ya;
    float c = N.x * sy;

    float3 T = float3(c * N.x * a - 1.0f, sy * b, c);
    float3 B = float3(b, N.y * ya - sy, N.y);

    // Note: due to the quaternion formulation, the generated frame is rotated by 180 degrees,
    // s.t. if N = (0, 0, 1), then T = (-1, 0, 0) and B = (0, -1, 0).
    return float3x3(T, B, N);
}


float3 GenerateRayAndUpdateThroughput(inout GeometryProps geometryProps, inout MaterialProps materialProps, inout float3 throughput, uint sampleMaxNum, bool isDiffuse, float2 rnd)
{
    bool isHair = false;
    // float3x3 mLocalBasis = isHair ? Hair_GetBasis( materialProps.N, materialProps.T ) : Geometry::GetBasis( materialProps.N );
    float3x3 mLocalBasis = Geometry::GetBasis(materialProps.N);
    float3 Vlocal = Geometry::RotateVector(mLocalBasis, geometryProps.V);


    // return geometryProps.V;
    // return Vlocal;
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
        // return float3(rnd,0);
        // return  Geometry::RotateVectorInverse(mLocalBasis, candidateRayLocal);
        // return candidateRayLocal;

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
    // if( USE_TRANSLUCENCY && geometryProps.Has( FLAG_LEAF ) && isDiffuse )
    // {
    //     if( Rng::Hash::GetFloat( ) < LEAF_TRANSLUCENCY )
    //     {
    //         rayLocal = -rayLocal;
    //         geometryProps.X -= LEAF_THICKNESS * geometryProps.N;
    //         throughput /= LEAF_TRANSLUCENCY;
    //     }
    //     else
    //         throughput /= 1.0 - LEAF_TRANSLUCENCY;
    // }
    #endif

    // Transform to world space
    float3 ray = Geometry::RotateVectorInverse(mLocalBasis, rayLocal);

    // Path termination or ray direction fix
    float NoLgeom = dot(geometryProps.N, ray);
    float roughnessThreshold = saturate(materialProps.roughness / 0.15);

    if (!isHair && NoLgeom < 0.0)
    {
        if (isDiffuse || Rng::Hash::GetFloat() < roughnessThreshold)
            throughput = 0.0; // terminate ray pointing inside the surface
        else
        {
            // If roughness is low, patch ray direction and shading normal to avoid self-intersections
            // ( https://arxiv.org/pdf/1705.01263.pdf, Appendix 3 )
            float b = abs(dot(geometryProps.N, materialProps.N)) * 0.99;

            ray = normalize(ray + geometryProps.N * abs(NoLgeom) * Math::PositiveRcp(b));
            materialProps.N = normalize(geometryProps.V + ray);
        }
    }

    return ray;
}


#define PT_THROUGHPUT_THRESHOLD             0.001


float2 GetConeAngleFromRoughness(float mip, float roughness)
{
    float tanConeAngle = roughness * roughness * 0.05; // TODO: tweaked to be accurate and give perf boost

    return GetConeAngleFromAngularRadius(mip, tanConeAngle);
}


float ApplyThinLensEquation(float hitDist, float curvature)
{
    return hitDist / (2.0 * curvature * hitDist + 1.0);
}


void CastRay(float3 origin, float3 direction, float Tmin, float Tmax, float2 mipAndCone, out GeometryProps props, out MaterialProps matProps)
{
    RayDesc rayDesc;
    rayDesc.Origin = origin;
    rayDesc.Direction = direction;
    rayDesc.TMin = Tmin;
    rayDesc.TMax = Tmax;

    MainRayPayload payload = (MainRayPayload)0;
    payload.mipAndCone = mipAndCone;

    TraceRay(g_AccelStruct, RAY_FLAG_NONE | RAY_FLAG_NONE, 0xFF, 0, 1, 0, rayDesc, payload);


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


#define LIGHTING    0x01
#define SHADOW      0x02
#define SSS         0x04


 



float3 GetLighting(GeometryProps geometryProps, inout MaterialProps materialProps, uint flags, out float3 Xshadow)
{
    float3 lighting = 0.0;

    // Lighting
    Xshadow = geometryProps.X;

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
    if( ( flags & SHADOW ) != 0 && Color::Luminance( lighting ) != 0 )
    {
        
        float2 rnd = Rng::Hash::GetFloat2( );
        rnd = ImportanceSampling::Cosine::GetRay( rnd ).xy;
        rnd *= gTanSunAngularRadius;
        
        float3 sunDirection = normalize( gSunBasisX.xyz * rnd.x + gSunBasisY.xyz * rnd.y + gSunDirection.xyz );
        float2 mipAndCone = GetConeAngleFromAngularRadius( geometryProps.mip, gTanSunAngularRadius );

        GeometryProps shadowRayGeometryProps;
        MaterialProps shadowRayMaterialProps;
        
        CastRay(Xshadow,sunDirection,0.0,INF,mipAndCone,shadowRayGeometryProps,shadowRayMaterialProps);
        float hitT = shadowRayGeometryProps.hitT;
        lighting *= float( hitT == INF );
    }

    return lighting;
}

TraceOpaqueResult TraceOpaque(GeometryProps geometryProps0, MaterialProps materialProps0, uint2 pixelPos, float3x3 mirrorMatrix, float4 Lpsr)
{
    TraceOpaqueResult result = (TraceOpaqueResult)0;
    #if( NRD_MODE < OCCLUSION )
    result.specHitDist = NRD_FrontEnd_SpecHitDistAveraging_Begin();
    #endif

    float viewZ0 = Geometry::AffineTransform(gWorldToView, geometryProps0.X).z;
    float roughness0 = materialProps0.roughness;

    // Material de-modulation ( convert irradiance into radiance )
    float3 diffFactor0, specFactor0;
    {
        float3 albedo, Rf0;
        BRDF::ConvertBaseColorMetalnessToAlbedoRf0(materialProps0.baseColor, materialProps0.metalness, albedo, Rf0);

        NRD_MaterialFactors(materialProps0.N, geometryProps0.V, albedo, Rf0, materialProps0.roughness, diffFactor0, specFactor0);
    }


    // uint checkerboard = Sequence::CheckerBoard(pixelPos, g_FrameIndex) != 0;

    // uint pathNum = gSampleNum << (gTracingMode == RESOLUTION_FULL ? 1 : 0);
    // 两条路径
    uint pathNum = 2;
    uint diffPathNum = 0;

    pathNum = 2;
    [loop]
    for (uint path = 0; path < pathNum; path++)
    {
        GeometryProps geometryProps = geometryProps0;
        MaterialProps materialProps = materialProps0;

        float accumulatedHitDist = 0;
        float accumulatedDiffuseLikeMotion = 0;
        float accumulatedCurvature = 0;

        float3 Lsum = Lpsr.xyz;
        float3 pathThroughput = 1.0 - Lpsr.w;
        bool isDiffusePath = false;

        [loop]
        for (uint bounce = 1; bounce <= gBounceNum && !geometryProps.IsMiss(); bounce++)
        {
            //=============================================================================================================================================================
            // Origin point
            //=============================================================================================================================================================

            bool isDiffuse = false;
            float lobeTanHalfAngleAtOrigin = 0.0;
            {
                // Diffuse probability
                float diffuseProbability = EstimateDiffuseProbability(geometryProps, materialProps);

                float rnd = Rng::Hash::GetFloat();
                if (gTracingMode == RESOLUTION_FULL_PROBABILISTIC && bounce == 1 && !gRR)
                {
                    // Clamp probability to a sane range to guarantee a sample in 3x3 ( or 5x5 ) area ( see NRD docs )
                    diffuseProbability = float(diffuseProbability != 0.0) * clamp(diffuseProbability, gMinProbability, 1.0 - gMinProbability);
                    rnd = Sequence::Bayer4x4(pixelPos, gFrameIndex) + rnd / 16.0;
                }

                // Diffuse or specular?
                isDiffuse = rnd < diffuseProbability; // TODO: if "diffuseProbability" is clamped, "pathThroughput" should be adjusted too
                if (gTracingMode == RESOLUTION_FULL_PROBABILISTIC || bounce > 1)
                    pathThroughput /= isDiffuse ? diffuseProbability : (1.0 - diffuseProbability);
                else
                    // 第1次 是镜面反射 第2次 是漫反射
                    isDiffuse = (path & 0x1);

                // // This is not needed in case of "RESOLUTION_FULL_PROBABILISTIC", since hair doesn't have diffuse component
                // if( geometryProps.Has( FLAG_HAIR ) && isDiffuse )
                //     break;

                // Importance sampling
                uint sampleMaxNum = 0;
                // if( bounce == 1 && gDisableShadowsAndEnableImportanceSampling && NRD_MODE < OCCLUSION )
                //     sampleMaxNum = PT_IMPORTANCE_SAMPLES_NUM * ( isDiffuse ? 1.0 : GetSpecMagicCurve( materialProps.roughness ) );
                // sampleMaxNum = max( sampleMaxNum, 1 );
                sampleMaxNum = 1;

                #if( NRD_MODE < OCCLUSION )
                float2 rnd2 = Rng::Hash::GetFloat2();
                #else
                uint2 blueNoisePos = pixelPos + uint2(Sequence::Weyl2D(0.0, path * gBounceNum + bounce) * (BLUE_NOISE_SPATIAL_DIM - 1));
                float2 rnd2 = GetBlueNoise(blueNoisePos, gTracingMode == RESOLUTION_HALF);
                #endif

                float3 ray = GenerateRayAndUpdateThroughput(geometryProps, materialProps, pathThroughput, sampleMaxNum, isDiffuse, rnd2);



                // Special case for primary surface ( 1st bounce starts here )
                if (bounce == 1)
                {
                    isDiffusePath = isDiffuse;

                    if (gTracingMode == RESOLUTION_FULL)
                        Lsum *= isDiffuse ? diffuseProbability : (1.0 - diffuseProbability);
                }

                // Abort tracing if the current bounce contribution is low

                /*
                GOOD PRACTICE:
                - terminate path if "pathThroughput" is smaller than some threshold
                - approximate ambient at the end of the path
                - re-use data from the previous frame
                */

                if (PT_THROUGHPUT_THRESHOLD != 0.0 && Color::Luminance(pathThroughput) < PT_THROUGHPUT_THRESHOLD)
                    break;


                //=========================================================================================================================================================
                // Trace to the next hit
                //=========================================================================================================================================================

                float roughnessTemp = isDiffuse ? 1.0 : materialProps.roughness;
                lobeTanHalfAngleAtOrigin = roughnessTemp * roughnessTemp / (1.0 + roughnessTemp * roughnessTemp);

                // float2 mipAndCone = GetConeAngleFromRoughness( geometryProps.mip, isDiffuse ? 1.0 : materialProps.roughness );
                float2 mipAndCone = GetConeAngleFromRoughness(geometryProps.mip, isDiffuse ? 1.0 : materialProps.roughness);

                CastRay(geometryProps.GetXoffset(geometryProps.N), ray, 0.0, INF, mipAndCone, geometryProps, materialProps);
            }


            //=============================================================================================================================================================
            // Hit point
            //=============================================================================================================================================================

            {
                //=============================================================================================================================================================
                // Lighting
                //=============================================================================================================================================================

                float4 Lcached = float4(materialProps.Lemi, 0.0);
                if (!geometryProps.IsMiss())
                {
                    // Cache miss - compute lighting, if not found in caches
                    // if (Rng::Hash::GetFloat() > Lcached.w)
                    {
                        float3 nouse;
                        float3 L = GetLighting(geometryProps, materialProps, LIGHTING | SHADOW, nouse) + materialProps.Lemi;
                        Lcached.xyz = bounce < gBounceNum ? L : max(Lcached.xyz, L);
                    }
                }




                //=============================================================================================================================================================
                // Other
                //=============================================================================================================================================================

                // Accumulate lighting
                float3 L = Lcached.xyz * pathThroughput;

                Lsum += L;

                // ( Biased ) Reduce contribution of next samples if previous frame is sampled, which already has multi-bounce information
                pathThroughput *= 1.0 - Lcached.w;

                // Accumulate path length for NRD ( see "README/NOISY INPUTS" )
                float a = Color::Luminance(L);
                float b = Color::Luminance(Lsum); // already includes L
                float importance = a / (b + 1e-6);

                importance *= 1.0 - Color::Luminance(materialProps.Lemi) / (a + 1e-6);

                float diffuseLikeMotion = EstimateDiffuseProbability(geometryProps, materialProps, true);
                diffuseLikeMotion = isDiffuse ? 1.0 : diffuseLikeMotion;

                accumulatedHitDist += ApplyThinLensEquation(geometryProps.hitT, accumulatedCurvature) * Math::SmoothStep(0.2, 0.0, accumulatedDiffuseLikeMotion);
                accumulatedDiffuseLikeMotion += 1.0 - importance * (1.0 - diffuseLikeMotion);
                accumulatedCurvature += materialProps.curvature; // yes, after hit

                #if( USE_CAMERA_ATTACHED_REFLECTION_TEST == 1 && NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_R10G10B10A2_UNORM )
                // IMPORTANT: lazy ( no checkerboard support ) implementation of reflections masking for objects attached to the camera
                // TODO: better find a generic solution for tracking of reflections for objects attached to the camera
                if (bounce == 1 && !isDiffuse && desc.materialProps.roughness < 0.01)
                {
                    if (!geometryProps.IsMiss() && !geometryProps.Has(FLAG_STATIC))
                        gOut_Normal_Roughness[desc.pixelPos].w = MATERIAL_ID_SELF_REFLECTION;
                }
                #endif
            }
        }

        // // Debug visualization: specular mip level at the end of the path
        // if( gOnScreen == SHOW_MIP_SPECULAR )
        // {
        //     float mipNorm = Math::Sqrt01( geometryProps.mip / MAX_MIP_LEVEL );
        //     Lsum = Color::ColorizeZucconi( mipNorm );
        // }

        // Normalize hit distances for REBLUR before averaging ( needed only for AO for REFERENCE )
        float normHitDist = accumulatedHitDist;
        // if( gDenoiserType != DENOISER_RELAX )
        //     normHitDist = REBLUR_FrontEnd_GetNormHitDist( accumulatedHitDist, viewZ0, gHitDistParams, isDiffusePath ? 1.0 : materialProps0.roughness );

        // result.debug = float3(normHitDist,normHitDist,normHitDist);
        
        // Accumulate diffuse and specular separately for denoising
        if (!USE_SANITIZATION || NRD_IsValidRadiance(Lsum))
        {
            if (isDiffusePath)
            {
                result.diffRadiance += Lsum;
                result.diffHitDist += normHitDist;
                diffPathNum++;
            }
            else
            {
                result.specRadiance += Lsum;

                #if( NRD_MODE < OCCLUSION )
                NRD_FrontEnd_SpecHitDistAveraging_Add(result.specHitDist, normHitDist);

                result.debug = float3(result.specHitDist,result.specHitDist,result.specHitDist);
                #else
                result.specHitDist += normHitDist;
                #endif
            }
        }
    }
    
     
    // Material de-modulation ( convert irradiance into radiance )
    // if( gOnScreen != SHOW_MIP_SPECULAR )
    {
        result.diffRadiance /= diffFactor0;
        result.specRadiance /= specFactor0;
    }

    // Radiance is already divided by sampling probability, we need to average across all paths
    float radianceNorm = 1.0 / float(gSampleNum);
    result.diffRadiance *= radianceNorm;
    result.specRadiance *= radianceNorm;

    
    // Others are not divided by sampling probability, we need to average across diffuse / specular only paths
    float diffNorm = diffPathNum == 0 ? 0.0 : 1.0 / float(diffPathNum);
    float specNorm = pathNum == diffPathNum ? 0.0 : 1.0 / float(pathNum - diffPathNum);

    result.diffHitDist *= diffNorm;

    #if( NRD_MODE < OCCLUSION )
    NRD_FrontEnd_SpecHitDistAveraging_End(result.specHitDist);
    #else
    result.specHitDist *= specNorm;
    #endif
    
    // result.debug = float3(0.5,0.5,0.5);

    #if( NRD_MODE == SH || NRD_MODE == DIRECTIONAL_OCCLUSION )
    result.diffDirection *= diffNorm;
    result.specDirection *= specNorm;
    #endif

    return result;
}


#define MATERIAL_ID_DEFAULT                 0.0f
#define MATERIAL_ID_METAL                   1.0f

float GetMaterialID(GeometryProps geometryProps, MaterialProps materialProps)
{
    bool isMetal = materialProps.metalness > 0.5;
    return (isMetal ? MATERIAL_ID_METAL : MATERIAL_ID_DEFAULT);
}

[shader("raygeneration")]
void MainRayGenShader()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;
    float2 frameCoord = float2(launchIndex.x, launchIndex.y) + float2(0.5, 0.5);

    uint rngState = uint(
        uint(launchIndex.x) * uint(1973) + uint(launchIndex.y) * uint(9277) + uint(
            g_ConvergenceStep + g_FrameIndex * g_SampleCount) *
        uint(26699)) | uint(1);
    float2 jitter = float2(RandomFloat01(rngState), RandomFloat01(rngState)) - float2(0.5, 0.5);

    jitter = float2(0.0, 0.0); // 关闭抖动以便调试


    // 0 - 1
    float2 ndcCoord = (frameCoord + jitter) / float2(launchDim.x - 1, launchDim.y - 1);
    // -1 - 1
    ndcCoord = ndcCoord * 2.0 - 1.0;

    ndcCoord.y = -ndcCoord.y; // NDC Y 轴向上，纹理 UV Y 轴向下，需要翻转

    float4 viewPos = mul(_CInverseProjection, float4(ndcCoord.x, ndcCoord.y, 1.0, 1.0));
    viewPos /= viewPos.w;
    // viewPos.z = viewPos.z;
    float3 viewDirection = normalize(viewPos.xyz);

    float3 rayDirection = mul((float3x3)_CCameraToWorld, viewDirection);


    // Initialize RNG
    Rng::Hash::Initialize(launchIndex, gFrameIndex);

    //================================================================================================================================================================================
    // Primary ray
    //================================================================================================================================================================================

    float3 cameraRayOrigin = _CameraPosition;
    float3 cameraRayDirection = rayDirection;

    GeometryProps geometryProps0;
    MaterialProps materialProps0;

    CastRay(cameraRayOrigin, cameraRayDirection, 0.0, 1000.0, GetConeAngleFromRoughness(0.0, 0.0), geometryProps0, materialProps0);

    //================================================================================================================================================================================
    // G-buffer ( guides )
    //================================================================================================================================================================================

    // Motion
    float3 X0 = geometryProps0.X;
    gOut_Mv[launchIndex] = float4(GetMotion(geometryProps0.X, geometryProps0.Xprev), 1);

    // ViewZ
    float viewZ = -Geometry::AffineTransform(gWorldToView, X0).z;
    viewZ = geometryProps0.IsMiss() ? Math::Sign(viewZ) * INF : viewZ;

    gOut_ViewZ[launchIndex] = viewZ;

    // Emission
    gOut_DirectEmission[launchIndex] = materialProps0.Lemi;

    // Early out
    if (geometryProps0.IsMiss())
    {
        return;
    }

    // Normal, roughness and material ID
    float materialID = GetMaterialID(geometryProps0, materialProps0);
    gOut_Normal_Roughness[launchIndex] = NRD_FrontEnd_PackNormalAndRoughness(materialProps0.N, materialProps0.roughness, materialID);

    // Base color and metalness
    gOut_BaseColor_Metalness[launchIndex] = float4(Color::ToSrgb(materialProps0.baseColor), materialProps0.metalness);

    // Direct lighting
    float3 Xshadow;
    float3 Ldirect = GetLighting(geometryProps0, materialProps0, LIGHTING, Xshadow);

    gOut_DirectLighting[launchIndex] = Ldirect;

    float3x3 mirrorMatrix = Geometry::GetMirrorMatrix(0); // identity
    float4 Lpsr = 0;

    //================================================================================================================================================================================
    // Secondary rays
    //================================================================================================================================================================================
    TraceOpaqueResult result = TraceOpaque(geometryProps0, materialProps0, launchIndex, mirrorMatrix, Lpsr);

    //================================================================================================================================================================================
    // Sun shadow
    //================================================================================================================================================================================
    geometryProps0.X = Xshadow;

    float2 rnd = GetBlueNoise(launchIndex);
    rnd = ImportanceSampling::Cosine::GetRay(rnd).xy;
    rnd *= gTanSunAngularRadius;

    float3 sunDirection = normalize(gSunBasisX.xyz * rnd.x + gSunBasisY.xyz * rnd.y + gSunDirection.xyz);
    float3 Xoffset = geometryProps0.GetXoffset(sunDirection, PT_SHADOW_RAY_OFFSET);
    float2 mipAndCone = GetConeAngleFromAngularRadius(geometryProps0.mip, gTanSunAngularRadius);

    float shadowTranslucency = (Color::Luminance(Ldirect) != 0.0) ? 1.0 : 0.0;
    float shadowHitDist = 0.0;

    if (shadowTranslucency > 0.1)
    {
        GeometryProps geometryPropsShadow;
        MaterialProps materialPropsShadow;

        CastRay(Xoffset, sunDirection, 0.0, INF, mipAndCone, geometryPropsShadow, materialPropsShadow);

        /*
        // RayDesc rayDesc;
        // rayDesc.Origin = Xoffset;
        // rayDesc.Direction = sunDirection;
        // rayDesc.TMin = 0;
        // rayDesc.TMax = 1000;
        //
        // MainRayPayload shadowPayload = (MainRayPayload)0;
        // TraceRay(g_AccelStruct, RAY_FLAG_NONE | RAY_FLAG_NONE, 0xFF, 0, 1, 1, rayDesc, shadowPayload);
        */

        shadowHitDist = geometryPropsShadow.hitT;
    }

    float penumbra = SIGMA_FrontEnd_PackPenumbra(shadowHitDist, gTanSunAngularRadius);

    gOut_ShadowData[launchIndex] = penumbra;

    //================================================================================================================================================================================
    // Output
    //================================================================================================================================================================================
    gOut_Diff[launchIndex] = RELAX_FrontEnd_PackRadianceAndHitDist(result.diffRadiance, result.diffHitDist, USE_SANITIZATION);
    gOut_Spec[launchIndex] = RELAX_FrontEnd_PackRadianceAndHitDist(result.specRadiance, result.specHitDist, USE_SANITIZATION);

    result.debug = float3(result.diffHitDist,result.diffHitDist,result.diffHitDist);
    // result.debug = float3(result.specHitDist,result.specHitDist,result.specHitDist);

    g_Output[launchIndex] = float4(result.debug, 1);
}
