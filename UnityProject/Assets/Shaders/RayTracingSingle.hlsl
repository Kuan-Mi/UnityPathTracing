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

// Inputs
Texture2D<float4> gIn_ScramblingRanking;
Texture2D<float4> gIn_Sobol;

// Output
RWTexture2D<float3> g_Output;

// 运动矢量（Motion Vector），用于描述像素在当前帧与上一帧之间的运动，以及视深（ViewZ）和TAA遮罩信息。
RWTexture2D<float4> gOut_Mv;

// 视空间深度（ViewZ），即像素在视空间中的Z值。
RWTexture2D<float> gOut_ViewZ;

// 法线、粗糙度和材质ID的打包信息。用于后续的去噪和材质区分。
RWTexture2D<float4> gOut_Normal_Roughness;

// 半影信息（Penumbra），用于软阴影的计算和去噪。
RWTexture2D<float> gOut_Penumbra;



// 基础色（BaseColor，已转为sRGB）和金属度（Metalness）。
RWTexture2D<float4> gOut_BaseColor_Metalness;
// 直接光照（Direct Lighting），即主光线命中点的直接光照结果。
RWTexture2D<float3> gOut_DirectLighting;
// 直接自发光（Direct Emission），即材质的自发光分量。
RWTexture2D<float3> gOut_DirectEmission;
// 主表面替换（PSR）后的路径通量（Throughput），用于镜面跳跃等特殊路径的能量追踪。
RWTexture2D<float3> gOut_PsrThroughput;
// 阴影数据（Shadow Data），如半影宽度等，用于软阴影和去噪。
RWTexture2D<float2> gOut_ShadowData;
// 阴影穿透率（Shadow Translucency），描述光线穿透透明物体后的能量变化。
RWTexture2D<float4> gOut_Shadow_Translucency;
// 漫反射光照结果（Diffuse Radiance），包含去噪和打包后的信息。
RWTexture2D<float4> gOut_Diff;
// 高光反射光照结果（Specular Radiance），包含去噪和打包后的信息。
RWTexture2D<float4> gOut_Spec;


struct TraceOpaqueResult
{
    float3 diffRadiance;
    float diffHitDist;

    float3 specRadiance;
    float specHitDist;
};


[shader("miss")]
void MainMissShader(inout MainRayPayload payload : SV_RayPayload)
{
    payload.hitT = INF;
    // payload.emission = g_EnvTex.SampleLevel(sampler_g_EnvTex, WorldRayDirection(), 0).xyz;
    //
    // // payload.emission = float3(0.5,0.5,0.5); // 固定背景色，便于调试
    // payload.bounceIndexOpaque = -1;
}

// struct ShadowPayload
// {
//     bool hit;
// };

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

    // return float2(sampleIndex/4.0,0); // 仅用于调试，显示采样索引

    // The algorithm
    uint3 A = gIn_ScramblingRanking[pixelPos & (BLUE_NOISE_SPATIAL_DIM - 1)] * 255;


    uint rankedSampleIndex = sampleIndex ^ A.z;
    uint4 B = gIn_Sobol[uint2(rankedSampleIndex & 255, 0)] * 255;
    float4 blue = (float4(B ^ A.xyxy) + 0.5) * (1.0 / 256.0);

    // ( Optional ) Randomize in [ 0; 1 / 256 ] area to get rid of possible banding
    uint d = Bayer4x4ui(pixelPos, g_FrameIndex);
    float2 dither = (float2(d & 3, d >> 2) + 0.5) * (1.0 / 4.0);
    blue += (dither.xyxy - 0.5) * (1.0 / 256.0);


    return saturate(blue.xy);
}

// float4 NRD_FrontEnd_PackNormalAndRoughness(float3 N, float roughness)
// {
//     float4 p;
//
//     // Best fit ( optional )
//     N /= max(abs(N.x), max(abs(N.y), abs(N.z)));
//
//     #if( NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_RGBA8_UNORM || NRD_NORMAL_ENCODING == NRD_NORMAL_ENCODING_RGBA16_UNORM )
//     N = N * 0.5 + 0.5;
//     #endif
//
//     p.xyz = N;
//     p.w = roughness;
//
//
//     return p;
// }


#define PT_SHADOW_RAY_OFFSET                1.0 // pixels


float2 GetConeAngleFromAngularRadius(float mip, float tanConeAngle)
{
    // In any case, we are limited by the output resolution
    tanConeAngle = max(tanConeAngle, gTanPixelAngularRadius);

    return float2(mip, tanConeAngle);
}

float3 GetMotion( float3 X, float3 Xprev )
{
    float3 motion = Xprev - X;

    float viewZ = Geometry::AffineTransform( gWorldToView, X ).z;
    float2 sampleUv = Geometry::GetScreenUv( gWorldToClip, X );

    float viewZprev = Geometry::AffineTransform( gWorldToViewPrev, Xprev ).z;
    float2 sampleUvPrev = Geometry::GetScreenUv( gWorldToClipPrev, Xprev );

    // viewZ =  mul(gWorldToView, float4(X, 1)).z;
    // viewZprev = mul(gWorldToViewPrev, float4(Xprev, 1)).z;
    
    // IMPORTANT: scaling to "pixel" unit significantly improves utilization of FP16
    motion.xy = ( sampleUvPrev - sampleUv ) * gRectSize;

    // IMPORTANT: 2.5D motion is preferred over 3D motion due to imprecision issues caused by FP16 rounding negative effects
    motion.z = viewZprev - viewZ;

    // return 0;
    return motion;
}


// float3 GetMotion( float3 X, float3 Xprev ) {
//     
//     // ---------------------------------------------------------
//     // 1. 计算当前帧 (Current) 的信息
//     // ---------------------------------------------------------
//     float4 viewPos = mul(gWorldToView, float4(X, 1.0));
//     float4 clipPos = mul(gWorldToClip, float4(X, 1.0));
//     
//     // 剔除无效点(可选，防止除以0)
//     float kEpsilon = 1e-7;
//     float rcpW = 1.0 / max(clipPos.w, kEpsilon);
//     
//     // NDC (Normalized Device Coordinates)
//     // DX12 Clip Space: X[-1, 1], Y[-1, 1], Z[0, 1]
//     float2 ndc = clipPos.xy * rcpW;
//
//     // NDC -> UV [0, 1]
//     // DX12 Texture UV 原点在左上角 (0,0)，Clip Space Y轴向上 (+1)
//     // 因此 Y 轴需要翻转: V = (1 - Y_ndc) * 0.5  =>  -0.5 * Y_ndc + 0.5
//     float2 sampleUv = ndc * float2(0.5, -0.5) + 0.5;
//     
//     // NRD 需要 View Space Z (通常是线性深度)
//     float viewZ = viewPos.z;
//
//
//     // ---------------------------------------------------------
//     // 2. 计算上一帧 (Previous) 的信息
//     // ---------------------------------------------------------
//     float4 viewPosPrev = mul(gWorldToViewPrev, float4(Xprev, 1.0));
//     float4 clipPosPrev = mul(gWorldToClipPrev, float4(Xprev, 1.0));
//     
//     float rcpWPrev = 1.0 / max(clipPosPrev.w, kEpsilon);
//     float2 ndcPrev = clipPosPrev.xy * rcpWPrev;
//     
//     // 同样的 NDC -> UV 转换
//     float2 sampleUvPrev = ndcPrev * float2(0.5, -0.5) + 0.5;
//     
//     float viewZprev = viewPosPrev.z;
//
//
//     // ---------------------------------------------------------
//     // 3. 计算 Motion Vector (符合 NRD 要求)
//     // ---------------------------------------------------------
//     float3 motion;
//
//     // XY: Screen-space motion in Pixels
//     // Direction: Previous position - Current position (指向历史帧)
//     motion.xy = ( sampleUvPrev - sampleUv ) * gRectSize;
//     
//     // Z: View-space depth difference
//     motion.z = viewZprev - viewZ;
//
//     return motion;
// }


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

    float4 viewPos = mul(_CInverseProjection, float4(ndcCoord.x, ndcCoord.y, 1.0, 1.0));
    viewPos /= viewPos.w;
    // viewPos.z = viewPos.z;
    float3 viewDirection = normalize(viewPos.xyz);

    float3 rayDirection = mul((float3x3)_CCameraToWorld, viewDirection);

    RayDesc ray;
    ray.Origin = _CameraPosition;
    ray.Direction = rayDirection;
    ray.TMin = 0.0f;
    ray.TMax = 1000.0f;

    MainRayPayload payload = (MainRayPayload)0;


    TraceRay(g_AccelStruct, RAY_FLAG_NONE | RAY_FLAG_NONE, 0xFF, 0, 1, 0, ray, payload);

    GeometryProps geometryProps0 = (GeometryProps)0;
    geometryProps0.X = payload.X;
    geometryProps0.Xprev = payload.X;
    geometryProps0.V = -rayDirection;
    geometryProps0.N = payload.N;
    geometryProps0.T = payload.T;
    geometryProps0.hitT = payload.hitT;
    geometryProps0.curvature = payload.curvature;
    geometryProps0.instanceIndex = payload.instanceIndex;

    MaterialProps materialProps0 = (MaterialProps)0;
    materialProps0.baseColor = payload.baseColor;
    materialProps0.roughness = payload.roughness;
    materialProps0.metalness = payload.metalness;
    materialProps0.Lemi = payload.Lemi;
    // 这三个应该从贴图再计算一次
    materialProps0.curvature = payload.curvature;
    materialProps0.N = payload.N;
    materialProps0.T = payload.T;

    float3 X0 = payload.X;
    float3 V0 = -rayDirection;
    float viewZ0 = abs(mul(gWorldToView, float4(X0, 1)).z);
    // float viewZ0 = payload.hitT;

    gOut_Mv[launchIndex] = float4(GetMotion(geometryProps0.X, geometryProps0.Xprev), 1);
    gOut_ViewZ[launchIndex] = viewZ0;


    gOut_Normal_Roughness[launchIndex] = NRD_FrontEnd_PackNormalAndRoughness(payload.N, payload.roughness, 0);
    gOut_BaseColor_Metalness[launchIndex] = float4(payload.baseColor, payload.metalness);

    float3 Ldirect = GetLighting(geometryProps0, materialProps0, LIGHTING, X0);

    gOut_DirectLighting[launchIndex] = Ldirect;
    gOut_DirectEmission[launchIndex] = payload.Lemi;
    gOut_PsrThroughput[launchIndex] = float3(1, 1, 1);
    gOut_ShadowData[launchIndex] = float2(0, 0);


    float2 rnd = GetBlueNoise(launchIndex);
    
    // rnd = float2(RandomFloat01(rngState), RandomFloat01(rngState));
    rnd = ImportanceSampling::Cosine::GetRay(rnd).xy;
    rnd *= gTanSunAngularRadius;

    float3 sunDirection = normalize(gSunBasisX.xyz * rnd.x + gSunBasisY.xyz * rnd.y + gSunDirection.xyz);
    float3 Xoffset = payload.GetXoffset(sunDirection, PT_SHADOW_RAY_OFFSET);
    float2 mipAndCone = GetConeAngleFromAngularRadius(1, gTanSunAngularRadius);

    float shadowTranslucency = (Color::Luminance(Ldirect) != 0.0 && !true) ? 1.0 : 0.0;
    float shadowHitDist = 0.0;

    RayDesc rayDesc;
    rayDesc.Origin = Xoffset;
    rayDesc.Direction = sunDirection;
    rayDesc.TMin = 0;
    rayDesc.TMax = 1000;


    MainRayPayload shadowPayload = (MainRayPayload)0;
    TraceRay(g_AccelStruct, RAY_FLAG_NONE | RAY_FLAG_NONE, 0xFF, 0, 1, 1, rayDesc, shadowPayload);
    shadowHitDist = shadowPayload.hitT;

    float penumbra = SIGMA_FrontEnd_PackPenumbra(shadowHitDist, gTanSunAngularRadius);
    gOut_Penumbra[launchIndex] = penumbra;

    // float4 translucency = SIGMA_FrontEnd_PackTranslucency(shadowHitDist, shadowTranslucency);
    gOut_ShadowData[launchIndex] = penumbra;
    gOut_Shadow_Translucency[launchIndex] = gOut_Shadow_Translucency[launchIndex];

    float3x3 mirrorMatrix = Geometry::GetMirrorMatrix(0); // identity


    float3 v = GetSunIntensity(-V0);
    float3 s2 = GetSkyIntensity(-V0);


    // float s = viewZ0 * 0.1;
    float s = payload.metalness;
    s = penumbra;
    g_Output[launchIndex] = float3(s, s, s);
    // g_Output[launchIndex] = gOut_DirectEmission[launchIndex];
}
