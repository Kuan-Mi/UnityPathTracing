#include "ml.hlsli"

#define INF                                 1e5

struct RayPayload
{
    float k;
    float3 albedo;
    float3 emission;
    uint bounceIndexOpaque;
    uint bounceIndexTransparent;
    float3 bounceRayOrigin;
    float3 bounceRayDirection;
    // float3 worldFaceNormal;
    uint rngState;
};


cbuffer PathTracingParams : register(b0)
{
    float g_Zoom;
    uint g_ConvergenceStep;
    uint g_FrameIndex;
    uint g_SampleCount;

    float lightOffset;
    float3 _CameraPosition;
    float4x4 _CCameraToWorld;
    float4x4 gWorldToView;
    float4x4 gWorldToClip;
    float4x4 gWorldToViewPrev;
    float4x4 gWorldToClipPrev;
    float2 gRectSize;
    float2 pad1;
    float4x4 _CInverseProjection;

    float4 gSunBasisX;
    float4 gSunBasisY;
    float4 gSunDirection;

    float gTanPixelAngularRadius;
    float gUnproject;
    float gTanSunAngularRadius;
};


struct MainRayPayload
{
    float3 X; // 命中点的世界空间坐标
    float3 Xprev;
    float4 T; // 切线向量（xyz）和副切线符号（w）
    float3 N; // 法线向量（世界空间）
    float hitT; // 光线命中的距离（t值），INF表示未命中
    float curvature; // 曲率估算值（用于材质、去噪等）
    uint instanceIndex; // 命中的实例索引（用于查找InstanceData）
    
    float3 matN;

    float3 Lemi;
    float3 baseColor;
    float roughness;
    float metalness;

    float3 GetXoffset(float3 offsetDir, float amount)
    {
        float viewZ = Geometry::AffineTransform(gWorldToView, X).z;
        amount *= gUnproject * abs(viewZ);
        return X + offsetDir * max(amount, 0.00001);
    }

    bool IsMiss()
    {
        return hitT == INF;
    }
};

struct GeometryProps
{
    float3 X; // 命中点的世界空间坐标
    float3 Xprev; // 命中点在上一帧的世界空间坐标（用于时序去噪/运动矢量）
    float3 V; // 视线方向（通常为 -ray 方向）
    float4 T; // 切线向量（xyz）和副切线符号（w）
    float3 N; // 法线向量（世界空间）
    float hitT; // 光线命中的距离（t值），INF表示未命中
    float curvature; // 曲率估算值（用于材质、去噪等）
    uint instanceIndex; // 命中的实例索引（用于查找InstanceData）

    float3 GetXoffset(float3 offsetDir, float amount)
    {
        float viewZ = Geometry::AffineTransform(gWorldToView, X).z;
        amount *= gUnproject * abs(viewZ);

        return X + offsetDir * max(amount, 0.00001);
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


#define SKY_INTENSITY 1.0
#define SUN_INTENSITY 10.0



float3 GetSunIntensity(float3 v)
{
    float b = dot(v, gSunDirection.xyz);
    float d = length(v - gSunDirection.xyz * b);

    float glow = saturate(1.015 - d);
    glow *= b * 0.5 + 0.5;
    glow *= 0.6;

    float a = Math::Sqrt01(1.0 - b * b) / b;
    float sun = 1.0 - Math::SmoothStep(gTanSunAngularRadius * 0.9, gTanSunAngularRadius * 1.66 + 0.01, a);
    sun *= float(b > 0.0);
    sun *= 1.0 - Math::Pow01(1.0 - v.y, 4.85);
    sun *= Math::SmoothStep(0.0, 0.1, gSunDirection.y);
    sun += glow;

    float3 sunColor = lerp(float3(1.0, 0.6, 0.3), float3(1.0, 0.9, 0.7), Math::Sqrt01(gSunDirection.y));
    sunColor *= sun;

    sunColor *= Math::SmoothStep(-0.01, 0.05, gSunDirection.y);

    return Color::FromGamma(sunColor) * SUN_INTENSITY;
}


float3 GetSkyIntensity(float3 v)
{
    float atmosphere = sqrt(1.0 - saturate(v.y));

    float scatter = pow(saturate(gSunDirection.y), 1.0 / 15.0);
    scatter = 1.0 - clamp(scatter, 0.8, 1.0);

    float3 scatterColor = lerp(float3(1.0, 1.0, 1.0), float3(1.0, 0.3, 0.0) * 1.5, scatter);
    float3 skyColor = lerp(float3(0.2, 0.4, 0.8), float3(scatterColor), atmosphere / 1.3);
    skyColor *= saturate(1.0 + gSunDirection.y);

    float ground = 0.5 + 0.5 * Math::SmoothStep(-1.0, 0.0, v.y);
    skyColor *= ground;

    return Color::FromGamma(skyColor) * SKY_INTENSITY + GetSunIntensity(v);
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

    return lighting;
}
