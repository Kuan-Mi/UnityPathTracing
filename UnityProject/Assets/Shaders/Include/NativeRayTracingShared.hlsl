#include "../Sharc/SharcCommon.h"


RaytracingAccelerationStructure gWorldTlas : register(t0, space2);

ByteAddressBuffer t_BindlessBuffers[] : register(t0, space3);
Texture2D<float4> t_BindlessTextures[] : register(t0, space4);


struct GeometryData
{
    uint numIndices;
    uint numVertices;
    int indexBufferIndex; // index into t_BindlessBuffers
    uint indexOffset; // byte offset into index buffer

    int vertexBufferIndex; // index into t_BindlessBuffers
    uint positionOffset; // byte offset to first position (float3 per vertex)
    uint normalOffset; // byte offset to first normal  (float3 per vertex, or ~0u if absent)
    uint texCoord1Offset; // byte offset to first texcoord (float2 per vertex, or ~0u if absent)

    uint tangentOffset; // byte offset to first tangent (float4 per vertex, or ~0u if absent)
    uint vertexStride; // stride in bytes between consecutive vertices
    uint indexStride; // 2 (16-bit) or 4 (32-bit)
    uint materialIndex;
};

struct InstanceData
{
    uint firstGeometryIndex; // index into t_GeometryData for the first geometry of this instance
    uint numGeometries;
    uint pad0;
    uint pad1;
    row_major float3x4 transform; // object-to-world (row-major)
};

struct MaterialConstants
{
    // row 0
    float3 baseOrDiffuseColor;
    int flags;

    // row 1
    float3 emissiveColor;
    int domain;

    // row 2
    float opacity;
    float roughness;
    float metalness;
    float normalTextureScale;

    // row 3
    float occlusionStrength;
    float alphaCutoff;
    float transmissionFactor;
    int baseOrDiffuseTextureIndex; // index into t_BindlessTextures[], -1 = none

    // row 4
    int metalRoughOrSpecularTextureIndex;
    int emissiveTextureIndex;
    int normalTextureIndex; // index into t_BindlessTextures[], -1 = none
    int occlusionTextureIndex;
};


StructuredBuffer<InstanceData> t_InstanceData : register(t3, space2);
StructuredBuffer<GeometryData> t_GeometryData : register(t4, space2);
StructuredBuffer<MaterialConstants> t_MaterialConstants : register(t5, space2);


uint LoadIndex(ByteAddressBuffer buf, uint byteOffset, uint stride)
{
    if (stride == 4u)
    {
        return buf.Load(byteOffset);
    }
    else
    {
        // 16-bit indices: Load() requires 4-byte alignment, unpack manually.
        uint aligned = byteOffset & ~3u;
        uint dw = buf.Load(aligned);
        return (byteOffset & 2u) ? (dw >> 16u) : (dw & 0xFFFFu);
    }
}

struct GeometrySample
{
    InstanceData instance;
    GeometryData geometry;
    MaterialConstants material;

    float3 vertexPositions[3];
    float2 vertexTexcoords[3];

    float3 objectSpacePosition;
    float2 texcoord;
    float3 flatNormal;
    float3 geometryNormal;
    float4 tangent;
};

GeometrySample getGeometryFromHit(
    uint instanceIndex,
    uint geometryIndex,
    uint triangleIndex,
    float2 rayBarycentrics,
    StructuredBuffer<InstanceData> instanceBuffer,
    StructuredBuffer<GeometryData> geometryBuffer,
    StructuredBuffer<MaterialConstants> materialBuffer
)
{
    GeometrySample gs = (GeometrySample)0;

    gs.instance = instanceBuffer[instanceIndex];
    gs.geometry = geometryBuffer[gs.instance.firstGeometryIndex + geometryIndex];
    gs.material = materialBuffer[gs.geometry.materialIndex];

    ByteAddressBuffer indexBuffer = t_BindlessBuffers[NonUniformResourceIndex(gs.geometry.indexBufferIndex)];
    ByteAddressBuffer vertexBuffer = t_BindlessBuffers[NonUniformResourceIndex(gs.geometry.vertexBufferIndex)];

    // Load triangle indices
    uint baseByteIdx = gs.geometry.indexOffset + triangleIndex * 3u * gs.geometry.indexStride;
    uint i0 = LoadIndex(indexBuffer, baseByteIdx, gs.geometry.indexStride);
    uint i1 = LoadIndex(indexBuffer, baseByteIdx + gs.geometry.indexStride, gs.geometry.indexStride);
    uint i2 = LoadIndex(indexBuffer, baseByteIdx + gs.geometry.indexStride * 2u, gs.geometry.indexStride);

    // Barycentrics
    float3 barycentrics;
    barycentrics.yz = rayBarycentrics;
    barycentrics.x = 1.0f - barycentrics.y - barycentrics.z;

    {
        gs.vertexPositions[0] = asfloat(vertexBuffer.Load3(gs.geometry.positionOffset + i0 * gs.geometry.vertexStride));
        gs.vertexPositions[1] = asfloat(vertexBuffer.Load3(gs.geometry.positionOffset + i1 * gs.geometry.vertexStride));
        gs.vertexPositions[2] = asfloat(vertexBuffer.Load3(gs.geometry.positionOffset + i2 * gs.geometry.vertexStride));
        gs.objectSpacePosition = interpolate(gs.vertexPositions, barycentrics);
    }

    {
        gs.vertexTexcoords[0] = asfloat(vertexBuffer.Load2(gs.geometry.texCoord1Offset + i0 * gs.geometry.vertexStride));
        gs.vertexTexcoords[1] = asfloat(vertexBuffer.Load2(gs.geometry.texCoord1Offset + i1 * gs.geometry.vertexStride));
        gs.vertexTexcoords[2] = asfloat(vertexBuffer.Load2(gs.geometry.texCoord1Offset + i2 * gs.geometry.vertexStride));
        gs.texcoord = interpolate(gs.vertexTexcoords, barycentrics);
    }

    {
        float3 normals[3];
        normals[0] = asfloat(vertexBuffer.Load3(gs.geometry.normalOffset + i0 * gs.geometry.vertexStride));
        normals[1] = asfloat(vertexBuffer.Load3(gs.geometry.normalOffset + i1 * gs.geometry.vertexStride));
        normals[2] = asfloat(vertexBuffer.Load3(gs.geometry.normalOffset + i2 * gs.geometry.vertexStride));
        gs.geometryNormal = interpolate(normals, barycentrics);
        gs.geometryNormal = mul(gs.instance.transform, float4(gs.geometryNormal, 0.0f)).xyz;
        gs.geometryNormal = normalize(gs.geometryNormal);
    }

    {
        float4 tangents[3];
        tangents[0] = asfloat(vertexBuffer.Load4(gs.geometry.tangentOffset + i0 * gs.geometry.vertexStride));
        tangents[1] = asfloat(vertexBuffer.Load4(gs.geometry.tangentOffset + i1 * gs.geometry.vertexStride));
        tangents[2] = asfloat(vertexBuffer.Load4(gs.geometry.tangentOffset + i2 * gs.geometry.vertexStride));
        gs.tangent.xyz = interpolate(tangents, barycentrics).xyz;
        gs.tangent.xyz = mul(gs.instance.transform, float4(gs.tangent.xyz, 0.0f)).xyz;
        gs.tangent.xyz = normalize(gs.tangent.xyz);
        gs.tangent.w = tangents[0].w;
    }

    {
        float3 objectSpaceFlatNormal = normalize(cross(
            gs.vertexPositions[1] - gs.vertexPositions[0],
            gs.vertexPositions[2] - gs.vertexPositions[0]));
        gs.flatNormal = normalize(mul(gs.instance.transform, float4(objectSpaceFlatNormal, 0.0f)).xyz);
    }

    return gs;
}

struct MaterialProps
{
    float3 Lemi;
    float3 scatteringColor;
    float3 N;
    float3 T;
    float3 baseColor;
    float alpha;
    float roughness;
    float metalness;
    float curvature;
};

SamplerState s_LinearRepeat : register(s1);
static const int MaterialFlags_DoubleSided = 0x00000002;
static const int MaterialFlags_UseMetalRoughOrSpecularTexture = 0x00000004;
static const int MaterialFlags_UseBaseOrDiffuseTexture = 0x00000008;
static const int MaterialFlags_UseEmissiveTexture = 0x00000010;
static const int MaterialFlags_UseNormalTexture = 0x00000020;

float3 UnpackNormalAG(float4 packedNormal, float scale = 1.0)
{
    float3 normal;
    normal.xy = packedNormal.ag * 2.0 - 1.0;
    normal.z = max(1.0e-16, sqrt(1.0 - saturate(dot(normal.xy, normal.xy))));

    // must scale after reconstruction of normal.z which also
    // mirrors UnpackNormalRGB(). This does imply normal is not returned
    // as a unit length vector but doesn't need it since it will get normalized after TBN transformation.
    // If we ever need to blend contributions with built-in shaders for URP
    // then we should consider using UnpackDerivativeNormalAG() instead like
    // HDRP does since derivatives do not use renormalization and unlike tangent space
    // normals allow you to blend, accumulate and scale contributions correctly.
    normal.xy *= scale;
    return normal;
}

float3 UnpackNormalRGB(float4 packedNormal, float scale = 1.0)
{
    float3 normal;
    normal.xyz = packedNormal.rgb * 2.0 - 1.0;
    normal.xy *= scale;
    return normal;
}

float3 UnpackNormalMapRGorAG(float4 packedNormal, float scale = 1.0)
{
    // Convert to (?, y, 0, x)
    packedNormal.a *= packedNormal.r;
    return UnpackNormalAG(packedNormal, scale);
}

float3 SafeNormalize(float3 inVec)
{
    float dp3 = max(1.175494351e-38, dot(inVec, inVec));
    return inVec * rsqrt(dp3);
}

float3 TransformTangentToWorld(float3 normalTS, float3x3 tangentToWorld, bool doNormalize = false)
{
    // Note matrix is in row major convention with left multiplication as it is build on the fly
    float3 result = mul(normalTS, tangentToWorld);
    if (doNormalize)
        return SafeNormalize(result);
    return result;
}


RWStructuredBuffer<uint64_t> gInOut_SharcHashEntriesBuffer: register(u12, space1);
RWStructuredBuffer<SharcAccumulationData> gInOut_SharcAccumulated: register(u13, space1);
RWStructuredBuffer<SharcPackedData> gInOut_SharcResolved: register(u14, space1);

#define RTXCR_INTEGRATION 1

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

#if( RTXCR_INTEGRATION == 1 )
// Required by RTXCR
float luminance(float3 x)
{
    return Color::Luminance(x);
}

#include "../rtxcr/HairFarFieldBCSDF.hlsli"
#include "../rtxcr/SubsurfaceScattering.hlsli"
#include "../rtxcr/Transmission.hlsli"
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
    uint primitiveIndex; // 命中的三角形索引
    float2 barycentrics; // 命中的三角形的重心坐标（uv）
    float2 texcoord;
    MaterialConstants material;

    float3 GetXoffset(float3 offsetDir, float amount = PT_BOUNCE_RAY_OFFSET)
    {
        float viewZ = Geometry::AffineTransform(gWorldToView, X).z;
        amount *= gUnproject * abs(viewZ);

        return X + offsetDir * max(amount, 0.00001);
    }

    float3 GetXoffset2(float3 offsetDir, float amount = 0.001)
    {
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



MaterialProps sampleGeometryMaterial(
    GeometryProps gs,
    SamplerState materialSampler,
    float normalMapScale = 1.0)
{
    MaterialProps props = (MaterialProps)0;


    {
        props.baseColor = gs.material.baseOrDiffuseColor;
        props.alpha = 1;
        if ((gs.material.flags & MaterialFlags_UseBaseOrDiffuseTexture) != 0 &&
            gs.material.baseOrDiffuseTextureIndex >= 0)
        {
            Texture2D<float4> tex = t_BindlessTextures[NonUniformResourceIndex(gs.material.baseOrDiffuseTextureIndex)];
            props.baseColor = tex.SampleLevel(materialSampler, gs.texcoord, 0).rgb * props.baseColor;
            props.alpha *= tex.SampleLevel(materialSampler, gs.texcoord, 0).a;
        }
    }

    {
        props.N = gs.N;
        if ((gs.material.flags & MaterialFlags_UseNormalTexture) != 0 &&
            gs.material.normalTextureIndex >= 0)
        {
            Texture2D<float4> normalTex = t_BindlessTextures[NonUniformResourceIndex(gs.material.normalTextureIndex)];
            float4 n = normalTex.SampleLevel(materialSampler, gs.texcoord, 0);
            // float3 tangentNormal = UnpackNormalMapRGorAG(n, normalMapScale);
            float3 tangentNormal = UnpackNormalMapRGorAG(n, normalMapScale);

            float3 T = normalize(gs.T.xyz);
            float3 B = - cross(props.N, T) *  sign(gs.T.w);
            half3x3 tangentToWorld = half3x3(T, B, props.N);

            float3 matWorldNormal = TransformTangentToWorld(tangentNormal, tangentToWorld);

            props.N = matWorldNormal;
            // props.N = gs.geometryNormal;
        }
    }

    {
        props.Lemi = gs.material.emissiveColor;
        if ((gs.material.flags & MaterialFlags_UseEmissiveTexture) != 0 &&
            gs.material.emissiveTextureIndex >= 0)
        {
            Texture2D<float4> emissiveTex = t_BindlessTextures[NonUniformResourceIndex(gs.material.emissiveTextureIndex)];
            props.Lemi *= emissiveTex.SampleLevel(materialSampler, gs.texcoord, 0).rgb;
        }
    }

    {
        props.roughness = gs.material.roughness;
        props.metalness = gs.material.metalness;
        if ((gs.material.flags & MaterialFlags_UseMetalRoughOrSpecularTexture) != 0 &&
            gs.material.metalRoughOrSpecularTextureIndex >= 0)
        {
            Texture2D<float4> metalRoughTex = t_BindlessTextures[NonUniformResourceIndex(gs.material.metalRoughOrSpecularTextureIndex)];
            float4 mrSample = metalRoughTex.SampleLevel(materialSampler, gs.texcoord, 0);
            props.roughness = 1 - (1 - mrSample.g) * (1 - props.roughness);
            props.metalness = mrSample.b;
        }
    }


    return props;
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

struct RayPayload
{
    float committedRayT;
    uint instanceID;
    uint geometryIndex;
    uint triangleIndex;
    float2 barycentrics;
};

[shader("miss")]
void MainMissShader(inout RayPayload payload : SV_RayPayload)
{
    payload.instanceID = ~0u;
    payload.committedRayT = INF;
}

uint ToRayFlag(uint flag)
{
    if (flag == FLAG_TRANSPARENT)
        return RAY_FLAG_CULL_OPAQUE;
    else if (flag == FLAG_NON_TRANSPARENT)
        return RAY_FLAG_CULL_NON_OPAQUE;
    else
        return RAY_FLAG_NONE;
}

uint ToRayFlag2(uint flag)
{
    if (flag == FLAG_TRANSPARENT)
        return RAY_FLAG_CULL_OPAQUE;
    else
        return RAY_FLAG_NONE;
}

struct AttributeData
{
    float2 barycentrics;
};

[shader("closesthit")]
void ClosestHit_0_Shader(inout RayPayload payload, AttributeData attribs : SV_IntersectionAttributes)
{
    payload.instanceID = InstanceID();
    payload.geometryIndex = GeometryIndex();
    payload.triangleIndex = PrimitiveIndex();
    payload.barycentrics = attribs.barycentrics;
    payload.committedRayT = RayTCurrent();
}

float CastVisibilityRay_AnyHit(float3 origin, float3 direction, float Tmin, float Tmax, float2 mipAndCone, RaytracingAccelerationStructure accelerationStructure, uint mask, uint rayFlags)
{
    RayDesc rayDesc;
    rayDesc.Origin = origin;
    rayDesc.Direction = direction;
    rayDesc.TMin = Tmin;
    rayDesc.TMax = Tmax;

    RayPayload payload = (RayPayload)0;
    // payload.mipAndCone = mipAndCone;

    uint flag = ToRayFlag2(mask);
    flag = flag | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;

    TraceRay(accelerationStructure, flag, mask, 0, 0, 0, rayDesc, payload);

    return payload.committedRayT;
}

GeometryProps CastRay(float3 origin, float3 direction, float Tmin, float Tmax, float2 mipAndCone, uint mask)
{
    RayDesc rayDesc;
    rayDesc.Origin = origin;
    rayDesc.Direction = direction;
    rayDesc.TMin = Tmin;
    rayDesc.TMax = Tmax;

    RayPayload payload = (RayPayload)0;
    // payload.mipAndCone = mipAndCone;

    // if (mask == FLAG_NON_TRANSPARENT)
    //     payload.SetFlag(FLAG_IGNORE_WHEN_TRANSPARENT);

    TraceRay(gWorldTlas, ToRayFlag2(mask), mask, 0, 0, 0, rayDesc, payload);
    GeometryProps props = (GeometryProps)0;
    props.hitT = payload.committedRayT;
    
    if (payload.committedRayT != INF)
    {
        GeometrySample geo = getGeometryFromHit(
            payload.instanceID,
            payload.geometryIndex,
            payload.triangleIndex,
            payload.barycentrics,
            t_InstanceData,
            t_GeometryData,
            t_MaterialConstants
        );

        props.instanceIndex = payload.instanceID;
        props.N = geo.geometryNormal;
        props.curvature = 1;

        props.mip = 0;

        props.T = geo.tangent;
        props.X = origin + direction * payload.committedRayT;

        props.Xprev = props.X;

        if (gIsEditor)
        {
            props.Xprev = props.X;
        }

        props.V = -direction;
        props.textureOffsetAndFlags = 0;
        props.primitiveIndex = payload.triangleIndex;
        props.barycentrics = payload.barycentrics;
        props.texcoord = geo.texcoord;
        props.material = geo.material;
    }
    return props;
}


MaterialProps GetMaterialProps( GeometryProps geometryProps )
{
    
    MaterialProps props = ( MaterialProps )0;
    
    [branch]
    if( geometryProps.IsMiss( ) )
    {
        props.Lemi = GetSkyIntensity( -geometryProps.V );

        return props;
    }
    
    MaterialProps mat = sampleGeometryMaterial(geometryProps, s_LinearRepeat);
    
    props = (MaterialProps)0;
    props.baseColor = mat.baseColor;

    props.roughness = mat.roughness;
    props.metalness = mat.metalness;

    // if (props.Has(FLAG_SKIN))
    // {
    //     matProps.scatteringColor = Packing::DecodeRgbe(payload.Lemi);
    //     matProps.Lemi = 0;
    // }
    // else
    // {
    //     matProps.Lemi = Packing::DecodeRgbe(payload.Lemi);
    // }
    props.Lemi = mat.Lemi;

    // 这三个应该从贴图再计算一次
    props.curvature = mat.curvature;
    props.N = mat.N;
    props.T = mat.T;
    
    
    return  props;
}

#include "DirectionalLights.hlsl"
#include "SpotLights.hlsl"
#include "AreaLights.hlsl"
#include "PointLights.hlsl"

// Compile-time flags for "GetLighting"
#define LIGHTING    0x01
#define SHADOW      0x02
#define SSS         0x04

float3 GetLighting(GeometryProps geometryProps, MaterialProps materialProps, uint flags)
{
    float3 lighting = 0.0;

    if ((flags & LIGHTING) != 0)
    {
        bool isSSS = (flags & SSS) != 0 && geometryProps.Has(FLAG_SKIN);
        lighting += EvaluateDirectionalLights(geometryProps, materialProps, isSSS);
        lighting += EvaluateSpotLights(geometryProps, materialProps, isSSS);
        lighting += EvaluateAreaLights(geometryProps, materialProps, isSSS);
        lighting += EvaluatePointLights(geometryProps, materialProps, isSSS);
    }

    return lighting;
}

// Compile-time flags for "GenerateRayAndUpdateThroughput"
#define HAIR 0x1

float3 GenerateRayAndUpdateThroughput(inout GeometryProps geometryProps, inout MaterialProps materialProps, inout float3 throughput, bool isDiffuse, float2 rnd)
{
    float3x3 mLocalBasis = Geometry::GetBasis(materialProps.N);
    float3 Vlocal = Geometry::RotateVector(mLocalBasis, geometryProps.V);

    // Importance sampling

    float3 candidateRayLocal;

    if (isDiffuse)
        candidateRayLocal = ImportanceSampling::Cosine::GetRay(rnd);
    else
    {
        float3 Hlocal = ImportanceSampling::VNDF::GetRay(rnd, materialProps.roughness, Vlocal, PT_SPEC_LOBE_ENERGY);
        candidateRayLocal = reflect(-Vlocal, Hlocal);
    }

    float3 rayLocal = candidateRayLocal;

    // Update throughput
    float3 albedo, Rf0;
    BRDF::ConvertBaseColorMetalnessToAlbedoRf0(materialProps.baseColor, materialProps.metalness, albedo, Rf0);

    float3 Nlocal = float3(0, 0, 1);
    float3 Hlocal = normalize(Vlocal + rayLocal);

    float NoL = saturate(dot(Nlocal, rayLocal));
    float VoH = abs(dot(Vlocal, Hlocal));

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

    // Transform to world space
    float3 ray = Geometry::RotateVectorInverse(mLocalBasis, rayLocal);

    return ray;
}


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
        // #if( RTXCR_INTEGRATION == 1 )
        // if (isHair)
        // {
        //     float2 rand[2] = {Rng::Hash::GetFloat2(), Rng::Hash::GetFloat2()};
        //
        //     float3 specular = 0.0;
        //     float3 diffuse = 0.0;
        //     float pdf = 0.0;
        //
        //     RTXCR_HairInteractionSurface hairSurface = Hair_GetSurface(Vlocal);
        //     RTXCR_HairMaterialInteractionBcsdf hairMaterial = Hair_GetMaterial();
        //     RTXCR_SampleFarFieldBcsdf(hairSurface, hairMaterial, Vlocal, 2.0 * rnd.x - 1.0, rnd.y, rand, candidateRayLocal, specular, diffuse, pdf);
        // }
        // else
        // #endif
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

    // #if( RTXCR_INTEGRATION == 1 )
    // if (isHair)
    // {
    //     float3 specular = 0.0;
    //     float3 diffuse = 0.0;
    //     float pdf = 0.0;
    //
    //     RTXCR_HairInteractionSurface hairGeometry = Hair_GetSurface(Vlocal);
    //     RTXCR_HairMaterialInteractionBcsdf hairMaterial = Hair_GetMaterial();
    //     RTXCR_HairFarFieldBcsdfEval(hairGeometry, hairMaterial, rayLocal, Vlocal, specular, diffuse, pdf);
    //
    //     throughput *= pdf > 0.0 ? (specular + diffuse) / pdf : 0.0;
    // }
    // else
    // #endif
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
