#ifndef SCENE_GEOMETRY_HLSL
#define SCENE_GEOMETRY_HLSL

#include "bindless.hlsl"

#include <donut/shaders/utils.hlsli>

ByteAddressBuffer t_BindlessBuffers[] : register(t0, space1);
Texture2D<float4> t_BindlessTextures[] : register(t0, space2);



enum GeometryAttributes
{
    GeomAttr_Position     = 0x01,
    GeomAttr_TexCoord     = 0x02,
    GeomAttr_Normal       = 0x04,
    GeomAttr_Tangents     = 0x08,
    GeomAttr_PrevPosition = 0x10,

    GeomAttr_All          = 0x1F
};

struct GeometrySample
{
    InstanceData     instance;
    GeometryData     geometry;
    MaterialConstants material;

    float3 vertexPositions[3];
    float2 vertexTexcoords[3];

    float3 objectSpacePosition;
    float2 texcoord;
    float3 flatNormal;
    float3 geometryNormal;
    float4 tangent;
};

// ---- Helpers ----

// Load a 32-bit or 16-bit index from a ByteAddressBuffer
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


GeometrySample getGeometryFromHit(
    uint instanceIndex,
    uint geometryIndex,
    uint triangleIndex,
    float2 rayBarycentrics,
    GeometryAttributes attributes,
    StructuredBuffer<InstanceData> instanceBuffer,
    StructuredBuffer<GeometryData> geometryBuffer,
    StructuredBuffer<MaterialConstants> materialBuffer
    )
{
    GeometrySample gs = (GeometrySample)0;

    gs.instance = instanceBuffer[instanceIndex];
    gs.geometry = geometryBuffer[gs.instance.firstGeometryIndex + geometryIndex];
    gs.material = materialBuffer[gs.geometry.materialIndex];

    ByteAddressBuffer indexBuffer  = t_BindlessBuffers[NonUniformResourceIndex(gs.geometry.indexBufferIndex)];
    ByteAddressBuffer vertexBuffer = t_BindlessBuffers[NonUniformResourceIndex(gs.geometry.vertexBufferIndex)];

    // Load triangle indices
    uint baseByteIdx = gs.geometry.indexOffset + triangleIndex * 3u * gs.geometry.indexStride;
    uint i0 = LoadIndex(indexBuffer, baseByteIdx,                                gs.geometry.indexStride);
    uint i1 = LoadIndex(indexBuffer, baseByteIdx + gs.geometry.indexStride,      gs.geometry.indexStride);
    uint i2 = LoadIndex(indexBuffer, baseByteIdx + gs.geometry.indexStride * 2u, gs.geometry.indexStride);

    // Barycentrics
    float3 barycentrics;
    barycentrics.yz = rayBarycentrics;
    barycentrics.x  = 1.0f - barycentrics.y - barycentrics.z;

    if (attributes & GeomAttr_Position)
    {
        gs.vertexPositions[0] = asfloat(vertexBuffer.Load3(gs.geometry.positionOffset + i0 * gs.geometry.vertexStride));
        gs.vertexPositions[1] = asfloat(vertexBuffer.Load3(gs.geometry.positionOffset + i1 * gs.geometry.vertexStride));
        gs.vertexPositions[2] = asfloat(vertexBuffer.Load3(gs.geometry.positionOffset + i2 * gs.geometry.vertexStride));
        gs.objectSpacePosition = interpolate(gs.vertexPositions, barycentrics);
    }

    if ((attributes & GeomAttr_TexCoord) && gs.geometry.texCoord1Offset != ~0u)
    {
        gs.vertexTexcoords[0] = asfloat(vertexBuffer.Load2(gs.geometry.texCoord1Offset + i0 * gs.geometry.vertexStride));
        gs.vertexTexcoords[1] = asfloat(vertexBuffer.Load2(gs.geometry.texCoord1Offset + i1 * gs.geometry.vertexStride));
        gs.vertexTexcoords[2] = asfloat(vertexBuffer.Load2(gs.geometry.texCoord1Offset + i2 * gs.geometry.vertexStride));
        gs.texcoord = interpolate(gs.vertexTexcoords, barycentrics);
    }

    if ((attributes & GeomAttr_Normal) && gs.geometry.normalOffset != ~0u)
    {
        float3 normals[3];
        normals[0] = asfloat(vertexBuffer.Load3(gs.geometry.normalOffset + i0 * gs.geometry.vertexStride));
        normals[1] = asfloat(vertexBuffer.Load3(gs.geometry.normalOffset + i1 * gs.geometry.vertexStride));
        normals[2] = asfloat(vertexBuffer.Load3(gs.geometry.normalOffset + i2 * gs.geometry.vertexStride));
        gs.geometryNormal = interpolate(normals, barycentrics);
        gs.geometryNormal = mul(gs.instance.transform, float4(gs.geometryNormal, 0.0f)).xyz;
        gs.geometryNormal = normalize(gs.geometryNormal);
    }

    if ((attributes & GeomAttr_Tangents) && gs.geometry.tangentOffset != ~0u)
    {
        float4 tangents[3];
        tangents[0] = asfloat(vertexBuffer.Load4(gs.geometry.tangentOffset + i0 * gs.geometry.vertexStride));
        tangents[1] = asfloat(vertexBuffer.Load4(gs.geometry.tangentOffset + i1 * gs.geometry.vertexStride));
        tangents[2] = asfloat(vertexBuffer.Load4(gs.geometry.tangentOffset + i2 * gs.geometry.vertexStride));
        gs.tangent.xyz = interpolate(tangents, barycentrics).xyz;
        gs.tangent.xyz = mul(gs.instance.transform, float4(gs.tangent.xyz, 0.0f)).xyz;
        gs.tangent.xyz = normalize(gs.tangent.xyz);
        gs.tangent.w   = tangents[0].w;
    }

    if (attributes & GeomAttr_Position)
    {
        float3 objectSpaceFlatNormal = normalize(cross(
            gs.vertexPositions[1] - gs.vertexPositions[0],
            gs.vertexPositions[2] - gs.vertexPositions[0]));
        gs.flatNormal = normalize(mul(gs.instance.transform, float4(objectSpaceFlatNormal, 0.0f)).xyz);
    }

    return gs;
}


enum MaterialAttributes
{
    MatAttr_BaseColor    = 0x01,
    MatAttr_Emissive     = 0x02,
    MatAttr_Normal       = 0x04,
    MatAttr_MetalRough   = 0x08,
    MatAttr_Transmission = 0x10,

    MatAttr_All          = 0x1F
};

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

MaterialProps sampleGeometryMaterial(
GeometrySample gs, 
MaterialAttributes attributes, 
SamplerState materialSampler,
float normalMapScale = 1.0)
{
    MaterialProps props = (MaterialProps)0;
    
    
    if ((attributes & MatAttr_BaseColor) != 0)
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
    
    if ((attributes & MatAttr_Normal) != 0)
    {
        props.N = gs.geometryNormal;
        if ((gs.material.flags & MaterialFlags_UseNormalTexture) != 0 &&
            gs.material.normalTextureIndex >= 0)
        {
            Texture2D<float4> normalTex = t_BindlessTextures[NonUniformResourceIndex(gs.material.normalTextureIndex)];
            float3 normalSample = normalTex.SampleLevel(materialSampler, gs.texcoord, 0).xyz * 2.0f - 1.0f;
            normalSample.xy *= normalMapScale;
            float3 T = normalize(gs.tangent.xyz);
            float3 B = cross(props.N, T) * gs.tangent.w;
            props.N = normalize(normalSample.x * T + normalSample.y * B + normalSample.z * props.N);
        }
    }
    
    if ((attributes & MatAttr_Emissive) != 0)
    {
        props.Lemi = gs.material.emissiveColor;
        if ((gs.material.flags & MaterialFlags_UseEmissiveTexture) != 0 &&
            gs.material.emissiveTextureIndex >= 0)
        {
            Texture2D<float4> emissiveTex = t_BindlessTextures[NonUniformResourceIndex(gs.material.emissiveTextureIndex)];
            props.Lemi *= emissiveTex.SampleLevel(materialSampler, gs.texcoord, 0).rgb;
        }
    }
    
    if ((attributes & MatAttr_MetalRough) != 0)
    {
        props.roughness = gs.material.roughness;
        props.metalness = gs.material.metalness;
        if ((gs.material.flags & MaterialFlags_UseMetalRoughOrSpecularTexture) != 0 &&
            gs.material.metalRoughOrSpecularTextureIndex >= 0)
        {
            Texture2D<float4> metalRoughTex = t_BindlessTextures[NonUniformResourceIndex(gs.material.metalRoughOrSpecularTextureIndex)];
            float4 mrSample = metalRoughTex.SampleLevel(materialSampler, gs.texcoord, 0);
            props.roughness *= mrSample.g;
            props.metalness *= mrSample.b;
        }
    }
    
    
    return props;
}

#endif // SCENE_GEOMETRY_HLSL