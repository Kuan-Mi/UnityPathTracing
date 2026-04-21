// ---- Resource bindings ----
RaytracingAccelerationStructure SceneBVH : register(t0);

RWTexture2D<float4> OutputTexture : register(u0);

ByteAddressBuffer t_BindlessBuffers[] : register(t0, space1);
Texture2D<float4> t_BindlessTextures[] : register(t0, space2);

#define  NonUniformResourceIndex 

struct GeometryData
{
    uint numIndices;
    uint numVertices;
    int  indexBufferIndex;   // index into t_BindlessBuffers
    uint indexOffset;        // byte offset into index buffer

    int  vertexBufferIndex;  // index into t_BindlessBuffers
    uint positionOffset;     // byte offset to first position (float3 per vertex)
    uint normalOffset;       // byte offset to first normal  (float3 per vertex, or ~0u if absent)
    uint texCoord1Offset;    // byte offset to first texcoord (float2 per vertex, or ~0u if absent)

    uint tangentOffset;      // byte offset to first tangent (float4 per vertex, or ~0u if absent)
    uint vertexStride;       // stride in bytes between consecutive vertices
    uint indexStride;        // 2 (16-bit) or 4 (32-bit)
    uint materialIndex;
};

struct InstanceData
{
    uint firstGeometryIndex; // index into t_GeometryData for the first geometry of this instance
    uint numGeometries;
    uint pad0;
    uint pad1;
    row_major float3x4 transform;      // object-to-world (row-major)
};

struct MaterialConstants
{
    // row 0
    float3 baseOrDiffuseColor;
    int    flags;

    // row 1
    float3 emissiveColor;
    int    domain;

    // row 2
    float  opacity;
    float  roughness;
    float  metalness;
    float  normalTextureScale;

    // row 3
    float  occlusionStrength;
    float  alphaCutoff;
    float  transmissionFactor;
    int    baseOrDiffuseTextureIndex;   // index into t_BindlessTextures[], -1 = none

    // row 4
    int    metalRoughOrSpecularTextureIndex;
    int    emissiveTextureIndex;
    int    normalTextureIndex;          // index into t_BindlessTextures[], -1 = none
    int    occlusionTextureIndex;
};


StructuredBuffer<InstanceData>     t_InstanceData : register(t1);
StructuredBuffer<GeometryData>     t_GeometryData : register(t2);
StructuredBuffer<MaterialConstants> t_MaterialConstants : register(t3);

cbuffer SceneConstants : register(b0)
{
    float4x4 viewProjInv;
    float3 cameraPos;
    float _scenePad;
};

// struct [raypayload] RayPayload
// {
//     uint hit : write(caller, closesthit, miss) : read(caller, closesthit, miss);
//     float2 barycentrics : write(caller, closesthit) : read(caller, closesthit);
// };

// struct RayPayload
// {
//     uint hit;
//     float2 barycentrics;
// };


// ---- Payloads / hit attributes ----
struct  RayPayload
{
    float committedRayT;
    uint instanceID ;
    uint geometryIndex ;
    uint triangleIndex ;
    float2 barycentrics;
};

float3 HashColor(uint id)
{
    uint hash = id;
    hash ^= 2747636419u;
    hash *= 2654435769u;
    hash ^= hash >> 16;
    hash *= 2654435769u;
    hash ^= hash >> 16;
    hash *= 2654435769u;

    return float3(hash & 0xFFu, (hash >> 8) & 0xFFu, (hash >> 16) & 0xFFu) / 255.0f;
}


float interpolate(float vertices[3], float3 bary)
{
    return vertices[0] * bary[0] + vertices[1] * bary[1] + vertices[2] * bary[2];
}

float2 interpolate(float2 vertices[3], float3 bary)
{
    return vertices[0] * bary[0] + vertices[1] * bary[1] + vertices[2] * bary[2];
}

float3 interpolate(float3 vertices[3], float3 bary)
{
    return vertices[0] * bary[0] + vertices[1] * bary[1] + vertices[2] * bary[2];
}

float4 interpolate(float4 vertices[3], float3 bary)
{
    return vertices[0] * bary[0] + vertices[1] * bary[1] + vertices[2] * bary[2];
}

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
        gs.tangent.w   = tangents[0].w;
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

SamplerState s_LinearRepeat  : register(s1);
static const int MaterialFlags_DoubleSided                      = 0x00000002;
static const int MaterialFlags_UseMetalRoughOrSpecularTexture   = 0x00000004;
static const int MaterialFlags_UseBaseOrDiffuseTexture          = 0x00000008;
static const int MaterialFlags_UseEmissiveTexture               = 0x00000010;
static const int MaterialFlags_UseNormalTexture                 = 0x00000020;

MaterialProps sampleGeometryMaterial(
GeometrySample gs, 
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
            props.roughness *= mrSample.g;
            props.metalness *= mrSample.b;
        }
    }
    
    
    return props;
}

float3 RamdomizeRayDirection(uint2 idx)
{
    uint seed = idx.x * 73856093u ^ idx.y * 19349663u; // simple hash to get a pseudo-random seed per pixel
    seed = (seed ^ 2747636419u) * 2654435769u; // further scramble the bits
    float randX = ((seed & 0xFFFFu) / 65535.0f - 0.5f) * 0.01f; // small random offset
    float randY = (((seed >> 16) & 0xFFFFu) / 65535.0f - 0.5f) * 0.01f; 
    float randZ = (((seed >> 8) & 0xFFFFu) / 65535.0f - 0.5f) * 0.01f;
    return float3(randX, randY, randZ);
}

using namespace dx; // dx::HitObject

[shader("raygeneration")]
void RayGenShader()
{
    uint2 idx = DispatchRaysIndex().xy;
    uint2 dim = DispatchRaysDimensions().xy;

    float2 uv = (float2(idx) + 0.5f) / float2(dim);
    float2 ndc = uv * 2.0f - 1.0f;

    float4 target = mul(viewProjInv, float4(ndc, 1, 1));
    target /= target.w;

    RayDesc ray;
    ray.Origin = cameraPos;
    ray.Direction = normalize(target.xyz - cameraPos);
    ray.TMin = 0.001f;
    ray.TMax = 1000.0f;

    // ray.Direction = RamdomizeRayDirection(idx); // for testing - add some noise to ray direction to verify that rays are not getting excessively coherent after primary hit (which would indicate potential issues with payload handling or thread reordering)

    RayPayload payload = (RayPayload)0;

    float3 color = 0;
    uint flag = RAY_FLAG_NONE;

    for (int i = 0; i < 1; ++i)
    {
        payload.instanceID = ~0u;
        
        // TraceRay(SceneBVH, flag, 0xFF, 0, 0, 0, ray, payload);
        // if (payload.instanceID == ~0u)
        // {
        //     color = float3(0, 0, 0);
        //     break;
        // }

        HitObject hit = HitObject::TraceRay(SceneBVH, flag, 0xFF, 0, 0, 0, ray, payload);

        // MaybeReorderThread(hit);
        // MaybeReorderThread(hit, hit.GetInstanceID(), 32);
        
        if (hit.IsMiss())
        {
            color = float3(0, 0, 0);
            break;
        }
        HitObject::Invoke(hit, payload);
        color = float3(1, 1, 1); // hit - white for testing
        
        // {
        //     GeometryData geomData = t_GeometryData[t_InstanceData[payload.instanceID].firstGeometryIndex + payload.geometryIndex];
        //     MaterialConstants matConst = t_MaterialConstants[geomData.materialIndex];
        //
        //     
        //     GeometrySample geo = getGeometryFromHit (
        //         payload.instanceID,
        //         payload.geometryIndex,
        //         payload.triangleIndex,
        //         payload.barycentrics,
        //         t_InstanceData,
        //         t_GeometryData,
        //         t_MaterialConstants
        //     );
        //     
        //     // Test1 - OK
        //     // color += HashColor(payload.instanceID + payload.geometryIndex * 9973 + payload.triangleIndex * 99991);
        //     
        //     // Test2 - OK
        //     // color += geo.flatNormal;
        //     
        //     // Test3 - not pass
        //     MaterialProps mat = sampleGeometryMaterial(geo, s_LinearRepeat);
        //     
        //     ray.Origin = ray.Origin + ray.Direction * payload.committedRayT;
        //     // ray.Direction = normalize(reflect(ray.Direction, mat.N));
        //     ray.Direction = normalize(reflect(ray.Direction, mat.N));
        //     
        //     color = mat.baseColor;
        // }
    }

    OutputTexture[idx] = float4(color, 1.0f);
}

[shader("miss")]
void Miss_0_Shader(inout RayPayload payload)
{
    payload.instanceID = ~0u;
}

struct AttributeData
{
    float2 barycentrics;
};

[shader("anyhit")]
void AnyHit_0_Shader(inout RayPayload payload, AttributeData attribs : SV_IntersectionAttributes)
{
}

[shader("closesthit")]
void ClosestHit_0_Shader(inout RayPayload payload, AttributeData attribs : SV_IntersectionAttributes)
{
    payload.instanceID = InstanceID();
    payload.geometryIndex = GeometryIndex();
    payload.triangleIndex = PrimitiveIndex();
    payload.barycentrics = attribs.barycentrics;
    payload.committedRayT = RayTCurrent();
}
