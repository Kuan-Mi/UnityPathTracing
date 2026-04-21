// ---- Resource bindings ----
RaytracingAccelerationStructure gWorldTlas : register(t0);
RaytracingAccelerationStructure gLightTlas : register(t1);

#include "ml.hlsli"

struct PrimitiveData
{
    float16_t2 uv0;
    float16_t2 uv1;
    float16_t2 uv2;
    float worldArea;

    float16_t2 n0;
    float16_t2 n1;
    float16_t2 n2;
    float uvArea;

    float16_t2 t0;
    float16_t2 t1;
    float16_t2 t2;
    float bitangentSign;
};

struct InstanceData
{
    // For static: mObjectToWorld
    // For dynamic: mWorldToWorldPrev
    float4 mOverloadedMatrix0;
    float4 mOverloadedMatrix1;
    float4 mOverloadedMatrix2;

    float16_t4 baseColorAndMetalnessScale;
    float16_t4 emissionAndRoughnessScale;

    float16_t2 normalUvScale;
    uint32_t textureOffsetAndFlags;
    uint32_t primitiveOffset;
    float scale; // TODO: handling object scale embedded into the transformation matrix (assuming uniform scale), sign represents triangle winding

    uint32_t morphPrimitiveOffset;
    uint32_t unused1;
    uint32_t unused2;
    uint32_t unused3;
};

struct MorphVertex // same as utils::MorphVertex
{
    float16_t4 pos;
    float16_t2 N;
    float16_t2 T;
};

struct MorphAttributes
{
    float16_t2 N;
    float16_t2 T;
};

struct MorphPrimitivePositions
{
    float16_t4 pos0;
    float16_t4 pos1;
    float16_t4 pos2;
};


StructuredBuffer<InstanceData> gIn_InstanceData : register(t2);
StructuredBuffer<PrimitiveData> gIn_PrimitiveData : register(t3);
StructuredBuffer<MorphPrimitivePositions> gIn_MorphPrimitivePositionsPrev : register(t4);


Texture2D<float4> gIn_Textures[]: register(t0, space2 );

RWTexture2D<float4> OutputTexture : register(u0);


#define  NonUniformResourceIndex 

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
struct RayPayload
{
    float committedRayT;
    uint instanceID;
    uint geometryIndex;
    uint triangleIndex;
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

using namespace dx; // dx::HitObject


// Instance flags
#define FLAG_FIRST_BIT                      24 // this + number of flags must be <= 32
#define NON_FLAG_MASK                       ( ( 1 << FLAG_FIRST_BIT ) - 1 )

#define INF                                 1e5 // IMPORTANT: INF * FP16_VIEWZ_SCALE < FP16_MAX!

struct GeometryProps
{
    float3 X;
    float3 Xprev;
    float3 V;
    float4 T;
    float3 N;
    float2 uv;
    float mip;
    float hitT;
    float curvature;
    uint textureOffsetAndFlags;
    uint instanceIndex;

    bool Has( uint flag )
    { return ( textureOffsetAndFlags & ( flag << FLAG_FIRST_BIT ) ) != 0; }

    uint GetBaseTexture( )
    { return textureOffsetAndFlags & NON_FLAG_MASK; }

    float3 GetForcedEmissionColor( )
    { return ( ( textureOffsetAndFlags >> 2 ) & 0x1 ) ? float3( 1.0, 0.0, 0.0 ) : float3( 0.0, 1.0, 0.0 ); }

    bool IsMiss( )
    { return hitT == INF; }
};

#define FLAG_NON_TRANSPARENT                0x01 // geometry flag: non-transparent
#define FLAG_TRANSPARENT                    0x02 // geometry flag: transparent
#define FLAG_FORCED_EMISSION                0x04 // animated emissive cube
#define FLAG_STATIC                         0x08 // no velocity
#define FLAG_HAIR                           0x10 // hair
#define FLAG_LEAF                           0x20 // leaf
#define FLAG_SKIN                           0x40 // skin
#define FLAG_MORPH                          0x80 // morph

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

        HitObject hit = HitObject::TraceRay(gWorldTlas, flag, 0xFF, 0, 0, 0, ray, payload);

        // MaybeReorderThread(hit);
        // MaybeReorderThread(hit, hit.GetInstanceID(), 32);

        if (hit.IsMiss())
        {
            color = float3(0, 0, 0);
            break;
        }
        HitObject::Invoke(hit, payload);
        
        
        color = float3(1, 1, 1); // hit - white for testing

        color = payload.geometryIndex / 4.0f; 
        
        
        
        GeometryProps props = ( GeometryProps )0;
        
        props.hitT = hit.GetRayTCurrent();
        
        uint instanceIndex = hit.GetInstanceID() + hit.GetGeometryIndex();
        props.instanceIndex = instanceIndex;
        
        
        InstanceData instanceData = gIn_InstanceData[ instanceIndex ];
        
        props.textureOffsetAndFlags = instanceData.textureOffsetAndFlags;
        
        
        float3x3 mObjectToWorld = ( float3x3 )hit.GetObjectToWorld3x4();
        float3x4 mOverloaded = float3x4( instanceData.mOverloadedMatrix0, instanceData.mOverloadedMatrix1, instanceData.mOverloadedMatrix2 ); \
        
        if( props.Has( FLAG_STATIC ) )
            mObjectToWorld = ( float3x3 )mOverloaded;
        
         
        float flip = ( (hit.GetHitKind() == HIT_KIND_TRIANGLE_FRONT_FACE) ? -1.0 : 1.0 );
        
        
        uint primitiveIndex = instanceData.primitiveOffset + hit.GetPrimitiveIndex( );
        PrimitiveData primitiveData = gIn_PrimitiveData[ primitiveIndex ];
        
        float worldArea = primitiveData.worldArea * instanceData.scale * instanceData.scale;
        
        
        // Barycentrics
        float3 barycentrics;
        barycentrics.yz =  payload.barycentrics;
        barycentrics.x = 1.0 - barycentrics.y - barycentrics.z;
        
        // Normal
        float3 n0 = Packing::DecodeUnitVector( primitiveData.n0, true );
        float3 n1 = Packing::DecodeUnitVector( primitiveData.n1, true );
        float3 n2 = Packing::DecodeUnitVector( primitiveData.n2, true );
        
        
        float3 N = barycentrics.x * n0 + barycentrics.y * n1 + barycentrics.z * n2;
        N = Geometry::RotateVector( mObjectToWorld, N );
        N = normalize( N * flip );
        props.N = -N; // TODO: why negated?
        
        
        
        color = HashColor( instanceIndex + primitiveIndex * 9973 + hit.GetPrimitiveIndex( ) * 99991 );
        color = N;
        
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
