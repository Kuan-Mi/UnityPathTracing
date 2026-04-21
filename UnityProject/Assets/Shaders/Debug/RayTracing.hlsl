#include "SceneGeometry.hlsl"

// ---- Resource bindings ----
RaytracingAccelerationStructure SceneBVH : register(t0);
RWTexture2D<float4> OutputTexture : register(u0);
// Per-instance and per-geometry metadata
StructuredBuffer<InstanceData>     t_InstanceData : register(t1);
StructuredBuffer<GeometryData>     t_GeometryData : register(t2);
StructuredBuffer<MaterialConstants> t_MaterialConstants : register(t3);

SamplerState s_LinearRepeat  : register(s1);

cbuffer SceneConstants : register(b0)
{
    float4x4 viewProjInv;
    float3 cameraPos;
    float _scenePad;
};

struct  TestConstants
{
    float4 dummy;
};

ConstantBuffer<TestConstants> testConstants : register(b1, space3);


Texture2D<float4> t_TestBindless[] : register(t0, space3);


// ---- Constants ----
static const uint c_SizeOfPosition = 12u; // float3
static const uint c_SizeOfTexcoord = 8u; // float2
static const uint c_SizeOfNormal = 12u; // float3

// // ---- Payloads / hit attributes ----
// struct [raypayload] RayPayload
// {
//     // float3 throughput: write() : read();
//     float committedRayT: write(caller, closesthit) : read(caller);
//
//     uint instanceID : write(caller, closesthit,miss) : read(caller);
//     uint geometryIndex : write(caller, closesthit) : read(caller);
//     uint triangleIndex : write(caller, closesthit) : read(caller);
//     // bool frontFace: write(caller, closesthit, miss) : read(caller);
//
//     float2 barycentrics: write(caller, closesthit) : read(caller);
//
//
// };
//
// struct [raypayload] RayPayload2
// {
//     uint instanceID : write( caller,closesthit,miss) : read(caller);
//     uint geometryIndex : write( closesthit) : read(caller);
//     uint triangleIndex : write( closesthit) : read(caller);
//     // bool frontFace: write(caller, closesthit, miss) : read(caller);
//
//     float2 barycentrics: write( closesthit) : read(caller);
// };


struct  RayPayload
{
    float committedRayT;
    uint instanceID ;
    uint geometryIndex ;
    uint triangleIndex ;
    float2 barycentrics;
};

struct  RayPayload2
{
    uint instanceID ;
    uint geometryIndex ;
    uint triangleIndex ;

    float2 barycentrics;
};

// ---- Shaders ----

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

    RayPayload payload = (RayPayload)0;

    payload.instanceID = ~0u;

    // SER: decouple traversal from shading to allow thread reordering
    // HitObject hit = HitObject::TraceRay(SceneBVH, RAY_FLAG_FORCE_NON_OPAQUE, 0xFF, 0, 0, 0, ray, payload);
    HitObject hit = HitObject::TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

    // Reorder threads: sort key 1 = hit (expensive), 0 = miss (cheap).
    // Separates divergent ClosestHit / Miss execution paths.
    // Note: explicitly scoped as dx::MaybeReorderThread to avoid a DXC bug
    // where 'using namespace dx' generates bad DXIL for this intrinsic.
    // dx::MaybeReorderThread(hit.IsHit() ? 1 : 0, 1);

    // Dispatch ClosestHit or Miss shader for the reordered thread
    HitObject::Invoke(hit, payload);

    if(payload.instanceID == ~0u)
    {
        // Miss shader was invoked, write sky color
        OutputTexture[idx] = float4(0.1f, 0.1f, 0.1f, 1.0f);
        return;
    }
    
    GeometrySample gs = getGeometryFromHit(
        payload.instanceID,
        payload.geometryIndex,
        payload.triangleIndex,
        payload.barycentrics,
        GeomAttr_All,
        t_InstanceData, t_GeometryData,t_MaterialConstants);

    // Sample base color from material texture (or tint color if no texture)
    MaterialProps props = sampleGeometryMaterial(gs,MatAttr_All,s_LinearRepeat);

    // Simple diffuse shading with a fixed light direction
    float3 lightDir = normalize(float3(1, 1, -1));
    float NdotL = saturate(dot(gs.geometryNormal, lightDir));
    float3 color = props.baseColor.rgb * (0.2f + 0.8f * NdotL) + payload.committedRayT*0.0001;

    // color *= testConstants.dummy.xyz;
    
    float3 tt = t_TestBindless[0].SampleLevel(s_LinearRepeat, gs.texcoord, 0).xyz;
    
    // float3 reflectDir = reflect(ray.Direction, props.N);
    // // reflectDir = ray.Direction;
    //
    // ray.Origin = ray.Origin + ray.Direction * payload.committedRayT + gs.geometryNormal * 0.001f; // offset ray origin to avoid self-intersection
    // ray.Direction = normalize(reflectDir.xyz);
    //
    // RayPayload2 pp;
    // pp.instanceID = ~0u;
    //
    // HitObject reflectedHit = HitObject::TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 1, 0, 1, ray, pp);
    // dx::MaybeReorderThread(reflectedHit.IsHit() ? 1 : 0, 1);
    // HitObject::Invoke(reflectedHit, pp);
    //
    // if(pp.instanceID != ~0u)
    // {
    //     GeometrySample reflectedGS = getGeometryFromHit(
    //         pp.instanceID,
    //         pp.geometryIndex,
    //         pp.triangleIndex,
    //         pp.barycentrics,
    //         GeomAttr_All,
    //         t_InstanceData, t_GeometryData,t_MaterialConstants);
    //
    //     MaterialProps reflectedProps = sampleGeometryMaterial(reflectedGS,MatAttr_All,s_LinearRepeat);
    //     color = reflectedProps.baseColor ; // simple reflection contribution
    // }else
    // {
    //     color = float3(0.05f, 0.05f, 0.1f); // environment color for rays that miss on reflection
    // }
    //
    //
    //  reflectDir = reflect(ray.Direction, props.N);
    // // reflectDir = ray.Direction;
    //
    // ray.Origin = ray.Origin + ray.Direction * payload.committedRayT + gs.geometryNormal * 0.001f; // offset ray origin to avoid self-intersection
    // ray.Direction = normalize(reflectDir.xyz);
    //
    //
    // pp.instanceID = ~0u;
    //
    // reflectedHit = HitObject::TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 1, 0, 1, ray, pp);
    // dx::MaybeReorderThread(reflectedHit.IsHit() ? 1 : 0, 1);
    // HitObject::Invoke(reflectedHit, pp);
    //
    // if(pp.instanceID != ~0u)
    // {
    //     GeometrySample reflectedGS = getGeometryFromHit(
    //         pp.instanceID,
    //         pp.geometryIndex,
    //         pp.triangleIndex,
    //         pp.barycentrics,
    //         GeomAttr_All,
    //         t_InstanceData, t_GeometryData,t_MaterialConstants);
    //
    //     MaterialProps reflectedProps = sampleGeometryMaterial(reflectedGS,MatAttr_All,s_LinearRepeat);
    //     color = reflectedProps.baseColor ; // simple reflection contribution
    // }else
    // {
    //     color = float3(0.05f, 0.05f, 0.1f); // environment color for rays that miss on reflection
    // }
    //
    
    
    
    // color = ray.Direction;
    // color = gs.geometryNormal;
    OutputTexture[idx] = float4(color, 1.0f);
}

[shader("miss")]
void Miss_0_Shader(inout RayPayload payload)
{
    payload.instanceID = ~0u; // indicate miss with invalid instance ID
}

[shader("anyhit")]
void AnyHit_0_Shader(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    //IgnoreHit();
    GeometrySample gs = getGeometryFromHit(
        InstanceID(), GeometryIndex(), PrimitiveIndex(),
        attr.barycentrics, GeomAttr_TexCoord,
        t_InstanceData, t_GeometryData, t_MaterialConstants);
    
    float alphaCutoff = gs.material.alphaCutoff;
    if (alphaCutoff > 0.0f)
    {
        float alpha = sampleGeometryMaterial(gs, MatAttr_BaseColor, s_LinearRepeat).alpha;
        if (alpha < alphaCutoff)
            IgnoreHit();
    }
}

[shader("closesthit")]
void ClosestHit_0_Shader(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    payload.committedRayT = RayTCurrent();
    payload.instanceID = InstanceID();
    payload.geometryIndex = GeometryIndex();
    payload.triangleIndex = PrimitiveIndex();
    // payload.frontFace = HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE;
    payload.barycentrics = attr.barycentrics;
}

[shader("miss")]
void Miss_1_Shader(inout RayPayload2 payload)
{
    payload.instanceID = ~0u;
}

[shader("anyhit")]
void AnyHit_1_Shader(inout RayPayload2 payload, in BuiltInTriangleIntersectionAttributes attr)
{
    //IgnoreHit();
    GeometrySample gs = getGeometryFromHit(
        InstanceID(), GeometryIndex(), PrimitiveIndex(),
        attr.barycentrics, GeomAttr_TexCoord,
        t_InstanceData, t_GeometryData, t_MaterialConstants);
    
    float alphaCutoff = gs.material.alphaCutoff;
    if (alphaCutoff > 0.0f)
    {
        float alpha = sampleGeometryMaterial(gs, MatAttr_BaseColor, s_LinearRepeat).alpha;
        if (alpha < alphaCutoff)
            IgnoreHit();
    }
}

[shader("closesthit")]
void ClosestHit_1_Shader(inout RayPayload2 payload, in BuiltInTriangleIntersectionAttributes attr)
{
    payload.instanceID = InstanceID();
    payload.geometryIndex = GeometryIndex();
    payload.triangleIndex = PrimitiveIndex();
    // payload.frontFace = HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE;
    payload.barycentrics = attr.barycentrics;
}
