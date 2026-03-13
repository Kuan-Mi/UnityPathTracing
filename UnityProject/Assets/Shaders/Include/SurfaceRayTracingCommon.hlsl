// SurfaceRayTracingCommon.hlsl
// Shared vertex structures and ray-tracing utility functions used by all
// Surface ray-tracing shaders (Lit, Skin, Fabric).
//
// Requirements before including this file:
//   - UnityRaytracingMeshUtils.cginc must already be included.
//   - Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl must already be included.
//
// Optional keywords:
//   _SKINNEDMESH  – extends the Vertex struct with a 'lastPos' field (previous-frame
//                   object-space position stored in TexCoord4) used for motion vectors.

#ifndef SURFACE_RT_COMMON_HLSL
#define SURFACE_RT_COMMON_HLSL

// ---------------------------------------------------------------------------
// Triangle attribute data (intersection barycentrics)
// ---------------------------------------------------------------------------
struct AttributeData
{
    float2 barycentrics;
};

// ---------------------------------------------------------------------------
// Base vertex layout: position, normal, tangent, UV0.
// When _SKINNEDMESH is defined, 'lastPos' (previous-frame OS position) is also
// fetched from TexCoord4 and used for accurate motion-vector computation.
// ---------------------------------------------------------------------------
struct Vertex
{
    float3 position;
    float3 normal;
    float4 tangent;
    float2 uv;
#if _SKINNEDMESH
    float3 lastPos; // previous-frame object-space position (TexCoord4)
#endif
};

Vertex FetchVertex(uint vertexIndex)
{
    Vertex v;
    v.position = UnityRayTracingFetchVertexAttribute3(vertexIndex, kVertexAttributePosition);
    v.normal   = UnityRayTracingFetchVertexAttribute3(vertexIndex, kVertexAttributeNormal);
    v.tangent  = UnityRayTracingFetchVertexAttribute4(vertexIndex, kVertexAttributeTangent);
    v.uv       = UnityRayTracingFetchVertexAttribute2(vertexIndex, kVertexAttributeTexCoord0);
#if _SKINNEDMESH
    v.lastPos  = UnityRayTracingFetchVertexAttribute3(vertexIndex, kVertexAttributeTexCoord4);
#endif
    return v;
}

Vertex InterpolateVertices(Vertex v0, Vertex v1, Vertex v2, float3 barycentrics)
{
    Vertex v;
    #define INTERPOLATE_ATTRIBUTE(attr) v.attr = v0.attr * barycentrics.x + v1.attr * barycentrics.y + v2.attr * barycentrics.z
    INTERPOLATE_ATTRIBUTE(position);
    INTERPOLATE_ATTRIBUTE(normal);
    INTERPOLATE_ATTRIBUTE(tangent);
    INTERPOLATE_ATTRIBUTE(uv);
#if _SKINNEDMESH
    INTERPOLATE_ATTRIBUTE(lastPos);
#endif
    #undef INTERPOLATE_ATTRIBUTE
    return v;
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

float LengthSquared(float3 v)
{
    return dot(v, v);
}

// Curvature estimate derived from the angular spread of three vertex normals.
float ComputeVertexCurvature(float3 n0, float3 n1, float3 n2)
{
    float dnSq0 = LengthSquared(n0 - n1);
    float dnSq1 = LengthSquared(n1 - n2);
    float dnSq2 = LengthSquared(n2 - n0);
    return sqrt(max(dnSq0, max(dnSq1, dnSq2)));
}

// Ray-cone driven mip level for a textured triangle.
// uvScale: extra UV tiling multiplier applied to the edge vectors (pass 1.0 when using raw UV).
#define MAX_MIP_LEVEL 11.0

float ComputeRayMipLevel(
    float2 uv0, float2 uv1, float2 uv2,
    float3 pos0, float3 pos1, float3 pos2,
    float3 normalWS, float3 rayDir,
    float hitT, float coneTan,
    float uvScale)
{
    float2 uvE1 = (uv1 - uv0) * uvScale;
    float2 uvE2 = (uv2 - uv0) * uvScale;
    float uvArea = abs(uvE1.x * uvE2.y - uvE2.x * uvE1.y) * 0.5f;

    float3 edge1 = pos1 - pos0;
    float3 edge2 = pos2 - pos0;
    float worldArea = length(cross(edge1, edge2)) * 0.5f;

    float NoRay = abs(dot(rayDir, normalWS));
    float a = hitT * coneTan;
    a *= Math::PositiveRcp(NoRay);
    a *= sqrt(uvArea / max(worldArea, 1e-10f));

    return max(log2(a) + MAX_MIP_LEVEL, 0.0);
}

// Blend two tangent-space normals using reoriented normal mapping (RNM).
// Equivalent to Unity's NormalBlend graph node.
float3 BlendNormals(float3 n1, float3 n2)
{
    return SafeNormalize(float3(n1.rg + n2.rg, n1.b * n2.b));
}

#endif // SURFACE_RT_COMMON_HLSL
