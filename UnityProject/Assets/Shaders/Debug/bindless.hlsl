#ifndef BINDLESS_H_
#define BINDLESS_H_


// ---- Material flags (matches MaterialConstantsGPU in C++) ----

static const int MaterialFlags_UseBaseOrDiffuseTexture          = 0x00000008;
static const int MaterialFlags_UseNormalTexture                 = 0x00000020;
static const int MaterialFlags_UseMetalRoughOrSpecularTexture   = 0x00000004;
static const int MaterialFlags_UseEmissiveTexture               = 0x00000010;
static const int MaterialFlags_DoubleSided                      = 0x00000002;

// ---- MaterialConstants (per-material, indexed via GeometryData.materialIndex) ----
// Must match MaterialConstantsGPU in AccelerationStructure.h exactly (16-byte rows).

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


// ---- Data structures ----

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

#endif // BINDLESS_H_