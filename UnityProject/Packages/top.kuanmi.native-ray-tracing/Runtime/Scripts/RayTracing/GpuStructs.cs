using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;

namespace NativeRender
{
    /// <summary>
    /// GPU-side instance data. Must match InstanceDataGPU in bindless.hlsl exactly.
    /// Size: 64 bytes.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct InstanceDataGPU
    {
        public uint    firstGeometryIndex;  // +0
        public uint    numGeometries;       // +4
        public uint    pad0;                // +8
        public uint    pad1;                // +12
        public Vector4 transformRow0;       // +16  row-major 3x4 object-to-world
        public Vector4 transformRow1;       // +32
        public Vector4 transformRow2;       // +48
    }   // Total: 64 bytes

    /// <summary>
    /// GPU-side per-geometry data. Must match GeometryData in bindless.hlsl exactly.
    /// Size: 48 bytes.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct GeometryDataGPU
    {
        public uint numIndices;          // +0
        public uint numVertices;         // +4
        public int  indexBufferIndex;    // +8   index into t_BindlessBuffers
        public uint indexOffset;         // +12  byte offset to first index

        public int  vertexBufferIndex;   // +16  index into t_BindlessBuffers
        public uint positionOffset;      // +20  byte offset to first position (float3)
        public uint normalOffset;        // +24  byte offset to first normal,   or ~0u
        public uint texCoord1Offset;     // +28  byte offset to first texcoord, or ~0u

        public uint tangentOffset;       // +32  byte offset to first tangent,  or ~0u
        public uint vertexStride;        // +36
        public uint indexStride;         // +40  2 or 4
        public uint materialIndex;       // +44
    }   // Total: 48 bytes

    /// <summary>
    /// GPU-side per-material data. Must match MaterialConstants in bindless.hlsl exactly.
    /// Size: 80 bytes.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct MaterialConstantsGPU
    {
        // row 0
        public Vector3 baseOrDiffuseColor;      // +0
        public int     flags;                   // +12  MaterialFlags_* bitmask

        // row 1
        public Vector3 emissiveColor;           // +16
        public int     domain;                  // +28  0=Opaque 1=AlphaTest 2=Transparent

        // row 2
        public float opacity;                   // +32
        public float roughness;                 // +36
        public float metalness;                 // +40
        public float normalTextureScale;        // +44

        // row 3
        public float occlusionStrength;         // +48
        public float alphaCutoff;               // +52
        public float transmissionFactor;        // +56
        public int   baseOrDiffuseTextureIndex; // +60  index into t_BindlessTextures, -1 = none

        // row 4
        public int metalRoughOrSpecularTextureIndex; // +64
        public int emissiveTextureIndex;             // +68
        public int normalTextureIndex;               // +72
        public int occlusionTextureIndex;            // +76
    }   // Total: 80 bytes

    // Material flag constants (match bindless.hlsl)
    public static class MaterialFlags
    {
        public const int DoubleSided                    = 0x00000002;
        public const int UseMetalRoughOrSpecularTexture = 0x00000004;
        public const int UseBaseOrDiffuseTexture        = 0x00000008;
        public const int UseEmissiveTexture             = 0x00000010;
        public const int UseNormalTexture               = 0x00000020;
    }

    // =====================================================================
    // NRDSample-compatible layouts (strict match of NRDSample.cpp structs)
    // =====================================================================

    /// <summary>
    /// Per-instance data matching NRDSample.cpp's <c>InstanceData</c> exactly.
    /// Size: 96 bytes.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct InstanceDataNRD
    {
        // For static: mObjectToWorld; for dynamic: mWorldToWorldPrev.
        public Vector4 mOverloadedMatrix0;        // +0   (col 0 of transposed 3x4)
        public Vector4 mOverloadedMatrix1;        // +16
        public Vector4 mOverloadedMatrix2;        // +32

        // float16_t4 baseColorAndMetalnessScale  (+48, 8 B)
        public ushort baseColorR, baseColorG, baseColorB, metalnessScaleH;
        // float16_t4 emissionAndRoughnessScale   (+56, 8 B)
        public ushort emissionR,  emissionG,  emissionB,  roughnessScaleH;
        // float16_t2 normalUvScale               (+64, 4 B)
        public ushort normalUvScaleX, normalUvScaleY;

        public uint  textureOffsetAndFlags;       // +68
        public uint  primitiveOffset;             // +72
        public float scale;                       // +76  sign = winding, magnitude = max scale

        public uint morphPrimitiveOffset;         // +80
        public uint unused1;                      // +84
        public uint unused2;                      // +88
        public uint unused3;                      // +92
    }   // Total: 96 bytes

    /// <summary>
    /// Per-triangle data matching NRDSample.cpp's <c>PrimitiveData</c> exactly.
    /// Size: 48 bytes.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct PrimitiveDataNRD
    {
        // float16_t2 uv0/uv1/uv2 + float worldArea
        public half2 uv0;
        public half2 uv1;
        public half2 uv2;
        public float worldArea;

        // float16_t2 n0/n1/n2 (octahedral-signed encoded) + float uvArea
        public half2 n0;
        public half2 n1;
        public half2 n2;
        public float uvArea;

        // float16_t2 t0/t1/t2 (octahedral-signed encoded) + float bitangentSign
        public half2 t0;
        public half2 t1;
        public half2 t2;
        public float bitangentSign;
    }   // Total: 48 bytes

    /// <summary>
    /// Morph primitive positions stub matching NRDSample.cpp's
    /// <c>MorphPrimitivePositions</c>. Size: 24 bytes.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct MorphPrimitivePositionsNRD
    {
        public ushort p0x, p0y, p0z, p0w;   // float16_t4
        public ushort p1x, p1y, p1z, p1w;
        public ushort p2x, p2y, p2z, p2w;
    }   // Total: 24 bytes
}
