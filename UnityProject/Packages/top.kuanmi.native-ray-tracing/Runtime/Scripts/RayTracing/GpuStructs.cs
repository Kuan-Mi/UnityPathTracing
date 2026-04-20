using System.Runtime.InteropServices;
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

    /// <summary>
    /// Per-triangle data, populated once for static scenes.
    /// Layout uses float4 for positions to avoid HLSL float3 alignment ambiguity
    /// inside StructuredBuffer&lt;PrimitiveData&gt;.
    /// Size: 96 bytes.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct PrimitiveDataGPU
    {
        public Vector2 uv0;          // +0
        public Vector2 uv1;          // +8
        public Vector2 uv2;          // +16
        public Vector2 _pad0;        // +24

        public Vector4 pos0;         // +32   .xyz = position
        public Vector4 pos1;         // +48
        public Vector4 pos2;         // +64

        public uint    instanceId;   // +80
        public uint    _pad1;        // +84
        public uint    _pad2;        // +88
        public uint    _pad3;        // +92
    }   // Total: 96 bytes
}
