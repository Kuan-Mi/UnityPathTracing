using System.Runtime.InteropServices;
using UnityEngine;

namespace PathTracing
{
    // =========================================================================
    // RTXPT-specific GPU structs
    // Must exactly mirror the HLSL/C++ definitions in:
    //   SubInstanceData.h  (SUBINSTANCEDATA_EXTENDED = 1)
    //   MaterialPT.h       (PTMaterialData)
    //   OmmGeometryDebugData.hlsli (GeometryDebugData)
    // =========================================================================

    /// <summary>
    /// Per-geometry-instance data that avoids an indirection through InstanceData+GeometryData
    /// for hot code paths (alpha testing, NEE exclusion).
    ///
    /// Mirrors <c>SubInstanceData</c> in <c>SubInstanceData.h</c> with
    /// <c>SUBINSTANCEDATA_EXTENDED = 1</c>. Size = 8 × 4 = 32 bytes.
    ///
    /// Shader binding: <c>StructuredBuffer&lt;SubInstanceData&gt; t_SubInstanceData : register(t1)</c>
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct SubInstanceData
    {
        // Flags layout:
        //   bits 0-15  : AlphaTextureIndex (bindless texture index for alpha test)
        //   bit  16    : Flags_AlphaTested
        //   bit  17    : Flags_ExcludeFromNEE
        //   bit  18    : Flags_Dummy0  (reserved)
        //   bits 24-31 : quantized AlphaCutoff (uint8, divide by 255.0)
        public uint FlagsAndAlphaInfo;

        // bits 31-16 : GlobalGeometryIndex  (index into t_GeometryData / t_GeometryDebugData)
        // bits 15-0  : PTMaterialDataIndex  (index into t_PTMaterialData)
        public uint GlobalGeometryIndex_PTMaterialDataIndex;

        // Index of the first emissive light for this geometry in the LightDataBuffer.
        // 0xFFFFFFFF means "not emissive".
        public uint EmissiveLightMappingOffset;

        // Index of the analytic proxy light, if this geometry stands in for one.
        public uint AnalyticProxyLightIndex;

        // ---- SUBINSTANCEDATA_EXTENDED fields --------------------------------
        // bits 31-16 : IndexBufferIndex   (into t_BindlessBuffers)
        // bits 15-0  : VertexBufferIndex  (into t_BindlessBuffers)
        public uint IndexBufferIndex_VertexBufferIndex;

        // Byte offset of the first index inside the index ByteAddressBuffer.
        public uint IndexOffset;

        // Byte offset of the first texcoord inside the vertex ByteAddressBuffer.
        public uint TexCoord1Offset;

        public uint padding0;
    } // 32 bytes

    // =========================================================================

    /// <summary>
    /// Per-material GPU data for the RTXPT path tracer.
    ///
    /// Mirrors <c>PTMaterialData</c> in <c>MaterialPT.h</c>. Size = 24 × 4 = 96 bytes.
    ///
    /// Shader binding: <c>StructuredBuffer&lt;PTMaterialData&gt; t_PTMaterialData : register(t5)</c>
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct PTMaterialData
    {
        // Row 0
        public Vector3 BaseOrDiffuseColor; // float3
        public uint    Flags;              // PTMaterialFlags_* bitmask

        // Row 1
        public Vector3 SpecularColor;
        public int     _padding0;

        // Row 2
        public Vector3 EmissiveColor;
        public float   ShadowNoLFadeout;

        // Row 3
        public float Opacity;
        public float Roughness;
        public float Metalness;
        public float NormalTextureScale;

        // Row 4
        public float _padding1;
        public float AlphaCutoff;
        public float TransmissionFactor;
        public uint  BaseOrDiffuseTextureIndex;  // bindless index, ~0u = none

        // Row 5
        public uint MetalRoughOrSpecularTextureIndex;
        public uint EmissiveTextureIndex;
        public uint NormalTextureIndex;
        public uint OcclusionTextureIndex;

        // Row 6
        public uint  TransmissionTextureIndex;
        public float IoR;
        public float ThicknessFactor;
        public float DiffuseTransmissionFactor;

        // Row 7  (VolumePTConstants)
        public Vector3 VolumeAttenuationColor;
        public float   VolumeAttenuationDistance;
    } // 8 rows × 16 bytes = 128 bytes

    // =========================================================================

    /// <summary>
    /// Per-geometry OMM (Opacity Micro-Map) debug data.
    ///
    /// Mirrors <c>GeometryDebugData</c> in <c>OmmGeometryDebugData.hlsli</c>. Size = 8 × 4 = 32 bytes.
    ///
    /// Shader binding: <c>StructuredBuffer&lt;GeometryDebugData&gt; t_GeometryDebugData : register(t4)</c>
    ///
    /// When OMM is not used all fields should be set to 0 / ~0u as appropriate.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct GeometryDebugData
    {
        public uint OmmArrayDataBufferIndex;    // bindless index into t_BindlessBuffers
        public uint OmmArrayDataBufferOffset;   // byte offset
        public uint OmmDescArrayBufferIndex;    // bindless index
        public uint OmmDescArrayBufferOffset;   // byte offset

        public uint OmmIndexBufferIndex;        // bindless index
        public uint OmmIndexBufferOffset;       // byte offset
        public uint OmmIndexBuffer16Bit;        // 1 = 16-bit indices, 0 = 32-bit
        public uint _pad0;
    } // 32 bytes

    // =========================================================================
    // PTMaterialFlags constants  (match PTMaterialFlags_* in MaterialPT.h)
    // =========================================================================
    public static class PTMaterialFlags
    {
        public const uint UseSpecularGlossModel          = 0x00000001u;
        public const uint UseMetalRoughOrSpecularTexture = 0x00000004u;
        public const uint UseBaseOrDiffuseTexture        = 0x00000008u;
        public const uint UseEmissiveTexture             = 0x00000010u;
        public const uint UseNormalTexture               = 0x00000020u;
        public const uint UseTransmissionTexture         = 0x00000080u;
        public const uint MetalnessInRedChannel          = 0x00000100u;
        public const uint ThinSurface                    = 0x00000200u;
        public const uint PSDExclude                     = 0x00000400u;
        public const uint EnableAsAnalyticLightProxy     = 0x00000800u;
        public const uint IgnoreMeshTangentSpace         = 1u << 12;
        public const uint PSDBlockMVsAtSurfaceTypeB0     = 1u << 13;
        public const uint PSDBlockMVsAtSurfaceTypeB1     = 1u << 14;
        public const uint NestedPriorityMask             = 0xF0000000u;
        public const int  NestedPriorityShift            = 28;
        public const uint PSDDominantDeltaLobeP1Mask     = 0x0F000000u;
        public const int  PSDDominantDeltaLobeP1Shift    = 24;
    }

    // =========================================================================
    // SubInstanceData flag helpers  (match SubInstanceData::Flags_* in SubInstanceData.h)
    // =========================================================================
    public static class SubInstanceFlags
    {
        public const uint AlphaTested       = 1u << 16;
        public const uint ExcludeFromNEE    = 1u << 17;
        public const uint AlphaOffsetMask   = 0xFF000000u;
        public const int  AlphaOffsetOffset = 24;
    }
}
