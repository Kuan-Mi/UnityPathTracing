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

    // =========================================================================
    // RTXPT Lighting structs
    // Mirror PolymorphicLight.h / LightingTypes.hlsli
    // =========================================================================

    /// <summary>
    /// RTXPT polymorphic light type codes.
    /// Mirrors <c>PolymorphicLightType</c> in <c>PolymorphicLight.h</c>.
    /// </summary>
    public enum RtxptLightType : uint
    {
        Sphere         = 0,
        Triangle       = 1,
        Directional    = 2,
        Environment    = 3,
        Point          = 4,
        EnvironmentQuad = 5,
    }

    /// <summary>
    /// Per-light GPU data. Mirrors <c>PolymorphicLightInfo</c> in <c>PolymorphicLight.h</c>.
    /// Size = 2 × uint4 = 32 bytes.
    /// Shader binding: <c>StructuredBuffer&lt;PolymorphicLightInfo&gt; t_Lights : register(t13)</c>
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct RtxptPolymorphicLightInfo
    {
        // uint4[0]: position + packed color/type
        public float CenterX;
        public float CenterY;
        public float CenterZ;
        public uint  ColorTypeAndFlags;  // R8G8B8 chroma | type<<24 | flags<<28

        // uint4[1]: direction / scalar encodings + log-radiance
        public uint Direction1;    // oct-encoded normal (for triangles/directionals)
        public uint Direction2;    // oct-encoded or fp16 cone angle pair (for spots)
        public uint Scalars;       // fp16 pair: e.g. sphere radius
        public uint LogRadiance;   // uint16 log2-encoded radiance magnitude
    } // 32 bytes

    /// <summary>
    /// Extended per-light data (shaping / IES). Mirrors <c>PolymorphicLightInfoEx</c>.
    /// Size = 1 × uint4 = 16 bytes.
    /// Shader binding: <c>StructuredBuffer&lt;PolymorphicLightInfoEx&gt; t_LightsEx : register(t14)</c>
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct RtxptPolymorphicLightInfoEx
    {
        public uint IesProfileIndex;
        public uint PrimaryAxis;               // oct-encoded spot/IES primary axis
        public uint CosConeAngleAndSoftness;   // fp16 pair: cos(outerAngle) | softness
        public uint UniqueID;                  // debug-only hash
    } // 16 bytes

    /// <summary>
    /// Single-element control buffer read by the path tracer each frame.
    /// Mirrors <c>LightingControlData</c> in <c>LightingTypes.hlsli</c>.
    /// Size = 112 + 464 (paddingBK) = 576 bytes.
    /// Shader binding: <c>StructuredBuffer&lt;LightingControlData&gt; t_LightsCB : register(t12)</c>
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public unsafe struct RtxptLightingControlData
    {
        public uint  TotalLightCount;
        public uint  EnvmapQuadNodeCount;
        public uint  AnalyticLightCount;
        public uint  TriangleLightCount;

        public uint  SamplingProxyCount;
        public uint  HistoricTotalLightCount;
        public uint  LastFrameTemporalFeedbackAvailable;
        public uint  LastFrameLocalSamplesAvailable;

        public uint  ProxyBuildTaskCount;
        public uint  WeightsSumUINT;
        public uint  ImportanceSamplingType;   // 0=Uniform, 1=Power, 2=NEE-AT
        public uint  _padding0;

        public uint  TemporalFeedbackRequired;
        public uint  TotalMaxFeedbackCount;
        public float GlobalFeedbackUseWeight;
        public float LocalToGlobalSampleRatio;

        public uint  TileBufferHeight;
        public float ScreenSpaceVsWorldSpaceThreshold;
        public uint  LocalSamplingResolutionX;
        public uint  LocalSamplingResolutionY;

        public uint  LocalSamplingTileJitterX;
        public uint  LocalSamplingTileJitterY;
        public uint  LocalSamplingTileJitterPrevX;
        public uint  LocalSamplingTileJitterPrevY;

        public uint  ValidFeedbackCount;
        public uint  _padding1;
        public uint  _padding2;
        public uint  _padding3;

        // Padding for LightsBakerConstants (unused in non-baker path, 464 bytes = 116 uints)
        public fixed uint _paddingBK[116];
    } // 576 bytes
}
