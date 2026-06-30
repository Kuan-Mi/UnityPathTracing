using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;

namespace PathTracing.Rendering.Data.RTXDI
{
    // Shared GPU scene/light data structs consumed across the path-tracing pipelines
    // (Core, RTXPT, NRD-Sample) as well as RTXDI. Kept in the Core assembly so every
    // feature assembly can reference them; the RTXDI-specific LightScene logic lives in
    // the RTXDI assembly.
    [StructLayout(LayoutKind.Sequential)]
    public struct PrimitiveData
    {
        public float2 uv0;
        public float2 uv1;
        public float2 uv2;

        public float3 pos0;
        public float3 pos1;
        public float3 pos2;

        public uint instanceId;
    }

    // 对应不是每个 MeshRenderer，而是每个发光 SubMesh 实例（一个 Renderer 多个发光 SubMesh 会产生多条记录）
    // 是transform + 材质信息
    [StructLayout(LayoutKind.Sequential)]
    public struct InstanceData
    {
        public float4x4 transform;

        public float3 emissiveColor;
        public uint   emissiveTextureIndex;
    }


    public enum PolymorphicLightType
    {
        kSphere = 0,
        kCylinder,
        kDisk,
        kRect,
        kTriangle,
        kDirectional,
        kEnvironment,
        kPoint
    };


    [StructLayout(LayoutKind.Sequential)]
    public struct PolymorphicLightInfo
    {
        // uint4[0]
        public float3 center;
        public uint   colorTypeAndFlags; // RGB8 + uint8 (see the kPolymorphicLight... constants above)

        // uint4[1]
        public uint direction1; // oct-encoded
        public uint direction2; // oct-encoded
        public uint scalars; // 2x float16
        public uint logRadiance; // uint16

        // uint4[2] -- optional, contains only shaping data
        public uint iesProfileIndex;
        public uint primaryAxis; // oct-encoded
        public uint cosConeAngleAndSoftness; // 2x float16
        public uint padding;

        public void SetColorAndType(Color color, PolymorphicLightType typeCode)
        {
            const float kMinLog2Radiance = -8f;
            const float kMaxLog2Radiance = 40f;
            const uint  kTypeShift       = 24u;

            float intensity = math.max(color.r, math.max(color.g, color.b));
            if (intensity > 0f)
            {
                // log-encode radiance (mirrors packLightColor in HLSL)
                float normalizedLog = math.saturate(
                    (math.log2(intensity) - kMinLog2Radiance) / (kMaxLog2Radiance - kMinLog2Radiance));
                uint packedRadiance = (uint)math.min((uint)math.ceil(normalizedLog * 65534f) + 1u, 0xFFFFu);
                logRadiance = packedRadiance; // stored in low 16 bits

                // decode back to find actual encoded intensity, then normalize color to [0,1]
                float unpackedIntensity = math.exp2(
                    ((packedRadiance - 1f) / 65534f) * (kMaxLog2Radiance - kMinLog2Radiance) + kMinLog2Radiance);
                float3 normalizedColor = math.saturate(new float3(color.r, color.g, color.b) / unpackedIntensity);

                // Pack_R8G8B8_UFLOAT: R in bits[0:7], G in bits[8:15], B in bits[16:23]
                uint r = (uint)math.round(normalizedColor.x * 255f) & 0xFFu;
                uint g = (uint)math.round(normalizedColor.y * 255f) & 0xFFu;
                uint b = (uint)math.round(normalizedColor.z * 255f) & 0xFFu;
                colorTypeAndFlags = ((uint)typeCode) << (int)kTypeShift | r | (g << 8) | (b << 16);
            }
            else
            {
                logRadiance       = 0;
                colorTypeAndFlags = ((uint)typeCode) << (int)kTypeShift;
            }
        }
    };
}