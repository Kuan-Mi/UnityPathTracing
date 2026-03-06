using System.Runtime.InteropServices;
using UnityEngine;

namespace PathTracing
{
    // Layout must mirror the HLSL PointLight struct in Include/PointLights.hlsl exactly.
    // Total size: 2 * float3 + 2 * float = 2 * 12 + 2 * 4 = 32 bytes (2 × 16-byte rows).
    [StructLayout(LayoutKind.Sequential)]
    public struct PointLightData
    {
        public Vector3 position;    // World-space position
        public float   range;       // Maximum range (hard cutoff)

        public Vector3 color;       // Pre-multiplied color * intensity
        public float   pad;
    }
}
