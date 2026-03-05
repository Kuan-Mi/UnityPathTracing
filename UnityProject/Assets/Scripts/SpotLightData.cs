using System.Runtime.InteropServices;
using UnityEngine;

namespace PathTracing
{
    // Layout must mirror the HLSL SpotLight struct in Include/SpotLights.hlsl exactly.
    // Total size: 4 * float3 + 4 * float = 4 * 12 + 4 * 4 = 64 bytes (4 × 16-byte rows).
    [StructLayout(LayoutKind.Sequential)]
    public struct SpotLightData
    {
        public Vector3 position;       // World-space position
        public float   range;          // Maximum range

        public Vector3 direction;      // Normalized forward direction (light → scene)
        public float   cosOuterAngle;  // cos(outerHalfAngle)

        public Vector3 color;          // Pre-multiplied color * intensity
        public float   cosInnerAngle;  // cos(innerHalfAngle)
    }
}
