using System.Runtime.InteropServices;
using UnityEngine;

namespace PathTracing
{
    // Layout must mirror the HLSL AreaLight struct in Include/AreaLights.hlsl exactly.
    // Total size: 4 * float3 + 4 * float = 4 * 12 + 4 * 4 = 64 bytes (4 × 16-byte rows).
    // lightType: 0 = rectangle, 1 = disc
    [StructLayout(LayoutKind.Sequential)]
    public struct AreaLightData
    {
        public Vector3 position;    // World-space centre
        public float   halfWidth;   // Rect: half-extent along right. Disc: radius.

        public Vector3 right;       // Unit right vector (world-space)
        public float   halfHeight;  // Rect: half-extent along up. Disc: unused (0).

        public Vector3 up;          // Unit up vector (world-space)
        public float   lightType;   // 0 = rectangle, 1 = disc

        public Vector3 color;       // Pre-multiplied color * intensity
        public float   pad2;
    }
}
