using System.Runtime.InteropServices;
using UnityEngine;

namespace PathTracing
{
    // Layout must mirror the HLSL AreaLight struct in Include/AreaLights.hlsl exactly.
    // Total size: 4 * float3 + 4 * float = 4 * 12 + 4 * 4 = 64 bytes (4 × 16-byte rows).
    [StructLayout(LayoutKind.Sequential)]
    public struct AreaLightData
    {
        public Vector3 position;    // World-space centre of the rectangle
        public float   halfWidth;   // Half-extent along the right axis

        public Vector3 right;       // Unit right vector (world-space)
        public float   halfHeight;  // Half-extent along the up axis

        public Vector3 up;          // Unit up vector (world-space)
        public float   pad;

        public Vector3 color;       // Pre-multiplied color * intensity
        public float   pad2;
    }
}
