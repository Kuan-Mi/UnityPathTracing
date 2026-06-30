using System.Runtime.InteropServices;
using UnityEngine;

namespace PathTracing
{
    // Donut-style GPU scene layout structs shared by the RTXDI and RTXPT pipelines.
    // Kept in the Core assembly so both feature assemblies can reference them.

    /// <summary>
    /// Mirrors donut <c>InstanceData</c> from <c>bindless.h</c>.
    /// Size: 7 × 16 = 112 bytes  (<c>c_SizeOfInstanceData</c>).
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct DonutInstanceData
    {
        public uint flags; // +0
        public uint firstGeometryInstanceIndex; // +4   = firstGeometryIndex in our flat list
        public uint firstGeometryIndex; // +8

        public uint numGeometries; // +12

        // float3x4 transform — row 0..2 of object-to-world
        public Vector4 transformRow0; // +16
        public Vector4 transformRow1; // +32

        public Vector4 transformRow2; // +48

        // float3x4 prevTransform — same layout, previous frame
        public Vector4 prevTransformRow0; // +64
        public Vector4 prevTransformRow1; // +80
        public Vector4 prevTransformRow2; // +96
    } // Total: 112 bytes

    /// <summary>
    /// Mirrors donut <c>GeometryData</c> from <c>bindless.h</c>.
    /// Size: 4 × 16 = 64 bytes  (<c>c_SizeOfGeometryData</c>).
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct DonutGeometryData
    {
        public uint numIndices; // +0
        public uint numVertices; // +4
        public int  indexBufferIndex; // +8
        public uint indexOffset; // +12  byte offset

        public int  vertexBufferIndex; // +16
        public uint positionOffset; // +20  byte offset (float3)
        public uint prevPositionOffset; // +24  byte offset; = positionOffset for static meshes
        public uint texCoord1Offset; // +28  byte offset (float2), or ~0u

        public uint texCoord2Offset; // +32  byte offset, ~0u (not used)
        public uint normalOffset; // +36  byte offset (oct-encoded or float3), or ~0u
        public uint tangentOffset; // +40  byte offset, or ~0u
        public uint curveRadiusOffset; // +44  ~0u (curves not supported)

        public uint materialIndex; // +48
        public uint pad0; // +52
        public uint pad1; // +56
        public uint pad2; // +60
    } // Total: 64 bytes

    /// <summary>
    /// One entry per emissive sub-mesh in the scene.
    /// Mirrors the per-geometry task record that PrepareLights.computeshader needs.
    /// </summary>
    public struct EmissiveGeometryEntry
    {
        /// <summary>Flat index into the t_InstanceData GPU buffer.</summary>
        public int InstanceIndex;

        /// <summary>Sub-geometry index within the instance (0 … numGeometries-1).</summary>
        public int GeometrySubIndex;

        /// <summary>Number of triangles (numIndices / 3) for this sub-mesh.</summary>
        public uint TriangleCount;

        /// <summary>instance.firstGeometryInstanceIndex — used to fill GeometryInstanceToLight.</summary>
        public uint FirstGeometryInstanceIndex;
    }
}