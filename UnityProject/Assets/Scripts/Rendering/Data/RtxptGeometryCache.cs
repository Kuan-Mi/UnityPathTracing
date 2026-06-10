using System;
using System.Collections.Generic;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

namespace PathTracing
{
    /// <summary>
    /// Byte offsets of each SoA stream inside a donut vertex buffer, mirroring the layout
    /// produced by <see cref="RtxptGeometryCache.GetOrCreate"/>:
    ///   [positions: vc*12][normals?: vc*4][uvs?: vc*8][tangents?: vc*4]
    /// Absent streams report 0xFFFFFFFF.
    /// </summary>
    internal readonly struct RtxptMeshStreamOffsets
    {
        public const uint Absent = 0xFFFFFFFFu;

        public readonly uint Pos;
        public readonly uint Normal;
        public readonly uint Uv;
        public readonly uint Tangent;

        public bool HasNormal  => Normal  != Absent;
        public bool HasUv      => Uv      != Absent;
        public bool HasTangent => Tangent != Absent;

        public RtxptMeshStreamOffsets(Mesh mesh)
        {
            uint vc     = (uint)mesh.vertexCount;
            uint offset = 0u;

            Pos     = offset;
            offset += vc * 12u;

            if (mesh.HasVertexAttribute(VertexAttribute.Normal)) { Normal = offset; offset += vc * 4u; }
            else                                                   Normal = Absent;

            if (mesh.HasVertexAttribute(VertexAttribute.TexCoord0)) { Uv = offset; offset += vc * 8u; }
            else                                                      Uv = Absent;

            Tangent = mesh.HasVertexAttribute(VertexAttribute.Tangent) ? offset : Absent;
        }
    }

    /// <summary>
    /// Per-mesh donut-compatible SoA vertex buffer + uint32 index buffer, persistent for the
    /// scene's lifetime (re-packing every mesh's vertex data on each topology change was the
    /// dominant rebuild cost). Packing reads through <see cref="Mesh.AcquireReadOnlyMeshData"/>
    /// into NativeArrays — no managed array copies, no per-element allocations.
    ///
    /// VB layout: [Position: float3 × vc][Normal: RGB8_SNORM × vc][TexCoord: float2 × vc][Tangent: RGBA8_SNORM × vc]
    /// IB layout: uint32 per index, same slot layout as Unity submesh indexStart.
    /// Both are <c>GraphicsBuffer.Target.Raw</c> (ByteAddressBuffer).
    /// </summary>
    internal sealed class RtxptGeometryCache : IDisposable
    {
        private readonly Dictionary<int, (GraphicsBuffer vb, GraphicsBuffer ib)> _cache = new();
        private readonly List<int> _evictScratch = new();

        public (GraphicsBuffer vb, GraphicsBuffer ib) GetOrCreate(Mesh src)
        {
            int key = src.GetInstanceID();
            if (_cache.TryGetValue(key, out var cached)) return cached;

            var streams = new RtxptMeshStreamOffsets(src);
            int vc      = src.vertexCount;

            int vbBytes = vc * 12;
            if (streams.HasNormal)  vbBytes += vc * 4;
            if (streams.HasUv)      vbBytes += vc * 8;
            if (streams.HasTangent) vbBytes += vc * 4;

            using var meshDataArray = Mesh.AcquireReadOnlyMeshData(src);
            var meshData = meshDataArray[0];

            var vbData = new NativeArray<uint>(vbBytes / 4, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            int w = 0; // write cursor, in uints

            // Position stream (float3, no compression)
            {
                var positions = new NativeArray<Vector3>(vc, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                meshData.GetVertices(positions);
                for (int i = 0; i < vc; i++)
                {
                    Vector3 p = positions[i];
                    vbData[w++] = (uint)BitConverter.SingleToInt32Bits(p.x);
                    vbData[w++] = (uint)BitConverter.SingleToInt32Bits(p.y);
                    vbData[w++] = (uint)BitConverter.SingleToInt32Bits(p.z);
                }
                positions.Dispose();
            }

            // Normal stream (RGB8_SNORM, 4 bytes each)
            if (streams.HasNormal)
            {
                var normals = new NativeArray<Vector3>(vc, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                meshData.GetNormals(normals);
                for (int i = 0; i < vc; i++)
                    vbData[w++] = PackRGB8Snorm(normals[i]);
                normals.Dispose();
            }

            // TexCoord stream (float2, 8 bytes each)
            if (streams.HasUv)
            {
                var uvs = new NativeArray<Vector2>(vc, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                meshData.GetUVs(0, uvs);
                for (int i = 0; i < vc; i++)
                {
                    Vector2 uv = uvs[i];
                    vbData[w++] = (uint)BitConverter.SingleToInt32Bits(uv.x);
                    vbData[w++] = (uint)BitConverter.SingleToInt32Bits(uv.y);
                }
                uvs.Dispose();
            }

            // Tangent stream (RGBA8_SNORM, 4 bytes each). The w (handedness) sign is flipped to
            // match the original RTXPT glTF import convention.
            if (streams.HasTangent)
            {
                var tangents = new NativeArray<Vector4>(vc, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                meshData.GetTangents(tangents);
                for (int i = 0; i < vc; i++)
                {
                    Vector4 t = tangents[i];
                    vbData[w++] = PackRGBA8Snorm(new Vector4(t.x, t.y, t.z, -t.w));
                }
                tangents.Dispose();
            }

            var vbGfx = new GraphicsBuffer(GraphicsBuffer.Target.Raw, vbBytes / 4, 4) { name = "VertexBuffer" };
            vbGfx.SetData(vbData);
            vbData.Dispose();

            // ---- IB (uint32, matching Unity submesh indexStart layout) ----
            int totalIndexSlots = 0;
            for (int s = 0; s < src.subMeshCount; s++)
            {
                var sub = src.GetSubMesh(s);
                totalIndexSlots = Mathf.Max(totalIndexSlots, sub.indexStart + sub.indexCount);
            }

            var ibData = new NativeArray<uint>(Mathf.Max(totalIndexSlots, 3), Allocator.Persistent, NativeArrayOptions.ClearMemory);
            for (int s = 0; s < src.subMeshCount; s++)
            {
                var sub    = src.GetSubMesh(s);
                var subIdx = new NativeArray<int>(sub.indexCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                meshData.GetIndices(subIdx, s); // applyBaseVertex defaults to true
                for (int k = 0; k < subIdx.Length; k++)
                    ibData[sub.indexStart + k] = (uint)subIdx[k];
                subIdx.Dispose();
            }

            var ibGfx = new GraphicsBuffer(GraphicsBuffer.Target.Raw, ibData.Length, 4) { name = "IndexBuffer" };
            ibGfx.SetData(ibData);
            ibData.Dispose();

            var result = (vbGfx, ibGfx);
            _cache[key] = result;
            return result;
        }

        /// <summary>Releases cached buffers for meshes that are no longer in the scene.</summary>
        public void EvictUnused(Func<int, bool> isMeshUsed)
        {
            _evictScratch.Clear();
            foreach (var key in _cache.Keys)
                if (!isMeshUsed(key))
                    _evictScratch.Add(key);
            foreach (var key in _evictScratch)
            {
                var (vb, ib) = _cache[key];
                vb?.Release();
                ib?.Release();
                _cache.Remove(key);
            }
        }

        public void Dispose()
        {
            foreach (var (vb, ib) in _cache.Values)
            {
                vb?.Release();
                ib?.Release();
            }
            _cache.Clear();
        }

        // Matches donut's dm::vectorToSnorm8 (vector.cpp) bit-for-bit: scale = 127/length,
        // then TRUNCATE toward zero via (int) cast (donut uses int(v*scale), not rounding),
        // and keep the low byte with &0xFF. This is what the original RTXPT importer
        // (GltfImporter.cpp -> vectorToSnorm8) does, so normals/tangents quantize identically.
        private static uint PackRGB8Snorm(Vector3 v)
        {
            float scale = 127.0f / Mathf.Sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
            int r = (int)(v.x * scale) & 0xFF;
            int g = (int)(v.y * scale) & 0xFF;
            int b = (int)(v.z * scale) & 0xFF;
            return (uint)(r | (g << 8) | (b << 16));
        }

        private static uint PackRGBA8Snorm(Vector4 v)
        {
            // donut scales all four channels by 127/length(xyz) (w shares the xyz-based scale).
            float scale = 127.0f / Mathf.Sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
            int r = (int)(v.x * scale) & 0xFF;
            int g = (int)(v.y * scale) & 0xFF;
            int b = (int)(v.z * scale) & 0xFF;
            int a = (int)(v.w * scale) & 0xFF;
            return (uint)(r | (g << 8) | (b << 16) | (a << 24));
        }
    }
}
