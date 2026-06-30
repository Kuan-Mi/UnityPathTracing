using System;
using System.Collections.Generic;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

namespace PathTracing
{
    /// <summary>
    /// Byte offsets of each SoA stream inside a donut vertex buffer, mirroring the layout
    /// produced by <see cref="RtxptGeometryCache"/>:
    ///   [positions: vc*12][prevPositions?: vc*12][normals?: vc*4][uvs?: vc*8][tangents?: vc*4]
    /// The PrevPosition stream exists only for skinned buffers; static buffers report
    /// <see cref="PrevPos"/> == <see cref="Pos"/> (no per-vertex motion).
    /// Absent optional streams report 0xFFFFFFFF.
    /// </summary>
    internal readonly struct RtxptMeshStreamOffsets
    {
        public const uint Absent = 0xFFFFFFFFu;

        public readonly uint Pos;
        public readonly uint PrevPos;
        public readonly uint Normal;
        public readonly uint Uv;
        public readonly uint Tangent;
        public readonly uint TotalBytes;

        public bool HasPrevPos => PrevPos != Pos;
        public bool HasNormal => Normal != Absent;
        public bool HasUv => Uv != Absent;
        public bool HasTangent => Tangent != Absent;

        public RtxptMeshStreamOffsets(Mesh mesh, bool withPrevPosition = false)
        {
            uint vc     = (uint)mesh.vertexCount;
            uint offset = 0u;

            Pos    =  offset;
            offset += vc * 12u;

            if (withPrevPosition)
            {
                PrevPos =  offset;
                offset  += vc * 12u;
            }
            else PrevPos = Pos;

            if (mesh.HasVertexAttribute(VertexAttribute.Normal))
            {
                Normal =  offset;
                offset += vc * 4u;
            }
            else Normal = Absent;

            if (mesh.HasVertexAttribute(VertexAttribute.TexCoord0))
            {
                Uv     =  offset;
                offset += vc * 8u;
            }
            else Uv = Absent;

            if (mesh.HasVertexAttribute(VertexAttribute.Tangent))
            {
                Tangent =  offset;
                offset  += vc * 4u;
            }
            else Tangent = Absent;

            TotalBytes = offset;
        }
    }

    /// <summary>
    /// A skinned instance's GPU geometry: a per-instance donut SoA vertex buffer (with the
    /// PrevPosition stream), initialised to the rest pose and rewritten every frame by the
    /// skinned-repack compute, plus the shared per-mesh uint32 donut index buffer.
    /// </summary>
    internal sealed class RtxptSkinnedGeometry
    {
        public GraphicsBuffer Vb; // owned by the cache
        public GraphicsBuffer Ib; // shared per-mesh donut IB (cache-owned)
        public int            VertexCount;
        public int            MeshId; // rest-pose mesh, for staleness detection

        /// <summary>True until the first repack dispatch (prev = current on that dispatch).</summary>
        public bool PendingFirstFrame = true;
    }

    /// <summary>
    /// Per-mesh donut-compatible SoA vertex buffers + uint32 index buffers, persistent for the
    /// scene's lifetime (re-packing every mesh's vertex data on each topology change was the
    /// dominant rebuild cost). Skinned instances get their own per-renderer SoA buffer with a
    /// PrevPosition stream (donut SkinningPass model). Packing reads through
    /// <see cref="Mesh.AcquireReadOnlyMeshData"/> into NativeArrays — no managed array copies,
    /// no per-element allocations.
    ///
    /// VB layout: [Position: float3][PrevPosition: float3 (skinned only)][Normal: RGB8_SNORM]
    ///            [TexCoord: float2][Tangent: RGBA8_SNORM], each stream × vertexCount.
    /// IB layout: uint32 per index, same slot layout as Unity submesh indexStart.
    /// All buffers are <c>GraphicsBuffer.Target.Raw</c> (ByteAddressBuffer / RWByteAddressBuffer).
    /// </summary>
    internal sealed class RtxptGeometryCache : IDisposable
    {
        private readonly Dictionary<int, GraphicsBuffer>       _staticVbCache = new(); // mesh id
        private readonly Dictionary<int, GraphicsBuffer>       _ibCache       = new(); // mesh id
        private readonly Dictionary<int, RtxptSkinnedGeometry> _skinnedCache  = new(); // renderer id
        private readonly List<int>                             _evictScratch  = new();

        /// <summary>Static path: shared per-mesh SoA VB + donut IB.</summary>
        public (GraphicsBuffer vb, GraphicsBuffer ib) GetOrCreate(Mesh src)
        {
            int key = src.GetInstanceID();

            if (!_staticVbCache.TryGetValue(key, out var vb))
            {
                vb                  = BuildSoAVertexBuffer(src, new RtxptMeshStreamOffsets(src));
                _staticVbCache[key] = vb;
            }

            return (vb, GetOrCreateIndexBuffer(src, key));
        }

        /// <summary>
        /// Skinned path: per-renderer SoA VB (rest-pose initialised, PrevPosition stream) plus
        /// the shared per-mesh donut IB. Recreated automatically if the rest-pose mesh changed.
        /// </summary>
        public RtxptSkinnedGeometry GetOrCreateSkinned(int rendererId, Mesh restPose)
        {
            int meshId = restPose.GetInstanceID();

            if (_skinnedCache.TryGetValue(rendererId, out var entry))
            {
                if (entry.MeshId == meshId)
                    return entry;
                entry.Vb?.Release(); // mesh changed — rebuild (Ib is cache-owned, not released here)
                _skinnedCache.Remove(rendererId);
            }

            entry = new RtxptSkinnedGeometry
            {
                Vb          = BuildSoAVertexBuffer(restPose, new RtxptMeshStreamOffsets(restPose, withPrevPosition: true)),
                Ib          = GetOrCreateIndexBuffer(restPose, meshId),
                VertexCount = restPose.vertexCount,
                MeshId      = meshId,
            };
            _skinnedCache[rendererId] = entry;
            return entry;
        }

        /// <summary>
        /// Releases cached buffers no longer referenced by the scene: static VBs and IBs for
        /// meshes failing <paramref name="isMeshUsed"/>, skinned VBs for renderers failing
        /// <paramref name="isSkinnedRendererUsed"/>.
        /// </summary>
        public void EvictUnused(Func<int, bool> isMeshUsed, Func<int, bool> isSkinnedRendererUsed)
        {
            _evictScratch.Clear();
            foreach (var key in _staticVbCache.Keys)
                if (!isMeshUsed(key))
                    _evictScratch.Add(key);
            foreach (var key in _evictScratch)
            {
                _staticVbCache[key]?.Release();
                _staticVbCache.Remove(key);
            }

            _evictScratch.Clear();
            foreach (var key in _ibCache.Keys)
                if (!isMeshUsed(key))
                    _evictScratch.Add(key);
            foreach (var key in _evictScratch)
            {
                _ibCache[key]?.Release();
                _ibCache.Remove(key);
            }

            _evictScratch.Clear();
            foreach (var key in _skinnedCache.Keys)
                if (!isSkinnedRendererUsed(key))
                    _evictScratch.Add(key);
            foreach (var key in _evictScratch)
            {
                _skinnedCache[key].Vb?.Release();
                _skinnedCache.Remove(key);
            }
        }

        public void Dispose()
        {
            foreach (var vb in _staticVbCache.Values) vb?.Release();
            _staticVbCache.Clear();
            foreach (var ib in _ibCache.Values) ib?.Release();
            _ibCache.Clear();
            foreach (var e in _skinnedCache.Values) e.Vb?.Release();
            _skinnedCache.Clear();
        }

        // -------------------------------------------------------------------
        // Packing
        // -------------------------------------------------------------------

        private GraphicsBuffer GetOrCreateIndexBuffer(Mesh src, int meshId)
        {
            if (_ibCache.TryGetValue(meshId, out var cached)) return cached;

            using var meshDataArray = Mesh.AcquireReadOnlyMeshData(src);
            var       meshData      = meshDataArray[0];

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

            _ibCache[meshId] = ibGfx;
            return ibGfx;
        }

        // Packs the rest-pose SoA vertex buffer for the given stream layout. When the layout
        // has a PrevPosition stream it is initialised to the rest-pose positions.
        private static GraphicsBuffer BuildSoAVertexBuffer(Mesh src, in RtxptMeshStreamOffsets streams)
        {
            int vc      = src.vertexCount;
            int vbBytes = (int)streams.TotalBytes;

            using var meshDataArray = Mesh.AcquireReadOnlyMeshData(src);
            var       meshData      = meshDataArray[0];

            var vbData = new NativeArray<uint>(vbBytes / 4, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            int w      = 0; // write cursor, in uints (streams are packed in declaration order)

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

                // PrevPosition stream (skinned only): rest pose = current pose
                if (streams.HasPrevPos)
                {
                    for (int i = 0; i < vc; i++)
                    {
                        Vector3 p = positions[i];
                        vbData[w++] = (uint)BitConverter.SingleToInt32Bits(p.x);
                        vbData[w++] = (uint)BitConverter.SingleToInt32Bits(p.y);
                        vbData[w++] = (uint)BitConverter.SingleToInt32Bits(p.z);
                    }
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
            return vbGfx;
        }

        // Matches donut's dm::vectorToSnorm8 (vector.cpp) bit-for-bit: scale = 127/length,
        // then TRUNCATE toward zero via (int) cast (donut uses int(v*scale), not rounding),
        // and keep the low byte with &0xFF. This is what the original RTXPT importer
        // (GltfImporter.cpp -> vectorToSnorm8) does, so normals/tangents quantize identically.
        private static uint PackRGB8Snorm(Vector3 v)
        {
            float scale = 127.0f / Mathf.Sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
            int   r     = (int)(v.x * scale) & 0xFF;
            int   g     = (int)(v.y * scale) & 0xFF;
            int   b     = (int)(v.z * scale) & 0xFF;
            return (uint)(r | (g << 8) | (b << 16));
        }

        private static uint PackRGBA8Snorm(Vector4 v)
        {
            // donut scales all four channels by 127/length(xyz) (w shares the xyz-based scale).
            float scale = 127.0f / Mathf.Sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
            int   r     = (int)(v.x * scale) & 0xFF;
            int   g     = (int)(v.y * scale) & 0xFF;
            int   b     = (int)(v.z * scale) & 0xFF;
            int   a     = (int)(v.w * scale) & 0xFF;
            return (uint)(r | (g << 8) | (b << 16) | (a << 24));
        }
    }
}