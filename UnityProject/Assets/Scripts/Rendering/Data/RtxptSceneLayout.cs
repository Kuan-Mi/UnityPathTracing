using System.Collections.Generic;
using NativeRender;
using UnityEngine;
using UnityEngine.Rendering;

namespace PathTracing
{
    /// <summary>
    /// One TLAS instance in the RTXPT scene: a (renderer, SubmeshGroup) pair with its
    /// pre-resolved BLAS inputs. Built once per topology change by <see cref="RtxptSceneLayout"/>
    /// and consumed by BOTH the acceleration-structure sync (<see cref="RtxptAccelRegistry"/>)
    /// and the GPU-buffer build (<see cref="NativeRtxptGPUScene"/>), so the TLAS emission order,
    /// t_InstanceData, the flat per-geometry arrays, and the hit-group shader table all derive
    /// from the same list and can never drift apart.
    /// </summary>
    internal sealed class RtxptInstanceRecord
    {
        public RtxptRenderer       Renderer;
        public Renderer            TargetRenderer; // MeshRenderer or SkinnedMeshRenderer
        public SkinnedMeshRenderer Skinned;        // non-null for the skinned/dynamic path
        public Mesh                Mesh;
        public int                 RendererId; // TargetRenderer.GetInstanceID()
        public int                 GroupIndex; // index into Renderer.SubmeshGroups

        public bool IsSkinned => Skinned != null;

        // Skinned path only — the per-instance donut SoA buffer (refreshed each frame by the
        // repack compute) and the shared uint32 donut IB, attached by
        // NativeRtxptGPUScene.PrepareSkinnedRecords before the AS sync. The BLAS is built from
        // these (positions at offset 0, stride 12) instead of the mesh's native buffers.
        public GraphicsBuffer SkinnedVb;
        public GraphicsBuffer SkinnedIb;
        public int            SkinnedVertexCount;

        /// <summary>Assigned sub-mesh indices of this group, ascending. Never empty.</summary>
        public int[] SubmeshIndices;

        /// <summary>BLAS submesh descriptors, one per <see cref="SubmeshIndices"/> entry.</summary>
        public NativeRenderPlugin.SubmeshDesc[] Descs;

        /// <summary>
        /// Hit-group variant: 0 = emissive (HitEmissive), 1 = non-emissive. Analytic-light-proxy
        /// materials use variant 0 — the proxy NEE branch is compiled out of the non-emissive
        /// variant (RTXPT_MATERIAL_IS_ANALYTIC_LIGHT_PROXY=0).
        /// </summary>
        public uint HitGroupVariant;

        /// <summary>
        /// FNV-1a over the BLAS-affecting inputs (mesh identity + submesh subset + geometry
        /// flags). Equal hash ⇒ the already-registered native instance/BLAS is reused as-is.
        /// </summary>
        public ulong ContentHash;

        /// <summary>Stable native TLAS handle, assigned by <see cref="RtxptAccelRegistry.Sync"/>.</summary>
        public uint Handle;
    }

    /// <summary>
    /// Builds the flat instance-record list for the current renderer set. This is the single
    /// scene traversal: every filter decision (unassigned sub-meshes, empty groups, missing
    /// meshes) is made exactly once, here.
    /// </summary>
    internal static class RtxptSceneLayout
    {
        public static List<RtxptInstanceRecord> Build(IReadOnlyList<RtxptRenderer> targets)
        {
            var records = new List<RtxptInstanceRecord>(targets.Count);

            foreach (var t in targets)
            {
                if (t == null) continue;
                var r = t.TargetRenderer;
                if (r == null) continue;
                var mesh = t.SharedMesh;
                if (mesh == null) continue;

                var groups = t.SubmeshGroups;
                if (groups == null || groups.Length == 0)
                {
                    // Groups are normally built in OnEnable/OnValidate; rebuild defensively in
                    // case neither ran yet. Still empty afterwards = no assigned RtxptMaterial
                    // slots, i.e. nothing to render for this component.
                    t.RebuildGroups();
                    groups = t.SubmeshGroups;
                    if (groups == null || groups.Length == 0) continue;
                }

                bool skinned = t.Skinned != null;
                // Static BLASes read the mesh's native IB (16/32-bit); the skinned BLAS reads
                // our donut IB, which is always uint32.
                uint indexStride = skinned ? 4u : (mesh.indexFormat == IndexFormat.UInt16 ? 2u : 4u);
                int  mrId        = r.GetInstanceID();

                for (int gi = 0; gi < groups.Length; gi++)
                {
                    var grp = groups[gi];

                    // Only sub-meshes with a pre-baked RtxptMaterial enter the BLAS; null slots
                    // are skipped (not rendered).
                    var submeshIdx = new List<int>(grp.submeshIndices.Length);
                    var descs      = new List<NativeRenderPlugin.SubmeshDesc>(grp.submeshIndices.Length);
                    foreach (int sIdx in grp.submeshIndices)
                    {
                        if (!SubmeshHasMaterial(t, sIdx))
                        {
                            Debug.LogWarning($"[RtxptSceneLayout] '{r.name}' sub-mesh {sIdx} has no RtxptMaterial assigned — skipping (not rendered).");
                            continue;
                        }

                        var sub = mesh.GetSubMesh(sIdx);
                        submeshIdx.Add(sIdx);
                        descs.Add(new NativeRenderPlugin.SubmeshDesc
                        {
                            indexCount      = (uint)sub.indexCount,
                            indexByteOffset = (uint)sub.indexStart * indexStride,
                            baseVertex      = (uint)sub.baseVertex,
                            flags           = grp.isAlphaClip ? 0u : NativeRenderPlugin.SUBMESH_FLAG_GEOMETRY_OPAQUE,
                        });
                    }

                    if (descs.Count == 0) continue; // every sub-mesh in this group is unassigned

                    var descArray = descs.ToArray();
                    records.Add(new RtxptInstanceRecord
                    {
                        Renderer        = t,
                        TargetRenderer  = r,
                        Skinned         = t.Skinned,
                        Mesh            = mesh,
                        RendererId      = mrId,
                        GroupIndex      = gi,
                        SubmeshIndices  = submeshIdx.ToArray(),
                        Descs           = descArray,
                        HitGroupVariant = (grp.isEmissive || grp.isAnalyticProxy) ? 0u : 1u,
                        ContentHash     = HashRegistration(mesh.GetInstanceID(), descArray, skinned),
                    });
                }
            }

            return records;
        }

        /// <summary>True when the renderer has a pre-baked RtxptMaterial for the given sub-mesh.</summary>
        public static bool SubmeshHasMaterial(RtxptRenderer rr, int subMesh)
            => rr != null && subMesh < rr.Slots.Count && rr.Slots[subMesh] != null;

        /// <summary>
        /// World transform for a record's TLAS instance. GPU-skinned vertices are in root-bone
        /// space (Unity skins with <c>bone.localToWorldMatrix * bindpose</c>), so skinned
        /// instances must use the root bone's transform; static instances use the renderer's.
        /// </summary>
        public static Matrix4x4 GetRootTransform(RtxptInstanceRecord rec)
        {
            if (rec.Skinned != null && rec.Skinned.rootBone != null)
                return rec.Skinned.rootBone.localToWorldMatrix;
            return rec.TargetRenderer.transform.localToWorldMatrix;
        }

        private static ulong HashRegistration(int meshId, NativeRenderPlugin.SubmeshDesc[] descs, bool skinned)
        {
            const ulong Prime = 1099511628211UL;
            ulong h = 14695981039346656037UL;
            h = (h ^ (uint)meshId) * Prime;
            h = (h ^ (skinned ? 1u : 0u)) * Prime;
            h = (h ^ (uint)descs.Length) * Prime;
            foreach (var d in descs)
            {
                h = (h ^ d.indexCount) * Prime;
                h = (h ^ d.indexByteOffset) * Prime;
                h = (h ^ d.baseVertex) * Prime;
                h = (h ^ d.flags) * Prime;
            }
            return h;
        }
    }
}
