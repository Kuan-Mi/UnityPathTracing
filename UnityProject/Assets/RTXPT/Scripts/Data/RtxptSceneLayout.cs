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
    /// and the GPU-buffer build (<see cref="RtxptGPUScene"/>), so the TLAS emission order,
    /// t_InstanceData, the flat per-geometry arrays, and the hit-group shader table all derive
    /// from the same list and can never drift apart.
    /// </summary>
    internal sealed class RtxptInstanceRecord
    {
        public RtxptRenderer       Renderer;
        public Renderer            TargetRenderer; // MeshRenderer or SkinnedMeshRenderer
        public SkinnedMeshRenderer Skinned; // non-null for the skinned/dynamic path
        public Mesh                Mesh;
        public int                 RendererId; // TargetRenderer.GetInstanceID()
        public int                 GroupIndex; // index into Renderer.SubmeshGroups

        public bool IsSkinned => Skinned != null;

        // Skinned path only — the per-instance donut SoA buffer (refreshed each frame by the
        // repack compute) and the shared uint32 donut IB, attached by
        // RtxptGPUScene.PrepareSkinnedRecords before the AS sync. The BLAS is built from
        // these (positions at offset 0, stride 12) instead of the mesh's native buffers.
        public GraphicsBuffer SkinnedVb;
        public GraphicsBuffer SkinnedIb;
        public int            SkinnedVertexCount;

        /// <summary>Assigned sub-mesh indices of this group, ascending. Never empty.</summary>
        public int[] SubmeshIndices;

        /// <summary>BLAS submesh descriptors, one per <see cref="SubmeshIndices"/> entry.</summary>
        public NativeRenderPlugin.SubmeshDesc[] Descs;

        /// <summary>
        /// Hit-group variant in shader-table order:
        /// 0 = opaque emissive/proxy, 1 = opaque non-emissive,
        /// 2 = alpha/custom emissive/proxy, 3 = alpha/custom non-emissive.
        /// Analytic-light-proxy materials use the emissive/proxy branch — the proxy NEE path is
        /// compiled out of the non-emissive variant (RTXPT_MATERIAL_IS_ANALYTIC_LIGHT_PROXY=0).
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
        private static readonly HashSet<string> s_StaticParityWarnings = new();

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

                        var sub  = mesh.GetSubMesh(sIdx);
                        var slot = t.Slots[sIdx];
                        submeshIdx.Add(sIdx);

                        // BLAS geometry is non-opaque when the any-hit shader must run for it:
                        // alpha-tested OR ExcludeFromNEE (the latter lets NEE shadow rays pass
                        // through transmissive surfaces like glass). Mirrors RTXPT's rule in
                        // AccelerationStructureUtil.h: flags = (EnableAlphaTesting || ExcludeFromNEE)
                        // ? GeometryFlags::None : GeometryFlags::Opaque. Without the ExcludeFromNEE
                        // term, glass is built opaque and wrongly blocks NEE light.
                        bool nonOpaque = grp.isAlphaClip || (slot != null && slot.ExcludeFromNEE);

                        descs.Add(new NativeRenderPlugin.SubmeshDesc
                        {
                            indexCount      = (uint)sub.indexCount,
                            indexByteOffset = (uint)sub.indexStart * indexStride,
                            // The donut IB already has baseVertex baked into its uint32 indices
                            // (GetIndices applyBaseVertex), so the skinned BLAS — which reads that
                            // IB — must not offset the vertex buffer by baseVertex a second time.
                            // The native mesh IB (static path) stores baseVertex-relative indices,
                            // so there the offset is required.
                            baseVertex = skinned ? 0u : (uint)sub.baseVertex,
                            flags      = nonOpaque ? 0u : NativeRenderPlugin.SUBMESH_FLAG_GEOMETRY_OPAQUE,
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
                        HitGroupVariant = ComputeHitGroupVariant(t, grp, descArray),
                        ContentHash     = HashRegistration(mesh.GetInstanceID(), descArray, skinned),
                    });
                }
            }

            SortRecordsForParity(records);
            return records;
        }

        // ---- TEMPORARY: instance-ordering parity test -----------------------------
        // Force a deterministic scene order (hierarchy depth-first pre-order) instead of
        // RtxptRenderer.OnEnable registration order, so the Unity TLAS / sub-instance /
        // hit-group / material sequence is reproducible and lines up with the original
        // RTXPT GetMeshInstances() (glTF scene-graph load) order. Everything downstream
        // (RtxptAccelRegistry.Sync: SetInstanceOrderIndex + runningContribution + the
        // per-geometry variant array, and RtxptGPUScene: t_InstanceData/t_SubInstanceData/
        // t_GeometryData) derives from this list, so reordering here reorders them all
        // consistently. Purpose: remove instance ordering as a variable when diffing PIX
        // captures. Flip kSortRecordsForParity to false to restore OnEnable order. Remove
        // once ordering is ruled out as a cause.
        private const bool kSortRecordsForParity = true;

        private static void SortRecordsForParity(List<RtxptInstanceRecord> records)
        {
            if (!kSortRecordsForParity || records.Count < 2) return;

            // Cache each transform's root→node sibling-index path (unique per node), so the
            // O(n log n) comparisons don't re-walk the hierarchy repeatedly.
            var pathCache = new Dictionary<Transform, int[]>();
            int[] PathOf(Transform t)
            {
                if (t == null) return System.Array.Empty<int>();
                if (pathCache.TryGetValue(t, out var cached)) return cached;
                var chain = new List<int>();
                for (var c = t; c != null; c = c.parent)
                    chain.Add(c.GetSiblingIndex());
                chain.Reverse(); // root → node
                var arr = chain.ToArray();
                pathCache[t] = arr;
                return arr;
            }

            records.Sort((a, b) =>
            {
                int[] pa = PathOf(a.TargetRenderer != null ? a.TargetRenderer.transform : null);
                int[] pb = PathOf(b.TargetRenderer != null ? b.TargetRenderer.transform : null);
                int n = Mathf.Min(pa.Length, pb.Length);
                for (int i = 0; i < n; i++)
                    if (pa[i] != pb[i]) return pa[i] < pb[i] ? -1 : 1;
                if (pa.Length != pb.Length) return pa.Length < pb.Length ? -1 : 1;
                // Same renderer: keep its groups ascending so the per-geometry
                // (sub-instance / SBT) order stays native-submesh order.
                return a.GroupIndex.CompareTo(b.GroupIndex);
            });
        }

        /// <summary>True when the renderer has a pre-baked RtxptMaterial for the given sub-mesh.</summary>
        public static bool SubmeshHasMaterial(RtxptRenderer rr, int subMesh)
            => rr != null && subMesh < rr.Slots.Count && rr.Slots[subMesh] != null;

        private static uint ComputeHitGroupVariant(RtxptRenderer renderer, RtxptSubmeshGroup grp, NativeRenderPlugin.SubmeshDesc[] descs)
        {
            if (TryGetHardcodedRtxptMaterialVariant(renderer, grp, out uint materialVariant))
                return materialVariant;

            return ComputeStaticParityFallbackVariant(renderer, grp, descs);
        }

        private static bool TryGetHardcodedRtxptMaterialVariant(RtxptRenderer renderer, RtxptSubmeshGroup grp, out uint variant)
        {
            // Static-scene RTXPT parity mode. These indices must match the hit-group blob order
            // supplied by RtxptFeature.AutoFillShaders:
            //   0 = LMBR0000002Mesh_b12400a3
            //   1 = LMBR0000040black_eae3d639
            // These are material shader permutations, not literal material names. Mesh is the
            // representative name for the ordinary non-emissive/non-analytic permutation.
            variant = 0;
            if (renderer?.Slots == null || grp?.submeshIndices == null) return false;

            bool sawBlack = false;
            bool sawMesh  = false;
            foreach (int submesh in grp.submeshIndices)
            {
                if (submesh < 0 || submesh >= renderer.Slots.Count) continue;
                var mat = renderer.Slots[submesh];
                if (mat == null) continue;

                string name = NormalizeMaterialName(mat.name);
                if (name.Contains("lmbr0000040black"))
                    sawBlack = true;
                if (name.Contains("lmbr0000002mesh"))
                    sawMesh = true;
            }

            if (sawBlack && !sawMesh)
            {
                variant = 1u;
                return true;
            }

            if (sawMesh && !sawBlack)
            {
                variant = 0u;
                return true;
            }

            if (sawMesh && sawBlack)
                LogStaticParityMaterialWarning(renderer, grp, "contains both hard-coded RTXPT representative material names in one submesh group", 0u);

            return false;
        }

        private static string NormalizeMaterialName(string name)
        {
            if (string.IsNullOrEmpty(name)) return string.Empty;
            return name
                .Replace("_", "")
                .Replace("-", "")
                .Replace(".", "")
                .Replace(" ", "")
                .ToLowerInvariant();
        }

        private static uint ComputeFallbackHitGroupVariant(RtxptSubmeshGroup grp, NativeRenderPlugin.SubmeshDesc[] descs)
        {
            bool customHit = grp.isAlphaClip;
            if (!customHit)
            {
                foreach (var d in descs)
                {
                    if ((d.flags & NativeRenderPlugin.SUBMESH_FLAG_GEOMETRY_OPAQUE) == 0)
                    {
                        customHit = true;
                        break;
                    }
                }
            }

            bool emissiveOrProxy = grp.isEmissive || grp.isAnalyticProxy;
            if (customHit)
                return emissiveOrProxy ? 2u : 3u;
            return emissiveOrProxy ? 0u : 1u;
        }

        private static uint ComputeStaticParityFallbackVariant(RtxptRenderer renderer, RtxptSubmeshGroup grp, NativeRenderPlugin.SubmeshDesc[] descs)
        {
            uint legacyVariant = ComputeFallbackHitGroupVariant(grp, descs);
            // Static parity mode currently bakes exactly two RTXPT material hit groups:
            //   0 = ordinary non-emissive/non-analytic material permutation (Mesh representative)
            //   1 = material permutation with emissive/analytic paths compiled in (black representative)
            // Alpha testing affects whether AnyHit is wired, but not the material shader
            // permutation index in the original RTXPT baker.
            uint fallbackVariant = (grp != null && (grp.isEmissive || grp.isAnalyticProxy)) ? 1u : 0u;
            return fallbackVariant;
        }

        private static void LogStaticParityMaterialWarning(RtxptRenderer renderer, RtxptSubmeshGroup grp, string reason, uint fallbackVariant)
        {
            string rendererName = renderer != null ? renderer.name : "<null renderer>";
            string groupInfo = FormatGroupInfo(grp);
            string submeshInfo = FormatSubmeshMaterials(renderer, grp);
            string key = $"{rendererName}|{groupInfo}|{submeshInfo}|{reason}";
            if (!s_StaticParityWarnings.Add(key))
                return;

            Debug.LogError(
                $"[RtxptSceneLayout] Static RTXPT parity mode only has hit groups 0/1 " +
                $"(0=LMBR0000002Mesh_b12400a3, 1=LMBR0000040black_eae3d639), but renderer '{rendererName}' {groupInfo} {reason}. " +
                $"Using fallback variant {fallbackVariant} to avoid an out-of-range shader table. Submeshes/materials: {submeshInfo}");
        }

        private static string FormatGroupInfo(RtxptSubmeshGroup grp)
        {
            if (grp == null)
                return "group=<null>";

            return $"groupFlags=(alphaClip={grp.isAlphaClip}, emissive={grp.isEmissive}, analyticProxy={grp.isAnalyticProxy})";
        }

        private static string FormatSubmeshMaterials(RtxptRenderer renderer, RtxptSubmeshGroup grp)
        {
            if (renderer?.Slots == null || grp?.submeshIndices == null)
                return "<unavailable>";

            var parts = new List<string>(grp.submeshIndices.Length);
            foreach (int submesh in grp.submeshIndices)
            {
                string matName = "<missing>";
                if (submesh >= 0 && submesh < renderer.Slots.Count && renderer.Slots[submesh] != null)
                    matName = renderer.Slots[submesh].name;
                parts.Add($"{submesh}:{matName}");
            }

            return string.Join(", ", parts);
        }

        /// <summary>
        /// World transform for a record's TLAS instance. GPU-skinned vertices are in the root
        /// bone's RIGID frame: Unity bakes all bone scale (including the root's) into the
        /// skinned positions and draws the renderer with the root bone's unscaled
        /// rotation+translation. Using localToWorldMatrix here would apply the root's scale a
        /// second time (e.g. a glTF unit-scale root of ~0.0018 shrinks the instance to nothing).
        /// Static instances use the renderer's full transform.
        /// </summary>
        public static Matrix4x4 GetRootTransform(RtxptInstanceRecord rec)
        {
            if (rec.Skinned != null)
            {
                Transform root = rec.Skinned.rootBone != null ? rec.Skinned.rootBone : rec.TargetRenderer.transform;
                return Matrix4x4.TRS(root.position, root.rotation, Vector3.one);
            }

            return rec.TargetRenderer.transform.localToWorldMatrix;
        }

        private static ulong HashRegistration(int meshId, NativeRenderPlugin.SubmeshDesc[] descs, bool skinned)
        {
            const ulong Prime = 1099511628211UL;
            ulong       h     = 14695981039346656037UL;
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
