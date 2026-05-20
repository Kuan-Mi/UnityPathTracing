using System.Collections.Generic;
using UnityEngine;

namespace NativeRender
{
    /// <summary>
    /// Pure scene classification layer: sorts all active <see cref="NativeRayTracingTarget"/>s
    /// and <see cref="NativeRayTracingSkinnedTarget"/>s into rendering buckets.
    ///
    /// No GPU resources, no acceleration-structure references.  A consumer (e.g.
    /// <c>NRDSampleResource</c>) calls <see cref="Rebuild"/> or <see cref="ApplyIncremental"/>
    /// each frame and then reads the resulting lists.
    /// </summary>
    internal sealed class SceneLayout
    {
        // =====================================================================
        // Public data types
        // =====================================================================

        /// <summary>Identifies one submesh within a <see cref="NativeRayTracingTarget"/>.</summary>
        public readonly struct SubmeshRef
        {
            public readonly NativeRayTracingTarget Target;
            public readonly int                    SubIndex;

            public SubmeshRef(NativeRayTracingTarget t, int s) { Target = t; SubIndex = s; }
        }

        // =====================================================================
        // Classification buckets
        // =====================================================================

        /// <summary>Opaque submeshes from static MeshRenderers (merged-BLAS bucket, play-mode only).</summary>
        public readonly List<SubmeshRef> StaticOpaque      = new();

        /// <summary>Transparent submeshes from static MeshRenderers (merged-BLAS bucket, play-mode only).</summary>
        public readonly List<SubmeshRef> StaticTransparent = new();

        /// <summary>Emissive submeshes from static MeshRenderers (may overlap with opaque/transparent).</summary>
        public readonly List<SubmeshRef> StaticEmissive    = new();

        /// <summary>Non-static (or edit-mode) MeshRenderer targets that get their own per-target BLAS.</summary>
        public readonly List<NativeRayTracingTarget> DynamicTargets = new();

        /// <summary>All active skinned targets.</summary>
        public readonly List<NativeRayTracingSkinnedTarget> SkinnedTargets = new();

        // =====================================================================
        // Public API
        // =====================================================================

        /// <summary>
        /// Fully reclassifies all targets.  Clears every bucket, then repopulates from
        /// <paramref name="targets"/> and the current <see cref="NativeRayTracingSkinnedTarget.All"/> list.
        /// </summary>
        public void Rebuild(IReadOnlyList<NativeRayTracingTarget> targets)
        {
            StaticOpaque.Clear();
            StaticTransparent.Clear();
            StaticEmissive.Clear();
            DynamicTargets.Clear();
            SkinnedTargets.Clear();

            bool mergeStatics = Application.isPlaying;

            foreach (var t in targets)
            {
                if (t == null) continue;
                var mr = t.GetComponent<MeshRenderer>();
                if (mr == null) continue;
                var mf = mr.GetComponent<MeshFilter>();
                if (mf == null || mf.sharedMesh == null) continue;

                if (mergeStatics && t.IsStatic)
                {
                    ClassifyStaticSubmeshes(t, mr, mf.sharedMesh);
                }
                else
                {
                    DynamicTargets.Add(t);
                }
            }

            foreach (var st in NativeRayTracingSkinnedTarget.All)
                if (st != null) SkinnedTargets.Add(st);
        }

        /// <summary>
        /// Processes the static-mesh add/remove queues and updates the dynamic and skinned buckets
        /// without touching the static buckets.
        ///
        /// Returns <c>true</c> when a change was detected that requires a full GPU scene rebuild
        /// (static object added or removed while in play mode).  The caller should call
        /// <see cref="Rebuild"/> again before proceeding with the GPU build in that case.
        /// </summary>
        public bool ApplyIncremental()
        {
            bool needsFullRebuild = false;

            // ---- MeshRenderer removes ----
            while (NativeRayTracingTarget.RemoveQueue.Count > 0)
            {
                var ev = NativeRayTracingTarget.RemoveQueue.Dequeue();
                int id = ev.RendererInstanceId;

                if (DynamicTargets.RemoveAll(t =>
                        t != null && t.GetComponent<MeshRenderer>()?.GetInstanceID() == id) > 0)
                {
                    // Removed a dynamic target — no full rebuild needed.
                }
                else
                {
                    // Was a merged static — full rebuild required.
                    needsFullRebuild = true;
                }
            }

            // ---- MeshRenderer adds ----
            while (NativeRayTracingTarget.AddQueue.Count > 0)
            {
                var ev = NativeRayTracingTarget.AddQueue.Dequeue();
                if (ev.Target == null || ev.Renderer == null) continue;

                bool mergeStatics = Application.isPlaying;
                if (mergeStatics && ev.Target.IsStatic)
                {
                    needsFullRebuild = true;
                    continue;
                }

                if (!DynamicTargets.Contains(ev.Target))
                    DynamicTargets.Add(ev.Target);
            }

            // ---- Skinned removes ----
            while (NativeRayTracingSkinnedTarget.RemoveQueue.Count > 0)
            {
                var ev = NativeRayTracingSkinnedTarget.RemoveQueue.Dequeue();
                int id = ev.RendererInstanceId;
                SkinnedTargets.RemoveAll(st =>
                    st != null && st.GetComponent<SkinnedMeshRenderer>()?.GetInstanceID() == id);
            }

            // ---- Skinned adds ----
            while (NativeRayTracingSkinnedTarget.AddQueue.Count > 0)
            {
                var ev = NativeRayTracingSkinnedTarget.AddQueue.Dequeue();
                if (ev.Target == null) continue;
                if (!SkinnedTargets.Contains(ev.Target))
                    SkinnedTargets.Add(ev.Target);
            }

            return needsFullRebuild;
        }

        // =====================================================================
        // Triangle-count helpers (mirror CountGroupTriangles in NRDSampleResource)
        // =====================================================================

        /// <summary>Total triangles for a list of per-submesh refs (used for merged-BLAS buffers).</summary>
        public static int CountTriangles(List<SubmeshRef> submeshRefs)
        {
            int count = 0;
            foreach (var sr in submeshRefs)
            {
                if (sr.Target == null) continue;
                var mf = sr.Target.GetComponent<MeshFilter>();
                if (mf?.sharedMesh == null) continue;
                count += (int)(mf.sharedMesh.GetIndexCount(sr.SubIndex) / 3);
            }
            return count;
        }

        /// <summary>Total triangles across all submeshes of a list of dynamic targets.</summary>
        public static int CountTriangles(List<NativeRayTracingTarget> targets)
        {
            int count = 0;
            foreach (var t in targets)
            {
                if (t == null) continue;
                var mf = t.GetComponent<MeshFilter>();
                if (mf?.sharedMesh == null) continue;
                Mesh mesh = mf.sharedMesh;
                for (int s = 0; s < mesh.subMeshCount; s++)
                    count += (int)(mesh.GetIndexCount(s) / 3);
            }
            return count;
        }

        // =====================================================================
        // Helpers
        // =====================================================================

        /// <summary>
        /// Encodes a unique TLAS instance handle for a specific submesh group of a MeshRenderer.
        /// High 4 bits = groupIndex (max 16 groups per renderer), low 28 bits = mrInstanceId.
        /// </summary>
        public static uint MakeGroupHandle(int mrInstanceId, int groupIndex)
            => (uint)(mrInstanceId & 0x0FFFFFFF) | ((uint)groupIndex << 28);

        private void ClassifyStaticSubmeshes(NativeRayTracingTarget t, MeshRenderer mr, Mesh mesh)
        {
            int subCnt = mesh.subMeshCount;
            for (int s = 0; s < subCnt; s++)
            {
                if (s >= t.SubmeshMaterialInfos.Length)
                {
                    Debug.LogError(
                        $"[SceneLayout] Submesh {s} of '{mr.name}' has no material assigned; skipping");
                    continue;
                }

                bool isTrans    = t.SubmeshMaterialInfos[s].isTransparent;
                bool isEmissive = t.SubmeshMaterialInfos[s].isEmissive;

                var sr = new SubmeshRef(t, s);
                if (isTrans) StaticTransparent.Add(sr);
                else         StaticOpaque.Add(sr);

                if (isEmissive) StaticEmissive.Add(sr);
            }
        }

    }
}
