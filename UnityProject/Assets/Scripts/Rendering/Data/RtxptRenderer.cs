using System;
using System.Collections.Generic;
using UnityEngine;

namespace PathTracing
{
    // =========================================================================
    // RtxptRenderer  —  attach to any MeshRenderer
    // =========================================================================

    /// <summary>
    /// Per-renderer RTXPT material override component.
    ///
    /// Attach to any <see cref="MeshRenderer"/> to supply the RTXPT material
    /// properties sent to the GPU. Each sub-mesh slot references a
    /// <see cref="RtxptMaterial"/> that can be shared across multiple renderers.
    /// A null entry means the sub-mesh is skipped — it is not added to the
    /// acceleration structure and does not render. Materials must be authored in
    /// advance (imported from RTXPT JSON, or baked once with the button below);
    /// there is no runtime Unity-material fallback.
    ///
    /// Use the "Bake from Renderer" button in the Inspector to create and
    /// populate slot assets from the current Unity materials as a starting point.
    /// </summary>
    [RequireComponent(typeof(MeshRenderer))]
    [DisallowMultipleComponent]
    [ExecuteAlways]
    public class RtxptRenderer : MonoBehaviour
    {
        [Tooltip("One RtxptMaterial per sub-mesh. Index matches MeshRenderer.sharedMaterials. Null = sub-mesh is skipped (not rendered) — there is no runtime Unity-material fallback.")]
        public List<RtxptMaterial> Slots = new();

        // ---- Active-renderer registry (replaces NativeRayTracingTarget for the Rtxpt path) ----

        private static readonly List<RtxptRenderer> s_All = new();

        /// <summary>All currently enabled RTXPT renderers. Consumed by <see cref="NativeRtxptGPUScene"/>.</summary>
        public static IReadOnlyList<RtxptRenderer> All => s_All;

        /// <summary>
        /// Bumped whenever any renderer's set membership or sub-mesh grouping changes, so the GPU
        /// scene can detect topology changes (material (un)assigned, emissive/alpha-test toggled)
        /// that a plain set-equality check would miss.
        /// </summary>
        public static int TopologyVersion { get; private set; }

        private static void BumpTopology() => TopologyVersion++;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        private static void ResetStatics()
        {
            s_All.Clear();
            TopologyVersion = 0;
        }

        /// <summary>The MeshRenderer on the same GameObject (cached on enable).</summary>
        public MeshRenderer MeshRenderer { get; private set; }

        /// <summary>
        /// Sub-meshes grouped by (isTransparent, isEmissive, isAlphaClip, isAnalyticProxy), derived from
        /// the assigned <see cref="RtxptMaterial"/> slots. Each group becomes a separate TLAS instance.
        /// Null slots are excluded (their sub-mesh is not rendered). Rebuilt on enable / validate / bake.
        /// </summary>
        public RtxptSubmeshGroup[] SubmeshGroups { get; private set; } = Array.Empty<RtxptSubmeshGroup>();

        /// <summary>
        /// True when any slot parameter has changed since the last GPU upload.
        /// Set automatically when any referenced asset changes (via its Modified event)
        /// or via OnValidate. Cleared by <see cref="NativeRtxptGPUScene"/> after upload.
        /// Note: texture assignment changes require a full scene rebuild.
        /// </summary>
        public bool IsDirty { get; private set; }

        public void MarkDirty()  => IsDirty = true;
        public void ClearDirty() => IsDirty = false;

        private readonly List<RtxptMaterial> _subscribedAssets = new();

        private void OnEnable()
        {
            MeshRenderer = GetComponent<MeshRenderer>();
            if (!s_All.Contains(this)) s_All.Add(this);
            RefreshSubscriptions();
            RebuildGroups(); // also bumps TopologyVersion (covers set growth)
        }

        private void OnDisable()
        {
            ClearSubscriptions();
            s_All.Remove(this);
            // No BumpTopology: the set shrank, which TargetSetChanged already detects.
        }

        private void OnValidate()
        {
            IsDirty      = true;
            MeshRenderer = GetComponent<MeshRenderer>();
            RefreshSubscriptions();
            RebuildGroups(); // slot (un)assignment or flag changes can alter grouping
        }

        private void RefreshSubscriptions()
        {
            ClearSubscriptions();
            if (Slots == null) return;
            foreach (var asset in Slots)
            {
                if (asset == null || _subscribedAssets.Contains(asset)) continue;
                asset.Modified += OnSlotMaterialModified;
                _subscribedAssets.Add(asset);
            }
        }

        private void ClearSubscriptions()
        {
            foreach (var asset in _subscribedAssets)
                if (asset != null)
                    asset.Modified -= OnSlotMaterialModified;
            _subscribedAssets.Clear();
        }

        // A referenced material changed: flag the cheap scalar re-upload, and re-evaluate grouping
        // (RebuildGroups bumps TopologyVersion only if an emissive/transmission/alpha-test flag flip
        // actually changed the groups, forcing the full rebuild those changes require).
        private void OnSlotMaterialModified()
        {
            IsDirty = true;
            RebuildGroups();
        }

        /// <summary>
        /// Rebuilds <see cref="SubmeshGroups"/> from the assigned slots, grouping sub-meshes that
        /// share the same (isTransparent, isEmissive, isAlphaClip, isAnalyticProxy) tuple while
        /// preserving sub-mesh order. Null slots are skipped (their sub-mesh is not rendered). Bumps
        /// <see cref="TopologyVersion"/> only when the grouping actually changed, so scalar-only
        /// edits keep using the lightweight material-update path instead of a full scene rebuild.
        /// </summary>
        public void RebuildGroups()
        {
            var mf     = GetComponent<MeshFilter>();
            int subCnt = mf != null && mf.sharedMesh != null ? mf.sharedMesh.subMeshCount : (Slots?.Count ?? 0);

            var order = new List<(bool t, bool e, bool a, bool p)>();
            var lists = new Dictionary<(bool, bool, bool, bool), List<int>>();

            for (int s = 0; s < subCnt; s++)
            {
                RtxptMaterial mat = Slots != null && s < Slots.Count ? Slots[s] : null;
                if (mat == null) continue; // unassigned sub-mesh — excluded (not rendered)

                var key = (IsTransparent(mat), IsEmissive(mat), mat.EnableAlphaTesting, IsAnalyticProxy(mat));
                if (!lists.TryGetValue(key, out var list))
                {
                    list       = new List<int>();
                    lists[key] = list;
                    order.Add(key);
                }

                list.Add(s);
            }

            var groups = new RtxptSubmeshGroup[order.Count];
            for (int i = 0; i < order.Count; i++)
            {
                var key = order[i];
                groups[i] = new RtxptSubmeshGroup
                {
                    isTransparent   = key.t,
                    isEmissive      = key.e,
                    isAlphaClip     = key.a,
                    isAnalyticProxy = key.p,
                    submeshIndices  = lists[key].ToArray(),
                };
            }

            if (!GroupsEqual(SubmeshGroups, groups))
            {
                SubmeshGroups = groups;
                BumpTopology();
            }
        }

        private static bool GroupsEqual(RtxptSubmeshGroup[] a, RtxptSubmeshGroup[] b)
        {
            if (a == null || a.Length != b.Length) return false;
            for (int i = 0; i < a.Length; i++)
            {
                var x = a[i];
                var y = b[i];
                if (x.isTransparent != y.isTransparent || x.isEmissive != y.isEmissive ||
                    x.isAlphaClip != y.isAlphaClip || x.isAnalyticProxy != y.isAnalyticProxy ||
                    x.submeshIndices.Length != y.submeshIndices.Length)
                    return false;
                for (int k = 0; k < x.submeshIndices.Length; k++)
                    if (x.submeshIndices[k] != y.submeshIndices[k])
                        return false;
            }

            return true;
        }

        // Classification mirrors the RTXPT material semantics (not Unity shader sniffing).
        private static bool IsTransparent(RtxptMaterial m) => m.EnableTransmission;

        private static bool IsEmissive(RtxptMaterial m)
            => m.EmissiveIntensity > 0f &&
               (m.EmissiveColor.r > 0f || m.EmissiveColor.g > 0f || m.EmissiveColor.b > 0f);

        // Analytic-light-proxy materials must use the emissive hit-group variant (the proxy NEE branch
        // is compiled out of the non-emissive variant), so they get their own grouping dimension.
        private static bool IsAnalyticProxy(RtxptMaterial m) => m.EnableAsAnalyticLightProxy;

        /// <summary>
        /// Bakes slot data from the renderer's current Unity materials into any
        /// already-assigned <see cref="RtxptMaterial"/> entries.
        /// Null slot entries are skipped — use the "Bake from Renderer" Inspector
        /// button to create missing assets automatically.
        /// </summary>
        public void BakeFromRenderer()
        {
            var mr = GetComponent<MeshRenderer>();
            var mf = GetComponent<MeshFilter>();
            if (mr == null) return;

            Material[] mats       = mr.sharedMaterials ?? Array.Empty<Material>();
            int        subMeshCnt = mf != null && mf.sharedMesh != null ? mf.sharedMesh.subMeshCount : mats.Length;

            while (Slots.Count < subMeshCnt) Slots.Add(null);
            if (Slots.Count > subMeshCnt) Slots.RemoveRange(subMeshCnt, Slots.Count - subMeshCnt);

            for (int s = 0; s < subMeshCnt; s++)
            {
                if (Slots[s] == null) continue;
                Material mat = s < mats.Length ? mats[s] : (mats.Length > 0 ? mats[^1] : null);
                BakeSlotFromMaterial(Slots[s], mat);
            }

            RebuildGroups();
        }

        public static void BakeSlotFromMaterial(RtxptMaterial slot, Material mat)
        {
            if (mat == null) return;

            bool isGltf = mat.shader.name == "Shader Graphs/glTF-pbrMetallicRoughness";
            if (isGltf)
            {
                slot.BaseOrDiffuseTexture              = TryGetTex(mat, "baseColorTexture");
                slot.NormalTexture                     = TryGetTex(mat, "normalTexture");
                slot.OcclusionRoughnessMetallicTexture = TryGetTex(mat, "metallicRoughnessTexture");
                slot.EmissiveTexture                   = TryGetTex(mat, "emissiveTexture");

                Color baseC = TryGetColor(mat, "baseColorFactor", Color.white);
                slot.BaseColorFactor    = baseC;
                slot.Opacity            = baseC.a;
                slot.EmissiveColor      = TryGetColor(mat, "emissiveFactor", Color.black);
                slot.Roughness          = TryGetFloat(mat, "roughnessFactor", 0.5f);
                slot.Metalness          = TryGetFloat(mat, "metallicFactor", 0f);
                slot.EnableAlphaTesting = mat.IsKeywordEnabled("_ALPHATEST_ON");
                slot.AlphaCutoff        = slot.EnableAlphaTesting ? TryGetFloat(mat, "alphaCutoff", 0.5f) : 0f;
                slot.NormalTextureScale = 1f;
            }
            else
            {
                // URP Lit / unknown fallback
                slot.BaseOrDiffuseTexture              = TryGetTex(mat, "_BaseMap");
                slot.NormalTexture                     = TryGetTex(mat, "_BumpMap");
                slot.OcclusionRoughnessMetallicTexture = TryGetTex(mat, "_MetallicGlossMap");
                slot.EmissiveTexture                   = TryGetTex(mat, "_EmissionMap");

                Color baseC = TryGetColor(mat, "_BaseColor", Color.white);
                slot.BaseColorFactor    = baseC;
                slot.Opacity            = baseC.a;
                slot.EmissiveColor      = TryGetColor(mat, "_EmissionColor", Color.black);
                slot.Roughness          = 1f - TryGetFloat(mat, "_Smoothness", 0.5f);
                slot.Metalness          = TryGetFloat(mat, "_Metallic", 0f);
                slot.AlphaCutoff        = TryGetFloat(mat, "_Cutoff", 0f);
                slot.EnableAlphaTesting = slot.AlphaCutoff > 0f;
                slot.NormalTextureScale = TryGetFloat(mat, "_BumpScale", 1f);
            }

            slot.EmissiveIntensity  = 1f;
            slot.EnableTransmission = false;
            slot.TransmissionFactor = 0f;

            float   met            = slot.Metalness;
            Vector3 dielectricF0   = new Vector3(0.04f, 0.04f, 0.04f);
            Vector3 metalBaseColor = new Vector3(slot.BaseColorFactor.r, slot.BaseColorFactor.g, slot.BaseColorFactor.b);
            Vector3 specF0         = Vector3.Lerp(dielectricF0, metalBaseColor, met);
            slot.SpecularColor = new Color(specF0.x, specF0.y, specF0.z, 1f);

            slot.UseSpecularGlossModel = false;
            slot.ThinSurface           = false;
            slot.MetalnessInRedChannel = slot.OcclusionRoughnessMetallicTexture != null;

            // Texture enables: on by default for all assigned textures
            slot.EnableBaseTexture                       = true;
            slot.EnableOcclusionRoughnessMetallicTexture = true;
            slot.EnableNormalTexture                     = true;
            slot.EnableEmissiveTexture                   = true;
            slot.EnableTransmissionTexture               = true;
        }

        private static Texture TryGetTex(Material mat, string prop)
            => mat != null && mat.HasProperty(prop) ? mat.GetTexture(prop) : null;

        private static Color TryGetColor(Material mat, string prop, Color fallback)
            => mat != null && mat.HasProperty(prop) ? mat.GetColor(prop) : fallback;

        private static float TryGetFloat(Material mat, string prop, float fallback)
            => mat != null && mat.HasProperty(prop) ? mat.GetFloat(prop) : fallback;
    }

    // =========================================================================
    // RtxptSubmeshGroup  —  sub-meshes sharing one (transparent, emissive, alphaClip, analyticProxy)
    // tuple, each becoming a separate TLAS instance. Derived from RtxptMaterial slots
    // by RtxptRenderer (replaces NativeRender.SubmeshGroupDesc for the Rtxpt path).
    // =========================================================================
    public sealed class RtxptSubmeshGroup
    {
        public bool isTransparent;
        public bool isEmissive;
        public bool isAlphaClip;

        /// <summary>
        /// True when the group's material is flagged <c>EnableAsAnalyticLightProxy</c>. Routed to the
        /// emissive hit-group variant (variant 0) so the analytic-proxy NEE branch is compiled in.
        /// </summary>
        public bool isAnalyticProxy;

        /// <summary>Indices into the Mesh's sub-meshes, in ascending order.</summary>
        public int[] submeshIndices;
    }
}