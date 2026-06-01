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
    /// Attach to any <see cref="MeshRenderer"/> to take manual control of the
    /// RTXPT material properties sent to the GPU. Each sub-mesh slot references
    /// a <see cref="RtxptMaterial"/> that can be shared across
    /// multiple renderers. A null entry means "use the default baking path" for
    /// that sub-mesh.
    ///
    /// Use the "Bake from Renderer" button in the Inspector to create and
    /// populate slot assets from the current Unity materials as a starting point.
    /// </summary>
    [RequireComponent(typeof(MeshRenderer))]
    [DisallowMultipleComponent]
    public class RtxptRenderer : MonoBehaviour
    {
        [Tooltip("One asset per sub-mesh. Index matches MeshRenderer.sharedMaterials. Null = use default material baking.")]
        public List<RtxptMaterial> Slots = new();

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

        private void OnEnable()  => RefreshSubscriptions();
        private void OnDisable() => ClearSubscriptions();

        private void OnValidate()
        {
            IsDirty = true;
            RefreshSubscriptions();
        }

        private void RefreshSubscriptions()
        {
            ClearSubscriptions();
            if (Slots == null) return;
            foreach (var asset in Slots)
            {
                if (asset == null || _subscribedAssets.Contains(asset)) continue;
                asset.Modified += MarkDirty;
                _subscribedAssets.Add(asset);
            }
        }

        private void ClearSubscriptions()
        {
            foreach (var asset in _subscribedAssets)
                if (asset != null) asset.Modified -= MarkDirty;
            _subscribedAssets.Clear();
        }

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
        }

        internal static void BakeSlotFromMaterial(RtxptMaterial slot, Material mat)
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
                slot.BaseColorFactor       = baseC;
                slot.Opacity               = baseC.a;
                slot.EmissiveColor         = TryGetColor(mat, "emissiveFactor", Color.black);
                slot.Roughness             = TryGetFloat(mat, "roughnessFactor", 0.5f);
                slot.Metalness             = TryGetFloat(mat, "metallicFactor", 0f);
                slot.EnableAlphaTesting    = mat.IsKeywordEnabled("_ALPHATEST_ON");
                slot.AlphaCutoff           = slot.EnableAlphaTesting ? TryGetFloat(mat, "alphaCutoff", 0.5f) : 0f;
                slot.NormalTextureScale    = 1f;
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

            slot.EmissiveIntensity   = 1f;
            slot.EnableTransmission  = false;
            slot.TransmissionFactor  = 0f;

            float   met            = slot.Metalness;
            Vector3 dielectricF0   = new Vector3(0.04f, 0.04f, 0.04f);
            Vector3 metalBaseColor = new Vector3(slot.BaseColorFactor.r, slot.BaseColorFactor.g, slot.BaseColorFactor.b);
            Vector3 specF0         = Vector3.Lerp(dielectricF0, metalBaseColor, met);
            slot.SpecularColor     = new Color(specF0.x, specF0.y, specF0.z, 1f);

            slot.UseSpecularGlossModel  = false;
            slot.ThinSurface            = false;
            slot.MetalnessInRedChannel  = slot.OcclusionRoughnessMetallicTexture != null;

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
}
