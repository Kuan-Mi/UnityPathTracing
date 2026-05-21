using System;
using System.Collections.Generic;
using UnityEngine;

namespace PathTracing
{
    // =========================================================================
    // RtxptMaterialSlot  —  per-sub-mesh material override
    // =========================================================================

    [Serializable]
    public class RtxptMaterialSlot
    {
        [Header("Source (read-only reference)")]
        public Material SourceMaterial;

        [Header("Textures")]
        public Texture BaseOrDiffuseTexture;
        public Texture NormalTexture;
        public Texture MetalRoughTexture;   // R=Metalness, G=Roughness (URP convention)
        public Texture EmissiveTexture;
        public Texture OcclusionTexture;
        public Texture TransmissionTexture;

        [Header("Colors")]
        public Color BaseOrDiffuseColor   = Color.white;
        public Color SpecularColor        = new Color(0.04f, 0.04f, 0.04f, 1f);
        public Color EmissiveColor        = Color.black;
        public Color VolumeAttenuationColor = Color.white;

        [Header("PBR Scalars")]
        [Range(0f, 1f)]  public float Roughness              = 0.5f;
        [Range(0f, 1f)]  public float Metalness              = 0f;
        [Range(0f, 1f)]  public float Opacity                = 1f;
        [Min(0f)]        public float NormalTextureScale     = 1f;
        [Range(0f, 1f)]  public float AlphaCutoff            = 0f;
        [Range(1f, 3f)]  public float IoR                    = 1.5f;

        [Header("Transmission")]
        [Range(0f, 1f)]  public float TransmissionFactor          = 0f;
        [Range(0f, 1f)]  public float DiffuseTransmissionFactor   = 0f;
        [Min(0f)]        public float ThicknessFactor             = 0f;
        [Min(0f)]        public float VolumeAttenuationDistance   = float.MaxValue;

        [Header("Shadows")]
        [Range(0f, 0.25f)] public float ShadowNoLFadeout = 0f;

        [Header("PT Flags")]
        public bool ThinSurface             = true;
        public bool MetalnessInRedChannel   = false;
        public bool PSDExclude             = false;
        public bool IgnoreMeshTangentSpace  = false;

        /// <summary>
        /// Dominant delta lobe for Stable Planes.
        /// -1 = None (surface itself), 0 = Surface, 1 = Transparency, 2 = Reflection.
        /// Packed as (value+1) into bits [27:24].
        /// </summary>
        [Range(-1, 6)] public int PSDDominantDeltaLobe = -1;

        /// <summary>
        /// Dielectric nesting priority (0–15). Used for overlapping IOR regions.
        /// </summary>
        [Range(0, 15)] public int NestedPriority = 0;

        /// <summary>
        /// Controls motion-vector blocking at surface type boundaries (0–3).
        /// </summary>
        [Range(0, 3)] public int PSDBlockMotionVectorsAtSurfaceType = 0;
    }

    // =========================================================================
    // NativeRtxptMaterialOverride  —  attach to any MeshRenderer
    // =========================================================================

    /// <summary>
    /// Per-renderer RTXPT material override component.
    ///
    /// Attach to any <see cref="MeshRenderer"/> to take manual control of the
    /// RTXPT material properties that <see cref="NativeRtxptGPUScene"/> sends to
    /// the GPU. If this component is present on the renderer, its <see cref="Slots"/>
    /// values are used instead of auto-deriving them from the Unity material.
    ///
    /// Use the "Bake from Renderer" button in the Inspector to populate the slots
    /// from the current Unity materials as a starting point, then tweak as needed.
    /// </summary>
    [RequireComponent(typeof(MeshRenderer))]
    [DisallowMultipleComponent]
    public class NativeRtxptMaterialOverride : MonoBehaviour
    {
        [Tooltip("One slot per sub-mesh. Index matches MeshRenderer.sharedMaterials.")]
        public List<RtxptMaterialSlot> Slots = new();

        // Called by the Editor button and also useful at runtime.
        public void BakeFromRenderer()
        {
            var mr   = GetComponent<MeshRenderer>();
            var mf   = GetComponent<MeshFilter>();
            if (mr == null) return;

            Material[] mats      = mr.sharedMaterials ?? Array.Empty<Material>();
            int        subMeshCnt = mf != null && mf.sharedMesh != null ? mf.sharedMesh.subMeshCount : mats.Length;

            // Keep existing slots if count matches; otherwise rebuild.
            if (Slots.Count != subMeshCnt)
                Slots = new List<RtxptMaterialSlot>(subMeshCnt);

            while (Slots.Count < subMeshCnt)
                Slots.Add(new RtxptMaterialSlot());

            for (int s = 0; s < subMeshCnt; s++)
            {
                Material mat  = s < mats.Length ? mats[s] : (mats.Length > 0 ? mats[^1] : null);
                RtxptMaterialSlot slot = Slots[s] ?? new RtxptMaterialSlot();

                slot.SourceMaterial = mat;
                BakeSlotFromMaterial(slot, mat);
                Slots[s] = slot;
            }
        }

        private static void BakeSlotFromMaterial(RtxptMaterialSlot slot, Material mat)
        {
            if (mat == null) return;

            bool isGltf = mat.shader.name == "Shader Graphs/glTF-pbrMetallicRoughness";
            if (isGltf)
            {
                slot.BaseOrDiffuseTexture = TryGetTex(mat, "baseColorTexture");
                slot.NormalTexture        = TryGetTex(mat, "normalTexture");
                slot.MetalRoughTexture    = TryGetTex(mat, "metallicRoughnessTexture");
                slot.EmissiveTexture      = TryGetTex(mat, "emissiveTexture");
                slot.OcclusionTexture     = TryGetTex(mat, "occlusionTexture");

                Color baseC = TryGetColor(mat, "baseColorFactor", Color.white);
                slot.BaseOrDiffuseColor = baseC;
                slot.Opacity            = baseC.a;
                slot.EmissiveColor      = TryGetColor(mat, "emissiveFactor", Color.black);
                slot.Roughness          = TryGetFloat(mat, "roughnessFactor", 0.5f);
                slot.Metalness          = TryGetFloat(mat, "metallicFactor", 0f);
                slot.AlphaCutoff        = mat.IsKeywordEnabled("_ALPHATEST_ON") ? TryGetFloat(mat, "alphaCutoff", 0.5f) : 0f;
                slot.NormalTextureScale = 1f;
            }
            else
            {
                // URP Lit / unknown fallback
                slot.BaseOrDiffuseTexture = TryGetTex(mat, "_BaseMap");
                slot.NormalTexture        = TryGetTex(mat, "_BumpMap");
                slot.MetalRoughTexture    = TryGetTex(mat, "_MetallicGlossMap");
                slot.EmissiveTexture      = TryGetTex(mat, "_EmissionMap");
                slot.OcclusionTexture     = TryGetTex(mat, "_OcclusionMap");

                Color baseC = TryGetColor(mat, "_BaseColor", Color.white);
                slot.BaseOrDiffuseColor = baseC;
                slot.Opacity            = baseC.a;
                slot.EmissiveColor      = TryGetColor(mat, "_EmissionColor", Color.black);
                slot.Roughness          = 1f - TryGetFloat(mat, "_Smoothness", 0.5f);
                slot.Metalness          = TryGetFloat(mat, "_Metallic", 0f);
                slot.AlphaCutoff        = TryGetFloat(mat, "_Cutoff", 0f);
                slot.NormalTextureScale = TryGetFloat(mat, "_BumpScale", 1f);
            }

            // Derived fields
            float met = slot.Metalness;
            Vector3 dielectricF0   = new Vector3(0.04f, 0.04f, 0.04f);
            Vector3 metalBaseColor = new Vector3(slot.BaseOrDiffuseColor.r,
                                                  slot.BaseOrDiffuseColor.g,
                                                  slot.BaseOrDiffuseColor.b);
            Vector3 specF0 = Vector3.Lerp(dielectricF0, metalBaseColor, met);
            slot.SpecularColor = new Color(specF0.x, specF0.y, specF0.z, 1f);

            // PT flags
            slot.ThinSurface           = true;  // TransmissionFactor = 0 → always thin
            slot.MetalnessInRedChannel = slot.MetalRoughTexture != null;
        }

        private static Texture TryGetTex(Material mat, string prop)
            => mat != null && mat.HasProperty(prop) ? mat.GetTexture(prop) : null;

        private static Color TryGetColor(Material mat, string prop, Color fallback)
            => mat != null && mat.HasProperty(prop) ? mat.GetColor(prop) : fallback;

        private static float TryGetFloat(Material mat, string prop, float fallback)
            => mat != null && mat.HasProperty(prop) ? mat.GetFloat(prop) : fallback;
    }
}
