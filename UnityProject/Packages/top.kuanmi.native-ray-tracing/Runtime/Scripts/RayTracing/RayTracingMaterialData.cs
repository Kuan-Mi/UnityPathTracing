using System;
using System.Collections.Generic;
using UnityEngine;

namespace NativeRender
{
    /// <summary>
    /// Pre-computed material properties for one submesh.
    /// Cached by <see cref="NativeRayTracingTarget"/> and <see cref="NativeRayTracingSkinnedTarget"/>;
    /// consumed by <see cref="NRDSampleResource"/> during scene build and incremental add.
    /// </summary>
    [Serializable]
    public class SubmeshMaterialData
    {
        /// <summary>Source material — used as the cache key in NRDSampleResource's bindless texture atlas.</summary>
        public Material material;

        public bool isTransparent;
        public bool isEmissive;

        /// <summary>
        /// 4 native texture pointers in gIn_Textures order:
        /// [0] BaseMap,  [1] MetallicGlossMap,  [2] BumpMap,  [3] EmissionMap.
        /// Placeholder textures fill missing slots (never IntPtr.Zero).
        /// </summary>
        public IntPtr[] texturePtrs; // length = 4

        // Scalar fields written into InstanceDataNRD.
        public Color baseColor;
        public Color emissionColor;
        public float metallic;
        public float roughnessScale;  // 1 − smoothness (URP) or roughnessFactor (glTF)
        public float normalScale;
    }

    /// <summary>
    /// A group of submeshes that share the same (isTransparent, isEmissive) pair.
    /// Each group is registered as a separate TLAS entry by <see cref="NRDSampleResource"/>.
    /// </summary>
    
    [Serializable]
    public class SubmeshGroupDesc
    {
        public bool isTransparent;
        public bool isEmissive;

        /// <summary>Indices into the Mesh's subMeshCount — also indices into SubmeshMaterialInfos.</summary>
        public int[] submeshIndices;

        /// <summary>Per-submesh material data, parallel to <see cref="submeshIndices"/>.</summary>
        public SubmeshMaterialData[] materialDatas;
    }

    /// <summary>
    /// Static helpers for classifying materials and building <see cref="SubmeshMaterialData"/>.
    /// Shared between <see cref="NativeRayTracingTarget"/>, <see cref="NativeRayTracingSkinnedTarget"/>,
    /// and <see cref="NRDSampleResource"/>.
    /// </summary>
    internal static class RayTracingMaterialHelper
    {
        // ── Placeholder textures ──────────────────────────────────────────────

        private static Texture2D _phWhite;
        private static Texture2D _phFlatNormal;
        private static Texture2D _phBlack;

        public static Texture2D White
        {
            get
            {
                if (_phWhite == null)
                {
                    _phWhite = new Texture2D(1, 1, TextureFormat.RGBA32, false, true)
                        { name = "NRD_Placeholder_White", hideFlags = HideFlags.HideAndDontSave };
                    _phWhite.SetPixel(0, 0, Color.white);
                    _phWhite.Apply(false, true);
                }
                return _phWhite;
            }
        }

        public static Texture2D FlatNormal
        {
            get
            {
                if (_phFlatNormal == null)
                {
                    _phFlatNormal = new Texture2D(1, 1, TextureFormat.RGBA32, false, true)
                        { name = "NRD_Placeholder_FlatNormal", hideFlags = HideFlags.HideAndDontSave };
                    _phFlatNormal.SetPixel(0, 0, new Color(0.5f, 0.5f, 1f, 1f));
                    _phFlatNormal.Apply(false, true);
                }
                return _phFlatNormal;
            }
        }

        public static Texture2D Black
        {
            get
            {
                if (_phBlack == null)
                {
                    _phBlack = new Texture2D(1, 1, TextureFormat.RGBA32, false, true)
                        { name = "NRD_Placeholder_Black", hideFlags = HideFlags.HideAndDontSave };
                    _phBlack.SetPixel(0, 0, Color.black);
                    _phBlack.Apply(false, true);
                }
                return _phBlack;
            }
        }

        // ── Material classification ───────────────────────────────────────────

        public static bool IsMaterialTransparent(Material mat)
        {
            if (mat == null) return false;

            if (mat.shader.name is "Universal Render Pipeline/Lit" or "RayTracing/Lit")
                return mat.HasProperty("_Surface") && mat.GetFloat("_Surface") > 0.5f;

            if (mat.shader.name == "Shader Graphs/glTF-pbrMetallicRoughness")
                return mat.IsKeywordEnabled("_SURFACE_TYPE_TRANSPARENT");

            Debug.LogWarning($"[RayTracingMaterialHelper] Unrecognized shader '{mat.shader.name}' when checking transparency for material '{mat.name}'. Defaulting to opaque.");
            return false;
        }

        public static bool IsMaterialEmissive(Material mat)
        {
            if (mat == null) return false;

            if (mat.shader.name is "Universal Render Pipeline/Lit" or "RayTracing/Lit")
            {
                if (mat.HasProperty("_EmissionColor"))
                {
                    Color e = mat.GetColor("_EmissionColor").linear;
                    if (e.r > 0f || e.g > 0f || e.b > 0f) return true;
                }
                return mat.HasProperty("_EmissionMap") && mat.GetTexture("_EmissionMap") != null;
            }

            if (mat.shader.name == "Shader Graphs/glTF-pbrMetallicRoughness")
                return mat.IsKeywordEnabled("_EMISSIVE");

            Debug.LogWarning($"[RayTracingMaterialHelper] Unrecognized shader '{mat.shader.name}' when checking emissiveness for material '{mat.name}'. Defaulting to non-emissive.");
            return false;
        }

        // ── Private property helpers ──────────────────────────────────────────

        private static Texture TryGetTex(Material mat, string prop) =>
            mat != null && mat.HasProperty(prop) ? mat.GetTexture(prop) : null;

        private static Color TryGetColor(Material mat, string prop, Color def) =>
            mat != null && mat.HasProperty(prop) ? mat.GetColor(prop).linear : def;

        private static float TryGetFloat(Material mat, string prop, float def) =>
            mat != null && mat.HasProperty(prop) ? mat.GetFloat(prop) : def;

        private static IntPtr ResolveTexPtr(Texture tex, Texture2D placeholder) =>
            (tex != null ? tex : placeholder).GetNativeTexturePtr();

        // ── Main SubmeshMaterialData builder ─────────────────────────────────

        /// <summary>
        /// Builds a fully-populated <see cref="SubmeshMaterialData"/> from a single Material.
        /// Must be called on the main thread (accesses native texture pointers).
        /// </summary>
        public static SubmeshMaterialData BuildSubmeshMaterialData(Material mat)
        {
            var data = new SubmeshMaterialData
            {
                material      = mat,
                isTransparent = IsMaterialTransparent(mat),
                isEmissive    = IsMaterialEmissive(mat),
                texturePtrs   = new IntPtr[4],
            };

            if (mat != null && mat.shader.name == "Shader Graphs/glTF-pbrMetallicRoughness")
            {
                data.baseColor      = TryGetColor(mat, "baseColorFactor", Color.white);
                data.emissionColor  = TryGetColor(mat, "emissiveFactor", Color.black);
                data.metallic       = TryGetFloat(mat, "metallicFactor", 0f);
                data.roughnessScale = TryGetFloat(mat, "roughnessFactor", 0.5f);
                data.normalScale    = 1f;

                data.texturePtrs[0] = ResolveTexPtr(TryGetTex(mat, "baseColorTexture"),          White);
                data.texturePtrs[1] = ResolveTexPtr(TryGetTex(mat, "metallicRoughnessTexture"),  Black);
                data.texturePtrs[2] = ResolveTexPtr(TryGetTex(mat, "normalTexture"),             FlatNormal);
                data.texturePtrs[3] = ResolveTexPtr(TryGetTex(mat, "emissiveTexture"),           Black);
            }
            else
            {
                // URP/Lit, RayTracing/Lit, and unknown-shader fallback.
                data.baseColor      = TryGetColor(mat, "_BaseColor", Color.white);
                data.emissionColor  = TryGetColor(mat, "_EmissionColor", Color.black);
                data.metallic       = TryGetFloat(mat, "_Metallic", 0f);
                float smooth        = TryGetFloat(mat, "_Smoothness", 0.5f);
                data.roughnessScale = 1f - smooth;
                data.normalScale    = TryGetFloat(mat, "_BumpScale", 1f);

                data.texturePtrs[0] = ResolveTexPtr(TryGetTex(mat, "_BaseMap"),           White);
                data.texturePtrs[1] = ResolveTexPtr(TryGetTex(mat, "_MetallicGlossMap"),  Black);
                data.texturePtrs[2] = ResolveTexPtr(TryGetTex(mat, "_BumpMap"),           FlatNormal);
                data.texturePtrs[3] = ResolveTexPtr(TryGetTex(mat, "_EmissionMap"),       Black);
            }

            return data;
        }

        // ── Submesh group classifier ─────────────────────────────────────────

        /// <summary>
        /// Groups per-submesh material data by (isTransparent, isEmissive) pair.
        /// Entries in each group's materialDatas are parallel to submeshIndices.
        /// </summary>
        public static SubmeshGroupDesc[] BuildSubmeshGroupDescs(SubmeshMaterialData[] infos)
        {
            var dict = new Dictionary<(bool, bool), (List<int> indices, List<SubmeshMaterialData> datas)>();

            for (int s = 0; s < infos.Length; s++)
            {
                var d   = infos[s];
                var key = (d.isTransparent, d.isEmissive);
                if (!dict.TryGetValue(key, out var pair))
                {
                    pair = (new List<int>(), new List<SubmeshMaterialData>());
                    dict[key] = pair;
                }
                pair.indices.Add(s);
                pair.datas.Add(d);
            }

            var result = new SubmeshGroupDesc[dict.Count];
            int gi     = 0;
            foreach (var kv in dict)
            {
                result[gi++] = new SubmeshGroupDesc
                {
                    isTransparent  = kv.Key.Item1,
                    isEmissive     = kv.Key.Item2,
                    submeshIndices = kv.Value.indices.ToArray(),
                    materialDatas  = kv.Value.datas.ToArray(),
                };
            }
            return result;
        }
    }
}
