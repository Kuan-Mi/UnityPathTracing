using System;
using UnityEngine;

namespace PathTracing
{
    // =========================================================================
    // RtxptMaterial  —  one RTXPT material asset
    // Mirrors PTMaterial in MaterialPT.h. Holds the scalar/flag/colour fields plus
    // Unity texture references. Created from RTXPT material JSON (directly via the
    // *.material.json ScriptedImporter, or baked from a Unity material) and shared
    // across renderers / sub-meshes by a RtxptRenderer component.
    // =========================================================================

    [CreateAssetMenu(menuName = "RTXPT/Material", fileName = "RtxptMaterial")]
    public class RtxptMaterial : ScriptableObject
    {
        [Header("Textures")]
        public Texture BaseOrDiffuseTexture;
        public Texture OcclusionRoughnessMetallicTexture; // R=Occlusion G=Roughness B=Metalness  (or spec-gloss fallback)
        public Texture NormalTexture;
        public Texture EmissiveTexture;
        public Texture TransmissionTexture;

        [Header("Texture Toggles")]
        public bool EnableBaseTexture                       = true;
        public bool EnableOcclusionRoughnessMetallicTexture = true;
        public bool EnableNormalTexture                     = true;
        public bool EnableEmissiveTexture                   = true;
        public bool EnableTransmissionTexture               = true;

        [Header("Base")]
        [ColorUsage(false, false)]
        public Color BaseColorFactor = Color.white;     // metal-rough: base color; spec-gloss: diffuse color

        [Range(0f, 1f)]
        public float Metalness = 0f;

        [Range(0f, 1f)]
        public float Roughness = 0.5f;

        public Color SpecularColor = new Color(0.04f, 0.04f, 0.04f, 1f); // spec-gloss: F0

        [Header("Emissive")]
        public Color EmissiveColor     = Color.black;
        [Min(0f)]
        public float EmissiveIntensity = 1f;

        [Header("Surface")]
        [Range(0f, 1f)]
        public float Opacity = 1f;

        [Range(0f, 1f)]
        public float AlphaCutoff = 0f;

        public bool EnableAlphaTesting = false;

        [Range(1f, 3f)]
        public float IoR = 1.5f;

        [Min(0f)]
        public float NormalTextureScale = 1f;

        [Header("Transmission")]
        public bool EnableTransmission = false;

        [Range(0f, 1f)]
        public float TransmissionFactor = 0f;

        [Range(0f, 1f)]
        public float DiffuseTransmissionFactor = 0f;

        [Min(0f)]
        public float ThicknessFactor = 0f;

        [Header("Volume")]
        public Color VolumeAttenuationColor = Color.white;

        [Min(0f)]
        public float VolumeAttenuationDistance = float.MaxValue;

        [Header("Shadows")]
        [Range(0f, 0.25f)]
        public float ShadowNoLFadeout = 0f;

        [Header("PT Flags")]
        public bool UseSpecularGlossModel  = false;
        public bool MetalnessInRedChannel  = false;
        public bool ThinSurface            = false;
        public bool ExcludeFromNEE         = false;
        public bool PSDExclude             = true;
        public bool IgnoreMeshTangentSpace = false;
        public bool EnableAsAnalyticLightProxy = false;
        public bool UseDonutEmissiveIntensity  = false;
        public bool SkipRender                 = false;

        /// <summary>
        /// Dominant delta lobe for Stable Planes.
        /// -1 = None (surface itself), 0 = Surface, 1 = Transparency, 2 = Reflection.
        /// </summary>
        [Range(-1, 6)]
        public int PSDDominantDeltaLobe = -1;

        /// <summary>Dielectric nesting priority (0–15).</summary>
        [Range(0, 15)]
        public int NestedPriority = 14;

        /// <summary>Controls motion-vector blocking at surface type boundaries (0–3).</summary>
        [Range(0, 3)]
        public int PSDBlockMotionVectorsAtSurfaceType = 0;

        /// <summary>Fired when the asset's data changes — subscribers (renderers) use this to mark themselves dirty.</summary>
        public event Action Modified;

        public void MarkModified() => Modified?.Invoke();

#if UNITY_EDITOR
        private void OnValidate() => Modified?.Invoke();
#endif

        // ---- JSON import ----

        /// <summary>
        /// Overwrites all non-texture fields from a RTXPT material JSON string.
        /// When <paramref name="textureResolver"/> is null, existing texture assignments are preserved.
        /// When supplied, each texture path present in the JSON is resolved and assigned (paths the
        /// resolver cannot find become null); texture entries absent from the JSON are left unchanged.
        /// </summary>
        public void LoadFromJson(string json, Func<RtxptTextureRef, Texture> textureResolver = null)
        {
            ApplyJsonData(JsonUtility.FromJson<SlotJson>(json), textureResolver);
        }

        private void ApplyJsonData(SlotJson d, Func<RtxptTextureRef, Texture> textureResolver = null)
        {
            BaseColorFactor             = Rgb(d.BaseOrDiffuseColor, 1f);
            SpecularColor               = Rgb(d.SpecularColor, 1f);
            EmissiveColor               = Rgb(d.EmissiveColor, 0f);
            EmissiveIntensity           = d.EmissiveIntensity;
            Opacity                     = d.Opacity;
            Metalness                   = d.Metalness;
            Roughness                   = d.Roughness;
            EnableAlphaTesting          = d.EnableAlphaTesting;
            AlphaCutoff                 = d.AlphaCutoff;
            NormalTextureScale          = d.NormalTextureScale;
            IoR                         = d.IoR;
            EnableTransmission          = d.EnableTransmission;
            TransmissionFactor          = d.TransmissionFactor;
            DiffuseTransmissionFactor   = d.DiffuseTransmissionFactor;
            VolumeAttenuationColor      = Rgb(d.VolumeAttenuationColor, 1f);
            VolumeAttenuationDistance   = d.VolumeAttenuationDistance;
            ShadowNoLFadeout            = d.ShadowNoLFadeout;
            EnableBaseTexture                       = d.EnableBaseTexture;
            EnableOcclusionRoughnessMetallicTexture = d.EnableOcclusionRoughnessMetallicTexture;
            EnableNormalTexture                     = d.EnableNormalTexture;
            EnableEmissiveTexture                   = d.EnableEmissiveTexture;
            EnableTransmissionTexture               = d.EnableTransmissionTexture;
            UseSpecularGlossModel       = d.UseSpecularGlossModel;
            MetalnessInRedChannel       = d.MetalnessInRedChannel;
            ThinSurface                 = d.ThinSurface;
            ExcludeFromNEE              = d.ExcludeFromNEE;
            PSDExclude                  = d.PSDExclude;
            IgnoreMeshTangentSpace      = d.IgnoreMeshTangentSpace;
            EnableAsAnalyticLightProxy  = d.EnableAsAnalyticLightProxy;
            UseDonutEmissiveIntensity   = d.UseDonutEmissiveIntensity;
            SkipRender                  = d.SkipRender;
            PSDDominantDeltaLobe        = d.PSDDominantDeltaLobe;
            NestedPriority              = d.NestedPriority;
            PSDBlockMotionVectorsAtSurfaceType = d.PSDBlockMotionVectorsAtSurfaceType;

            if (textureResolver != null)
            {
                ResolveTex(d.BaseTexture,                       textureResolver, ref BaseOrDiffuseTexture);
                ResolveTex(d.OcclusionRoughnessMetallicTexture, textureResolver, ref OcclusionRoughnessMetallicTexture);
                ResolveTex(d.NormalTexture,                     textureResolver, ref NormalTexture);
                ResolveTex(d.EmissiveTexture,                   textureResolver, ref EmissiveTexture);
                ResolveTex(d.TransmissionTexture,               textureResolver, ref TransmissionTexture);
            }
        }

        // Resolves one JSON texture entry into a Unity asset. Entries absent from the JSON (null or
        // empty path) leave the existing assignment untouched; present paths are always (re)assigned.
        private static void ResolveTex(TexJson tex, Func<RtxptTextureRef, Texture> resolver, ref Texture dst)
        {
            if (tex == null || string.IsNullOrEmpty(tex.path)) return;
            dst = resolver(new RtxptTextureRef { Path = tex.path, IsSRGB = tex.sRGB, IsNormalMap = tex.NormalMap });
        }

        private static Color Rgb(float[] arr, float a)
            => arr != null && arr.Length >= 3 ? new Color(arr[0], arr[1], arr[2], a) : new Color(0f, 0f, 0f, a);

        // Mirror of the JSON schema.
        [Serializable]
        private class SlotJson
        {
            public float   AlphaCutoff                             = 0f;
            public float[] BaseOrDiffuseColor                      = { 1f, 1f, 1f };
            public float   DiffuseTransmissionFactor               = 0f;
            public float[] EmissiveColor                           = { 0f, 0f, 0f };
            public float   EmissiveIntensity                       = 1f;
            public bool    EnableAlphaTesting                      = false;
            public bool    EnableAsAnalyticLightProxy              = false;
            public bool    EnableBaseTexture                       = true;
            public bool    EnableEmissiveTexture                   = true;
            public bool    EnableNormalTexture                     = true;
            public bool    EnableOcclusionRoughnessMetallicTexture = true;
            public bool    EnableTransmission                      = false;
            public bool    EnableTransmissionTexture               = true;
            public bool    ExcludeFromNEE                          = false;
            public bool    IgnoreMeshTangentSpace                  = false;
            public float   IoR                                     = 1.5f;
            public float   Metalness                               = 0f;
            public bool    MetalnessInRedChannel                   = false;
            public int     NestedPriority                          = 14;
            public float   NormalTextureScale                      = 1f;
            public float   Opacity                                 = 1f;
            public int     PSDBlockMotionVectorsAtSurfaceType      = 0;
            public int     PSDDominantDeltaLobe                    = -1;
            public bool    PSDExclude                              = true;
            public float   Roughness                               = 0.5f;
            public float   ShadowNoLFadeout                        = 0f;
            public bool    SkipRender                              = false;
            public float[] SpecularColor                           = { 0.04f, 0.04f, 0.04f };
            public bool    ThinSurface                             = false;
            public float   TransmissionFactor                      = 0f;
            public bool    UseDonutEmissiveIntensity               = false;
            public bool    UseSpecularGlossModel                   = false;
            public float[] VolumeAttenuationColor                  = { 1f, 1f, 1f };
            public float   VolumeAttenuationDistance               = float.MaxValue;

            // Texture path entries. Keys mirror the RTXPT exporter; absent entries deserialize with
            // an empty path and are treated as "no texture" by ResolveTex.
            public TexJson BaseTexture;
            public TexJson OcclusionRoughnessMetallicTexture;
            public TexJson NormalTexture;
            public TexJson EmissiveTexture;
            public TexJson TransmissionTexture;
        }

        // One texture reference as written by the RTXPT material exporter.
        [Serializable]
        private class TexJson
        {
            public bool   NormalMap = false;
            public string path      = "";   // exporter-relative, backslash-separated, e.g. "Models\\Kitchen\\foo.dds"
            public bool   sRGB      = false;
        }
    }

    // =========================================================================
    // RtxptTextureRef  —  a texture path entry parsed from RTXPT material JSON
    // =========================================================================

    /// <summary>
    /// Describes one texture slot as written by the RTXPT material exporter. Passed to the texture
    /// resolver delegate of <see cref="RtxptMaterial.LoadFromJson"/> so editor code can map the
    /// exporter-relative <see cref="Path"/> onto a Unity <see cref="Texture"/> asset.
    /// </summary>
    public struct RtxptTextureRef
    {
        /// <summary>Exporter-relative, backslash-separated path, e.g. "Models\Kitchen\foo.dds".</summary>
        public string Path;
        /// <summary>True if the source texture is sRGB-encoded (color), false for linear data.</summary>
        public bool   IsSRGB;
        /// <summary>True if the texture is a normal map.</summary>
        public bool   IsNormalMap;
    }
}
