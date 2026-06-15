#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEngine;

namespace PathTracing
{
    /// <summary>
    /// Maps an RTXPT exporter-relative texture path (e.g. "Models\Kitchen\foo.dds") onto a Unity
    /// <see cref="Texture"/> asset that lives under a chosen project-relative root folder.
    ///
    /// Shared by the <see cref="RtxptMaterialJsonImporter"/> ScriptedImporter and the
    /// <see cref="RtxptMaterialOverrideBatchImport"/> window so both resolve paths identically.
    /// </summary>
    public static class RtxptTextureResolver
    {
        /// <summary>
        /// Resolves <paramref name="exporterRelativePath"/> against <paramref name="assetRoot"/>.
        /// Tries the exact combined path first, then falls back to a filename search under the root.
        /// </summary>
        /// <param name="assetRoot">Project-relative folder the JSON paths are relative to (e.g. "Assets/Art/RTXPTAssets").</param>
        /// <param name="exporterRelativePath">Backslash- or slash-separated path from the JSON.</param>
        /// <param name="resolvedAssetPath">The project-relative path of the texture that was found, or null.</param>
        /// <returns>The loaded <see cref="Texture"/>, or null if nothing matched.</returns>
        public static Texture Resolve(string assetRoot, string exporterRelativePath, out string resolvedAssetPath)
        {
            resolvedAssetPath = null;
            if (string.IsNullOrEmpty(assetRoot) || string.IsNullOrEmpty(exporterRelativePath))
                return null;

            string root = assetRoot.Replace('\\', '/').TrimEnd('/');
            string rel  = exporterRelativePath.Replace('\\', '/').TrimStart('/');

            // 1. Exact path under the root.
            string combined = root + "/" + rel;
            var    tex      = AssetDatabase.LoadAssetAtPath<Texture>(combined);
            if (tex != null)
            {
                resolvedAssetPath = combined;
                return tex;
            }

            combined = Path.ChangeExtension(combined, "png");
            tex      = AssetDatabase.LoadAssetAtPath<Texture>(combined);
            if (tex != null)
            {
                resolvedAssetPath = combined;
                return tex;
            }

            combined = Path.ChangeExtension(combined, "jpg");
            tex      = AssetDatabase.LoadAssetAtPath<Texture>(combined);
            if (tex != null)
            {
                resolvedAssetPath = combined;
                return tex;
            }

            // 2. Fallback: search by filename anywhere under the root (handles relocated folders).
            string fileName = Path.GetFileNameWithoutExtension(rel);
            if (string.IsNullOrEmpty(fileName) || !AssetDatabase.IsValidFolder(root))
                return null;

            foreach (string guid in AssetDatabase.FindAssets($"\"{fileName}\" t:Texture", new[] { root }))
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                if (!string.Equals(Path.GetFileNameWithoutExtension(path), fileName,
                        System.StringComparison.OrdinalIgnoreCase))
                    continue;
                tex = AssetDatabase.LoadAssetAtPath<Texture>(path);
                if (tex != null)
                {
                    resolvedAssetPath = path;
                    return tex;
                }
            }

            return null;
        }
    }
}
#endif