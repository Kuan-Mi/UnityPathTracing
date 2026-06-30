#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;
using UnityEditor.AssetImporters;
using UnityEngine;

namespace PathTracing
{
    /// <summary>
    /// Imports RTXPT *.material.json files directly as <see cref="RtxptMaterial"/> objects,
    /// so each JSON file in the project IS an assignable RTXPT material (no separate generated .asset).
    /// Texture path entries are resolved to project textures via <see cref="RtxptTextureResolver"/>.
    ///
    /// The version number (first ctor arg) must be bumped whenever the import logic changes so Unity
    /// re-imports existing assets.
    /// </summary>
    [ScriptedImporter(2, "material.json")]
    public class RtxptMaterialJsonImporter : ScriptedImporter
    {
        [Tooltip("Project-relative folder the texture 'path' fields are resolved against.\n" +
                 "Leave empty to auto-use the parent of this file's folder " +
                 "(e.g. a file in Assets/Art/RTXPTAssets/Materials resolves textures under Assets/Art/RTXPTAssets).")]
        public string textureRoot = "";

        public override void OnImportAsset(AssetImportContext ctx)
        {
            var asset = ScriptableObject.CreateInstance<RtxptMaterial>();

            string json;
            try
            {
                json = File.ReadAllText(ctx.assetPath);
            }
            catch (Exception ex)
            {
                ctx.LogImportError($"Failed to read '{ctx.assetPath}': {ex.Message}");
                Finish(ctx, asset);
                return;
            }

            string root = ResolveRoot(ctx.assetPath, textureRoot);

            try
            {
                asset.LoadFromJson(json, texRef =>
                {
                    var tex = RtxptTextureResolver.Resolve(root, texRef.Path, out string resolvedPath);
                    if (resolvedPath != null)
                        // Re-resolve this material if the referenced texture is moved/renamed/replaced.
                        ctx.DependsOnSourceAsset(resolvedPath);
                    else if (!string.IsNullOrEmpty(texRef.Path))
                        ctx.LogImportWarning($"Texture not found under '{root}': {texRef.Path}");
                    return tex;
                });
            }
            catch (Exception ex)
            {
                ctx.LogImportError($"Failed to parse RTXPT material JSON '{ctx.assetPath}': {ex.Message}");
            }

            Finish(ctx, asset);
        }

        private static void Finish(AssetImportContext ctx, RtxptMaterial asset)
        {
            ctx.AddObjectToAsset("material", asset);
            ctx.SetMainObject(asset);
        }

        // Empty textureRoot => parent folder of the file's directory.
        private static string ResolveRoot(string assetPath, string textureRoot)
        {
            if (!string.IsNullOrWhiteSpace(textureRoot))
                return textureRoot.Replace('\\', '/').TrimEnd('/');

            string dir    = Path.GetDirectoryName(assetPath); // .../Materials
            string parent = Path.GetDirectoryName(dir); // .../RTXPTAssets
            return (parent ?? dir ?? "Assets").Replace('\\', '/').TrimEnd('/');
        }
    }
}
#endif