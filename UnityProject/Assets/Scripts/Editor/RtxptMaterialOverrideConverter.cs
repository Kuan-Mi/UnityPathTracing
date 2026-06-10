#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace PathTracing
{
    /// <summary>
    /// Converts the RTXPT material JSON files already living under
    /// <c>Assets/RtxptMaterialOverrides</c> into sibling <see cref="RtxptMaterial"/>
    /// <c>.asset</c> files, in place, so their values can be tweaked in the Inspector.
    ///
    /// Unlike <see cref="RtxptMaterialOverrideBatchImport"/> (which imports from an
    /// arbitrary on-disk folder into a chosen destination), this command operates on
    /// the JSON that is already part of the project: each <c>Foo.material.json</c>
    /// becomes a <c>Foo.asset</c> next to it. Texture paths are resolved against
    /// <see cref="TextureRoot"/> via the shared <see cref="RtxptTextureResolver"/>.
    ///
    /// Open via  RTXPT ▸ Convert RtxptMaterialOverrides JSON → Assets
    /// </summary>
    public static class RtxptMaterialOverrideConverter
    {
        private const string SourceFolder = "Assets/RtxptMaterialOverrides";
        private const string TextureRoot  = "Assets/Art/RTXPTAssets";

        [MenuItem("RTXPT/Convert RtxptMaterialOverrides JSON → Assets")]
        public static void Convert()
        {
            if (!AssetDatabase.IsValidFolder(SourceFolder))
            {
                EditorUtility.DisplayDialog("Convert RTXPT Material Overrides",
                    $"Folder not found:\n{SourceFolder}", "OK");
                return;
            }

            // Gather every *.material.json under the folder (recursive).
            string[] jsonFiles = Directory.GetFiles(SourceFolder, "*.material.json", SearchOption.AllDirectories);
            if (jsonFiles.Length == 0)
            {
                EditorUtility.DisplayDialog("Convert RTXPT Material Overrides",
                    $"No *.material.json files found under {SourceFolder}.", "OK");
                return;
            }

            // DisplayDialogComplex: 0 = Skip existing, 1 = Cancel, 2 = Overwrite existing.
            int choice = EditorUtility.DisplayDialogComplex("Convert RTXPT Material Overrides",
                $"Found {jsonFiles.Length} JSON file(s) under {SourceFolder}.\n\n" +
                "Create a sibling .asset for each. How should existing assets be handled?",
                "Skip existing", "Cancel", "Overwrite existing");
            if (choice == 1) return;
            bool overwrite = choice == 2;

            int created = 0, updated = 0, skipped = 0, failed = 0, texMissing = 0;
            var errors        = new List<string>();
            var missingTexErr = new HashSet<string>();

            try
            {
                AssetDatabase.StartAssetEditing();

                for (int i = 0; i < jsonFiles.Length; i++)
                {
                    string jsonPath  = jsonFiles[i].Replace('\\', '/');
                    string dir       = Path.GetDirectoryName(jsonPath)!.Replace('\\', '/');
                    // Strip both ".json" and the ".material" sub-extension: Foo.material.json -> Foo.
                    string baseName  = Path.GetFileNameWithoutExtension(Path.GetFileNameWithoutExtension(jsonPath));
                    string assetPath = $"{dir}/{baseName}.asset";

                    if (EditorUtility.DisplayCancelableProgressBar("Converting RTXPT Material Overrides",
                            baseName, (float)i / jsonFiles.Length))
                        break;

                    var existing = AssetDatabase.LoadAssetAtPath<RtxptMaterial>(assetPath);
                    if (existing != null && !overwrite)
                    {
                        skipped++;
                        continue;
                    }

                    Func<RtxptTextureRef, Texture> resolver = texRef =>
                    {
                        var tex = RtxptTextureResolver.Resolve(TextureRoot, texRef.Path, out _);
                        if (tex == null && !string.IsNullOrEmpty(texRef.Path))
                        {
                            texMissing++;
                            missingTexErr.Add(texRef.Path);
                        }
                        return tex;
                    };

                    try
                    {
                        string json = File.ReadAllText(jsonPath);

                        if (existing != null)
                        {
                            // Overwrite values on the existing asset, preserving its GUID/references.
                            existing.LoadFromJson(json, resolver);
                            EditorUtility.SetDirty(existing);
                            updated++;
                        }
                        else
                        {
                            var asset = ScriptableObject.CreateInstance<RtxptMaterial>();
                            asset.LoadFromJson(json, resolver);
                            AssetDatabase.CreateAsset(asset, assetPath);
                            created++;
                        }
                    }
                    catch (Exception ex)
                    {
                        errors.Add($"{Path.GetFileName(jsonPath)}: {ex.Message}");
                        failed++;
                    }
                }
            }
            finally
            {
                AssetDatabase.StopAssetEditing();
                EditorUtility.ClearProgressBar();
                AssetDatabase.SaveAssets();
                AssetDatabase.Refresh();
            }

            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"Source : {SourceFolder}");
            sb.AppendLine($"Found  : {jsonFiles.Length} JSON file(s)");
            sb.AppendLine($"Created: {created}");
            if (updated > 0) sb.AppendLine($"Updated: {updated}");
            if (skipped > 0) sb.AppendLine($"Skipped: {skipped}  (already existed)");
            if (failed  > 0)
            {
                sb.AppendLine($"Failed : {failed}");
                foreach (var e in errors) sb.AppendLine($"  • {e}");
            }
            if (texMissing > 0)
            {
                sb.AppendLine();
                sb.AppendLine($"Textures unresolved: {texMissing} reference(s), {missingTexErr.Count} unique path(s) under '{TextureRoot}'.");
            }

            Debug.Log("[RtxptMaterialOverrideConverter]\n" + sb);
            EditorUtility.DisplayDialog("Convert RTXPT Material Overrides", sb.ToString().TrimEnd(), "OK");
        }
    }
}
#endif
