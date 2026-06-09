#if UNITY_EDITOR
using System.Collections.Generic;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace PathTracing
{
    /// <summary>
    /// Scans every <see cref="RtxptMaterial"/> asset in the project and verifies that the texture
    /// assigned to its <see cref="RtxptMaterial.NormalTexture"/> slot is imported as a Normal Map
    /// (<see cref="TextureImporterType.NormalMap"/>). Mismatches are reported and, when fixing, the
    /// importer type is corrected and the texture re-imported.
    ///
    /// Open via  RTXPT ▸ Validate Normal Map Texture Types…
    /// </summary>
    public static class RtxptNormalMapFixer
    {
        [MenuItem("RTXPT/Validate Normal Map Texture Types…")]
        private static void Validate() => Run(fix: false);

        [MenuItem("RTXPT/Fix Normal Map Texture Types")]
        private static void Fix() => Run(fix: true);

        private static void Run(bool fix)
        {
            string[] guids = AssetDatabase.FindAssets("t:RtxptMaterial");

            // Dedupe by texture asset path: many materials may share one normal map.
            var checkedPaths = new HashSet<string>();
            var wrong        = new List<string>(); // "<texPath>  (used by <materialName>)"
            int fixedCount   = 0;
            int okCount      = 0;
            int missingImporter = 0;

            try
            {
                for (int i = 0; i < guids.Length; i++)
                {
                    string matPath = AssetDatabase.GUIDToAssetPath(guids[i]);
                    var    mat     = AssetDatabase.LoadAssetAtPath<RtxptMaterial>(matPath);
                    if (mat == null || mat.NormalTexture == null) continue;

                    string texPath = AssetDatabase.GetAssetPath(mat.NormalTexture);
                    if (string.IsNullOrEmpty(texPath) || !checkedPaths.Add(texPath)) continue;

                    EditorUtility.DisplayProgressBar(
                        fix ? "Fixing normal map types" : "Validating normal map types",
                        texPath, (float)i / Mathf.Max(1, guids.Length));

                    var importer = AssetImporter.GetAtPath(texPath) as TextureImporter;
                    if (importer == null)
                    {
                        // E.g. a render texture or non-2D texture asset with no TextureImporter.
                        missingImporter++;
                        wrong.Add($"{texPath}  (no TextureImporter — used by {mat.name})");
                        continue;
                    }

                    if (importer.textureType == TextureImporterType.NormalMap)
                    {
                        okCount++;
                        continue;
                    }

                    wrong.Add($"{texPath}  ({importer.textureType} → NormalMap — used by {mat.name})");

                    if (fix)
                    {
                        importer.textureType = TextureImporterType.NormalMap;
                        importer.SaveAndReimport();
                        fixedCount++;
                    }
                }
            }
            finally
            {
                EditorUtility.ClearProgressBar();
            }

            var sb = new StringBuilder();
            sb.AppendLine($"[RtxptNormalMapFixer] {(fix ? "Fix" : "Validation")} complete.");
            sb.AppendLine($"  Materials scanned : {guids.Length}");
            sb.AppendLine($"  Unique normal maps: {checkedPaths.Count}");
            sb.AppendLine($"  Already correct   : {okCount}");
            sb.AppendLine($"  Wrong type        : {wrong.Count}");
            if (fix) sb.AppendLine($"  Fixed             : {fixedCount}");
            if (missingImporter > 0) sb.AppendLine($"  No TextureImporter : {missingImporter}");
            foreach (string w in wrong) sb.AppendLine($"    - {w}");

            Debug.Log(sb.ToString().TrimEnd());

            EditorUtility.DisplayDialog(
                fix ? "Fix Normal Map Texture Types" : "Validate Normal Map Texture Types",
                fix
                    ? $"Scanned {checkedPaths.Count} unique normal maps.\n" +
                      $"Fixed {fixedCount}, already correct {okCount}.\n\nSee Console for details."
                    : $"Scanned {checkedPaths.Count} unique normal maps.\n" +
                      $"{wrong.Count} not set to Normal Map, {okCount} correct.\n\nSee Console for details.",
                "OK");
        }
    }
}
#endif
