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
    /// A second pair of menu items does the same for the
    /// <see cref="RtxptMaterial.OcclusionRoughnessMetallicTexture"/> slot's colour space: RTXPT loads
    /// the metal-rough texture as <b>linear</b> (so the path tracer reads roughness from the raw
    /// <c>.g</c> channel) and only the spec-gloss model loads it as sRGB. If Unity imported the map
    /// with sRGB enabled (the default for colour PNG/JPG), the GPU gamma-decodes <c>.g</c> on sample
    /// and roughness/metalness come out wrong versus the C++ RTXPT reference. This fixer forces
    /// <see cref="TextureImporterSettings.sRGBTexture"/> to match the material's lighting model.
    ///
    /// Open via  RTXPT ▸ Validate Normal Map Texture Types…
    /// </summary>
    public static class RtxptNormalMapFixer
    {
        [MenuItem("RTXPT/Validate Normal Map Texture Types…")]
        private static void Validate() => Run(fix: false);

        [MenuItem("RTXPT/Fix Normal Map Texture Types")]
        private static void Fix() => Run(fix: true);

        [MenuItem("RTXPT/Validate Metal-Rough Texture Color Space…")]
        private static void ValidateOrm() => RunOrm(fix: false);

        [MenuItem("RTXPT/Fix Metal-Rough Texture Color Space")]
        private static void FixOrm() => RunOrm(fix: true);

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

        private static void RunOrm(bool fix)
        {
            string[] guids = AssetDatabase.FindAssets("t:RtxptMaterial");

            // Dedupe by texture asset path: many materials may share one metal-rough map. Track the
            // desired sRGB flag per path so we can flag the (pathological) case where the same texture
            // is used by both a metal-rough and a spec-gloss material, which want opposite color spaces.
            var desiredByPath   = new Dictionary<string, bool>();
            var wrong           = new List<string>(); // "<texPath>  (...)"
            var conflicts       = new List<string>();
            int fixedCount      = 0;
            int okCount         = 0;
            int missingImporter = 0;

            try
            {
                for (int i = 0; i < guids.Length; i++)
                {
                    string matPath = AssetDatabase.GUIDToAssetPath(guids[i]);
                    var    mat     = AssetDatabase.LoadAssetAtPath<RtxptMaterial>(matPath);
                    if (mat == null || mat.OcclusionRoughnessMetallicTexture == null) continue;

                    string texPath = AssetDatabase.GetAssetPath(mat.OcclusionRoughnessMetallicTexture);
                    if (string.IsNullOrEmpty(texPath)) continue;

                    // Metal-rough model wants linear (sRGB off); spec-gloss stores specular color (sRGB on).
                    bool desiredSrgb = mat.UseSpecularGlossModel;

                    if (desiredByPath.TryGetValue(texPath, out bool prevDesired))
                    {
                        if (prevDesired != desiredSrgb)
                            conflicts.Add($"{texPath}  (used as both metal-rough and spec-gloss — review by hand; last seen: {mat.name})");
                        continue; // already processed this texture
                    }
                    desiredByPath.Add(texPath, desiredSrgb);

                    EditorUtility.DisplayProgressBar(
                        fix ? "Fixing metal-rough color space" : "Validating metal-rough color space",
                        texPath, (float)i / Mathf.Max(1, guids.Length));

                    var importer = AssetImporter.GetAtPath(texPath) as TextureImporter;
                    if (importer == null)
                    {
                        // E.g. a render texture or non-2D texture asset with no TextureImporter.
                        missingImporter++;
                        wrong.Add($"{texPath}  (no TextureImporter — used by {mat.name})");
                        continue;
                    }

                    if (importer.sRGBTexture == desiredSrgb)
                    {
                        okCount++;
                        continue;
                    }

                    string model = desiredSrgb ? "spec-gloss → sRGB" : "metal-rough → linear";
                    wrong.Add($"{texPath}  (sRGB {importer.sRGBTexture} → {desiredSrgb}; {model} — used by {mat.name})");

                    if (fix)
                    {
                        importer.sRGBTexture = desiredSrgb;
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
            sb.AppendLine($"[RtxptNormalMapFixer] Metal-rough color-space {(fix ? "fix" : "validation")} complete.");
            sb.AppendLine($"  Materials scanned     : {guids.Length}");
            sb.AppendLine($"  Unique metal-rough maps: {desiredByPath.Count}");
            sb.AppendLine($"  Already correct       : {okCount}");
            sb.AppendLine($"  Wrong color space     : {wrong.Count}");
            if (fix) sb.AppendLine($"  Fixed                 : {fixedCount}");
            if (missingImporter > 0) sb.AppendLine($"  No TextureImporter    : {missingImporter}");
            if (conflicts.Count > 0) sb.AppendLine($"  Model conflicts       : {conflicts.Count}");
            foreach (string w in wrong) sb.AppendLine($"    - {w}");
            foreach (string c in conflicts) sb.AppendLine($"    ! {c}");

            Debug.Log(sb.ToString().TrimEnd());

            EditorUtility.DisplayDialog(
                fix ? "Fix Metal-Rough Texture Color Space" : "Validate Metal-Rough Texture Color Space",
                fix
                    ? $"Scanned {desiredByPath.Count} unique metal-rough maps.\n" +
                      $"Fixed {fixedCount}, already correct {okCount}.\n\nSee Console for details."
                    : $"Scanned {desiredByPath.Count} unique metal-rough maps.\n" +
                      $"{wrong.Count} wrong color space, {okCount} correct.\n\nSee Console for details.",
                "OK");
        }
    }
}
#endif
