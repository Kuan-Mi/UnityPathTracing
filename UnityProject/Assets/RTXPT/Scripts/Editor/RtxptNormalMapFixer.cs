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
    /// The original RTXPT normal import mode keeps those same normal (DDNA) textures as Default,
    /// linear, unconverted textures, matching the reference shader's raw normal decode path, and
    /// BC7-compressed to match the original RTXPT asset format. (They were previously imported
    /// Uncompressed — at 2048² RGBA that is 32 bpp vs BC7's 8 bpp, and since the path-tracer
    /// ClosestHit samples them incoherently, the extra bandwidth/cache footprint measurably slowed
    /// the divergent PathTrace pass. BC7 keeps all four packed channels, so the raw decode is
    /// unaffected.)
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
        // BC7 (via CompressedHQ on desktop): 4-channel, high-quality, 8 bpp — matches the original
        // RTXPT BC7_UNORM asset format and keeps the packed DDNA channels intact for the raw decode.
        private const TextureImporterCompression OriginalNormalTextureCompression = TextureImporterCompression.CompressedHQ;

        // The normal, base color, and metal-rough source assets were re-exported at 4K; match that
        // resolution on import instead of Unity's default 2048 cap.
        private const int OriginalMaxTextureSize = 4096;

        [MenuItem("RTXPT/Validate Normal Map Texture Types…")]
        private static void Validate() => Run(fix: false);

        [MenuItem("RTXPT/Fix Normal Map Texture Types")]
        private static void Fix() => Run(fix: true);

        [MenuItem("RTXPT/Validate Original RTXPT Normal Texture Import…")]
        private static void ValidateOriginalNormalImport() => RunOriginalNormalImport(fix: false);

        [MenuItem("RTXPT/Fix Original RTXPT Normal Texture Import")]
        private static void FixOriginalNormalImport() => RunOriginalNormalImport(fix: true);

        [MenuItem("RTXPT/Validate Metal-Rough Texture Color Space…")]
        private static void ValidateOrm() => RunOrm(fix: false);

        [MenuItem("RTXPT/Fix Metal-Rough Texture Color Space")]
        private static void FixOrm() => RunOrm(fix: true);

        [MenuItem("RTXPT/Validate Base Color & Metal-Rough Texture Compression…")]
        private static void ValidateAlbedoOrmCompression() => RunAlbedoOrmCompression(fix: false);

        [MenuItem("RTXPT/Fix Base Color & Metal-Rough Texture Compression")]
        private static void FixAlbedoOrmCompression() => RunAlbedoOrmCompression(fix: true);

        private static void Run(bool fix)
        {
            string[] guids = AssetDatabase.FindAssets("t:RtxptMaterial");

            // Dedupe by texture asset path: many materials may share one normal map.
            var checkedPaths    = new HashSet<string>();
            var wrong           = new List<string>(); // "<texPath>  (used by <materialName>)"
            int fixedCount      = 0;
            int okCount         = 0;
            int missingImporter = 0;

            try
            {
                if (fix) AssetDatabase.StartAssetEditing();

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
                if (fix) AssetDatabase.StopAssetEditing();
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

        private static void RunOriginalNormalImport(bool fix)
        {
            string[] guids = AssetDatabase.FindAssets("t:RtxptMaterial");

            // The original RTXPT shader samples normal textures as raw linear texture data and
            // performs its own xy/z reconstruction. Unity's NormalMap importer can repack channels,
            // so this mode intentionally keeps the texture as a plain linear texture.
            var checkedPaths    = new HashSet<string>();
            var wrong           = new List<string>(); // "<texPath>  (...)"
            int fixedCount      = 0;
            int okCount         = 0;
            int missingImporter = 0;

            try
            {
                if (fix) AssetDatabase.StartAssetEditing();

                for (int i = 0; i < guids.Length; i++)
                {
                    string matPath = AssetDatabase.GUIDToAssetPath(guids[i]);
                    var    mat     = AssetDatabase.LoadAssetAtPath<RtxptMaterial>(matPath);
                    if (mat == null || mat.NormalTexture == null) continue;

                    string texPath = AssetDatabase.GetAssetPath(mat.NormalTexture);
                    if (string.IsNullOrEmpty(texPath) || !checkedPaths.Add(texPath)) continue;

                    EditorUtility.DisplayProgressBar(
                        fix ? "Fixing original RTXPT normal texture import" : "Validating original RTXPT normal texture import",
                        texPath, (float)i / Mathf.Max(1, guids.Length));

                    var importer = AssetImporter.GetAtPath(texPath) as TextureImporter;
                    if (importer == null)
                    {
                        missingImporter++;
                        wrong.Add($"{texPath}  (no TextureImporter — used by {mat.name})");
                        continue;
                    }

                    if (IsOriginalNormalImport(importer))
                    {
                        okCount++;
                        continue;
                    }

                    wrong.Add($"{texPath}  ({DescribeOriginalNormalImport(importer)} — used by {mat.name})");

                    if (fix)
                    {
                        importer.textureType         = TextureImporterType.Default;
                        importer.sRGBTexture         = false;
                        importer.convertToNormalmap  = false;
                        importer.textureCompression  = OriginalNormalTextureCompression;
                        importer.crunchedCompression = false; // plain BC7, matching RTXPT's BC7_UNORM
                        importer.maxTextureSize      = OriginalMaxTextureSize;
                        importer.SaveAndReimport();
                        fixedCount++;
                    }
                }
            }
            finally
            {
                if (fix) AssetDatabase.StopAssetEditing();
                EditorUtility.ClearProgressBar();
            }

            var sb = new StringBuilder();
            sb.AppendLine($"[RtxptNormalMapFixer] Original RTXPT normal texture import {(fix ? "fix" : "validation")} complete.");
            sb.AppendLine($"  Materials scanned : {guids.Length}");
            sb.AppendLine($"  Unique normal maps: {checkedPaths.Count}");
            sb.AppendLine($"  Already correct   : {okCount}");
            sb.AppendLine($"  Wrong import      : {wrong.Count}");
            if (fix) sb.AppendLine($"  Fixed             : {fixedCount}");
            if (missingImporter > 0) sb.AppendLine($"  No TextureImporter : {missingImporter}");
            foreach (string w in wrong) sb.AppendLine($"    - {w}");

            Debug.Log(sb.ToString().TrimEnd());

            EditorUtility.DisplayDialog(
                fix ? "Fix Original RTXPT Normal Texture Import" : "Validate Original RTXPT Normal Texture Import",
                fix
                    ? $"Scanned {checkedPaths.Count} unique normal maps.\n" +
                      $"Fixed {fixedCount}, already correct {okCount}.\n\nSee Console for details."
                    : $"Scanned {checkedPaths.Count} unique normal maps.\n" +
                      $"{wrong.Count} not in original RTXPT import mode, {okCount} correct.\n\nSee Console for details.",
                "OK");
        }

        private static bool IsOriginalNormalImport(TextureImporter importer)
        {
            return importer.textureType == TextureImporterType.Default
                   && !importer.sRGBTexture
                   && !importer.convertToNormalmap
                   && !importer.crunchedCompression
                   && importer.textureCompression == OriginalNormalTextureCompression
                   && importer.maxTextureSize == OriginalMaxTextureSize;
        }

        private static string DescribeOriginalNormalImport(TextureImporter importer)
        {
            return "expected Default/linear/BC7/4K; "
                   + $"type {importer.textureType} → {TextureImporterType.Default}, "
                   + $"sRGB {importer.sRGBTexture} → False, "
                   + $"convertToNormalmap {importer.convertToNormalmap} → False, "
                   + $"crunched {importer.crunchedCompression} → False, "
                   + $"compression {importer.textureCompression} → {OriginalNormalTextureCompression}, "
                   + $"maxSize {importer.maxTextureSize} → {OriginalMaxTextureSize}";
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
                if (fix) AssetDatabase.StartAssetEditing();

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
                if (fix) AssetDatabase.StopAssetEditing();
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

        // The original RTXPT base color, metallic, and roughness maps are all BC7_UNORM. Metalness and
        // roughness are packed into the same OcclusionRoughnessMetallicTexture, so fixing that one
        // texture's compression covers both channels; base color is BaseOrDiffuseTexture. Colour space
        // (sRGB/linear) is handled separately by RunOrm — this only touches compression format.
        private static void RunAlbedoOrmCompression(bool fix)
        {
            string[] guids = AssetDatabase.FindAssets("t:RtxptMaterial");

            var checkedPaths    = new HashSet<string>();
            var wrong           = new List<string>(); // "<texPath>  (...)"
            int fixedCount      = 0;
            int okCount         = 0;
            int missingImporter = 0;

            try
            {
                if (fix) AssetDatabase.StartAssetEditing();

                for (int i = 0; i < guids.Length; i++)
                {
                    string matPath = AssetDatabase.GUIDToAssetPath(guids[i]);
                    var    mat     = AssetDatabase.LoadAssetAtPath<RtxptMaterial>(matPath);
                    if (mat == null) continue;

                    EditorUtility.DisplayProgressBar(
                        fix ? "Fixing base color / metal-rough compression" : "Validating base color / metal-rough compression",
                        matPath, (float)i / Mathf.Max(1, guids.Length));

                    CheckOrFixBc7Compression(mat.BaseOrDiffuseTexture, mat.name, fix,
                        checkedPaths, wrong, ref fixedCount, ref okCount, ref missingImporter);
                    CheckOrFixBc7Compression(mat.OcclusionRoughnessMetallicTexture, mat.name, fix,
                        checkedPaths, wrong, ref fixedCount, ref okCount, ref missingImporter);
                }
            }
            finally
            {
                if (fix) AssetDatabase.StopAssetEditing();
                EditorUtility.ClearProgressBar();
            }

            var sb = new StringBuilder();
            sb.AppendLine($"[RtxptNormalMapFixer] Base color / metal-rough BC7 compression {(fix ? "fix" : "validation")} complete.");
            sb.AppendLine($"  Materials scanned : {guids.Length}");
            sb.AppendLine($"  Unique textures   : {checkedPaths.Count}");
            sb.AppendLine($"  Already correct   : {okCount}");
            sb.AppendLine($"  Wrong compression : {wrong.Count}");
            if (fix) sb.AppendLine($"  Fixed             : {fixedCount}");
            if (missingImporter > 0) sb.AppendLine($"  No TextureImporter : {missingImporter}");
            foreach (string w in wrong) sb.AppendLine($"    - {w}");

            Debug.Log(sb.ToString().TrimEnd());

            EditorUtility.DisplayDialog(
                fix ? "Fix Base Color & Metal-Rough Texture Compression" : "Validate Base Color & Metal-Rough Texture Compression",
                fix
                    ? $"Scanned {checkedPaths.Count} unique textures.\n" +
                      $"Fixed {fixedCount}, already correct {okCount}.\n\nSee Console for details."
                    : $"Scanned {checkedPaths.Count} unique textures.\n" +
                      $"{wrong.Count} not BC7 compressed, {okCount} correct.\n\nSee Console for details.",
                "OK");
        }

        private static void CheckOrFixBc7Compression(
            Texture texture, string materialName, bool fix,
            HashSet<string> checkedPaths, List<string> wrong,
            ref int fixedCount, ref int okCount, ref int missingImporter)
        {
            if (texture == null) return;

            string texPath = AssetDatabase.GetAssetPath(texture);
            if (string.IsNullOrEmpty(texPath) || !checkedPaths.Add(texPath)) return;

            var importer = AssetImporter.GetAtPath(texPath) as TextureImporter;
            if (importer == null)
            {
                missingImporter++;
                wrong.Add($"{texPath}  (no TextureImporter — used by {materialName})");
                return;
            }

            if (importer.textureCompression == OriginalNormalTextureCompression
                && !importer.crunchedCompression
                && importer.maxTextureSize == OriginalMaxTextureSize)
            {
                okCount++;
                return;
            }

            wrong.Add($"{texPath}  (compression {importer.textureCompression} → {OriginalNormalTextureCompression}, " +
                      $"crunched {importer.crunchedCompression} → False, " +
                      $"maxSize {importer.maxTextureSize} → {OriginalMaxTextureSize} — used by {materialName})");

            if (fix)
            {
                importer.textureCompression  = OriginalNormalTextureCompression;
                importer.crunchedCompression = false;
                importer.maxTextureSize      = OriginalMaxTextureSize;
                importer.SaveAndReimport();
                fixedCount++;
            }
        }
    }
}
#endif
