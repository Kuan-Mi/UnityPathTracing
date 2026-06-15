#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace PathTracing
{
    /// <summary>
    /// Batch-creates <see cref="RtxptMaterial"/> files from a folder of RTXPT
    /// material JSON files. Open via  RTXPT ▸ Batch Import Material Overrides from JSON…
    /// </summary>
    public class RtxptMaterialOverrideBatchImport : EditorWindow
    {
        private string _srcFolder      = "";
        private string _dstFolder      = "Assets/RtxptMaterialOverrides";
        private bool   _recurse        = true;
        private bool   _preserveLayout = true;
        private bool   _skipExisting   = true;
        private bool   _importTextures = true;
        private string _textureRoot    = "Assets/Art/RTXPTAssets";

        private Vector2 _scroll;
        private string  _lastReport = "";

        [MenuItem("RTXPT/Batch Import Material Overrides from JSON…")]
        public static void Open() => GetWindow<RtxptMaterialOverrideBatchImport>("Batch Import RTXPT Materials").minSize = new Vector2(440, 360);

        private void OnGUI()
        {
            EditorGUILayout.Space(6);
            EditorGUILayout.LabelField("Source", EditorStyles.boldLabel);

            // ---- Source folder (any path on disk) ----
            EditorGUILayout.BeginHorizontal();
            _srcFolder = EditorGUILayout.TextField("JSON Folder", _srcFolder);
            if (GUILayout.Button("…", GUILayout.Width(26)))
            {
                string start  = Directory.Exists(_srcFolder) ? _srcFolder : "";
                string picked = EditorUtility.OpenFolderPanel("Select folder containing JSON files", start, "");
                if (!string.IsNullOrEmpty(picked)) _srcFolder = Normalize(picked);
            }
            EditorGUILayout.EndHorizontal();

            bool srcOk = !string.IsNullOrWhiteSpace(_srcFolder) && Directory.Exists(_srcFolder);
            if (!string.IsNullOrWhiteSpace(_srcFolder) && !srcOk)
                EditorGUILayout.HelpBox("Folder not found.", MessageType.Warning);

            _recurse = EditorGUILayout.Toggle("Include Subdirectories", _recurse);

            EditorGUILayout.Space(8);
            EditorGUILayout.LabelField("Destination (inside project Assets/)", EditorStyles.boldLabel);

            // ---- Destination folder (must be under Assets/) ----
            EditorGUILayout.BeginHorizontal();
            _dstFolder = EditorGUILayout.TextField("Asset Folder", _dstFolder);
            if (GUILayout.Button("…", GUILayout.Width(26)))
            {
                // Compute an absolute start path for the dialog.
                string startAbs = ToAbsoluteIfRelative(_dstFolder);
                if (!Directory.Exists(startAbs)) startAbs = Application.dataPath;

                string picked = EditorUtility.OpenFolderPanel("Select destination inside the project", startAbs, "");
                if (!string.IsNullOrEmpty(picked))
                {
                    string rel = AbsToRelative(picked);
                    if (rel != null)
                        _dstFolder = rel;
                    else
                        EditorUtility.DisplayDialog("Invalid folder", "Destination must be inside the project's Assets folder.", "OK");
                }
            }
            EditorGUILayout.EndHorizontal();

            // Accept and convert absolute paths typed directly into the field.
            if (Path.IsPathRooted(_dstFolder))
            {
                string rel = AbsToRelative(_dstFolder);
                if (rel != null) _dstFolder = rel;
            }

            bool dstOk = !string.IsNullOrWhiteSpace(_dstFolder)
                      && _dstFolder.StartsWith("Assets", StringComparison.OrdinalIgnoreCase);
            if (!dstOk)
                EditorGUILayout.HelpBox("Must be a path starting with 'Assets'.", MessageType.Warning);

            if (_recurse)
                _preserveLayout = EditorGUILayout.Toggle("Mirror Subfolder Structure", _preserveLayout);
            _skipExisting = EditorGUILayout.Toggle("Skip Existing Assets", _skipExisting);

            EditorGUILayout.Space(8);
            EditorGUILayout.LabelField("Textures", EditorStyles.boldLabel);
            _importTextures = EditorGUILayout.Toggle(new GUIContent("Resolve Textures",
                "Resolve texture path entries in the JSON to project textures and assign them to each slot."),
                _importTextures);

            if (_importTextures)
            {
                EditorGUILayout.BeginHorizontal();
                _textureRoot = EditorGUILayout.TextField(new GUIContent("Texture Root",
                    "Project-relative folder the JSON 'path' fields are resolved against, e.g. \"Models\\Kitchen\\foo.dds\" under Assets/Art/RTXPTAssets."),
                    _textureRoot);
                if (GUILayout.Button("…", GUILayout.Width(26)))
                {
                    string picked = EditorUtility.OpenFolderPanel("Select texture root inside the project", ToAbsoluteIfRelative(_textureRoot), "");
                    if (!string.IsNullOrEmpty(picked))
                    {
                        string rel = AbsToRelative(picked);
                        if (rel != null) _textureRoot = rel;
                        else EditorUtility.DisplayDialog("Invalid folder", "Texture root must be inside the project's Assets folder.", "OK");
                    }
                }
                EditorGUILayout.EndHorizontal();

                if (Path.IsPathRooted(_textureRoot))
                {
                    string rel = AbsToRelative(_textureRoot);
                    if (rel != null) _textureRoot = rel;
                }

                if (!AssetDatabase.IsValidFolder(_textureRoot.TrimEnd('/')))
                    EditorGUILayout.HelpBox("Texture root is not an existing folder under Assets/.", MessageType.Warning);
            }

            EditorGUILayout.Space(10);

            GUI.enabled = srcOk && dstOk;
            if (GUILayout.Button("Import", GUILayout.Height(30)))
                RunImport();
            GUI.enabled = true;

            if (!string.IsNullOrEmpty(_lastReport))
            {
                EditorGUILayout.Space(8);
                EditorGUILayout.LabelField("Result", EditorStyles.boldLabel);
                _scroll = EditorGUILayout.BeginScrollView(_scroll, GUILayout.ExpandHeight(true));
                EditorGUILayout.HelpBox(_lastReport, MessageType.None);
                EditorGUILayout.EndScrollView();
            }
        }

        private void RunImport()
        {
            // Normalize the source to forward slashes, trimmed.
            string srcNorm = Normalize(_srcFolder.Trim());

            // Destination: strip any trailing slash; ensure it starts with Assets.
            string dst = _dstFolder.Trim().Replace('\\', '/').TrimEnd('/');
            if (!dst.StartsWith("Assets", StringComparison.OrdinalIgnoreCase))
            {
                _lastReport = $"ERROR: Destination '{dst}' is not inside Assets/.";
                return;
            }

            var    searchOption = _recurse ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
            string[] jsonFiles;
            try   { jsonFiles = Directory.GetFiles(srcNorm, "*.json", searchOption); }
            catch (Exception ex) { _lastReport = $"ERROR reading source folder:\n{ex.Message}"; return; }

            // Build report header so the user can always verify what was scanned.
            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"Source : {srcNorm}");
            sb.AppendLine($"Dest   : {dst}");
            sb.AppendLine($"Found  : {jsonFiles.Length} JSON file(s)");
            sb.AppendLine();

            if (jsonFiles.Length == 0)
            {
                _lastReport = sb.ToString().TrimEnd();
                return;
            }

            EnsureFolder(dst);

            string textureRoot = _textureRoot.Trim().Replace('\\', '/').TrimEnd('/');

            int created = 0, skipped = 0, failed = 0, texMissing = 0;
            var errors        = new List<string>();
            var missingTexErr = new HashSet<string>();

            foreach (string jsonPath in jsonFiles)
            {
                // Compute relative sub-directory by comparing normalized paths.
                string relSubDir = "";
                if (_recurse && _preserveLayout)
                {
                    string dir = Normalize(Path.GetDirectoryName(jsonPath)!);
                    // src is already normalized and trimmed.
                    if (dir.Length > srcNorm.Length)
                        relSubDir = dir.Substring(srcNorm.Length).TrimStart('/');
                }

                string assetDir  = string.IsNullOrEmpty(relSubDir) ? dst : dst + "/" + relSubDir;
                string name      = Path.GetFileNameWithoutExtension(jsonPath);
                string assetPath = assetDir + "/" + name + ".asset";

                if (!string.IsNullOrEmpty(relSubDir)) EnsureFolder(assetDir);

                if (_skipExisting && AssetDatabase.LoadAssetAtPath<RtxptMaterial>(assetPath) != null)
                {
                    skipped++;
                    continue;
                }

                try
                {
                    string json  = File.ReadAllText(jsonPath);
                    var    asset = CreateInstance<RtxptMaterial>();

                    Func<RtxptTextureRef, Texture> resolver = null;
                    if (_importTextures)
                        resolver = texRef =>
                        {
                            var tex = RtxptTextureResolver.Resolve(textureRoot, texRef.Path, out _);
                            if (tex == null && !string.IsNullOrEmpty(texRef.Path))
                            {
                                texMissing++;
                                missingTexErr.Add(texRef.Path);
                            }
                            return tex;
                        };

                    asset.LoadFromJson(json, resolver);
                    if (!_skipExisting)
                        assetPath = AssetDatabase.GenerateUniqueAssetPath(assetPath);
                    AssetDatabase.CreateAsset(asset, assetPath);
                    created++;
                }
                catch (Exception ex)
                {
                    errors.Add($"{Path.GetFileName(jsonPath)}: {ex.Message}");
                    failed++;
                }
            }

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();

            sb.AppendLine($"Created : {created}");
            if (skipped > 0) sb.AppendLine($"Skipped : {skipped}  (already existed)");
            if (failed  > 0)
            {
                sb.AppendLine($"Failed  : {failed}");
                foreach (var e in errors) sb.AppendLine($"  • {e}");
            }
            if (_importTextures && texMissing > 0)
            {
                sb.AppendLine();
                sb.AppendLine($"Textures unresolved : {texMissing} reference(s), {missingTexErr.Count} unique path(s) under '{textureRoot}':");
                foreach (var p in missingTexErr) sb.AppendLine($"  • {p}");
            }
            _lastReport = sb.ToString().TrimEnd();
        }

        // Normalize a file-system path: forward slashes, no trailing slash.
        private static string Normalize(string path)
            => path.Replace('\\', '/').TrimEnd('/');

        // Returns the project root with forward slashes and no trailing slash.
        private static string ProjectRoot()
            => Normalize(Path.GetFullPath(Path.Combine(Application.dataPath, "..")));

        // Converts an absolute path to a project-relative "Assets/…" path.
        // Returns null if the path is not inside the project.
        private static string AbsToRelative(string absPath)
        {
            if (string.IsNullOrEmpty(absPath)) return null;
            string norm = Normalize(Path.GetFullPath(absPath));
            string root = ProjectRoot();
            if (!norm.StartsWith(root, StringComparison.OrdinalIgnoreCase)) return null;
            string rel = norm.Substring(root.Length).TrimStart('/');
            return string.IsNullOrEmpty(rel) ? null : rel;
        }

        // Converts a relative Assets/… path to an absolute path for dialog start.
        private static string ToAbsoluteIfRelative(string path)
        {
            if (string.IsNullOrEmpty(path)) return "";
            if (Path.IsPathRooted(path)) return path;
            return Normalize(Path.GetFullPath(Path.Combine(ProjectRoot(), path)));
        }

        private static void EnsureFolder(string folderPath)
        {
            string[] parts   = folderPath.TrimEnd('/').Split('/');
            string   current = parts[0];
            for (int i = 1; i < parts.Length; i++)
            {
                string next = current + "/" + parts[i];
                if (!AssetDatabase.IsValidFolder(next))
                    AssetDatabase.CreateFolder(current, parts[i]);
                current = next;
            }
        }
    }
}
#endif
