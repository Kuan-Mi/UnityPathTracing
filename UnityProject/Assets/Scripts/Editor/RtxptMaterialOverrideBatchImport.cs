#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace PathTracing
{
    /// <summary>
    /// Batch-creates <see cref="RtxptMaterialOverrideAsset"/> files from a folder of RTXPT
    /// material JSON files. Open via  RTXPT ▸ Batch Import Material Overrides from JSON…
    /// </summary>
    public class RtxptMaterialOverrideBatchImport : EditorWindow
    {
        private string _srcFolder      = "";
        private string _dstFolder      = "Assets/RtxptMaterialOverrides";
        private bool   _recurse        = true;
        private bool   _preserveLayout = true;  // mirror subfolder structure in dst
        private bool   _skipExisting   = true;  // don't overwrite assets that already exist

        private Vector2 _scroll;
        private string  _lastReport = "";

        [MenuItem("RTXPT/Batch Import Material Overrides from JSON…")]
        public static void Open() => GetWindow<RtxptMaterialOverrideBatchImport>("Batch Import RTXPT Materials").minSize = new Vector2(420, 340);

        private void OnGUI()
        {
            EditorGUILayout.Space(6);
            EditorGUILayout.LabelField("Source", EditorStyles.boldLabel);

            // ---- Source folder ----
            EditorGUILayout.BeginHorizontal();
            _srcFolder = EditorGUILayout.TextField("JSON Folder", _srcFolder);
            if (GUILayout.Button("…", GUILayout.Width(26)))
            {
                string picked = EditorUtility.OpenFolderPanel("Select folder containing JSON files", _srcFolder, "");
                if (!string.IsNullOrEmpty(picked)) _srcFolder = picked;
            }
            EditorGUILayout.EndHorizontal();

            _recurse = EditorGUILayout.Toggle("Include Subdirectories", _recurse);

            EditorGUILayout.Space(8);
            EditorGUILayout.LabelField("Destination (must be under Assets/)", EditorStyles.boldLabel);

            // ---- Destination folder ----
            EditorGUILayout.BeginHorizontal();
            _dstFolder = EditorGUILayout.TextField("Asset Folder", _dstFolder);
            if (GUILayout.Button("…", GUILayout.Width(26)))
            {
                string startDir = AbsToRelative(_dstFolder) != null
                    ? Path.Combine(ProjectRoot(), _dstFolder)
                    : Application.dataPath;
                string picked = EditorUtility.OpenFolderPanel("Select destination folder inside the project", startDir, "");
                if (!string.IsNullOrEmpty(picked))
                {
                    string rel = AbsToRelative(picked);
                    if (rel != null)
                        _dstFolder = rel;
                    else
                        EditorUtility.DisplayDialog("Invalid folder", "Destination must be inside the project root.", "OK");
                }
            }
            EditorGUILayout.EndHorizontal();

            _preserveLayout = EditorGUILayout.Toggle("Mirror Subfolder Structure", _preserveLayout && _recurse);
            if (!_recurse) GUI.enabled = false;
            GUI.enabled = true;

            _skipExisting = EditorGUILayout.Toggle("Skip Existing Assets", _skipExisting);

            EditorGUILayout.Space(10);

            bool canImport = !string.IsNullOrWhiteSpace(_srcFolder) && Directory.Exists(_srcFolder)
                          && !string.IsNullOrWhiteSpace(_dstFolder);

            GUI.enabled = canImport;
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
            // Accept absolute paths typed directly — convert to relative if possible.
            if (Path.IsPathRooted(_dstFolder))
            {
                string rel = AbsToRelative(_dstFolder);
                if (rel == null) { EditorUtility.DisplayDialog("Invalid path", "Destination must be inside the project's Assets folder.", "OK"); return; }
                _dstFolder = rel;
            }
            if (!_dstFolder.StartsWith("Assets", StringComparison.OrdinalIgnoreCase))
            {
                EditorUtility.DisplayDialog("Invalid path", "Destination must be inside the project's Assets folder.", "OK");
                return;
            }

            var searchOption = _recurse ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
            string[] jsonFiles;
            try { jsonFiles = Directory.GetFiles(_srcFolder, "*.json", searchOption); }
            catch (Exception ex) { EditorUtility.DisplayDialog("Error", ex.Message, "OK"); return; }

            if (jsonFiles.Length == 0)
            {
                _lastReport = "No .json files found in the selected folder.";
                return;
            }

            EnsureFolder(_dstFolder);

            int created = 0, skipped = 0, failed = 0;
            var errors = new List<string>();

            foreach (string jsonPath in jsonFiles)
            {
                string relSubDir = "";
                if (_recurse && _preserveLayout)
                {
                    string rel = Path.GetDirectoryName(jsonPath)!.Substring(_srcFolder.TrimEnd('/', '\\').Length).TrimStart('/', '\\').Replace('\\', '/');
                    if (!string.IsNullOrEmpty(rel))
                    {
                        relSubDir = rel;
                        EnsureFolder(_dstFolder + "/" + relSubDir);
                    }
                }

                string name      = Path.GetFileNameWithoutExtension(jsonPath);
                string assetDir  = string.IsNullOrEmpty(relSubDir) ? _dstFolder : _dstFolder + "/" + relSubDir;
                string assetPath = assetDir + "/" + name + ".asset";

                if (_skipExisting && AssetDatabase.LoadAssetAtPath<RtxptMaterialOverrideAsset>(assetPath) != null)
                {
                    skipped++;
                    continue;
                }

                try
                {
                    string json  = File.ReadAllText(jsonPath);
                    var    asset = CreateInstance<RtxptMaterialOverrideAsset>();
                    asset.Slot.LoadFromJson(json);
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

            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"Processed {jsonFiles.Length} file(s):");
            sb.AppendLine($"  Created : {created}");
            if (skipped > 0) sb.AppendLine($"  Skipped : {skipped}  (already existed)");
            if (failed  > 0)
            {
                sb.AppendLine($"  Failed  : {failed}");
                foreach (var e in errors) sb.AppendLine($"    • {e}");
            }
            _lastReport = sb.ToString().TrimEnd();
        }

        // Returns the project root with forward slashes and no trailing slash.
        private static string ProjectRoot()
            => Path.GetFullPath(Path.Combine(Application.dataPath, "..")).Replace('\\', '/').TrimEnd('/');

        // Converts an absolute path to a project-relative "Assets/..." path.
        // Returns null if the path is not inside the project.
        private static string AbsToRelative(string absPath)
        {
            if (string.IsNullOrEmpty(absPath)) return null;
            string norm = Path.GetFullPath(absPath).Replace('\\', '/').TrimEnd('/');
            string root = ProjectRoot();
            if (!norm.StartsWith(root, StringComparison.OrdinalIgnoreCase)) return null;
            string rel = norm.Substring(root.Length).TrimStart('/');
            return string.IsNullOrEmpty(rel) ? null : rel;
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
