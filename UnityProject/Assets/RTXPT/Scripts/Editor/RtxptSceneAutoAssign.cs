#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace PathTracing
{
    /// <summary>
    /// Adds <see cref="RtxptRenderer"/> to every Renderer in the scene
    /// and auto-assigns slot assets by matching Unity material names to
    /// <see cref="RtxptMaterial"/> files found in a chosen project folder.
    /// Open via  RTXPT ▸ Auto-Assign Material Overrides to Scene…
    /// </summary>
    public class RtxptSceneAutoAssign : EditorWindow
    {
        private string _assetFolder   = "Assets/RtxptMaterialOverrides";
        private bool   _recurse       = true;
        private string _prefix        = ""; // e.g. "transparent-machines-pt0"
        private bool   _selectedOnly  = false;
        private bool   _skipExisting  = true;
        private bool   _caseSensitive = false;

        private Vector2 _scroll;
        private string  _lastReport = "";

        [MenuItem("RTXPT/Auto-Assign Material Overrides to Scene…")]
        public static void Open() => GetWindow<RtxptSceneAutoAssign>("Auto-Assign RTXPT Overrides").minSize = new Vector2(440, 360);

        private void OnGUI()
        {
            EditorGUILayout.Space(6);
            EditorGUILayout.LabelField("Asset Source Folder", EditorStyles.boldLabel);

            EditorGUILayout.BeginHorizontal();
            _assetFolder = EditorGUILayout.TextField("Folder", _assetFolder);
            if (GUILayout.Button("…", GUILayout.Width(26)))
            {
                string picked = EditorUtility.OpenFolderPanel("Select folder with RtxptMaterial files", ToAbsolute(_assetFolder), "");
                if (!string.IsNullOrEmpty(picked))
                {
                    string rel = AbsToRelative(picked);
                    if (rel != null) _assetFolder = rel;
                    else EditorUtility.DisplayDialog("Invalid folder", "Folder must be inside the project's Assets.", "OK");
                }
            }

            EditorGUILayout.EndHorizontal();

            if (!string.IsNullOrWhiteSpace(_assetFolder) && System.IO.Path.IsPathRooted(_assetFolder))
            {
                string rel                    = AbsToRelative(_assetFolder);
                if (rel != null) _assetFolder = rel;
            }

            bool folderOk = !string.IsNullOrWhiteSpace(_assetFolder)
                            && _assetFolder.StartsWith("Assets", StringComparison.OrdinalIgnoreCase)
                            && AssetDatabase.IsValidFolder(_assetFolder.TrimEnd('/'));
            if (!folderOk)
                EditorGUILayout.HelpBox("Select a valid folder inside Assets/.", MessageType.Warning);

            _recurse = EditorGUILayout.Toggle("Search Subfolders", _recurse);

            EditorGUILayout.Space(8);
            EditorGUILayout.LabelField("Name Matching", EditorStyles.boldLabel);

            _prefix = EditorGUILayout.TextField(new GUIContent("Asset Name Prefix",
                "Optional prefix to prepend when looking up an asset.\n" +
                "e.g. prefix \"transparent-machines-pt0\" makes \"RandomColor_803\" also try\n" +
                "\"transparent-machines-pt0.RandomColor_803\" and\n" +
                "\"transparent-machines-pt0.RandomColor_803\"."), _prefix);

            string previewName = string.IsNullOrWhiteSpace(_prefix) ? "<name>" : $"{_prefix.Trim()}.<name>";
            EditorGUILayout.HelpBox($"Matches asset named exactly: {previewName}", MessageType.None);

            _caseSensitive = EditorGUILayout.Toggle("Case-Sensitive Match", _caseSensitive);

            EditorGUILayout.Space(8);
            EditorGUILayout.LabelField("Scene Targets", EditorStyles.boldLabel);
            _selectedOnly = EditorGUILayout.Toggle("Selected GameObjects Only", _selectedOnly);
            _skipExisting = EditorGUILayout.Toggle("Skip Already-Assigned Slots", _skipExisting);

            EditorGUILayout.Space(10);

            GUI.enabled = folderOk;
            if (GUILayout.Button("Assign to Scene", GUILayout.Height(30)))
                Run();
            GUI.enabled = true;

            if (!string.IsNullOrEmpty(_lastReport))
            {
                EditorGUILayout.Space(8);
                _scroll = EditorGUILayout.BeginScrollView(_scroll, GUILayout.ExpandHeight(true));
                EditorGUILayout.HelpBox(_lastReport, MessageType.None);
                EditorGUILayout.EndScrollView();
            }
        }

        private void Run()
        {
            var comparison = _caseSensitive ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase;
            var comparer   = StringComparer.FromComparison(comparison);

            // ---- 1. Index: asset filename (without .asset) → asset ----
            var assetMap = new Dictionary<string, RtxptMaterial>(comparer);

            string[] guids = AssetDatabase.FindAssets("t:RtxptMaterial", new[] { _assetFolder.TrimEnd('/') });
            foreach (string guid in guids)
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                if (!_recurse)
                {
                    string parent = System.IO.Path.GetDirectoryName(path)?.Replace('\\', '/').TrimEnd('/');
                    if (!string.Equals(parent, _assetFolder.TrimEnd('/'), StringComparison.OrdinalIgnoreCase))
                        continue;
                }

                var asset = AssetDatabase.LoadAssetAtPath<RtxptMaterial>(path);
                if (asset == null) continue;
                string key                                    = System.IO.Path.GetFileNameWithoutExtension(path);
                if (!assetMap.ContainsKey(key)) assetMap[key] = asset;
            }

            // ---- 2. Collect renderers ----
            Renderer[] renderers;
            if (_selectedOnly)
            {
                var list = new List<Renderer>();
                foreach (var go in Selection.gameObjects)
                    list.AddRange(go.GetComponentsInChildren<Renderer>(includeInactive: true));
                renderers = list.ToArray();
            }
            else
            {
                renderers = FindObjectsByType<Renderer>(FindObjectsInactive.Include, FindObjectsSortMode.None);
            }

            // ---- 3. Assign ----
            string prefix = _prefix.Trim();

            int renderersTouched = 0, slotsAssigned = 0, slotsUnmatched = 0;
            var unmatched        = new HashSet<string>(comparer);

            foreach (var mr in renderers)
            {
                var mf   = mr.GetComponent<MeshFilter>();
                var mats = mr.sharedMaterials ?? Array.Empty<Material>();

                int slotCount = 0;
                if (mf == null)
                {
                    slotCount = mats.Length;
                }
                else
                {
                    slotCount = mf.sharedMesh.subMeshCount;
                }

                if (slotCount == 0) continue;

                var comp = mr.GetComponent<RtxptRenderer>();
                if (comp == null)
                    comp = Undo.AddComponent<RtxptRenderer>(mr.gameObject);

                Undo.RecordObject(comp, "Auto-Assign RTXPT Material Overrides");

                while (comp.Slots.Count < slotCount) comp.Slots.Add(null);
                if (comp.Slots.Count > slotCount) comp.Slots.RemoveRange(slotCount, comp.Slots.Count - slotCount);

                bool touched = false;
                for (int s = 0; s < slotCount; s++)
                {
                    if (_skipExisting && comp.Slots[s] != null) continue;

                    Material mat = s < mats.Length ? mats[s] : (mats.Length > 0 ? mats[^1] : null);
                    if (mat == null) continue;

                    var found = Lookup(assetMap, mat.name, prefix, comparison);
                    if (found != null)
                    {
                        comp.Slots[s] = found;
                        slotsAssigned++;
                        touched = true;
                    }
                    else
                    {
                        unmatched.Add(mat.name);
                        slotsUnmatched++;
                    }
                }

                if (touched)
                {
                    EditorUtility.SetDirty(comp);
                    renderersTouched++;
                }
            }

            EditorSceneManager.MarkAllScenesDirty();

            // ---- 4. Report ----
            var sb = new StringBuilder();
            sb.AppendLine($"Asset folder : {_assetFolder}  ({assetMap.Count} assets indexed)");
            if (!string.IsNullOrEmpty(prefix))
                sb.AppendLine($"Prefix       : {prefix}");
            sb.AppendLine($"Renderers    : {renderers.Length} scanned, {renderersTouched} updated");
            sb.AppendLine($"Slots        : {slotsAssigned} assigned, {slotsUnmatched} unmatched");
            if (unmatched.Count > 0)
            {
                sb.AppendLine();
                sb.AppendLine("Unmatched material names:");
                foreach (var name in unmatched) sb.AppendLine($"  • {name}");
            }

            _lastReport = sb.ToString().TrimEnd();
        }

        // Strict match: "{prefix}.{matName}" when prefix is set, else "{matName}".
        private static RtxptMaterial Lookup(
            Dictionary<string, RtxptMaterial> map,
            string matName, string prefix, StringComparison cmp)
        {
            string key = string.IsNullOrEmpty(prefix) ? matName : $"{prefix}.{matName}";
            return map.TryGetValue(key, out var asset) ? asset : null;
        }

        // ---- Path helpers ----

        private static string ProjectRoot()
            => System.IO.Path.GetFullPath(System.IO.Path.Combine(Application.dataPath, "..")).Replace('\\', '/').TrimEnd('/');

        private static string AbsToRelative(string absPath)
        {
            if (string.IsNullOrEmpty(absPath)) return null;
            string norm = System.IO.Path.GetFullPath(absPath).Replace('\\', '/').TrimEnd('/');
            string root = ProjectRoot();
            if (!norm.StartsWith(root, StringComparison.OrdinalIgnoreCase)) return null;
            string rel = norm.Substring(root.Length).TrimStart('/');
            return string.IsNullOrEmpty(rel) ? null : rel;
        }

        private static string ToAbsolute(string path)
        {
            if (string.IsNullOrEmpty(path)) return Application.dataPath;
            if (System.IO.Path.IsPathRooted(path)) return path;
            return System.IO.Path.GetFullPath(System.IO.Path.Combine(ProjectRoot(), path)).Replace('\\', '/');
        }
    }
}
#endif