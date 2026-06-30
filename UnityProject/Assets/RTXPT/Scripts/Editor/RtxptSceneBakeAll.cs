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
    /// One-click baker that adds <see cref="RtxptRenderer"/> to every Renderer in the
    /// scene and bakes a <see cref="RtxptMaterial"/> asset for every sub-mesh slot —
    /// reusing the same asset whenever multiple slots reference the same Unity <see cref="Material"/>.
    ///
    /// Open via  RTXPT ▸ Bake All Scene Materials…
    /// </summary>
    public class RtxptSceneBakeAll : EditorWindow
    {
        // ---- Settings ----
        private string _outputFolder = "Assets/RtxptMaterialOverrides";
        private bool   _selectedOnly = false;
        private bool   _skipExisting = true; // skip slots that already have an asset assigned
        private bool   _reBakeValues = true; // call BakeSlotFromMaterial on existing assets too

        private Vector2 _scroll;
        private string  _lastReport = "";

        [MenuItem("RTXPT/Bake All Scene Materials…")]
        public static void Open()
            => GetWindow<RtxptSceneBakeAll>("Bake All RTXPT Materials").minSize = new Vector2(420, 320);

        // ---- GUI ----

        private void OnGUI()
        {
            EditorGUILayout.Space(6);
            EditorGUILayout.LabelField("Output Folder (inside Assets/)", EditorStyles.boldLabel);

            EditorGUILayout.BeginHorizontal();
            _outputFolder = EditorGUILayout.TextField("Folder", _outputFolder);
            if (GUILayout.Button("…", GUILayout.Width(26)))
            {
                string abs    = ToAbsolute(_outputFolder);
                string picked = EditorUtility.OpenFolderPanel("Select output folder", abs, "");
                if (!string.IsNullOrEmpty(picked))
                {
                    string rel = AbsToRelative(picked);
                    if (rel != null) _outputFolder = rel;
                    else
                        EditorUtility.DisplayDialog("Invalid folder",
                            "Destination must be inside the project's Assets folder.", "OK");
                }
            }

            EditorGUILayout.EndHorizontal();

            // Normalise if the user typed an absolute path.
            if (System.IO.Path.IsPathRooted(_outputFolder))
            {
                string rel                     = AbsToRelative(_outputFolder);
                if (rel != null) _outputFolder = rel;
            }

            bool folderOk = !string.IsNullOrWhiteSpace(_outputFolder)
                            && _outputFolder.StartsWith("Assets", StringComparison.OrdinalIgnoreCase);
            if (!folderOk)
                EditorGUILayout.HelpBox("Must be a path starting with 'Assets'.", MessageType.Warning);

            EditorGUILayout.Space(8);
            EditorGUILayout.LabelField("Options", EditorStyles.boldLabel);
            _selectedOnly = EditorGUILayout.Toggle("Selected GameObjects Only", _selectedOnly);
            _skipExisting = EditorGUILayout.Toggle(
                new GUIContent("Skip Already-Assigned Slots",
                    "When enabled, slots that already reference a RtxptMaterial asset are left " +
                    "untouched (no new asset, no re-bake)."),
                _skipExisting);

            GUI.enabled = !_skipExisting || true; // always show but gate below
            _reBakeValues = EditorGUILayout.Toggle(
                new GUIContent("Re-bake Values on Existing Assets",
                    "When a Unity material is encountered again (same asset, slot already filled), " +
                    "overwrite the RtxptMaterial values from the Unity material. " +
                    "Ignored when 'Skip Already-Assigned Slots' is on."),
                _reBakeValues);
            GUI.enabled = true;

            EditorGUILayout.Space(4);
            EditorGUILayout.HelpBox(
                "Each unique Unity Material gets exactly one RtxptMaterial asset. " +
                "Slots sharing the same Unity material will reference the same asset.",
                MessageType.Info);

            EditorGUILayout.Space(10);

            GUI.enabled = folderOk;
            if (GUILayout.Button("Bake All", GUILayout.Height(32)))
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

        // ---- Core logic ----

        private void Run()
        {
            EnsureFolder(_outputFolder);

            // unity Material instanceID → RtxptMaterial asset path
            // Deduplication: same Material gets one RtxptMaterial, no matter how many slots use it.
            var materialCache = new Dictionary<int, RtxptMaterial>();

            // Pre-populate the cache with assets that already exist on disk so we can reuse them
            // across multiple bake runs without creating duplicates.
            string[] existingGuids = AssetDatabase.FindAssets("t:RtxptMaterial", new[] { _outputFolder.TrimEnd('/') });
            foreach (string guid in existingGuids)
            {
                string path  = AssetDatabase.GUIDToAssetPath(guid);
                var    asset = AssetDatabase.LoadAssetAtPath<RtxptMaterial>(path);
                if (asset == null) continue;
                // Store keyed by the asset's own instance ID (sentinel: negative) so we can look
                // it up later by the deterministic name we would generate.
            }

            // Build a name → asset map for name-based reuse.
            var nameToAsset = new Dictionary<string, RtxptMaterial>(StringComparer.OrdinalIgnoreCase);
            foreach (string guid in existingGuids)
            {
                string path  = AssetDatabase.GUIDToAssetPath(guid);
                var    asset = AssetDatabase.LoadAssetAtPath<RtxptMaterial>(path);
                if (asset == null) continue;
                string assetName                                                = System.IO.Path.GetFileNameWithoutExtension(path);
                if (!nameToAsset.ContainsKey(assetName)) nameToAsset[assetName] = asset;
            }

            // Collect renderers.
            Renderer[] renderers = CollectRenderers();

            int renderersTouched = 0, slotsCreated = 0, slotsReused = 0, slotsSkipped = 0;

            foreach (var mr in renderers)
            {
                var mf   = mr.GetComponent<MeshFilter>();
                var mats = mr.sharedMaterials ?? Array.Empty<Material>();


                int slotCount = 0;
                if (mf != null)
                {
                    slotCount = mf.sharedMesh.subMeshCount;
                }
                else
                {
                    slotCount = mats.Length;
                }

                if (slotCount == 0) continue;

                // Ensure RtxptRenderer exists.
                var comp = mr.GetComponent<RtxptRenderer>();
                if (comp == null)
                    comp = Undo.AddComponent<RtxptRenderer>(mr.gameObject);

                Undo.RecordObject(comp, "Bake All RTXPT Materials");

                // Resize slot list.
                while (comp.Slots.Count < slotCount) comp.Slots.Add(null);
                if (comp.Slots.Count > slotCount)
                    comp.Slots.RemoveRange(slotCount, comp.Slots.Count - slotCount);

                bool dirty = false;

                for (int s = 0; s < slotCount; s++)
                {
                    // Skip if already assigned and user chose to preserve.
                    if (_skipExisting && comp.Slots[s] != null)
                    {
                        slotsSkipped++;
                        continue;
                    }

                    Material mat = s < mats.Length ? mats[s] : (mats.Length > 0 ? mats[^1] : null);

                    RtxptMaterial rtxptAsset = ResolveAsset(mat, materialCache, nameToAsset,
                        out bool wasCreated);

                    if (wasCreated) slotsCreated++;
                    else slotsReused++;

                    // Bake values from the Unity material.
                    bool shouldBake = wasCreated || (!_skipExisting && _reBakeValues);
                    if (shouldBake)
                        RtxptRenderer.BakeSlotFromMaterial(rtxptAsset, mat);

                    comp.Slots[s] = rtxptAsset;
                    EditorUtility.SetDirty(rtxptAsset);
                    dirty = true;
                }

                if (dirty)
                {
                    EditorUtility.SetDirty(comp);
                    renderersTouched++;
                }
            }

            AssetDatabase.SaveAssets();
            EditorSceneManager.MarkAllScenesDirty();

            // Report.
            var sb = new StringBuilder();
            sb.AppendLine($"Output folder    : {_outputFolder}");
            sb.AppendLine($"Renderers updated: {renderersTouched} / {renderers.Length}");
            sb.AppendLine($"Slots — created  : {slotsCreated}");
            sb.AppendLine($"Slots — reused   : {slotsReused}");
            sb.AppendLine($"Slots — skipped  : {slotsSkipped}");
            _lastReport = sb.ToString().TrimEnd();

            Debug.Log($"[RtxptSceneBakeAll] Done. {slotsCreated} created, {slotsReused} reused, {slotsSkipped} skipped.");
        }

        // Returns (or creates) the RtxptMaterial asset for the given Unity material.
        // Reuse priority: 1) already in materialCache for this run, 2) existing on-disk asset
        // with the same deterministic name, 3) create new.
        private RtxptMaterial ResolveAsset(
            Material mat,
            Dictionary<int, RtxptMaterial> materialCache,
            Dictionary<string, RtxptMaterial> nameToAsset,
            out bool wasCreated)
        {
            wasCreated = false;

            // Null material → create a blank shared asset named "_NullMaterial".
            string assetName = mat != null ? SanitizeAssetName(mat.name) : "_NullMaterial";

            // 1. Cache hit from this bake run (same Material instance seen before).
            if (mat != null && materialCache.TryGetValue(mat.GetInstanceID(), out var cached))
                return cached;

            // 2. Existing on-disk asset with matching name (persists across bake runs).
            if (nameToAsset.TryGetValue(assetName, out var existing))
            {
                if (mat != null) materialCache[mat.GetInstanceID()] = existing;
                return existing;
            }

            // 3. Create new asset.
            wasCreated = true;
            var    asset = CreateInstance<RtxptMaterial>();
            string path  = AssetDatabase.GenerateUniqueAssetPath($"{_outputFolder}/{assetName}.asset");
            AssetDatabase.CreateAsset(asset, path);

            // Register in both lookup structures so subsequent slots with the same material reuse it.
            if (mat != null) materialCache[mat.GetInstanceID()] = asset;
            nameToAsset[assetName] = asset;

            return asset;
        }

        // ---- Helpers ----

        private Renderer[] CollectRenderers()
        {
            if (_selectedOnly)
            {
                var list = new List<Renderer>();
                foreach (var go in Selection.gameObjects)
                    list.AddRange(go.GetComponentsInChildren<Renderer>(includeInactive: true));
                return list.ToArray();
            }

            return FindObjectsByType<Renderer>(FindObjectsInactive.Include, FindObjectsSortMode.None);
        }

        /// <summary>Ensures every segment of <paramref name="folderPath"/> exists.</summary>
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

        /// <summary>Replaces characters that are invalid in Unity asset file names.</summary>
        private static string SanitizeAssetName(string name)
        {
            if (string.IsNullOrWhiteSpace(name)) return "_Unnamed";
            var sb = new StringBuilder(name.Length);
            foreach (char c in name)
                sb.Append(c is '/' or '\\' or ':' or '*' or '?' or '"' or '<' or '>' or '|' ? '_' : c);
            return sb.ToString();
        }

        private static string ProjectRoot()
            => System.IO.Path.GetFullPath(
                    System.IO.Path.Combine(Application.dataPath, ".."))
                .Replace('\\', '/').TrimEnd('/');

        private static string AbsToRelative(string absPath)
        {
            if (string.IsNullOrEmpty(absPath)) return null;
            string norm = System.IO.Path.GetFullPath(absPath).Replace('\\', '/').TrimEnd('/');
            string root = ProjectRoot();
            if (!norm.StartsWith(root, StringComparison.OrdinalIgnoreCase)) return null;
            string rel = norm.Substring(root.Length).TrimStart('/');
            return rel.StartsWith("Assets", StringComparison.OrdinalIgnoreCase) ? rel : null;
        }

        private static string ToAbsolute(string relPath)
        {
            if (string.IsNullOrWhiteSpace(relPath)) return Application.dataPath;
            if (System.IO.Path.IsPathRooted(relPath)) return relPath;
            return System.IO.Path.GetFullPath(
                    System.IO.Path.Combine(ProjectRoot(), relPath))
                .Replace('\\', '/');
        }
    }
}
#endif