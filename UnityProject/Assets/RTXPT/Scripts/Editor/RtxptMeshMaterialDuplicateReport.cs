using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace PathTracing
{
    public sealed class RtxptMeshMaterialDuplicateReport : EditorWindow
    {
        private enum MaterialSource
        {
            RtxptSlots,
            UnitySharedMaterials,
        }

        private sealed class Entry
        {
            public string Key;
            public Mesh Mesh;
            public string MaterialSummary;
            public readonly List<Renderer> Renderers = new();
        }

        private readonly List<Entry> _groups = new();
        private Vector2 _scroll;
        private MaterialSource _materialSource = MaterialSource.RtxptSlots;
        private bool _includeInactive = true;
        private bool _onlyDuplicates = true;
        private string _summary = "";

        [MenuItem("RTXPT/Diagnostics/Mesh Material Duplicate Report")]
        private static void Open()
        {
            GetWindow<RtxptMeshMaterialDuplicateReport>("RTXPT Mesh/Material");
        }

        private void OnGUI()
        {
            EditorGUILayout.LabelField("Current Scene Mesh + Material Groups", EditorStyles.boldLabel);

            using (new EditorGUILayout.HorizontalScope())
            {
                _materialSource = (MaterialSource)EditorGUILayout.EnumPopup("Material Source", _materialSource);
                _includeInactive = EditorGUILayout.ToggleLeft("Include Inactive", _includeInactive, GUILayout.Width(120));
                _onlyDuplicates = EditorGUILayout.ToggleLeft("Only Duplicates", _onlyDuplicates, GUILayout.Width(115));
            }

            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button("Scan Scene", GUILayout.Width(120)))
                    Scan();
                if (GUILayout.Button("Copy Report", GUILayout.Width(120)))
                    EditorGUIUtility.systemCopyBuffer = BuildTextReport();
            }

            if (!string.IsNullOrEmpty(_summary))
                EditorGUILayout.HelpBox(_summary, MessageType.Info);

            _scroll = EditorGUILayout.BeginScrollView(_scroll);
            foreach (var group in _groups)
            {
                if (_onlyDuplicates && group.Renderers.Count < 2)
                    continue;

                using (new EditorGUILayout.VerticalScope(EditorStyles.helpBox))
                {
                    using (new EditorGUILayout.HorizontalScope())
                    {
                        EditorGUILayout.LabelField($"{group.Renderers.Count} x {ObjectName(group.Mesh)}", EditorStyles.boldLabel);
                        if (GUILayout.Button("Select", GUILayout.Width(70)))
                            Selection.objects = group.Renderers.Select(r => r.gameObject).Cast<UnityEngine.Object>().ToArray();
                    }

                    EditorGUILayout.LabelField("Materials", group.MaterialSummary);
                    foreach (var renderer in group.Renderers)
                    {
                        using (new EditorGUILayout.HorizontalScope())
                        {
                            EditorGUILayout.ObjectField(renderer, typeof(Renderer), true);
                            if (GUILayout.Button("Ping", GUILayout.Width(55)))
                                EditorGUIUtility.PingObject(renderer.gameObject);
                        }
                    }
                }
            }
            EditorGUILayout.EndScrollView();
        }

        private void Scan()
        {
            _groups.Clear();
            var byKey = new Dictionary<string, Entry>();

            foreach (var renderer in FindObjectsByType<Renderer>(
                         _includeInactive ? FindObjectsInactive.Include : FindObjectsInactive.Exclude,
                         FindObjectsSortMode.None))
            {
                if (EditorUtility.IsPersistent(renderer))
                    continue;

                var mesh = GetMesh(renderer);
                if (mesh == null)
                    continue;

                string materialKey;
                string materialSummary;
                if (_materialSource == MaterialSource.RtxptSlots)
                {
                    if (!TryGetRtxptMaterials(renderer, out materialKey, out materialSummary))
                        continue;
                }
                else
                {
                    var materials = renderer.sharedMaterials ?? Array.Empty<Material>();
                    materialKey = string.Join("|", materials.Select(ObjectKey));
                    materialSummary = string.Join(", ", materials.Select(ObjectName));
                }

                string key = $"{ObjectKey(mesh)}::{materialKey}";
                if (!byKey.TryGetValue(key, out var entry))
                {
                    entry = new Entry
                    {
                        Key = key,
                        Mesh = mesh,
                        MaterialSummary = materialSummary,
                    };
                    byKey.Add(key, entry);
                    _groups.Add(entry);
                }
                entry.Renderers.Add(renderer);
            }

            _groups.Sort((a, b) => b.Renderers.Count.CompareTo(a.Renderers.Count));

            int duplicateGroups = _groups.Count(g => g.Renderers.Count > 1);
            int duplicateObjects = _groups.Where(g => g.Renderers.Count > 1).Sum(g => g.Renderers.Count);
            _summary = $"Groups: {_groups.Count}, duplicate groups: {duplicateGroups}, objects in duplicate groups: {duplicateObjects}.";
        }

        private static Mesh GetMesh(Renderer renderer)
        {
            if (renderer is MeshRenderer)
            {
                var filter = renderer.GetComponent<MeshFilter>();
                return filter != null ? filter.sharedMesh : null;
            }

            if (renderer is SkinnedMeshRenderer skinned)
                return skinned.sharedMesh;

            return null;
        }

        private static bool TryGetRtxptMaterials(Renderer renderer, out string key, out string summary)
        {
            key = "";
            summary = "";

            var rtxpt = renderer.GetComponent<RtxptRenderer>();
            if (rtxpt == null || rtxpt.Slots == null || rtxpt.Slots.Count == 0)
                return false;

            key = string.Join("|", rtxpt.Slots.Select(ObjectKey));
            summary = string.Join(", ", rtxpt.Slots.Select(ObjectName));
            return true;
        }

        private string BuildTextReport()
        {
            var sb = new StringBuilder();
            sb.AppendLine(_summary);
            sb.AppendLine($"Material source: {_materialSource}");

            foreach (var group in _groups)
            {
                if (_onlyDuplicates && group.Renderers.Count < 2)
                    continue;

                sb.AppendLine();
                sb.AppendLine($"{group.Renderers.Count} x {ObjectName(group.Mesh)}");
                sb.AppendLine($"Materials: {group.MaterialSummary}");
                foreach (var renderer in group.Renderers)
                    sb.AppendLine($"  {GetHierarchyPath(renderer.transform)}");
            }

            return sb.ToString();
        }

        private static string ObjectKey(UnityEngine.Object obj)
        {
            if (obj == null)
                return "<null>";

#if UNITY_2020_1_OR_NEWER
            if (AssetDatabase.TryGetGUIDAndLocalFileIdentifier(obj, out string guid, out long localId))
                return $"{guid}:{localId}";
#endif

            return $"{obj.GetInstanceID()}:{obj.name}";
        }

        private static string ObjectName(UnityEngine.Object obj)
        {
            return obj != null ? obj.name : "<null>";
        }

        private static string GetHierarchyPath(Transform transform)
        {
            var names = new Stack<string>();
            for (var t = transform; t != null; t = t.parent)
                names.Push(t.name);
            return string.Join("/", names);
        }
    }
}
