using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace NativeRender
{
    [CustomEditor(typeof(NativeRayTracingTarget))]
    [CanEditMultipleObjects]
    public class NativeRayTracingTargetEditor : Editor
    {
        // ── Shared bake settings ──────────────────────────────────────────
        private string _saveFolder  = "Assets/OMMCaches";
        private bool   _showBake    = true;
        private string _bakeStatus  = null;
        private bool   _bakeSuccess = false;

        // ── Per-target editor state ───────────────────────────────────────
        private class TargetState
        {
            public MeshFilter  meshFilter;
            public MeshRenderer renderer;
            public Mesh        prevMesh;
            public int         prevSubCount = -1;

            public Texture2D[]                 texOverride;
            public float[]                     alphaCutoff;
            public byte[]                      maxSubdiv;
            public float[]                     dynScale;
            public OMMCache.OmmFormat[]        ommFormat;
            public OMMCache.DownsampleFactor[] downsample;
        }

        private readonly Dictionary<int, TargetState> _states = new Dictionary<int, TargetState>();

        private void OnEnable()
        {
            foreach (var t in targets)
            {
                var nrt = (NativeRayTracingTarget)t;
                int id = nrt.GetInstanceID();
                if (!_states.ContainsKey(id))
                    _states[id] = CreateState(nrt);
            }
        }

        private TargetState CreateState(NativeRayTracingTarget nrt)
        {
            var state = new TargetState
            {
                meshFilter = nrt.GetComponent<MeshFilter>(),
                renderer   = nrt.GetComponent<MeshRenderer>()
            };
            RebuildArrays(state);
            return state;
        }

        // ──────────────────────────────────────────────────────────────────
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            EditorGUILayout.Space(8);
            _showBake = EditorGUILayout.BeginFoldoutHeaderGroup(_showBake, "OMM Bake");
            if (_showBake)
                DrawBakeSection();
            EditorGUILayout.EndFoldoutHeaderGroup();
        }

        private void DrawBakeSection()
        {
            // Save folder (shared across all selected objects)
            EditorGUILayout.BeginHorizontal();
            _saveFolder = EditorGUILayout.TextField("Save Folder", _saveFolder);
            if (GUILayout.Button("Browse", GUILayout.Width(60)))
            {
                string chosen = EditorUtility.SaveFolderPanel("Select OMMCache Save Folder",
                    _saveFolder.StartsWith("Assets/") ? Path.Combine(Application.dataPath, _saveFolder.Substring(7)) : Application.dataPath,
                    "");
                if (!string.IsNullOrEmpty(chosen))
                {
                    if (chosen.StartsWith(Application.dataPath))
                        chosen = "Assets" + chosen.Substring(Application.dataPath.Length).Replace('\\', '/');
                    _saveFolder = chosen;
                }
            }
            EditorGUILayout.EndHorizontal();
            EditorGUILayout.Space(4);

            bool multiSelect = targets.Length > 1;

            foreach (var t in targets)
            {
                var nrt = (NativeRayTracingTarget)t;
                int id  = nrt.GetInstanceID();

                if (!_states.TryGetValue(id, out var state))
                {
                    state = CreateState(nrt);
                    _states[id] = state;
                }

                Mesh mesh = state.meshFilter != null ? state.meshFilter.sharedMesh : null;

                // Detect mesh / subMesh count change
                if (mesh != state.prevMesh)
                {
                    state.prevMesh = mesh;
                    RebuildArrays(state);
                }
                else if (mesh != null && mesh.subMeshCount != state.prevSubCount)
                {
                    RebuildArrays(state);
                }

                if (multiSelect)
                {
                    EditorGUILayout.LabelField($"► {nrt.gameObject.name}", EditorStyles.boldLabel);
                    EditorGUI.indentLevel++;
                }

                if (mesh == null)
                {
                    EditorGUILayout.HelpBox("No Mesh found on MeshFilter.", MessageType.Warning);
                }
                else
                {
                    EditorGUILayout.LabelField("Submeshes", EditorStyles.boldLabel);
                    int subCount = mesh.subMeshCount;
                    for (int i = 0; i < subCount; i++)
                        DrawSubmeshRow(nrt, state, i, mesh, subCount);

                    EditorGUILayout.Space(4);
                    string bakeLabel = multiSelect ? $"Bake All Submeshes  ({nrt.gameObject.name})" : "Bake All Submeshes";
                    if (GUILayout.Button(bakeLabel, GUILayout.Height(24)))
                        BakeAll(nrt, mesh, state, updateStatus: !multiSelect);
                }

                if (multiSelect)
                {
                    EditorGUI.indentLevel--;
                    EditorGUILayout.Space(6);
                }
            }

            // Batch bake button shown only when multiple objects are selected
            if (multiSelect)
            {
                EditorGUILayout.Space(2);
                if (GUILayout.Button($"Bake All {targets.Length} Selected Objects", GUILayout.Height(28)))
                {
                    bool allOk = true;
                    foreach (var t in targets)
                    {
                        var nrt = (NativeRayTracingTarget)t;
                        if (!_states.TryGetValue(nrt.GetInstanceID(), out var state)) continue;
                        Mesh mesh = state.meshFilter != null ? state.meshFilter.sharedMesh : null;
                        if (mesh == null) continue;
                        if (!BakeAll(nrt, mesh, state, updateStatus: false))
                            allOk = false;
                    }
                    _bakeStatus  = allOk ? $"All {targets.Length} object(s) baked successfully." : "Some objects failed — check Console.";
                    _bakeSuccess = allOk;
                }
            }

            if (!string.IsNullOrEmpty(_bakeStatus))
                EditorGUILayout.HelpBox(_bakeStatus, _bakeSuccess ? MessageType.Info : MessageType.Error);
        }

        private void DrawSubmeshRow(NativeRayTracingTarget nrt, TargetState state, int i, Mesh mesh, int subCount)
        {
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);

            EditorGUILayout.BeginHorizontal();
            string label = subCount == 1 ? "Submesh 0 (only)" : $"Submesh {i}";
            EditorGUILayout.LabelField(label, EditorStyles.miniBoldLabel, GUILayout.Width(100));

            // Cache status badge
            OMMCache existing = (nrt.ommCaches != null && i < nrt.ommCaches.Length) ? nrt.ommCaches[i] : null;
            if (existing == null)
            {
                GUILayout.Label("[ no cache ]", EditorStyles.miniLabel);
            }
            else if (existing.IsValid)
            {
                var prev = GUI.color;
                GUI.color = new Color(0.4f, 1f, 0.4f);
                GUILayout.Label("✓ valid", EditorStyles.miniLabel);
                GUI.color = prev;
            }
            else
            {
                var prev = GUI.color;
                GUI.color = new Color(1f, 0.7f, 0.2f);
                GUILayout.Label("⚠ invalid", EditorStyles.miniLabel);
                GUI.color = prev;
            }

            GUILayout.FlexibleSpace();
            if (GUILayout.Button("Bake", GUILayout.Width(50), GUILayout.Height(18)))
                BakeSingle(nrt, state, i, mesh, updateStatus: true);
            EditorGUILayout.EndHorizontal();

            state.texOverride[i] = (Texture2D)EditorGUILayout.ObjectField("Alpha Texture",    state.texOverride[i], typeof(Texture2D), false);
            state.alphaCutoff[i] = EditorGUILayout.FloatField(               "Alpha Cutoff",    state.alphaCutoff[i]);
            state.maxSubdiv[i]   = (byte)EditorGUILayout.IntSlider(          "Max Subdivision", state.maxSubdiv[i], 0, 12);
            state.dynScale[i]    = EditorGUILayout.Slider(                   "Dynamic Scale",   state.dynScale[i], 0f, 12f);
            state.ommFormat[i]   = (OMMCache.OmmFormat)EditorGUILayout.EnumPopup(        "OMM Format",        state.ommFormat[i]);
            state.downsample[i]  = (OMMCache.DownsampleFactor)EditorGUILayout.EnumPopup( "Texture Downsample", state.downsample[i]);

            EditorGUILayout.EndVertical();
        }

        // ──────────────────────────────────────────────────────────────────
        private void RebuildArrays(TargetState state)
        {
            Mesh mesh = state.meshFilter != null ? state.meshFilter.sharedMesh : null;
            int n = (mesh != null) ? Mathf.Max(1, mesh.subMeshCount) : 1;
            state.prevSubCount = (mesh != null) ? mesh.subMeshCount : -1;

            int oldN = (state.texOverride != null) ? state.texOverride.Length : 0;

            System.Array.Resize(ref state.texOverride, n);
            System.Array.Resize(ref state.alphaCutoff,  n);
            System.Array.Resize(ref state.maxSubdiv,    n);
            System.Array.Resize(ref state.dynScale,     n);
            System.Array.Resize(ref state.ommFormat,    n);
            System.Array.Resize(ref state.downsample,   n);

            for (int i = oldN; i < n; i++)
            {
                state.texOverride[i] = AutoDetectTexture(state, i);
                state.alphaCutoff[i] = 0.5f;
                state.maxSubdiv[i]   = 8;
                state.dynScale[i]    = 2f;
                state.ommFormat[i]   = OMMCache.OmmFormat.FourState;
                state.downsample[i]  = OMMCache.DownsampleFactor.x1;
            }

            for (int i = 0; i < Mathf.Min(oldN, n); i++)
            {
                if (state.texOverride[i] == null)
                    state.texOverride[i] = AutoDetectTexture(state, i);
            }
        }

        private static Texture2D AutoDetectTexture(TargetState state, int submeshIndex)
        {
            if (state.renderer == null) return null;
            var mats = state.renderer.sharedMaterials;
            if (mats == null || submeshIndex >= mats.Length) return null;
            var mat = mats[submeshIndex];
            if (mat == null) return null;

            foreach (var propName in new[] { "_AlphaMap", "_AlphaTex", "_MaskTex", "_MainTex", "_BaseMap" })
            {
                if (mat.HasProperty(propName))
                {
                    var tex = mat.GetTexture(propName) as Texture2D;
                    if (tex != null) return tex;
                }
            }
            return mat.mainTexture as Texture2D;
        }

        // ──────────────────────────────────────────────────────────────────
        private bool BakeAll(NativeRayTracingTarget nrt, Mesh mesh, TargetState state, bool updateStatus)
        {
            bool allOk = true;
            for (int i = 0; i < mesh.subMeshCount; i++)
            {
                if (!BakeSingle(nrt, state, i, mesh, updateStatus: false))
                    allOk = false;
            }
            if (updateStatus)
            {
                _bakeStatus  = allOk ? $"All {mesh.subMeshCount} submesh(es) baked successfully." : "Some submeshes failed — check Console.";
                _bakeSuccess = allOk;
            }
            return allOk;
        }

        private bool BakeSingle(NativeRayTracingTarget nrt, TargetState state, int submeshIndex, Mesh mesh, bool updateStatus)
        {
            Texture2D tex = state.texOverride[submeshIndex];
            if (tex == null)
            {
                string msg = $"Submesh {submeshIndex}: No alpha texture specified.";
                if (updateStatus) { _bakeStatus = msg; _bakeSuccess = false; }
                Debug.LogError($"[NativeRayTracingTarget] ({nrt.gameObject.name}) {msg}");
                return false;
            }

            if (!AssetDatabase.IsValidFolder(_saveFolder))
                CreateFolderRecursive(_saveFolder);

            string assetName = $"{nrt.gameObject.name}_{mesh.name}_sub{submeshIndex}.asset";
            foreach (char c in Path.GetInvalidFileNameChars())
                assetName = assetName.Replace(c, '_');
            string assetPath = $"{_saveFolder}/{assetName}";

            OMMCache cache = AssetDatabase.LoadAssetAtPath<OMMCache>(assetPath);
            if (cache == null)
            {
                cache = ScriptableObject.CreateInstance<OMMCache>();
                AssetDatabase.CreateAsset(cache, assetPath);
            }

            cache.sourceMesh              = mesh;
            cache.submeshIndex            = submeshIndex;
            cache.sourceTexture           = tex;
            cache.alphaCutoff             = state.alphaCutoff[submeshIndex];
            cache.maxSubdivisionLevel     = state.maxSubdiv[submeshIndex];
            cache.dynamicSubdivisionScale = state.dynScale[submeshIndex];
            cache.ommFormat               = state.ommFormat[submeshIndex];
            cache.textureDownsampleFactor = state.downsample[submeshIndex];
            EditorUtility.SetDirty(cache);

            bool ok = OMMCacheEditor.BakeInto(cache, submeshIndex);

            if (ok)
            {
                EditorUtility.SetDirty(cache);
                AssetDatabase.SaveAssets();

                Undo.RecordObject(nrt, "Bake OMM");
                if (nrt.ommCaches == null || nrt.ommCaches.Length <= submeshIndex)
                {
                    int newLen = submeshIndex + 1;
                    var newArr = new OMMCache[newLen];
                    if (nrt.ommCaches != null)
                        System.Array.Copy(nrt.ommCaches, newArr, nrt.ommCaches.Length);
                    nrt.ommCaches = newArr;
                }
                nrt.ommCaches[submeshIndex] = cache;
                EditorUtility.SetDirty(nrt);
                if (!Application.isPlaying)
                    UnityEditor.SceneManagement.EditorSceneManager.MarkSceneDirty(nrt.gameObject.scene);

                string statusMsg = $"Submesh {submeshIndex} baked ✓  ({cache.bakedIndexCount / 3} triangles)";
                if (updateStatus) { _bakeStatus = statusMsg; _bakeSuccess = true; }
                Debug.Log($"[NativeRayTracingTarget] ({nrt.gameObject.name}) {statusMsg}  →  {assetPath}");
            }
            else
            {
                string statusMsg = $"Submesh {submeshIndex} bake FAILED. Check Console for details.";
                if (updateStatus) { _bakeStatus = statusMsg; _bakeSuccess = false; }
            }

            return ok;
        }

        // ──────────────────────────────────────────────────────────────────
        private static void CreateFolderRecursive(string folderPath)
        {
            string[] parts = folderPath.Split('/');
            string current = parts[0];
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
