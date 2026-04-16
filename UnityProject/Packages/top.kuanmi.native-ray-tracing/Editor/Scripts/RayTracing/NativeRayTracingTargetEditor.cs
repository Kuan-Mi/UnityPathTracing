using System.IO;
using UnityEditor;
using UnityEngine;

namespace NativeRender
{
    [CustomEditor(typeof(NativeRayTracingTarget))]
    public class NativeRayTracingTargetEditor : Editor
    {
        // ── Bake settings (editor-only, not serialized on the component) ──
        private string _saveFolder = "Assets/OMMCaches";

        // Per-submesh texture overrides and bake params
        private Texture2D[]            _texOverride;
        private float[]                _alphaCutoff;
        private byte[]                 _maxSubdiv;
        private float[]                _dynScale;
        private OMMCache.OmmFormat[]   _ommFormat;
        private OMMCache.DownsampleFactor[] _downsample;

        // UI state
        private bool   _showBake  = true;
        private string _bakeStatus = null;
        private bool   _bakeSuccess = false;

        // ── Cached component references ────────────────────────────────────
        private NativeRayTracingTarget Target     => (NativeRayTracingTarget)target;
        private MeshFilter             _meshFilter;
        private MeshRenderer           _renderer;
        private Mesh                   _prevMesh;
        private int                    _prevSubCount = -1;

        private void OnEnable()
        {
            _meshFilter = Target.GetComponent<MeshFilter>();
            _renderer   = Target.GetComponent<MeshRenderer>();
            RebuildPerSubmeshArrays();
        }

        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            EditorGUILayout.Space(8);
            _showBake = EditorGUILayout.BeginFoldoutHeaderGroup(_showBake, "OMM Bake");
            if (_showBake)
            {
                DrawBakeSection();
            }
            EditorGUILayout.EndFoldoutHeaderGroup();
        }

        // ──────────────────────────────────────────────────────────────────
        private void DrawBakeSection()
        {
            Mesh mesh = _meshFilter != null ? _meshFilter.sharedMesh : null;

            // Detect mesh change → re-init per-submesh arrays
            if (mesh != _prevMesh)
            {
                _prevMesh = mesh;
                RebuildPerSubmeshArrays();
            }

            if (mesh == null)
            {
                EditorGUILayout.HelpBox("No Mesh found on MeshFilter.", MessageType.Warning);
                return;
            }

            int subCount = mesh.subMeshCount;
            if (subCount != _prevSubCount)
            {
                RebuildPerSubmeshArrays();
            }

            // Save folder
            EditorGUILayout.BeginHorizontal();
            _saveFolder = EditorGUILayout.TextField("Save Folder", _saveFolder);
            if (GUILayout.Button("Browse", GUILayout.Width(60)))
            {
                string chosen = EditorUtility.SaveFolderPanel("Select OMMCache Save Folder",
                    _saveFolder.StartsWith("Assets/") ? Path.Combine(Application.dataPath, _saveFolder.Substring(7)) : Application.dataPath,
                    "");
                if (!string.IsNullOrEmpty(chosen))
                {
                    // Convert absolute path back to project-relative
                    if (chosen.StartsWith(Application.dataPath))
                        chosen = "Assets" + chosen.Substring(Application.dataPath.Length).Replace('\\', '/');
                    _saveFolder = chosen;
                }
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.Space(4);
            EditorGUILayout.LabelField("Submeshes", EditorStyles.boldLabel);

            // Per-submesh rows
            for (int i = 0; i < subCount; i++)
            {
                DrawSubmeshRow(i, mesh, subCount);
            }

            EditorGUILayout.Space(6);

            // Bake all button
            using (new EditorGUI.DisabledScope(mesh == null))
            {
                if (GUILayout.Button("Bake All Submeshes", GUILayout.Height(28)))
                {
                    BakeAll(mesh);
                }
            }

            // Status
            if (!string.IsNullOrEmpty(_bakeStatus))
            {
                var style = _bakeSuccess ? EditorStyles.helpBox : EditorStyles.helpBox;
                var msgType = _bakeSuccess ? MessageType.Info : MessageType.Error;
                EditorGUILayout.HelpBox(_bakeStatus, msgType);
            }
        }

        private void DrawSubmeshRow(int i, Mesh mesh, int subCount)
        {
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);

            EditorGUILayout.BeginHorizontal();
            string label = subCount == 1 ? "Submesh 0 (only)" : $"Submesh {i}";
            EditorGUILayout.LabelField(label, EditorStyles.miniBoldLabel, GUILayout.Width(100));

            // Cache status badge
            OMMCache existing = (Target.ommCaches != null && i < Target.ommCaches.Length)
                ? Target.ommCaches[i] : null;
            if (existing == null)
            {
                GUILayout.Label("[ no cache ]", EditorStyles.miniLabel);
            }
            else if (existing.IsValid)
            {
                var prevColor = GUI.color;
                GUI.color = new Color(0.4f, 1f, 0.4f);
                GUILayout.Label("✓ valid", EditorStyles.miniLabel);
                GUI.color = prevColor;
            }
            else
            {
                var prevColor = GUI.color;
                GUI.color = new Color(1f, 0.7f, 0.2f);
                GUILayout.Label("⚠ invalid", EditorStyles.miniLabel);
                GUI.color = prevColor;
            }

            GUILayout.FlexibleSpace();

            // Per-submesh bake button
            if (GUILayout.Button("Bake", GUILayout.Width(50), GUILayout.Height(18)))
            {
                BakeSingle(i, mesh);
            }
            EditorGUILayout.EndHorizontal();

            // Texture override
            _texOverride[i] = (Texture2D)EditorGUILayout.ObjectField(
                "Alpha Texture", _texOverride[i], typeof(Texture2D), false);

            // Alpha cutoff
            _alphaCutoff[i] = EditorGUILayout.FloatField("Alpha Cutoff", _alphaCutoff[i]);

            // Max subdivision
            _maxSubdiv[i] = (byte)EditorGUILayout.IntSlider("Max Subdivision", _maxSubdiv[i], 0, 12);

            // Dynamic subdivision scale
            _dynScale[i] = EditorGUILayout.Slider("Dynamic Scale", _dynScale[i], 0f, 12f);

            // OMM Format
            _ommFormat[i] = (OMMCache.OmmFormat)EditorGUILayout.EnumPopup("OMM Format", _ommFormat[i]);

            // Downsample factor
            _downsample[i] = (OMMCache.DownsampleFactor)EditorGUILayout.EnumPopup("Texture Downsample", _downsample[i]);

            EditorGUILayout.EndVertical();
        }

        // ──────────────────────────────────────────────────────────────────
        private void RebuildPerSubmeshArrays()
        {
            Mesh mesh = _meshFilter != null ? _meshFilter.sharedMesh : null;
            int n = (mesh != null) ? Mathf.Max(1, mesh.subMeshCount) : 1;
            _prevSubCount = (mesh != null) ? mesh.subMeshCount : -1;

            // Preserve existing values when growing
            int oldN = (_texOverride != null) ? _texOverride.Length : 0;

            System.Array.Resize(ref _texOverride, n);
            System.Array.Resize(ref _alphaCutoff,  n);
            System.Array.Resize(ref _maxSubdiv,    n);
            System.Array.Resize(ref _dynScale,     n);
            System.Array.Resize(ref _ommFormat,    n);
            System.Array.Resize(ref _downsample,   n);

            // Init new slots with defaults and auto-detect textures
            for (int i = oldN; i < n; i++)
            {
                _texOverride[i] = AutoDetectTexture(i);
                _alphaCutoff[i] = 0.5f;
                _maxSubdiv[i]   = 8;
                _dynScale[i]    = 2f;
                _ommFormat[i]   = OMMCache.OmmFormat.FourState;
                _downsample[i]  = OMMCache.DownsampleFactor.x1;
            }

            // Auto-fill textures for existing slots that are null
            for (int i = 0; i < Mathf.Min(oldN, n); i++)
            {
                if (_texOverride[i] == null)
                    _texOverride[i] = AutoDetectTexture(i);
            }
        }

        private Texture2D AutoDetectTexture(int submeshIndex)
        {
            if (_renderer == null) return null;
            var mats = _renderer.sharedMaterials;
            if (mats == null || submeshIndex >= mats.Length) return null;
            var mat = mats[submeshIndex];
            if (mat == null) return null;

            // Try common alpha/cutout texture property names first
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
        private void BakeAll(Mesh mesh)
        {
            bool allOk = true;
            for (int i = 0; i < mesh.subMeshCount; i++)
            {
                if (!BakeSingle(i, mesh))
                    allOk = false;
            }
            _bakeStatus  = allOk ? $"All {mesh.subMeshCount} submesh(es) baked successfully." : "Some submeshes failed — check Console.";
            _bakeSuccess = allOk;
        }

        private bool BakeSingle(int submeshIndex, Mesh mesh)
        {
            _bakeStatus = null;

            // Validate texture
            Texture2D tex = _texOverride[submeshIndex];
            if (tex == null)
            {
                _bakeStatus  = $"Submesh {submeshIndex}: No alpha texture specified.";
                _bakeSuccess = false;
                Debug.LogError($"[NativeRayTracingTarget] {_bakeStatus}");
                return false;
            }

            // Ensure save folder exists
            if (!AssetDatabase.IsValidFolder(_saveFolder))
            {
                CreateFolderRecursive(_saveFolder);
            }

            // Determine asset path
            string goName   = Target.gameObject.name;
            string meshName = mesh.name;
            string assetName = $"{goName}_{meshName}_sub{submeshIndex}.asset";
            // Sanitize
            foreach (char c in Path.GetInvalidFileNameChars())
                assetName = assetName.Replace(c, '_');
            string assetPath = $"{_saveFolder}/{assetName}";

            // Reuse existing asset or create new
            OMMCache cache = AssetDatabase.LoadAssetAtPath<OMMCache>(assetPath);
            if (cache == null)
            {
                cache = ScriptableObject.CreateInstance<OMMCache>();
                AssetDatabase.CreateAsset(cache, assetPath);
            }

            // Populate bake parameters
            cache.sourceMesh             = mesh;
            cache.submeshIndex           = submeshIndex;
            cache.sourceTexture          = tex;
            cache.alphaCutoff            = _alphaCutoff[submeshIndex];
            cache.maxSubdivisionLevel    = _maxSubdiv[submeshIndex];
            cache.dynamicSubdivisionScale = _dynScale[submeshIndex];
            cache.ommFormat              = _ommFormat[submeshIndex];
            cache.textureDownsampleFactor = _downsample[submeshIndex];
            EditorUtility.SetDirty(cache);

            // Bake
            bool ok = OMMCacheEditor.BakeInto(cache, submeshIndex);

            if (ok)
            {
                EditorUtility.SetDirty(cache);
                AssetDatabase.SaveAssets();

                // Write back into ommCaches array
                Undo.RecordObject(Target, "Bake OMM");
                if (Target.ommCaches == null || Target.ommCaches.Length <= submeshIndex)
                {
                    int newLen = submeshIndex + 1;
                    var newArr = new OMMCache[newLen];
                    if (Target.ommCaches != null)
                        System.Array.Copy(Target.ommCaches, newArr, Target.ommCaches.Length);
                    Target.ommCaches = newArr;
                }
                Target.ommCaches[submeshIndex] = cache;
                EditorUtility.SetDirty(Target);
                if (!Application.isPlaying)
                    UnityEditor.SceneManagement.EditorSceneManager.MarkSceneDirty(Target.gameObject.scene);

                _bakeStatus  = $"Submesh {submeshIndex} baked ✓  ({cache.bakedIndexCount / 3} triangles)";
                _bakeSuccess = true;
                Debug.Log($"[NativeRayTracingTarget] {_bakeStatus}  →  {assetPath}");
            }
            else
            {
                _bakeStatus  = $"Submesh {submeshIndex} bake FAILED. Check Console for details.";
                _bakeSuccess = false;
            }

            return ok;
        }

        // ──────────────────────────────────────────────────────────────────
        private static void CreateFolderRecursive(string folderPath)
        {
            // folderPath like "Assets/Foo/Bar/Baz"
            string[] parts = folderPath.Split('/');
            string current = parts[0]; // "Assets"
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
