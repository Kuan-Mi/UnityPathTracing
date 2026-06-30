#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace PathTracing
{
    [CustomEditor(typeof(RtxptRenderer))]
    [CanEditMultipleObjects]
    public class RtxptRendererEditor : UnityEditor.Editor
    {
        private bool[] _slotFoldouts = Array.Empty<bool>();

        public override void OnInspectorGUI()
        {
            bool multiEdit = targets.Length > 1;

            // ---- Top toolbar ----
            EditorGUILayout.BeginHorizontal();

            if (GUILayout.Button("Bake from Renderer", GUILayout.Height(28)))
            {
                foreach (var t in targets)
                {
                    var c = (RtxptRenderer)t;
                    EnsureSlotAssets(c);
                    Undo.RecordObject(c, "Bake RTXPT Materials from Renderer");
                    c.BakeFromRenderer();
                    EditorUtility.SetDirty(c);
                    foreach (var asset in c.Slots)
                        if (asset != null) EditorUtility.SetDirty(asset);
                }
                AssetDatabase.SaveAssets();
            }

            EditorGUILayout.EndHorizontal();
            EditorGUILayout.Space(4);

            if (multiEdit)
            {
                EditorGUILayout.HelpBox($"Editing {targets.Length} objects.", MessageType.None);
                EditorGUILayout.Space(2);
            }

            // Auto-sync slot count to sub-mesh count for the primary target.
            var primaryComp = (RtxptRenderer)target;
            SyncSlotCount(primaryComp);

            serializedObject.Update();
            var slotsProp    = serializedObject.FindProperty("Slots");
            int primaryCount = primaryComp.Slots?.Count ?? 0;

            if (primaryCount == 0)
            {
                EditorGUILayout.HelpBox("No sub-meshes found on the MeshRenderer.", MessageType.Info);
                serializedObject.ApplyModifiedProperties();
                return;
            }

            // Sync foldout array size.
            if (_slotFoldouts.Length != primaryCount)
            {
                Array.Resize(ref _slotFoldouts, primaryCount);
                for (int i = 0; i < _slotFoldouts.Length; i++) _slotFoldouts[i] = true;
            }

            for (int i = 0; i < primaryCount; i++)
            {
                var    assetRef = primaryComp.Slots[i];
                string label    = assetRef != null ? assetRef.name : $"Slot {i}  (not rendered)";

                _slotFoldouts[i] = EditorGUILayout.BeginFoldoutHeaderGroup(_slotFoldouts[i], $"Sub-mesh {i}  —  {label}");

                if (_slotFoldouts[i])
                {
                    EditorGUI.indentLevel++;

                    // Asset reference field — drag any RtxptMaterial here.
                    EditorGUILayout.BeginHorizontal();
                    EditorGUILayout.PropertyField(slotsProp.GetArrayElementAtIndex(i), new GUIContent("Override Asset"));

                    // "Create" button to make a new asset for this slot.
                    if (assetRef == null && GUILayout.Button("Create", GUILayout.Width(54), GUILayout.Height(18)))
                    {
                        var newAsset = CreateSlotAsset(primaryComp, i);
                        if (newAsset != null)
                        {
                            Undo.RecordObject(primaryComp, "Create RTXPT Slot Asset");
                            primaryComp.Slots[i] = newAsset;
                            EditorUtility.SetDirty(primaryComp);
                            serializedObject.Update();
                        }
                    }

                    EditorGUILayout.EndHorizontal();

                    // Inline-edit the assigned asset (draw all its fields except the script ref).
                    if (assetRef != null)
                    {
                        var assetSO = new SerializedObject(assetRef);
                        assetSO.Update();
                        SerializedProperty prop = assetSO.GetIterator();
                        bool enterChildren = true;
                        while (prop.NextVisible(enterChildren))
                        {
                            enterChildren = false;
                            if (prop.name == "m_Script") continue;
                            EditorGUILayout.PropertyField(prop, includeChildren: true);
                        }
                        if (assetSO.ApplyModifiedProperties())
                            EditorUtility.SetDirty(assetRef);
                    }
                    else
                    {
                        EditorGUILayout.HelpBox("No material assigned — this sub-mesh will not be rendered.", MessageType.Warning);
                    }

                    EditorGUI.indentLevel--;
                }

                EditorGUILayout.EndFoldoutHeaderGroup();
            }

            if (serializedObject.ApplyModifiedProperties())
            {
                foreach (var t in targets)
                    EditorUtility.SetDirty(t);
            }
        }

        // Resize the Slots list to match the renderer's sub-mesh count.
        private static void SyncSlotCount(RtxptRenderer comp)
        {
            var mr = comp.GetComponent<MeshRenderer>();
            var mf = comp.GetComponent<MeshFilter>();
            if (mr == null) return;

            var  mats      = mr.sharedMaterials ?? Array.Empty<Material>();
            int  slotCount = mf?.sharedMesh != null ? mf.sharedMesh.subMeshCount : mats.Length;

            if (comp.Slots.Count == slotCount) return;

            Undo.RecordObject(comp, "Sync RTXPT Slot Count");
            while (comp.Slots.Count < slotCount) comp.Slots.Add(null);
            if (comp.Slots.Count > slotCount)    comp.Slots.RemoveRange(slotCount, comp.Slots.Count - slotCount);
            EditorUtility.SetDirty(comp);
        }

        // Creates asset files for all null slot entries (used by "Bake from Renderer").
        private static void EnsureSlotAssets(RtxptRenderer comp)
        {
            SyncSlotCount(comp);
            string dir = ResolveAssetDir(comp.gameObject.scene.path);

            for (int s = 0; s < comp.Slots.Count; s++)
            {
                if (comp.Slots[s] != null) continue;
                comp.Slots[s] = CreateSlotAsset(comp, s, dir);
            }

            AssetDatabase.SaveAssets();
        }

        // Creates a single new asset for slot s.
        private static RtxptMaterial CreateSlotAsset(RtxptRenderer comp, int s, string dir = null)
        {
            dir ??= ResolveAssetDir(comp.gameObject.scene.path);
            var    asset = CreateInstance<RtxptMaterial>();
            string path  = AssetDatabase.GenerateUniqueAssetPath($"{dir}/{comp.gameObject.name}_Slot{s}.asset");
            AssetDatabase.CreateAsset(asset, path);
            AssetDatabase.SaveAssets();
            return asset;
        }

        private static string ResolveAssetDir(string scenePath)
        {
            string dir = string.IsNullOrEmpty(scenePath)
                ? "Assets/RtxptMaterialOverrides"
                : Path.GetDirectoryName(scenePath)?.Replace('\\', '/') + "/RtxptMaterialOverrides";

            if (!AssetDatabase.IsValidFolder(dir))
            {
                string[] parts   = dir.Split('/');
                string   current = parts[0];
                for (int i = 1; i < parts.Length; i++)
                {
                    string next = current + "/" + parts[i];
                    if (!AssetDatabase.IsValidFolder(next))
                        AssetDatabase.CreateFolder(current, parts[i]);
                    current = next;
                }
            }

            return dir;
        }
    }
}
#endif
