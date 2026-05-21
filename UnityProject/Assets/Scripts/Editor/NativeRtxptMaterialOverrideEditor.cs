#if UNITY_EDITOR
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace PathTracing
{
    [CustomEditor(typeof(NativeRtxptMaterialOverride))]
    public class NativeRtxptMaterialOverrideEditor : Editor
    {
        private static readonly GUIStyle s_headerStyle = null;
        private bool[] _slotFoldouts = System.Array.Empty<bool>();

        public override void OnInspectorGUI()
        {
            var comp = (NativeRtxptMaterialOverride)target;

            // ---- Top toolbar ----
            EditorGUILayout.BeginHorizontal();

            if (GUILayout.Button("Bake from Renderer", GUILayout.Height(28)))
            {
                Undo.RecordObject(comp, "Bake RTXPT Materials from Renderer");
                comp.BakeFromRenderer();
                EditorUtility.SetDirty(comp);
            }

            // if (GUILayout.Button("Force Scene Rebuild", GUILayout.Width(150), GUILayout.Height(28)))
            // {
            //     // Notify NativeRtxptGPUScene to rebuild next frame
            //     foreach (var scene in FindObjectsByType<NativeRtxptSceneHost>(FindObjectsSortMode.None))
            //         scene.MarkRebuildDirty();
            //     Debug.Log("[RTXPT] Scene rebuild scheduled.");
            // }

            EditorGUILayout.EndHorizontal();
            EditorGUILayout.Space(4);

            if (comp.Slots == null || comp.Slots.Count == 0)
            {
                EditorGUILayout.HelpBox("No slots. Press 'Bake from Renderer' to populate.", MessageType.Info);
                return;
            }

            // Sync foldout array size
            if (_slotFoldouts.Length != comp.Slots.Count)
            {
                System.Array.Resize(ref _slotFoldouts, comp.Slots.Count);
                for (int i = 0; i < _slotFoldouts.Length; i++)
                    _slotFoldouts[i] = true;
            }

            serializedObject.Update();
            var slotsProp = serializedObject.FindProperty("Slots");

            for (int i = 0; i < comp.Slots.Count; i++)
            {
                var slot    = comp.Slots[i];
                string name = slot?.SourceMaterial != null ? slot.SourceMaterial.name : $"Slot {i}";

                _slotFoldouts[i] = EditorGUILayout.BeginFoldoutHeaderGroup(_slotFoldouts[i],
                    $"Sub-mesh {i}  —  {name}");

                if (_slotFoldouts[i] && slotsProp.arraySize > i)
                {
                    EditorGUI.indentLevel++;
                    var slotProp = slotsProp.GetArrayElementAtIndex(i);
                    EditorGUILayout.PropertyField(slotProp, includeChildren: true);
                    EditorGUI.indentLevel--;
                }

                EditorGUILayout.EndFoldoutHeaderGroup();
            }

            if (serializedObject.ApplyModifiedProperties())
                EditorUtility.SetDirty(comp);
        }
    }
}
#endif
