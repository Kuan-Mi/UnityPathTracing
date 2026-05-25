#if UNITY_EDITOR
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace PathTracing
{
    [CustomEditor(typeof(NativeRtxptMaterialOverride))]
    [CanEditMultipleObjects]
    public class NativeRtxptMaterialOverrideEditor : Editor
    {
        private bool[] _slotFoldouts = System.Array.Empty<bool>();

        public override void OnInspectorGUI()
        {
            bool multiEdit = targets.Length > 1;

            // ---- Top toolbar ----
            EditorGUILayout.BeginHorizontal();

            if (GUILayout.Button("Bake from Renderer", GUILayout.Height(28)))
            {
                foreach (var t in targets)
                {
                    var c = (NativeRtxptMaterialOverride)t;
                    Undo.RecordObject(c, "Bake RTXPT Materials from Renderer");
                    c.BakeFromRenderer();
                    EditorUtility.SetDirty(c);
                }
            }

            EditorGUILayout.EndHorizontal();
            EditorGUILayout.Space(4);

            if (multiEdit)
            {
                EditorGUILayout.HelpBox($"Editing {targets.Length} objects. Slot properties are shown for objects with matching slot counts.", MessageType.None);
                EditorGUILayout.Space(2);
            }

            serializedObject.Update();
            var slotsProp = serializedObject.FindProperty("Slots");

            // Determine the slot count to display: use the primary target's count,
            // but show a warning when counts differ across selected objects.
            var primaryComp = (NativeRtxptMaterialOverride)target;
            int primaryCount = primaryComp.Slots?.Count ?? 0;

            if (primaryCount == 0)
            {
                EditorGUILayout.HelpBox("No slots. Press 'Bake from Renderer' to populate.", MessageType.Info);
                return;
            }

            bool countMismatch = false;
            if (multiEdit)
            {
                foreach (var t in targets)
                {
                    var c = (NativeRtxptMaterialOverride)t;
                    if ((c.Slots?.Count ?? 0) != primaryCount) { countMismatch = true; break; }
                }
            }

            if (countMismatch)
                EditorGUILayout.HelpBox("Selected objects have different slot counts. Only the primary object's slots are shown; slot edits apply to all objects with matching slot counts.", MessageType.Warning);

            // Sync foldout array size
            if (_slotFoldouts.Length != primaryCount)
            {
                System.Array.Resize(ref _slotFoldouts, primaryCount);
                for (int i = 0; i < _slotFoldouts.Length; i++)
                    _slotFoldouts[i] = true;
            }

            for (int i = 0; i < primaryCount; i++)
            {
                var slot    = primaryComp.Slots[i];
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
            {
                foreach (var t in targets)
                    EditorUtility.SetDirty(t);
            }
        }
    }
}
#endif
