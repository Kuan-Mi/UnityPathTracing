#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEngine;

namespace PathTracing
{
    [CustomEditor(typeof(RtxptMaterialOverrideAsset))]
    public class RtxptMaterialOverrideAssetEditor : Editor
    {
        private bool   _jsonFoldout = false;
        private string _jsonText    = "";
        private string _errorMsg    = null;

        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            EditorGUILayout.Space(6);
            EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);

            _jsonFoldout = EditorGUILayout.Foldout(_jsonFoldout, "Import from JSON", true);
            if (!_jsonFoldout) return;

            EditorGUI.indentLevel++;

            // File picker
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Load JSON File…", GUILayout.Height(22)))
            {
                string path = EditorUtility.OpenFilePanel("Select RTXPT material JSON", "", "json");
                if (!string.IsNullOrEmpty(path))
                {
                    _jsonText = File.ReadAllText(path);
                    _errorMsg = null;
                }
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.Space(2);

            // Paste area
            EditorGUILayout.LabelField("or paste JSON:");
            _jsonText = EditorGUILayout.TextArea(_jsonText, GUILayout.MinHeight(80));

            if (_errorMsg != null)
                EditorGUILayout.HelpBox(_errorMsg, MessageType.Error);

            EditorGUI.BeginDisabledGroup(string.IsNullOrWhiteSpace(_jsonText));
            if (GUILayout.Button("Apply to Slot", GUILayout.Height(26)))
            {
                var asset = (RtxptMaterialOverrideAsset)target;
                try
                {
                    Undo.RecordObject(asset, "Apply JSON to RTXPT Slot");
                    asset.Slot.LoadFromJson(_jsonText);
                    EditorUtility.SetDirty(asset);
                    _errorMsg = null;
                }
                catch (System.Exception ex)
                {
                    _errorMsg = ex.Message;
                }
            }
            EditorGUI.EndDisabledGroup();

            EditorGUI.indentLevel--;
        }
    }
}
#endif
