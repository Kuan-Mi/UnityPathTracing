using System.Collections.Generic;
using System.Reflection;
using NativeRender;
using UnityEditor;
using UnityEngine;

namespace PathTracing
{
    [CustomEditor(typeof(NativeRtxptFeature))]
    public class NativeRtxptFeatureEditor : Editor
    {
        private string GetKey(string headerName) =>
            $"PT_NativeRtxpt_Foldout_{target.GetInstanceID()}_{headerName}";

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            var feature = (NativeRtxptFeature)target;

            // Setting
            var settingsProp = serializedObject.FindProperty("setting");
            if (settingsProp != null)
                EditorGUILayout.PropertyField(settingsProp, includeChildren: true);

            EditorGUILayout.Space(4);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("renderPassEvent"));

            EditorGUILayout.Space(4);
            DrawGroupedAssetFields();

            EditorGUILayout.Space(8);
            if (GUILayout.Button("Auto Fill Shaders"))
            {
                Undo.RecordObject(feature, "Auto Fill Shaders");
                feature.AutoFillShaders();
            }

            serializedObject.ApplyModifiedProperties();
        }

        private void DrawGroupedAssetFields()
        {
            var skip = new HashSet<string> { "renderPassEvent", "setting" };

            var groupLabels = new Dictionary<System.Type, string>
            {
                { typeof(NativeComputeShader), "Native Compute Shaders" },
                { typeof(RayTraceShader),       "Ray Trace Shaders"      },
                { typeof(ComputeShader),         "Compute Shaders"        },
            };

            var groups = new Dictionary<string, List<string>>();

            var fields = typeof(NativeRtxptFeature)
                .GetFields(BindingFlags.Public | BindingFlags.Instance);

            foreach (var field in fields)
            {
                if (skip.Contains(field.Name)) continue;

                string groupName = null;
                foreach (var kv in groupLabels)
                {
                    if (kv.Key.IsAssignableFrom(field.FieldType))
                    {
                        groupName = kv.Value;
                        break;
                    }
                }

                if (groupName == null) groupName = "Other";

                if (!groups.ContainsKey(groupName))
                    groups[groupName] = new List<string>();
                groups[groupName].Add(field.Name);
            }

            var order = new[] { "Ray Trace Shaders", "Native Compute Shaders", "Compute Shaders", "Other" };
            foreach (var groupName in order)
            {
                if (!groups.TryGetValue(groupName, out var fieldNames) || fieldNames.Count == 0)
                    continue;

                string foldoutKey  = GetKey("AssetGroup_" + groupName);
                bool   isExpanded  = SessionState.GetBool(foldoutKey, true);
                bool   newExpanded = EditorGUILayout.Foldout(isExpanded, groupName, toggleOnLabelClick: true, EditorStyles.foldoutHeader);
                if (newExpanded != isExpanded)
                    SessionState.SetBool(foldoutKey, newExpanded);

                if (newExpanded)
                {
                    EditorGUI.indentLevel++;
                    foreach (var name in fieldNames)
                    {
                        var prop = serializedObject.FindProperty(name);
                        if (prop != null)
                            EditorGUILayout.PropertyField(prop);
                    }
                    EditorGUI.indentLevel--;
                }
            }
        }
    }
}
