using System;
using System.Collections.Generic;
using System.Reflection;
using NativeRender;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

namespace PathTracing
{
    [CustomEditor(typeof(NativeRtxdiFeature))]
    public class NativeRtxdiFeatureEditor : Editor
    {
        private string GetKey(string headerName) => $"PT_NativeRtxdi_Foldout_{target.GetInstanceID()}_{headerName}";

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            var feature = (NativeRtxdiFeature)target;

            // Settings (use default property field — RtxdiSetting has its own property drawers).
            var settingsProp = serializedObject.FindProperty("setting");
            if (settingsProp != null)
                EditorGUILayout.PropertyField(settingsProp, includeChildren: true);

            EditorGUILayout.Space(4);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("renderPassEvent"));

            EditorGUILayout.Space(4);
            DrawGroupedAssetFields();

            EditorGUILayout.Space(8);
            if (GUILayout.Button("Auto Fill Shaders & Materials"))
            {
                Undo.RecordObject(feature, "Auto Fill Shaders");
                feature.AutoFillShaders();
            }
            if (GUILayout.Button("InitializeBuffers"))
            {
                feature.InitializeBuffers();
            }

            EditorGUILayout.Space(8);
            var globalConstProp = serializedObject.FindProperty("globalConstants");
            if (globalConstProp != null)
                EditorGUILayout.PropertyField(globalConstProp, includeChildren: true);
            var resamplingConstProp = serializedObject.FindProperty("resamplingConstants");
            if (resamplingConstProp != null)
                EditorGUILayout.PropertyField(resamplingConstProp, includeChildren: true);

            serializedObject.ApplyModifiedProperties();
        }

        /// <summary>
        /// Reflection-driven asset grouping (mirrors RtxdiFeatureEditor) with NativeComputeShader added.
        /// </summary>
        private void DrawGroupedAssetFields()
        {
            var skip = new HashSet<string>
            {
                "globalConstants", "resamplingConstants", "renderPassEvent", "setting"
            };

            var groupLabels = new Dictionary<Type, string>
            {
                { typeof(Material),            "Materials" },
                { typeof(NativeComputeShader), "Native Compute Shaders" },
                { typeof(ComputeShader),       "Compute Shaders" },
                { typeof(Texture),             "Textures" },
                { typeof(Texture2D),           "Textures" },
                { typeof(Texture3D),           "Textures" },
                { typeof(RenderTexture),       "Textures" },
                { typeof(Cubemap),             "Textures" },
            };

            var groups = new Dictionary<string, List<string>>();

            FieldInfo[] fields = typeof(NativeRtxdiFeature)
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

            var order = new[] { "Materials", "Native Compute Shaders", "Compute Shaders", "Textures", "Other" };
            foreach (var groupName in order)
            {
                if (!groups.TryGetValue(groupName, out var fieldNames) || fieldNames.Count == 0)
                    continue;

                string foldoutKey = GetKey("AssetGroup_" + groupName);
                bool isExpanded = SessionState.GetBool(foldoutKey, true);
                bool newExpanded = EditorGUILayout.BeginFoldoutHeaderGroup(isExpanded, groupName);
                if (newExpanded != isExpanded)
                    SessionState.SetBool(foldoutKey, newExpanded);

                if (newExpanded)
                {
                    EditorGUI.indentLevel++;
                    foreach (var name in fieldNames)
                    {
                        SerializedProperty prop = serializedObject.FindProperty(name);
                        if (prop != null)
                            EditorGUILayout.PropertyField(prop);
                    }
                    EditorGUI.indentLevel--;
                }

                EditorGUILayout.EndFoldoutHeaderGroup();
            }
        }
    }
}
