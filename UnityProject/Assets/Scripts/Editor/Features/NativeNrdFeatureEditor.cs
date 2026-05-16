using System;
using System.Collections.Generic;
using System.Reflection;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

namespace PathTracing
{
    [CustomEditor(typeof(NativeNrdFeature))]
    public class NativeNrdFeatureEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            // DrawDefaultInspector();
            NativeNrdFeature feature = (NativeNrdFeature)target;

            // 1. 绘制 PathTracingSetting (带折叠 Header)
            SerializedProperty settingsProp = serializedObject.FindProperty("setting");
            if (settingsProp != null)
            {
                DrawSettingsWithFoldableHeaders(settingsProp);
            }

            DrawGroupedAssetFields();

            if (GUILayout.Button("Auto Fill Shaders & Materials"))
            {
                Undo.RecordObject(feature, "Auto Fill Shaders");
                feature.AutoFillShaders();
            }

            if (GUILayout.Button("Print NRDSampleResource Info"))
            {
                var res = feature.NrdSampleResource;
                if (res != null)
                    res.PrintDebugInfo();
                else
                    Debug.Log("[NRDFeatureEditor] NrdSampleResource is null (not initialized yet).");
            }


            EditorGUILayout.Space(10);

            DrawObjectHelper.Draw(target.GetInstanceID(), "Global Constants", feature.globalConstants);
            DrawObjectHelper.Draw(target.GetInstanceID(), "Info", feature.NrdSampleResource);

            serializedObject.ApplyModifiedProperties();
        }


        /// <summary>
        /// 通过反射扫描 NRDFeature 的所有公有字段，按类型自动分组显示。
        /// 新增字段无需修改此处代码。
        /// </summary>
        private void DrawGroupedAssetFields()
        {
            // 已在其他地方单独处理的字段名，跳过
            var skip = new HashSet<string>
            {
                "pathTracingSetting", "globalConstants", "resamplingConstants", "setting"
            };

            // 类型 → 分组标题
            var groupLabels = new Dictionary<Type, string>
            {
                { typeof(Material), "Materials" },
                { typeof(RayTracingShader), "Ray Tracing Shaders" },
                { typeof(ComputeShader), "Compute Shaders" },
                { typeof(Texture), "Textures" },
                { typeof(Texture2D), "Textures" },
                { typeof(Texture3D), "Textures" },
                { typeof(RenderTexture), "Textures" },
                { typeof(Cubemap), "Textures" },
            };

            // 收集分组
            var groups = new Dictionary<string, List<string>>();

            FieldInfo[] fields = typeof(NativeNrdFeature)
                .GetFields(BindingFlags.Public | BindingFlags.Instance);

            foreach (var field in fields)
            {
                if (skip.Contains(field.Name)) continue;

                string groupName = null;
                // 精确匹配或父类匹配
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

            // 按固定顺序渲染，最后渲染 Other
            var order = new[] { "Materials", "Ray Tracing Shaders", "Compute Shaders", "Textures", "Other" };
            foreach (var groupName in order)
            {
                if (!groups.TryGetValue(groupName, out var fieldNames) || fieldNames.Count == 0)
                    continue;

                string foldoutKey  = DrawObjectHelper.GetKey(target.GetInstanceID(),"AssetGroup_" + groupName);
                bool   isExpanded  = SessionState.GetBool(foldoutKey, true);
                bool   newExpanded = EditorGUILayout.BeginFoldoutHeaderGroup(isExpanded, groupName);
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

        /// <summary>
        /// 自动根据 [Header] 特性将属性分组并渲染为可折叠栏
        /// </summary>
        private void DrawSettingsWithFoldableHeaders(SerializedProperty parentProp)
        {
            EditorGUILayout.LabelField("Settings", EditorStyles.boldLabel);

            // 获取实际的类型以通过反射读取 Header
            Type type = typeof(NativeNrdSampleSetting);

            // 迭代所有子属性
            SerializedProperty childProp = parentProp.Copy();
            SerializedProperty endProp = childProp.GetEndProperty();

            bool currentFoldoutState = true;

            if (childProp.NextVisible(true)) // 进入对象内部
            {
                do
                {
                    if (SerializedProperty.EqualContents(childProp, endProp)) break;

                    // 通过反射查找该字段是否有 [Header] 标签
                    FieldInfo fieldInfo = type.GetField(childProp.name);
                    if (fieldInfo != null)
                    {
                        FoldoutHeaderAttribute header = fieldInfo.GetCustomAttribute<FoldoutHeaderAttribute>();
                        if (header != null)
                        {
                            // 如果有 Header，创建一个新的折叠组
                            // EditorGUILayout.Space(8);


                            // 从 SessionState 获取该 Header 的保存状态
                            string key        = DrawObjectHelper.GetKey(target.GetInstanceID(),header.Name);
                            bool   isExpanded = SessionState.GetBool(key, true);

                            // 绘制 Foldout
                            bool newState = EditorGUILayout.BeginFoldoutHeaderGroup(isExpanded, header.Name);

                            // 如果状态改变，存回 SessionState
                            if (newState != isExpanded)
                            {
                                SessionState.SetBool(key, newState);
                            }

                            currentFoldoutState = newState;
                            EditorGUILayout.EndFoldoutHeaderGroup();
                        }
                    }

                    // 如果当前组是打开的，则绘制属性
                    if (currentFoldoutState)
                    {
                        EditorGUI.indentLevel++;
                        EditorGUILayout.PropertyField(childProp, true);
                        EditorGUI.indentLevel--;
                    }
                } while (childProp.NextVisible(false)); // 只迭代当前层级
            }
        }
 }
}