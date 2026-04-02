using System;
using System.Collections.Generic;
using System.Reflection;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

namespace PathTracing
{
    [CustomEditor(typeof(RtxdiFeature))]
    public class RtxdiFeatureEditor : Editor
    {
        private string GetKey(string headerName)
        {
            return $"PT_Foldout_{target.GetInstanceID()}_{headerName}";
        }


        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            // DrawDefaultInspector();
            RtxdiFeature feature = (RtxdiFeature)target;

            // 1. 绘制 PathTracingSetting (带折叠 Header)
            SerializedProperty settingsProp = serializedObject.FindProperty("pathTracingSetting");
            if (settingsProp != null)
            {
                DrawSettingsWithFoldableHeaders(settingsProp);
            }

            DrawGroupedAssetFields();

            if (GUILayout.Button("InitializeBuffers"))
            {
                feature.InitializeBuffers();
            }

            if (GUILayout.Button("SetMask"))
            {
                feature.SetMask();
            }

            if (GUILayout.Button("TestPrepareLight"))
            {
                feature.Test();
            }

            EditorGUILayout.Space(10);
            

            DrawObjectRecursive("Global Constants", feature.globalConstants , "GlobalConstants");
            
  
            DrawObjectRecursive("Resampling Constants", feature.resamplingConstants,  "ResamplingConstants");
            

            serializedObject.ApplyModifiedProperties();
        }


        /// <summary>
        /// 通过反射扫描 RtxdiFeature 的所有公有字段，按类型自动分组显示。
        /// 新增字段无需修改此处代码。
        /// </summary>
        private void DrawGroupedAssetFields()
        {
            // 已在其他地方单独处理的字段名，跳过
            var skip = new HashSet<string>
            {
                "pathTracingSetting", "globalConstants", "resamplingConstants", "renderPassEvent"
            };

            // 类型 → 分组标题
            var groupLabels = new Dictionary<Type, string>
            {
                { typeof(Material),           "Materials" },
                { typeof(RayTracingShader),   "Ray Tracing Shaders" },
                { typeof(ComputeShader),      "Compute Shaders" },
                { typeof(Texture),            "Textures" },
                { typeof(Texture2D),          "Textures" },
                { typeof(Texture3D),          "Textures" },
                { typeof(RenderTexture),      "Textures" },
                { typeof(Cubemap),            "Textures" },
            };

            // 收集分组
            var groups = new Dictionary<string, List<string>>();

            FieldInfo[] fields = typeof(RtxdiFeature)
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

        /// <summary>
        /// 自动根据 [Header] 特性将属性分组并渲染为可折叠栏
        /// </summary>
        private void DrawSettingsWithFoldableHeaders(SerializedProperty parentProp)
        {
            EditorGUILayout.LabelField("Settings", EditorStyles.boldLabel);

            // 获取实际的类型以通过反射读取 Header
            Type type = typeof(PathTracingSetting);

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
                            string key = GetKey(header.Name);
                            bool isExpanded = SessionState.GetBool(key, false);

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

        /// <summary>
        /// 递归绘制对象的所有公有字段
        /// </summary>
        /// <summary>
        /// 递归绘制对象，增加了 path 参数用于保存折叠状态
        /// </summary>
        private void DrawObjectRecursive(string label, object obj, string path)
        {
            if (obj == null) return;

            Type type = obj.GetType();

            if (IsSimpleType(type))
            {
                DrawSimpleField(label, obj);
                return;
            }

            // 为当前层级生成唯一的 SessionState Key
            string foldoutKey = GetKey(path + "_" + label);
            bool isExpanded = SessionState.GetBool(foldoutKey, false); // 默认折叠

            EditorGUILayout.BeginVertical();
            
            // 绘制可点击的折叠标签
            isExpanded = EditorGUILayout.Foldout(isExpanded, label, true, EditorStyles.foldoutHeader);
            SessionState.SetBool(foldoutKey, isExpanded);

            if (isExpanded)
            {
                EditorGUI.indentLevel++;
                FieldInfo[] fields = type.GetFields(BindingFlags.Public | BindingFlags.Instance);
                foreach (var field in fields)
                {
                    object value = field.GetValue(obj);
                    // 递归时将当前 label 加入 path，保证子节点的 key 唯一
                    DrawObjectRecursive(field.Name, value, path + "_" + label);
                }
                EditorGUI.indentLevel--;
            }

            EditorGUILayout.EndVertical();
        }

        // 判断是否是直接绘制的底层类型
        private bool IsSimpleType(System.Type type)
        {
            return type.IsPrimitive ||type.IsEnum || 
                   type == typeof(float) || type == typeof(int) || type == typeof(uint) ||
                   type == typeof(float2) || type == typeof(float3) || type == typeof(float4) ||
                   type == typeof(float4x4) || type == typeof(Vector2) || type == typeof(Vector3) ||
                   type == typeof(Vector4) || type == typeof(bool) || type == typeof(string)|| type == typeof(int2) || type == typeof(uint2);
        }

        // 绘制具体的字段值
        private void DrawSimpleField(string label, object value)
        {
            if (value is System.Enum enumValue)
            {
                EditorGUILayout.EnumPopup(label, enumValue);
                return;
            }
            if (value is float4x4 m)
            {
                EditorGUILayout.LabelField(label);
                EditorGUI.indentLevel++;
                EditorGUILayout.Vector4Field("R0", new Vector4(m.c0.x, m.c1.x, m.c2.x, m.c3.x));
                EditorGUILayout.Vector4Field("R1", new Vector4(m.c0.y, m.c1.y, m.c2.y, m.c3.y));
                EditorGUILayout.Vector4Field("R2", new Vector4(m.c0.z, m.c1.z, m.c2.z, m.c3.z));
                EditorGUILayout.Vector4Field("R3", new Vector4(m.c0.w, m.c1.w, m.c2.w, m.c3.w));
                EditorGUI.indentLevel--;
            }
            else if (value is float4 v4) EditorGUILayout.Vector4Field(label, new Vector4(v4.x, v4.y, v4.z, v4.w));
            else if (value is float3 v3) EditorGUILayout.Vector3Field(label, new Vector3(v3.x, v3.y, v3.z));
            else if (value is float2 v2) EditorGUILayout.Vector2Field(label, new Vector2(v2.x, v2.y));
            else if (value is uint u) EditorGUILayout.LongField(label, (long)u); // uint转long显示
            else if (value is float f) EditorGUILayout.FloatField(label, f);
            else if (value is int i) EditorGUILayout.IntField(label, i);
            else if (value is bool b) EditorGUILayout.Toggle(label, b);
            else if (value is string s) EditorGUILayout.TextField(label, s);
            else if (value is int2 i2) EditorGUILayout.Vector2IntField(label, new Vector2Int(i2.x, i2.y));
            else if (value is uint2 u2) EditorGUILayout.Vector2IntField(label, new Vector2Int((int)u2.x, (int)u2.y));
            else EditorGUILayout.LabelField(label, value?.ToString() ?? "null");
        }
    }
}