using System;
using System.Reflection;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;

namespace PathTracing
{
    public static class DrawObjectHelper
    {
        public static string GetKey(int instanceID, string headerName)
        {
            return $"Foldout_{instanceID}_{headerName}";
        }
        
        /// <summary>
        /// 递归绘制对象的所有公有字段
        /// </summary>
        /// <summary>
        /// 递归绘制对象，增加了 path 参数用于保存折叠状态
        /// </summary>
        public static void Draw(int instanceID, string label, object obj, string path = "")
        {
            if (obj == null) return;

            Type type = obj.GetType();

            if (IsSimpleType(type))
            {
                DrawSimpleField(label, obj);
                return;
            }

            // 为当前层级生成唯一的 SessionState Key
            string foldoutKey = GetKey(instanceID,path + "_" + label);
            bool   isExpanded = SessionState.GetBool(foldoutKey, false); // 默认折叠

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
                    Draw(instanceID,field.Name, value, path + "_" + label);
                }

                EditorGUI.indentLevel--;
            }

            EditorGUILayout.EndVertical();
        }

        // 判断是否是直接绘制的底层类型
        private static bool IsSimpleType(System.Type type)
        {
            return type.IsPrimitive || type.IsEnum ||
                   type == typeof(float) || type == typeof(int) || type == typeof(uint) ||
                   type == typeof(float2) || type == typeof(float3) || type == typeof(float4) ||
                   type == typeof(float4x4) || type == typeof(Vector2) || type == typeof(Vector3) ||
                   type == typeof(Vector4) || type == typeof(bool) || type == typeof(string) || type == typeof(int2) || type == typeof(uint2);
        }

        // 绘制具体的字段值
        private static void DrawSimpleField(string label, object value)
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