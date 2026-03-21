using System.Reflection;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;

namespace PathTracing
{
    [CustomEditor(typeof(PathTracingFeature))]
    public class PathTracingFeatureEditor : Editor
    {
        
        private bool showDebug = true;
        
        // Asset paths relative to the project root.
        // Adjust these if assets are moved.
        private static readonly (string propName, string assetPath)[] AssetMappings =
        {
            ("finalMaterial",             "Assets/Shaders/Mat/KM_Final.mat"),
            ("opaqueTracingShader",       "Assets/Shaders/RayTracing/TraceOpaque.raytrace"),
            ("transparentTracingShader",  "Assets/Shaders/RayTracing/TraceTransparent.raytrace"),
            ("compositionComputeShader",  "Assets/Shaders/PostProcess/Composition.compute"),
            ("taaComputeShader",          "Assets/Shaders/PostProcess/Taa.compute"),
            ("dlssBeforeComputeShader",   "Assets/Shaders/PostProcess/DlssBefore.compute"),
            ("sharcResolveCs",            "Assets/Shaders/Sharc/SharcResolve.compute"),
            ("sharcUpdateTs",             "Assets/Shaders/Sharc/SharcUpdate.raytrace"),
            ("autoExposureShader",        "Assets/Shaders/PostProcess/AutoExposure.compute"),
            ("scramblingRankingTex",     "Assets/Textures/scrambling_ranking_128x128_2d_4spp.png"),
            ("sobolTex",                 "Assets/Textures/sobol_256_4d.png"),
        };

        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            EditorGUILayout.Space();

            PathTracingFeature ptFeature = (PathTracingFeature)target;

            GUI.backgroundColor = new Color(0.5f, 0.9f, 0.5f);
            if (GUILayout.Button("Auto Configure Assets", GUILayout.Height(30)))
            {
                AutoConfigure();
            }
            GUI.backgroundColor = Color.white;

            EditorGUILayout.Space();

            // if (GUILayout.Button("ReBuild"))
            // {
            //     ptFeature.ReBuild();
            // }
            if (GUILayout.Button("InitializeBuffers"))
            {
                ptFeature.InitializeBuffers();
            }
            if (GUILayout.Button("SetMask"))
            {
                ptFeature.SetMask();
            }
            if (GUILayout.Button("TestPrepareLight"))
            {
                ptFeature.Test();
            }
            
            EditorGUILayout.Space(10);
            showDebug = EditorGUILayout.BeginFoldoutHeaderGroup(showDebug, "Debug Constants (Non-Serialized)");
        
            if (showDebug)
            {
                EditorGUI.BeginDisabledGroup(true); // 设置为灰色只读，因为是 Debug 信息
            
                // 绘制第一个常量结构体
                DrawObjectRecursive("Global Constants", ptFeature.globalConstants);
            
                EditorGUILayout.Space(5);
            
                // 绘制第二个嵌套常量结构体
                DrawObjectRecursive("Resampling Constants", ptFeature.resamplingConstants);
            
                EditorGUI.EndDisabledGroup();
            }
            EditorGUILayout.EndFoldoutHeaderGroup();
            
        }
            /// <summary>
    /// 递归绘制对象的所有公有字段
    /// </summary>
    private void DrawObjectRecursive(string label, object obj)
    {
        if (obj == null) return;

        System.Type type = obj.GetType();

        // 如果是基础类型或数学类型，直接绘制
        if (IsSimpleType(type))
        {
            DrawSimpleField(label, obj);
            return;
        }

        // 如果是复杂结构体/类，开启一个 Foldout
        EditorGUILayout.BeginVertical(EditorStyles.helpBox);
        label = string.IsNullOrEmpty(label) ? type.Name : label;
        
        // 这里使用 LabelField 模拟标题，如果你想要点击折叠，可以加个布尔值字典记录状态
        EditorGUILayout.LabelField(label, EditorStyles.boldLabel);
        
        EditorGUI.indentLevel++;
        
        FieldInfo[] fields = type.GetFields(BindingFlags.Public | BindingFlags.Instance);
        foreach (var field in fields)
        {
            object value = field.GetValue(obj);
            // 递归调用
            DrawObjectRecursive(field.Name, value);
        }
        
        EditorGUI.indentLevel--;
        EditorGUILayout.EndVertical();
    }

    // 判断是否是直接绘制的底层类型
    private bool IsSimpleType(System.Type type)
    {
        return type.IsPrimitive || 
               type == typeof(float) || type == typeof(int) || type == typeof(uint) ||
               type == typeof(float2) || type == typeof(float3) || type == typeof(float4) ||
               type == typeof(float4x4) || type == typeof(Vector2) || type == typeof(Vector3) || 
               type == typeof(Vector4) || type == typeof(bool) || type == typeof(string);
    }

    // 绘制具体的字段值
    private void DrawSimpleField(string label, object value)
    {
        if (value is float4x4 m) {
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
        else if (value is uint u)    EditorGUILayout.LongField(label, (long)u); // uint转long显示
        else if (value is float f)   EditorGUILayout.FloatField(label, f);
        else if (value is int i)     EditorGUILayout.IntField(label, i);
        else if (value is bool b)    EditorGUILayout.Toggle(label, b);
        else EditorGUILayout.LabelField(label, value?.ToString() ?? "null");
    }
        private void AutoConfigure()
        {
            serializedObject.Update();

            int configured = 0;
            int missing = 0;

            foreach (var (propName, assetPath) in AssetMappings)
            {
                var prop = serializedObject.FindProperty(propName);
                if (prop == null)
                {
                    Debug.LogWarning($"[PathTracingFeature] Property '{propName}' not found on serialized object.");
                    continue;
                }

                var asset = AssetDatabase.LoadMainAssetAtPath(assetPath);
                if (asset == null)
                {
                    Debug.LogWarning($"[PathTracingFeature] Asset not found at path: {assetPath}");
                    missing++;
                }
                else
                {
                    prop.objectReferenceValue = asset;
                    configured++;
                }
            }

            serializedObject.ApplyModifiedProperties();
            EditorUtility.SetDirty(target);
            AssetDatabase.SaveAssets();

            Debug.Log($"[PathTracingFeature] Auto Configure complete: {configured} assigned, {missing} missing.");
        }
    }
}