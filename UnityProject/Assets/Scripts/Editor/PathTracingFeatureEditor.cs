using UnityEditor;
using UnityEngine;

namespace PathTracing
{
    [CustomEditor(typeof(PathTracingFeature))]
    public class PathTracingFeatureEditor : Editor
    {
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

            if (GUILayout.Button("ReBuild"))
            {
                ptFeature.ReBuild();
            }
            if (GUILayout.Button("InitializeBuffers"))
            {
                ptFeature.InitializeBuffers();
            }
            if (GUILayout.Button("SetMask"))
            {
                ptFeature.SetMask();
            }
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