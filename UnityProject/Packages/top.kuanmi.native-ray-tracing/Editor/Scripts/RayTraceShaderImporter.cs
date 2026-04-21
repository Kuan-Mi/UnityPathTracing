using System;
using System.IO;
using UnityEditor;
using UnityEditor.AssetImporters;
using UnityEngine;

namespace NativeRender
{
    /// <summary>
    /// Imports <c>.rayshader</c> files as <see cref="RayTraceShader"/> ScriptableObject assets.
    /// The file content is raw HLSL; the importer stores the absolute path so the native
    /// plugin can compile it at runtime via DXC.
    ///
    /// To create a new shader asset: right-click in the Project view and rename any text file
    /// to have the <c>.rayshader</c> extension, or duplicate an existing one.
    /// </summary>
    [ScriptedImporter(1, "rayshader", 0)]
    public class RayTraceShaderImporter : ScriptedImporter
    {
        [Tooltip("当设置为 True 时，该 .hlsl 文件将作为 RayTraceShader 资产导入。")]
        public bool isRayTraceShader = false;

        [Tooltip("Additional #include search directories (absolute paths). The shader file's own directory is always included automatically.")]
        public string[] additionalIncludePaths = Array.Empty<string>();

        [Tooltip("Additional DXC compiler arguments (e.g. -disable-payload-qualifiers, -HV 2021).")]
        public string[] extraArgs = Array.Empty<string>();

        [Tooltip("Preprocessor defines (e.g. FOO=1, BAR).")]
        public string[] defines = Array.Empty<string>();

        public override void OnImportAsset(AssetImportContext ctx)
        {
            if (!isRayTraceShader)
                return;

            var asset = ScriptableObject.CreateInstance<RayTraceShader>();

            // Write private serialized fields via SerializedObject.
            var so = new SerializedObject(asset);

            // Always prepend the Unity project root so shaders can include project-relative headers.
            string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            var allPaths = new string[1 + additionalIncludePaths.Length];
            allPaths[0] = projectRoot;
            for (int i = 0; i < additionalIncludePaths.Length; i++)
            {
                string p = Environment.ExpandEnvironmentVariables(additionalIncludePaths[i]);
                if (!Path.IsPathRooted(p))
                    p = Path.GetFullPath(Path.Combine(projectRoot, p));
                allPaths[1 + i] = p;
            }

            var pathsProp = so.FindProperty("additionalIncludePaths");
            pathsProp.arraySize = allPaths.Length;
            for (int i = 0; i < allPaths.Length; i++)
                pathsProp.GetArrayElementAtIndex(i).stringValue = allPaths[i];

            var extraArgsProp = so.FindProperty("_extraArgs");
            extraArgsProp.arraySize = extraArgs.Length;
            for (int i = 0; i < extraArgs.Length; i++)
                extraArgsProp.GetArrayElementAtIndex(i).stringValue = extraArgs[i];

            var definesProp = so.FindProperty("_defines");
            definesProp.arraySize = defines.Length;
            for (int i = 0; i < defines.Length; i++)
                definesProp.GetArrayElementAtIndex(i).stringValue = defines[i];

            so.ApplyModifiedPropertiesWithoutUndo();

            ctx.AddObjectToAsset("RayTraceShader", asset);
            ctx.SetMainObject(asset);
        }
    }

    [CustomEditor(typeof(RayTraceShaderImporter))]
    public class RayTraceShaderImporterEditor : ScriptedImporterEditor
    {
        public override void OnInspectorGUI()
        {
            base.OnInspectorGUI();

            // Only show compile button when a single asset is selected
            if (targets.Length != 1) return;

            var importer = (RayTraceShaderImporter)target;
            if (!importer.isRayTraceShader) return;

            var shader = AssetDatabase.LoadAssetAtPath<RayTraceShader>(importer.assetPath);
            if (shader == null) return;

            EditorGUILayout.Space(6);

            GUI.backgroundColor = shader.HasCompiledBytes ? new Color(0.4f, 0.8f, 0.4f) : new Color(1f, 0.6f, 0.3f);
            if (GUILayout.Button(shader.HasCompiledBytes ? "Recompile" : "Compile", GUILayout.Height(28)))
            {
                shader.ForceRecompile();
                EditorUtility.SetDirty(shader);
                AssetDatabase.SaveAssetIfDirty(shader);
            }
            GUI.backgroundColor = Color.white;

            EditorGUILayout.Space(2);
            if (shader.HasCompiledBytes)
                EditorGUILayout.HelpBox($"DXIL cached ({shader.CompiledByteCount:N0} bytes)", MessageType.Info);
            else
                EditorGUILayout.HelpBox("No compiled DXIL – click Compile to build.", MessageType.Warning);
        }
    }
}
