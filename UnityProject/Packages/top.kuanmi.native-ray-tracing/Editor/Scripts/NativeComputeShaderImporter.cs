using System;
using System.IO;
using UnityEditor;
using UnityEditor.AssetImporters;
using UnityEngine;

namespace NativeRender
{
    /// <summary>
    /// Imports <c>.computeshader</c> files as <see cref="NativeComputeShader"/> ScriptableObject assets.
    /// The file content is raw HLSL; the importer stores the absolute path so the native
    /// plugin can compile it at runtime via DXC.
    ///
    /// To create a new compute shader asset: right-click in the Project view and rename any
    /// text file to have the <c>.computeshader</c> extension, or duplicate an existing one.
    /// </summary>
    [ScriptedImporter(1, "computeshader", 0)]
    public class NativeComputeShaderImporter : ScriptedImporter
    {
        [Tooltip("当设置为 True 时，该 .hlsl 文件将作为 NativeComputeShader 资产导入。")]
        public bool isComputeShader = false;

        [Tooltip("Additional #include search directories (absolute paths). The shader file's own directory is always included automatically.")]
        public string[] additionalIncludePaths = Array.Empty<string>();

        [Tooltip("Additional DXC compiler arguments (e.g. -HV 2021).")]
        public string[] extraArgs = Array.Empty<string>();

        [Tooltip("Preprocessor defines (e.g. FOO=1, BAR).")]
        public string[] defines = Array.Empty<string>();

        [Tooltip("Entry point function name (e.g. main).")]
        public string entryPoint = "main";

        [Tooltip("DXC target profile (e.g. cs_6_6).")]
        public string targetProfile = "cs_6_6";

        public override void OnImportAsset(AssetImportContext ctx)
        {
            if (!isComputeShader)
                return;

            var asset = ScriptableObject.CreateInstance<NativeComputeShader>();

            // Write private serialized fields via SerializedObject.
            var so = new SerializedObject(asset);

            // Always prepend the Unity project root so shaders can include project-relative headers.
            string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            var allPaths = new string[1 + additionalIncludePaths.Length];
            allPaths[0] = projectRoot;
            additionalIncludePaths.CopyTo(allPaths, 1);

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

            var entryPointProp = so.FindProperty("_entryPoint");
            entryPointProp.stringValue = string.IsNullOrEmpty(entryPoint) ? "main" : entryPoint;

            var targetProfileProp = so.FindProperty("_targetProfile");
            targetProfileProp.stringValue = string.IsNullOrEmpty(targetProfile) ? "cs_6_6" : targetProfile;

            so.ApplyModifiedPropertiesWithoutUndo();

            ctx.AddObjectToAsset("NativeComputeShader", asset);
            ctx.SetMainObject(asset);
        }
    }

    [CustomEditor(typeof(NativeComputeShaderImporter))]
    public class NativeComputeShaderImporterEditor : ScriptedImporterEditor
    {
        public override void OnInspectorGUI()
        {
            base.OnInspectorGUI();

            // Only show compile button when a single asset is selected
            if (targets.Length != 1) return;

            var importer = (NativeComputeShaderImporter)target;
            if (!importer.isComputeShader) return;

            var shader = AssetDatabase.LoadAssetAtPath<NativeComputeShader>(importer.assetPath);
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
