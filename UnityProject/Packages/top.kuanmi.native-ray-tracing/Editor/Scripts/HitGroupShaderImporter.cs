using System;
using System.IO;
using UnityEditor;
using UnityEditor.AssetImporters;
using UnityEngine;

namespace NativeRender
{
    /// <summary>
    /// Imports <c>.hitgroupshader</c> files as <see cref="HitGroupShader"/> ScriptableObject assets.
    ///
    /// A hit-group shader contains only <b>ClosestHit</b> and/or <b>AnyHit</b> entry points
    /// and is merged with a primary <see cref="RayTraceShader"/> (raygen + miss) at pipeline
    /// creation time via <see cref="RayTracePipeline(RayTraceShader, HitGroupShader[])"/>.
    ///
    /// Unlike <c>.rayshader</c> assets, there is no <c>rayGenName</c> or
    /// <c>maxPayloadSizeInBytes</c> field — those belong to the primary pipeline shader.
    /// </summary>
    [ScriptedImporter(1, "hitgroupshader", 0)]
    public class HitGroupShaderImporter : ScriptedImporter
    {
        [Tooltip("Additional #include search directories (absolute paths). The shader file's own directory is always included automatically.")]
        public string[] additionalIncludePaths = Array.Empty<string>();

        [Tooltip("Additional DXC compiler arguments (e.g. -disable-payload-qualifiers, -HV 2021).")]
        public string[] extraArgs = Array.Empty<string>();

        [Tooltip("Preprocessor defines (e.g. FOO=1, RTXPT_MATERIAL_IS_EMISSIVE=1).")]
        public string[] defines = Array.Empty<string>();

        [Tooltip("DXC target profile (e.g. lib_6_6, lib_6_9).")]
        public string targetProfile = "lib_6_6";

        public override void OnImportAsset(AssetImportContext ctx)
        {
            var asset = ScriptableObject.CreateInstance<HitGroupShader>();
            var so    = new SerializedObject(asset);

            string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            var globalSettings = NativeShaderProjectSettings.instance;

            // ── Include paths ──
            var allIncludeSources = new string[additionalIncludePaths.Length + globalSettings.globalIncludePaths.Length];
            for (int i = 0; i < globalSettings.globalIncludePaths.Length; i++)
                allIncludeSources[i] = globalSettings.globalIncludePaths[i];
            for (int i = 0; i < additionalIncludePaths.Length; i++)
                allIncludeSources[globalSettings.globalIncludePaths.Length + i] = additionalIncludePaths[i];

            var allPaths = new string[1 + allIncludeSources.Length];
            allPaths[0] = projectRoot;
            for (int i = 0; i < allIncludeSources.Length; i++)
            {
                string p = Environment.ExpandEnvironmentVariables(allIncludeSources[i]);
                if (!Path.IsPathRooted(p))
                    p = Path.GetFullPath(Path.Combine(projectRoot, p));
                allPaths[1 + i] = p;
            }

            var pathsProp = so.FindProperty("additionalIncludePaths");
            pathsProp.arraySize = allPaths.Length;
            for (int i = 0; i < allPaths.Length; i++)
                pathsProp.GetArrayElementAtIndex(i).stringValue = allPaths[i];

            // ── Extra args ──
            var allExtraArgs = new string[globalSettings.globalExtraArgs.Length + extraArgs.Length];
            for (int i = 0; i < globalSettings.globalExtraArgs.Length; i++)
                allExtraArgs[i] = globalSettings.globalExtraArgs[i];
            for (int i = 0; i < extraArgs.Length; i++)
                allExtraArgs[globalSettings.globalExtraArgs.Length + i] = extraArgs[i];

            var extraArgsProp = so.FindProperty("_extraArgs");
            extraArgsProp.arraySize = allExtraArgs.Length;
            for (int i = 0; i < allExtraArgs.Length; i++)
                extraArgsProp.GetArrayElementAtIndex(i).stringValue = allExtraArgs[i];

            // ── Defines ──
            var allDefines = new string[globalSettings.globalDefines.Length + defines.Length];
            for (int i = 0; i < globalSettings.globalDefines.Length; i++)
                allDefines[i] = globalSettings.globalDefines[i];
            for (int i = 0; i < defines.Length; i++)
                allDefines[globalSettings.globalDefines.Length + i] = defines[i];

            var definesProp = so.FindProperty("_defines");
            definesProp.arraySize = allDefines.Length;
            for (int i = 0; i < allDefines.Length; i++)
                definesProp.GetArrayElementAtIndex(i).stringValue = allDefines[i];

            // ── Target profile ──
            so.FindProperty("_targetProfile").stringValue =
                string.IsNullOrEmpty(targetProfile) ? "lib_6_6" : targetProfile;

            so.ApplyModifiedPropertiesWithoutUndo();

            ctx.AddObjectToAsset("HitGroupShader", asset);
            asset.ForceRecompile(Path.GetFullPath(ctx.assetPath));
            ctx.SetMainObject(asset);
        }
    }

    // -----------------------------------------------------------------------
    // Inspector
    // -----------------------------------------------------------------------

    [CustomEditor(typeof(HitGroupShaderImporter))]
    public class HitGroupShaderImporterEditor : ScriptedImporterEditor
    {
        public override void OnInspectorGUI()
        {
            base.OnInspectorGUI();

            if (targets.Length != 1) return;
            var importer = (HitGroupShaderImporter)target;
            var shader   = AssetDatabase.LoadAssetAtPath<HitGroupShader>(importer.assetPath);
            if (shader == null) return;

            EditorGUILayout.Space(6);

            GUI.backgroundColor = shader.HasCompiledBytes
                ? new Color(0.4f, 0.8f, 0.4f)
                : new Color(1f, 0.6f, 0.3f);
            if (GUILayout.Button(shader.HasCompiledBytes ? "Recompile" : "Compile", GUILayout.Height(28)))
                importer.SaveAndReimport();
            GUI.backgroundColor = Color.white;

            EditorGUILayout.Space(2);
            if (shader.HasCompiledBytes)
                EditorGUILayout.HelpBox($"DXIL cached ({shader.CompiledByteCount:N0} bytes)", MessageType.Info);
            else
                EditorGUILayout.HelpBox("No compiled DXIL – click Compile to build.", MessageType.Warning);
        }
    }

    // -----------------------------------------------------------------------
    // Hot-reload notifier
    // -----------------------------------------------------------------------

    internal class HitGroupShaderPostprocessor : AssetPostprocessor
    {
        static void OnPostprocessAllAssets(
            string[] importedAssets, string[] deletedAssets,
            string[] movedAssets,    string[] movedFromAssetPaths)
        {
            foreach (string path in importedAssets)
            {
                if (!path.EndsWith(".hitgroupshader", StringComparison.OrdinalIgnoreCase)) continue;
                var shader = AssetDatabase.LoadAssetAtPath<HitGroupShader>(path);
                if (shader != null)
                    HitGroupShader.InvokeOnRecompiled(shader);
            }
        }
    }
}
