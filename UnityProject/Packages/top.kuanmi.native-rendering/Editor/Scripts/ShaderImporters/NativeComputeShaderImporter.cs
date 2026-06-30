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
    [ScriptedImporter(1, "computeshader", -1000)]
    public class NativeComputeShaderImporter : ScriptedImporter
    {
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

        [Tooltip("Promote these ConstantBuffer bindings to root 32-bit constants (SetComputeRoot32BitConstants). " +
                 "\"Name\" must match the HLSL variable name exactly. " +
                 "\"Count\" is the total number of 32-bit values in the buffer.")]
        public RootConstantsHint[] rootConstantsHints = Array.Empty<RootConstantsHint>();

        [Tooltip("Promote these buffer SRV / TLAS bindings to inline root descriptors (SetComputeRootShaderResourceView) " +
                 "instead of a descriptor-table entry. Only valid for buffer resources. " +
                 "Each string must match the HLSL variable name exactly.")]
        public string[] rootSRVHints = Array.Empty<string>();

        [Tooltip("Per-sampler references to shared NativeSampler assets (assigned via the reflected " +
                 "Sampler rows). \"Name\" matches the HLSL sampler variable exactly. Samplers with no " +
                 "reference fall back to the name-inference convention (sampler_LinearClamp, …).")]
        public SamplerBinding[] samplerBindings = Array.Empty<SamplerBinding>();

        public override void OnImportAsset(AssetImportContext ctx)
        {
            var asset = ScriptableObject.CreateInstance<NativeComputeShader>();

            // Write private serialized fields via SerializedObject.
            var so = new SerializedObject(asset);

            // Always prepend the Unity project root so shaders can include project-relative headers.
            string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));

            // Merge global project settings (prepended) with per-asset settings.
            var globalSettings = NativeShaderProjectSettings.instance;

            // ── Include paths: [projectRoot] + globalIncludePaths + additionalIncludePaths ──
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

            // ── Extra args: globalExtraArgs + extraArgs ──
            var allExtraArgs = new string[globalSettings.globalExtraArgs.Length + extraArgs.Length];
            for (int i = 0; i < globalSettings.globalExtraArgs.Length; i++)
                allExtraArgs[i] = globalSettings.globalExtraArgs[i];
            for (int i = 0; i < extraArgs.Length; i++)
                allExtraArgs[globalSettings.globalExtraArgs.Length + i] = extraArgs[i];

            var extraArgsProp = so.FindProperty("_extraArgs");
            extraArgsProp.arraySize = allExtraArgs.Length;
            for (int i = 0; i < allExtraArgs.Length; i++)
                extraArgsProp.GetArrayElementAtIndex(i).stringValue = allExtraArgs[i];

            // ── Defines: globalDefines + defines ──
            var allDefines = new string[globalSettings.globalDefines.Length + defines.Length];
            for (int i = 0; i < globalSettings.globalDefines.Length; i++)
                allDefines[i] = globalSettings.globalDefines[i];
            for (int i = 0; i < defines.Length; i++)
                allDefines[globalSettings.globalDefines.Length + i] = defines[i];

            var definesProp = so.FindProperty("_defines");
            definesProp.arraySize = allDefines.Length;
            for (int i = 0; i < allDefines.Length; i++)
                definesProp.GetArrayElementAtIndex(i).stringValue = allDefines[i];

            var entryPointProp = so.FindProperty("_entryPoint");
            entryPointProp.stringValue = string.IsNullOrEmpty(entryPoint) ? "main" : entryPoint;

            var targetProfileProp = so.FindProperty("_targetProfile");
            targetProfileProp.stringValue = string.IsNullOrEmpty(targetProfile) ? "cs_6_6" : targetProfile;

            var hintsProp = so.FindProperty("_rootConstantsHints");
            hintsProp.arraySize = rootConstantsHints?.Length ?? 0;
            for (int i = 0; i < (rootConstantsHints?.Length ?? 0); i++)
            {
                var elem = hintsProp.GetArrayElementAtIndex(i);
                elem.FindPropertyRelative("Name").stringValue = rootConstantsHints[i].Name ?? "";
                elem.FindPropertyRelative("Count").intValue   = (int)rootConstantsHints[i].Count;
            }

            var srvHintsProp = so.FindProperty("_rootSRVHints");
            srvHintsProp.arraySize = rootSRVHints?.Length ?? 0;
            for (int i = 0; i < (rootSRVHints?.Length ?? 0); i++)
                srvHintsProp.GetArrayElementAtIndex(i).stringValue = rootSRVHints[i] ?? "";

            SamplerBindingSerialization.Write(so.FindProperty("_samplerBindings"), samplerBindings);
            SamplerBindingSerialization.RegisterDependencies(ctx, samplerBindings);

            so.ApplyModifiedPropertiesWithoutUndo();

            ctx.AddObjectToAsset("NativeComputeShader", asset);
            var filePath = Path.GetFullPath(ctx.assetPath);
            asset.ForceRecompile(filePath);
            ctx.SetMainObject(asset);
        }
    }

    [CustomEditor(typeof(NativeComputeShaderImporter))]
    internal class NativeComputeShaderImporterEditor : NativeShaderImporterEditorBase
    {
        protected override bool TryGetStatus(string assetPath,
            out bool hasCompiledBytes, out int byteCount, out string reflectionJson, out string shaderHash)
        {
            hasCompiledBytes = false;
            byteCount        = 0;
            reflectionJson   = "";
            shaderHash       = "";
            var shader = AssetDatabase.LoadAssetAtPath<NativeComputeShader>(assetPath);
            if (shader == null) return false;

            hasCompiledBytes = shader.HasCompiledBytes;
            byteCount        = shader.CompiledByteCount;
            reflectionJson   = shader.ReflectionJson;
            shaderHash       = shader.ShaderHash;
            return true;
        }

        protected override void DrawImportSettings()
        {
            EditorGUILayout.PropertyField(serializedObject.FindProperty("entryPoint"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("targetProfile"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("defines"), true);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("additionalIncludePaths"), true);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("extraArgs"), true);
        }

        // Reflected CBV / buffer-SRV / TLAS rows get an inline "promote to root binding" checkbox
        // (shared with the ray-tracing importer), so a binding is configured right where it's listed.
        protected override bool TryDrawBindingRow(ShaderBindingEntry e)
            => RootBindingHintsGUI.TryDrawBindingRow((ScriptedImporter)target, e);

        protected override void DrawExtraReflection(ShaderReflectionInfo info)
            => RootBindingHintsGUI.DrawStaleHints((ScriptedImporter)target, info);
    }

    /// <summary>
    /// Notifies any live <see cref="NativeComputePipeline"/> instances that a .computeshader asset
    /// has been reimported so they can rebuild their native D3D12 handles using the new DXIL bytes.
    /// </summary>
    internal class NativeComputeShaderPostprocessor : AssetPostprocessor
    {
        static void OnPostprocessAllAssets(
            string[] importedAssets, string[] deletedAssets,
            string[] movedAssets, string[] movedFromAssetPaths)
        {
            foreach (string path in importedAssets)
            {
                if (!path.EndsWith(".computeshader", System.StringComparison.OrdinalIgnoreCase))
                    continue;

                var shader = AssetDatabase.LoadAssetAtPath<NativeComputeShader>(path);
                if (shader != null)
                    NativeComputeShader.InvokeOnRecompiled(shader);
            }
        }
    }
}