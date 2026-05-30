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
    [ScriptedImporter(1, "rayshader", -1000)]
    public class RayTraceShaderImporter : ScriptedImporter
    {
        [Tooltip("Additional #include search directories (absolute paths). The shader file's own directory is always included automatically.")]
        public string[] additionalIncludePaths = Array.Empty<string>();

        [Tooltip("Additional DXC compiler arguments (e.g. -disable-payload-qualifiers, -HV 2021).")]
        public string[] extraArgs = Array.Empty<string>();

        [Tooltip("Preprocessor defines (e.g. FOO=1, BAR).")]
        public string[] defines = Array.Empty<string>();

        [Tooltip("DXC target profile (e.g. lib_6_6, lib_6_9).")]
        public string targetProfile = "lib_6_6";

        [Tooltip("MaxPayloadSizeInBytes passed to D3D12 CreateStateObject. Must be >= the size of all payload structs used in the shader. Default 4 (uint = no real payload).")]
        public uint maxPayloadSizeInBytes = 4;

        [Tooltip("Name of the RayGeneration shader entry point to use for DispatchRays. Leave empty to use the first discovered RayGen shader.")]
        public string rayGenName = "";

        [Tooltip("Promote these ConstantBuffer bindings to root 32-bit constants. " +
                 "\"Name\" must match the HLSL variable name exactly. " +
                 "\"Count\" is the total number of 32-bit values in the buffer.")]
        public RootConstantsHint[] rootConstantsHints = Array.Empty<RootConstantsHint>();

        [Tooltip("Promote these buffer SRV / TLAS bindings to inline root descriptors instead of a " +
                 "descriptor-table entry. Only valid for buffer resources. " +
                 "Each string must match the HLSL variable name exactly.")]
        public string[] rootSRVHints = Array.Empty<string>();

        [Tooltip("Override static-sampler attributes per sampler instead of inferring them from the " +
                 "sampler name. \"Name\" must match the HLSL sampler variable exactly. Samplers not " +
                 "listed here fall back to the name-inference convention (sampler_LinearClamp, …).")]
        public SamplerHint[] samplerHints = Array.Empty<SamplerHint>();

        public override void OnImportAsset(AssetImportContext ctx)
        {
            var asset = ScriptableObject.CreateInstance<RayTraceShader>();

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

            var targetProfileProp = so.FindProperty("_targetProfile");
            targetProfileProp.stringValue = string.IsNullOrEmpty(targetProfile) ? "lib_6_6" : targetProfile;

            var payloadSizeProp = so.FindProperty("_maxPayloadSizeInBytes");
            payloadSizeProp.longValue = maxPayloadSizeInBytes;

            var rayGenNameProp = so.FindProperty("_rayGenName");
            rayGenNameProp.stringValue = rayGenName ?? "";

            var hintsProp = so.FindProperty("_rootConstantsHints");
            hintsProp.arraySize = rootConstantsHints?.Length ?? 0;
            for (int i = 0; i < (rootConstantsHints?.Length ?? 0); i++)
            {
                var elem = hintsProp.GetArrayElementAtIndex(i);
                elem.FindPropertyRelative("Name").stringValue = rootConstantsHints[i].Name ?? "";
                elem.FindPropertyRelative("Count").intValue    = (int)rootConstantsHints[i].Count;
            }

            var srvHintsProp = so.FindProperty("_rootSRVHints");
            srvHintsProp.arraySize = rootSRVHints?.Length ?? 0;
            for (int i = 0; i < (rootSRVHints?.Length ?? 0); i++)
                srvHintsProp.GetArrayElementAtIndex(i).stringValue = rootSRVHints[i] ?? "";

            SamplerHintSerialization.Write(so.FindProperty("_samplerHints"), samplerHints);

            so.ApplyModifiedPropertiesWithoutUndo();

            ctx.AddObjectToAsset("RayTraceShader", asset);
            var filePath = Path.GetFullPath(ctx.assetPath);
            asset.ForceRecompile(filePath);
            ctx.SetMainObject(asset);
        }
    }

    [CustomEditor(typeof(RayTraceShaderImporter))]
    internal class RayTraceShaderImporterEditor : NativeShaderImporterEditorBase
    {
        protected override bool TryGetStatus(string assetPath,
            out bool hasCompiledBytes, out int byteCount, out string reflectionJson)
        {
            hasCompiledBytes = false; byteCount = 0; reflectionJson = "";
            var shader = AssetDatabase.LoadAssetAtPath<RayTraceShader>(assetPath);
            if (shader == null) return false;

            hasCompiledBytes = shader.HasCompiledBytes;
            byteCount        = shader.CompiledByteCount;
            reflectionJson   = shader.ReflectionJson;
            return true;
        }

        protected override void DrawImportSettings()
        {
            EditorGUILayout.PropertyField(serializedObject.FindProperty("targetProfile"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("maxPayloadSizeInBytes"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("rayGenName"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("defines"), true);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("additionalIncludePaths"), true);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("extraArgs"), true);
        }

        // Inline "promote to root binding" checkboxes on the reflected CBV / buffer-SRV / TLAS rows,
        // shared with the compute importer. Pipeline-level hints belong to the primary ray shader.
        protected override bool TryDrawBindingRow(ShaderBindingEntry e)
            => RootBindingHintsGUI.TryDrawBindingRow((ScriptedImporter)target, e);

        protected override void DrawExtraReflection(ShaderReflectionInfo info)
            => RootBindingHintsGUI.DrawStaleHints((ScriptedImporter)target, info);
    }

    /// <summary>
    /// Notifies any live <see cref="RayTracePipeline"/> instances that a .rayshader asset has been
    /// reimported so they can rebuild their native D3D12 handles using the new DXIL bytes.
    /// </summary>
    internal class RayTraceShaderPostprocessor : AssetPostprocessor
    {
        static void OnPostprocessAllAssets(
            string[] importedAssets, string[] deletedAssets,
            string[] movedAssets,    string[] movedFromAssetPaths)
        {
            foreach (string path in importedAssets)
            {
                if (!path.EndsWith(".rayshader", System.StringComparison.OrdinalIgnoreCase))
                    continue;

                // Load the now-persisted asset and fire the event so RayTracePipeline can rebuild.
                var shader = AssetDatabase.LoadAssetAtPath<RayTraceShader>(path);
                if (shader != null)
                    RayTraceShader.InvokeOnRecompiled(shader);
            }
        }
    }
}
