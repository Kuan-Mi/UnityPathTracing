using System;
using System.IO;
using UnityEditor;
using UnityEditor.AssetImporters;
using UnityEngine;

namespace NativeRender
{
    /// <summary>
    /// Imports <c>.rastershader</c> files as <see cref="NativeRasterShader"/> assets.
    /// The file content is raw HLSL containing BOTH a vertex entry point and a pixel entry
    /// point; the importer stores compile settings so the asset can produce VS + PS DXIL.
    ///
    /// To create one: right-click in the Project view and rename a text file to the
    /// <c>.rastershader</c> extension, or duplicate an existing one. The source typically
    /// <c>#include</c>s shared HLSL and defines e.g. <c>VSMain</c> (a fullscreen-triangle
    /// vertex shader) and <c>PSMain</c> (the pixel shader).
    /// </summary>
    [ScriptedImporter(1, "rastershader", -1000)]
    public class NativeRasterShaderImporter : ScriptedImporter
    {
        [Tooltip("Additional #include search directories (absolute or project-relative). The shader's own directory is always included.")]
        public string[] additionalIncludePaths = Array.Empty<string>();

        [Tooltip("Additional DXC compiler arguments (e.g. -HV 2021).")]
        public string[] extraArgs = Array.Empty<string>();

        [Tooltip("Preprocessor defines (e.g. FOO=1, BAR).")]
        public string[] defines = Array.Empty<string>();

        [Tooltip("Vertex entry point function name.")]
        public string vsEntryPoint = "VSMain";

        [Tooltip("Pixel entry point function name.")]
        public string psEntryPoint = "PSMain";

        [Tooltip("DXC vertex target profile (e.g. vs_6_6).")]
        public string vsTargetProfile = "vs_6_6";

        [Tooltip("DXC pixel target profile (e.g. ps_6_6).")]
        public string psTargetProfile = "ps_6_6";

        [Tooltip("Promote these ConstantBuffer bindings to root 32-bit constants. Name must match the HLSL variable; Count is the total 32-bit value count.")]
        public RootConstantsHint[] rootConstantsHints = Array.Empty<RootConstantsHint>();

        [Tooltip("Promote these buffer SRV / TLAS bindings to inline root descriptors. Each string must match the HLSL variable name exactly.")]
        public string[] rootSRVHints = Array.Empty<string>();

        [Tooltip("Per-sampler references to shared NativeSampler assets (assigned via the reflected " +
                 "Sampler rows). \"Name\" matches the HLSL sampler variable exactly. Samplers with no " +
                 "reference fall back to the name-inference convention (sampler_LinearClamp, …).")]
        public SamplerBinding[] samplerBindings = Array.Empty<SamplerBinding>();

        public override void OnImportAsset(AssetImportContext ctx)
        {
            var asset = ScriptableObject.CreateInstance<NativeRasterShader>();
            var so    = new SerializedObject(asset);

            string projectRoot    = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            var    globalSettings = NativeShaderProjectSettings.instance;

            // ── Include paths: globalIncludePaths + additionalIncludePaths (resolved to absolute) ──
            var allIncludeSources = new string[additionalIncludePaths.Length + globalSettings.globalIncludePaths.Length];
            for (int i = 0; i < globalSettings.globalIncludePaths.Length; i++)
                allIncludeSources[i] = globalSettings.globalIncludePaths[i];
            for (int i = 0; i < additionalIncludePaths.Length; i++)
                allIncludeSources[globalSettings.globalIncludePaths.Length + i] = additionalIncludePaths[i];

            var allPaths = new string[allIncludeSources.Length];
            for (int i = 0; i < allIncludeSources.Length; i++)
            {
                string p = Environment.ExpandEnvironmentVariables(allIncludeSources[i]);
                if (!Path.IsPathRooted(p))
                    p = Path.GetFullPath(Path.Combine(projectRoot, p));
                allPaths[i] = p;
            }

            WriteStringArray(so, "additionalIncludePaths", allPaths);

            // ── Extra args / defines: globals + per-asset ──
            WriteStringArray(so, "_extraArgs", Concat(globalSettings.globalExtraArgs, extraArgs));
            WriteStringArray(so, "_defines", Concat(globalSettings.globalDefines, defines));

            so.FindProperty("_vsEntryPoint").stringValue    = string.IsNullOrEmpty(vsEntryPoint) ? "VSMain" : vsEntryPoint;
            so.FindProperty("_psEntryPoint").stringValue    = string.IsNullOrEmpty(psEntryPoint) ? "PSMain" : psEntryPoint;
            so.FindProperty("_vsTargetProfile").stringValue = string.IsNullOrEmpty(vsTargetProfile) ? "vs_6_6" : vsTargetProfile;
            so.FindProperty("_psTargetProfile").stringValue = string.IsNullOrEmpty(psTargetProfile) ? "ps_6_6" : psTargetProfile;

            var hintsProp = so.FindProperty("_rootConstantsHints");
            hintsProp.arraySize = rootConstantsHints?.Length ?? 0;
            for (int i = 0; i < (rootConstantsHints?.Length ?? 0); i++)
            {
                var elem = hintsProp.GetArrayElementAtIndex(i);
                elem.FindPropertyRelative("Name").stringValue = rootConstantsHints[i].Name ?? "";
                elem.FindPropertyRelative("Count").intValue   = (int)rootConstantsHints[i].Count;
            }

            WriteStringArray(so, "_rootSRVHints", rootSRVHints ?? Array.Empty<string>());
            SamplerBindingSerialization.Write(so.FindProperty("_samplerBindings"), samplerBindings);
            SamplerBindingSerialization.RegisterDependencies(ctx, samplerBindings);

            so.ApplyModifiedPropertiesWithoutUndo();

            ctx.AddObjectToAsset("NativeRasterShader", asset);
            var filePath = Path.GetFullPath(ctx.assetPath);
            asset.ForceRecompile(filePath);
            ctx.SetMainObject(asset);
        }

        private static string[] Concat(string[] a, string[] b)
        {
            var r = new string[(a?.Length ?? 0) + (b?.Length ?? 0)];
            int k = 0;
            if (a != null)
                foreach (var s in a)
                    r[k++] = s;
            if (b != null)
                foreach (var s in b)
                    r[k++] = s;
            return r;
        }

        private static void WriteStringArray(SerializedObject so, string prop, string[] values)
        {
            var p = so.FindProperty(prop);
            p.arraySize = values.Length;
            for (int i = 0; i < values.Length; i++)
                p.GetArrayElementAtIndex(i).stringValue = values[i];
        }
    }

    [CustomEditor(typeof(NativeRasterShaderImporter))]
    internal class NativeRasterShaderImporterEditor : NativeShaderImporterEditorBase
    {
        protected override bool TryGetStatus(string assetPath,
            out bool hasCompiledBytes, out int byteCount, out string reflectionJson, out string shaderHash)
        {
            hasCompiledBytes = false;
            byteCount        = 0;
            reflectionJson   = "";
            shaderHash       = "";
            var shader = AssetDatabase.LoadAssetAtPath<NativeRasterShader>(assetPath);
            if (shader == null) return false;

            hasCompiledBytes = shader.HasCompiledBytes;
            byteCount        = shader.CompiledByteCount;
            reflectionJson   = shader.ReflectionJson;
            shaderHash       = shader.ShaderHash;
            return true;
        }

        protected override void DrawImportSettings()
        {
            EditorGUILayout.PropertyField(serializedObject.FindProperty("vsEntryPoint"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("psEntryPoint"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("vsTargetProfile"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("psTargetProfile"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("defines"), true);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("additionalIncludePaths"), true);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("extraArgs"), true);
        }

        // Reflected CBV / buffer-SRV / TLAS rows get an inline "promote to root binding" checkbox
        // (shared with the compute / ray-tracing importers), so a binding is configured where it's listed.
        protected override bool TryDrawBindingRow(ShaderBindingEntry e)
            => RootBindingHintsGUI.TryDrawBindingRow((ScriptedImporter)target, e);

        protected override void DrawExtraReflection(ShaderReflectionInfo info)
            => RootBindingHintsGUI.DrawStaleHints((ScriptedImporter)target, info);
    }

    /// <summary>Notifies live pipelines that a .rastershader was reimported so they rebuild.</summary>
    internal class NativeRasterShaderPostprocessor : AssetPostprocessor
    {
        static void OnPostprocessAllAssets(
            string[] importedAssets, string[] deletedAssets,
            string[] movedAssets, string[] movedFromAssetPaths)
        {
            foreach (string path in importedAssets)
            {
                if (!path.EndsWith(".rastershader", StringComparison.OrdinalIgnoreCase)) continue;
                var shader = AssetDatabase.LoadAssetAtPath<NativeRasterShader>(path);
                if (shader != null) NativeRasterShader.InvokeOnRecompiled(shader);
            }
        }
    }
}