using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.AssetImporters;
using UnityEngine;
using Object = UnityEngine.Object;

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

        [Tooltip("DXC target profile (e.g. lib_6_6, lib_6_9).")]
        public string targetProfile = "lib_6_6";

        public override void OnImportAsset(AssetImportContext ctx)
        {
            if (!isRayTraceShader)
                return;

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

            so.ApplyModifiedPropertiesWithoutUndo();

            ctx.AddObjectToAsset("RayTraceShader", asset);
            var filePath = Path.GetFullPath(ctx.assetPath);
            asset.ForceRecompile(filePath);
            ctx.SetMainObject(asset);
        }
    }

    [CustomEditor(typeof(RayTraceShaderImporter))]
    public class RayTraceShaderImporterEditor : ScriptedImporterEditor
    {
        private bool _showSRV        = true;
        private bool _showUAV        = true;
        private bool _showCBV        = true;
        private bool _showSampler    = true;
        private bool _showTLAS       = true;
        private bool _showReflection = true;

        public override void OnInspectorGUI()
        {
            base.OnInspectorGUI();

            if (targets.Length != 1) return;

            var importer = (RayTraceShaderImporter)target;
            if (!importer.isRayTraceShader) return;

            var shader = AssetDatabase.LoadAssetAtPath<RayTraceShader>(importer.assetPath);
            if (shader == null) return;

            EditorGUILayout.Space(6);

            GUI.backgroundColor = shader.HasCompiledBytes ? new Color(0.4f, 0.8f, 0.4f) : new Color(1f, 0.6f, 0.3f);
            if (GUILayout.Button(shader.HasCompiledBytes ? "Recompile" : "Compile", GUILayout.Height(28)))
            {
                ApplyAndImport(importer);
            }
            GUI.backgroundColor = Color.white;

            EditorGUILayout.Space(2);
            if (shader.HasCompiledBytes)
                EditorGUILayout.HelpBox($"DXIL cached ({shader.CompiledByteCount:N0} bytes)", MessageType.Info);
            else
                EditorGUILayout.HelpBox("No compiled DXIL – click Compile to build.", MessageType.Warning);

            // ---------------------------------------------------------------
            // Reflection panel
            // ---------------------------------------------------------------
            string json = shader.ReflectionJson;
            if (string.IsNullOrEmpty(json)) return;

            var info = ShaderReflectionInfo.Parse(json);
            if (info == null) return;

            EditorGUILayout.Space(6);
            _showReflection = EditorGUILayout.Foldout(_showReflection, "Shader Reflection", true, EditorStyles.foldoutHeader);
            if (_showReflection)
            {
                EditorGUI.indentLevel++;

                DrawBindingGroup(ref _showSRV,     "SRV",     info.SRV);
                DrawBindingGroup(ref _showUAV,     "UAV",     info.UAV);
                DrawBindingGroup(ref _showCBV,     "CBV",     info.CBV);
                DrawBindingGroup(ref _showSampler, "Sampler", info.Sampler);
                DrawBindingGroup(ref _showTLAS,    "TLAS",    info.TLAS);

                EditorGUILayout.Space(4);
                if (GUILayout.Button("Print to Console"))
                    PrintReflectionToConsole(importer.assetPath, info);

                EditorGUI.indentLevel--;
            }
        }

        private static void ApplyAndImport(RayTraceShaderImporter importer)
        {
            importer.SaveAndReimport();
        }

        private static void PrintReflectionToConsole(string assetPath, ShaderReflectionInfo info)
        {
            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"=== Shader Reflection: {System.IO.Path.GetFileName(assetPath)} ===");
            sb.AppendLine();

            void AppendGroup(string label, List<ShaderBindingEntry> entries)
            {
                if (entries.Count == 0) return;
                sb.AppendLine($"-- {label} ({entries.Count}) --");
                foreach (var e in entries)
                    sb.AppendLine($"  {e.Name,-32}  {e.HlslType,-36}  space{e.Space}:{ResourcePrefix(label)}{e.Reg}");
                sb.AppendLine();
            }

            AppendGroup("SRV",     info.SRV);
            AppendGroup("UAV",     info.UAV);
            AppendGroup("CBV",     info.CBV);
            AppendGroup("Sampler", info.Sampler);
            AppendGroup("TLAS",    info.TLAS);

            Debug.Log(sb.ToString());
        }

        private static void DrawBindingGroup(ref bool foldout, string label, List<ShaderBindingEntry> entries)
        {
            if (entries.Count == 0) return;

            foldout = EditorGUILayout.Foldout(foldout, $"{label}  ({entries.Count})", true);
            if (!foldout) return;

            EditorGUI.indentLevel++;
            foreach (var e in entries)
                EditorGUILayout.LabelField(e.Name, $"{e.HlslType}    space{e.Space}:{ResourcePrefix(label)}{e.Reg}");
            EditorGUI.indentLevel--;
        }

        private static string ResourcePrefix(string type) => type switch
        {
            "SRV"     => "t",
            "UAV"     => "u",
            "CBV"     => "b",
            "Sampler" => "s",
            "TLAS"    => "t",
            _         => ""
        };

        // -------------------------------------------------------------------
        // Minimal JSON parser for the reflection JSON shape.
        // -------------------------------------------------------------------
        private class ShaderBindingEntry
        {
            public string Name;
            public string Type;
            public int    Space;
            public int    Reg;
            public string Dim;
            public string RetType;

            public string HlslType
            {
                get
                {
                    if (Type == "CBV")     return "ConstantBuffer";
                    if (Type == "Sampler") return "SamplerState";
                    if (Type == "TLAS")    return "RaytracingAccelerationStructure";

                    string prefix = Type == "UAV" ? "RW" : "";
                    string dim    = string.IsNullOrEmpty(Dim) ? "Buffer" : Dim;

                    if (dim == "ByteAddressBuffer")
                        return prefix + "ByteAddressBuffer";

                    if (string.IsNullOrEmpty(RetType))
                        return prefix + (dim == "Buffer" ? "StructuredBuffer" : dim);

                    return prefix + dim + "<" + RetType + "4>";
                }
            }
        }

        private class ShaderReflectionInfo
        {
            public List<ShaderBindingEntry> SRV     = new();
            public List<ShaderBindingEntry> UAV     = new();
            public List<ShaderBindingEntry> CBV     = new();
            public List<ShaderBindingEntry> Sampler = new();
            public List<ShaderBindingEntry> TLAS    = new();

            public static ShaderReflectionInfo Parse(string json)
            {
                try
                {
                    var result = new ShaderReflectionInfo();

                    int bindingsIdx = json.IndexOf("\"bindings\"", StringComparison.Ordinal);
                    if (bindingsIdx < 0) return result;

                    int arrayStart = json.IndexOf('[', bindingsIdx);
                    int arrayEnd   = json.LastIndexOf(']');
                    if (arrayStart < 0 || arrayEnd <= arrayStart) return result;

                    string arrayBody = json.Substring(arrayStart + 1, arrayEnd - arrayStart - 1);
                    int pos = 0;
                    while (pos < arrayBody.Length)
                    {
                        int objStart = arrayBody.IndexOf('{', pos);
                        if (objStart < 0) break;
                        int objEnd = arrayBody.IndexOf('}', objStart);
                        if (objEnd < 0) break;

                        string obj = arrayBody.Substring(objStart + 1, objEnd - objStart - 1);
                        var entry = new ShaderBindingEntry
                        {
                            Name    = ExtractString(obj, "name"),
                            Type    = ExtractString(obj, "type"),
                            Space   = ExtractInt   (obj, "space"),
                            Reg     = ExtractInt   (obj, "reg"),
                            Dim     = ExtractString(obj, "dim"),
                            RetType = ExtractString(obj, "retType"),
                        };

                        var list = entry.Type switch
                        {
                            "SRV"     => result.SRV,
                            "UAV"     => result.UAV,
                            "CBV"     => result.CBV,
                            "Sampler" => result.Sampler,
                            "TLAS"    => result.TLAS,
                            _         => null
                        };
                        list?.Add(entry);

                        pos = objEnd + 1;
                    }

                    return result;
                }
                catch
                {
                    return null;
                }
            }

            private static string ExtractString(string obj, string key)
            {
                string search = $"\"{key}\"";
                int ki = obj.IndexOf(search, StringComparison.Ordinal);
                if (ki < 0) return "";
                int colon = obj.IndexOf(':', ki + search.Length);
                if (colon < 0) return "";
                int q1 = obj.IndexOf('"', colon + 1);
                if (q1 < 0) return "";
                int q2 = obj.IndexOf('"', q1 + 1);
                if (q2 < 0) return "";
                return obj.Substring(q1 + 1, q2 - q1 - 1);
            }

            private static int ExtractInt(string obj, string key)
            {
                string search = $"\"{key}\"";
                int ki = obj.IndexOf(search, StringComparison.Ordinal);
                if (ki < 0) return 0;
                int colon = obj.IndexOf(':', ki + search.Length);
                if (colon < 0) return 0;
                int start = colon + 1;
                while (start < obj.Length && (obj[start] == ' ' || obj[start] == '\t')) start++;
                int end = start;
                while (end < obj.Length && (char.IsDigit(obj[end]) || obj[end] == '-')) end++;
                int.TryParse(obj.Substring(start, end - start), out int val);
                return val;
            }
        }
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
