using System;
using System.Collections.Generic;
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
        private bool _showSRV        = true;
        private bool _showUAV        = true;
        private bool _showCBV        = true;
        private bool _showSampler    = true;
        private bool _showTLAS       = true;
        private bool _showShaders    = true;
        private bool _showReflection = true;

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

                DrawShaderGroup(ref _showShaders, info.Shaders);

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

        private static void PrintReflectionToConsole(string assetPath, ShaderReflectionInfo info)
        {
            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"=== HitGroup Shader Reflection: {System.IO.Path.GetFileName(assetPath)} ===");
            sb.AppendLine();

            if (info.Shaders.Count > 0)
            {
                sb.AppendLine($"-- Shader Entry Points ({info.Shaders.Count}) --");
                foreach (var s in info.Shaders)
                    sb.AppendLine($"  [{s.Kind,-12}]  {s.Name}");
                sb.AppendLine();
            }

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

        private static void DrawShaderGroup(ref bool foldout, List<ShaderEntry> entries)
        {
            if (entries.Count == 0) return;
            foldout = EditorGUILayout.Foldout(foldout, $"Shaders  ({entries.Count})", true);
            if (!foldout) return;
            EditorGUI.indentLevel++;
            foreach (var s in entries)
                EditorGUILayout.LabelField(s.Name, s.Kind);
            EditorGUI.indentLevel--;
        }

        private static void DrawBindingGroup(ref bool foldout, string label, List<ShaderBindingEntry> entries)
        {
            if (entries.Count == 0) return;
            foldout = EditorGUILayout.Foldout(foldout, $"{label}  ({entries.Count})", true);
            if (!foldout) return;
            EditorGUI.indentLevel++;
            foreach (var e in entries)
                EditorGUILayout.LabelField(e.Name, $"{e.HlslType}   space{e.Space}:{ResourcePrefix(label)}{e.Reg}");
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

        private class ShaderEntry
        {
            public string Name;
            public string Kind;
        }

        private class ShaderReflectionInfo
        {
            public List<ShaderEntry>        Shaders = new();
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

                    // Parse "shaders" array
                    int shadersIdx = json.IndexOf("\"shaders\"", StringComparison.Ordinal);
                    if (shadersIdx >= 0)
                    {
                        int aStart = json.IndexOf('[', shadersIdx);
                        int aEnd   = json.IndexOf(']', aStart);
                        if (aStart >= 0 && aEnd > aStart)
                        {
                            string body = json.Substring(aStart + 1, aEnd - aStart - 1);
                            int pos2 = 0;
                            while (pos2 < body.Length)
                            {
                                int o1 = body.IndexOf('{', pos2);
                                if (o1 < 0) break;
                                int o2 = body.IndexOf('}', o1);
                                if (o2 < 0) break;
                                string obj = body.Substring(o1 + 1, o2 - o1 - 1);
                                result.Shaders.Add(new ShaderEntry
                                {
                                    Name = ExtractString(obj, "name"),
                                    Kind = ExtractString(obj, "kind"),
                                });
                                pos2 = o2 + 1;
                            }
                        }
                    }

                    // Parse "bindings" array
                    int bindingsIdx = json.IndexOf("\"bindings\"", StringComparison.Ordinal);
                    if (bindingsIdx < 0) return result;

                    int arrayStart = json.IndexOf('[', bindingsIdx);
                    int arrayEnd   = json.IndexOf(']', arrayStart);
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
