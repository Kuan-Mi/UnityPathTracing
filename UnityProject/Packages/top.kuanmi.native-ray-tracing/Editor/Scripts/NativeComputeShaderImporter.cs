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

            var entryPointProp = so.FindProperty("_entryPoint");
            entryPointProp.stringValue = string.IsNullOrEmpty(entryPoint) ? "main" : entryPoint;

            var targetProfileProp = so.FindProperty("_targetProfile");
            targetProfileProp.stringValue = string.IsNullOrEmpty(targetProfile) ? "cs_6_6" : targetProfile;

            so.ApplyModifiedPropertiesWithoutUndo();

            ctx.AddObjectToAsset("NativeComputeShader", asset);
            var filePath = Path.GetFullPath(ctx.assetPath);
            asset.ForceRecompile(filePath);
            ctx.SetMainObject(asset);
        }
    }

    [CustomEditor(typeof(NativeComputeShaderImporter))]
    public class NativeComputeShaderImporterEditor : ScriptedImporterEditor
    {
        // Foldout state — not persisted, resets on selection change (fine for a dev tool)
        private bool _showSRV     = true;
        private bool _showUAV     = true;
        private bool _showCBV     = true;
        private bool _showSampler = true;
        private bool _showTLAS    = true;
        private bool _showReflection = true;

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

            // ---------------------------------------------------------------
            // Reflection panel
            // ---------------------------------------------------------------
            string json = shader.ReflectionJson;
            if (string.IsNullOrEmpty(json)) return;

            var info = ShaderReflectionInfo.Parse(json);
            if (info == null) return;

            EditorGUILayout.Space(6);
            _showReflection = EditorGUILayout.BeginFoldoutHeaderGroup(_showReflection, "Shader Reflection");
            if (_showReflection)
            {
                EditorGUI.indentLevel++;

                // Thread group size
                EditorGUILayout.LabelField("numthreads",
                    $"[{info.NumThreadsX}, {info.NumThreadsY}, {info.NumThreadsZ}]",
                    EditorStyles.boldLabel);

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
            EditorGUILayout.EndFoldoutHeaderGroup();
        }

        private static void PrintReflectionToConsole(string assetPath, ShaderReflectionInfo info)
        {
            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"=== Shader Reflection: {System.IO.Path.GetFileName(assetPath)} ===");
            sb.AppendLine($"numthreads  [{info.NumThreadsX}, {info.NumThreadsY}, {info.NumThreadsZ}]");
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
        // Minimal JSON parser for our fixed reflection JSON shape.
        // Avoids a dependency on System.Text.Json / Newtonsoft.
        // -------------------------------------------------------------------
        private class ShaderBindingEntry
        {
            public string Name;
            public string Type;
            public int    Space;
            public int    Reg;
            public string Dim;     // e.g. "Texture2D", "Buffer", "ByteAddressBuffer"
            public string RetType; // e.g. "float", "uint", ""

            /// <summary>Human-readable HLSL-like type string, e.g. "Texture2D&lt;float4&gt;" or "StructuredBuffer".</summary>
            public string HlslType
            {
                get
                {
                    if (Type == "CBV")     return "ConstantBuffer";
                    if (Type == "Sampler") return "SamplerState";
                    if (Type == "TLAS")   return "RaytracingAccelerationStructure";

                    string prefix = Type == "UAV" ? "RW" : "";
                    string dim    = string.IsNullOrEmpty(Dim) ? "Buffer" : Dim;

                    // ByteAddressBuffer has no element type
                    if (dim == "ByteAddressBuffer")
                        return prefix + "ByteAddressBuffer";

                    // Buffers without a return type are StructuredBuffers
                    if (string.IsNullOrEmpty(RetType))
                        return prefix + (dim == "Buffer" ? "StructuredBuffer" : dim);

                    return prefix + dim + "<" + RetType + "4>";
                }
            }
        }

        private class ShaderReflectionInfo
        {
            public int NumThreadsX, NumThreadsY, NumThreadsZ;
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

                    // numthreads: [X, Y, Z]
                    int ntIdx = json.IndexOf("\"numthreads\"", StringComparison.Ordinal);
                    if (ntIdx >= 0)
                    {
                        int lb = json.IndexOf('[', ntIdx);
                        int rb = json.IndexOf(']', lb);
                        if (lb >= 0 && rb > lb)
                        {
                            var parts = json.Substring(lb + 1, rb - lb - 1).Split(',');
                            if (parts.Length >= 3)
                            {
                                int.TryParse(parts[0].Trim(), out result.NumThreadsX);
                                int.TryParse(parts[1].Trim(), out result.NumThreadsY);
                                int.TryParse(parts[2].Trim(), out result.NumThreadsZ);
                            }
                        }
                    }

                    // bindings array — iterate over { ... } objects
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
}
