using System;
using System.Collections.Generic;
using System.Text;
using UnityEditor;
using UnityEditor.AssetImporters;
using UnityEngine;

namespace NativeRender
{
    // =======================================================================
    // Shared reflection model + IMGUI helpers for the native-shader importer
    // inspectors (.computeshader / .rayshader / .hitgroupshader).
    //
    // All three importers compile HLSL to DXIL and emit the same reflection
    // JSON shape, so the parser, the binding drawers, the compile button and
    // the "Print to Console" logic live here once instead of being copied
    // into each editor.
    // =======================================================================

    /// <summary>
    /// Writes a <see cref="SamplerBinding"/>[] into the shader asset's serialized <c>_samplerBindings</c>
    /// property (HLSL name + a reference to a shared <see cref="NativeSampler"/> asset). Shared by all
    /// three shader importers so the field mapping lives in one place.
    /// </summary>
    internal static class SamplerBindingSerialization
    {
        public static void Write(SerializedProperty prop, SamplerBinding[] bindings)
        {
            if (prop == null) return;
            int count = bindings?.Length ?? 0;
            prop.arraySize = count;
            for (int i = 0; i < count; i++)
            {
                var elem = prop.GetArrayElementAtIndex(i);
                elem.FindPropertyRelative("Name").stringValue            = bindings[i].Name ?? "";
                elem.FindPropertyRelative("Sampler").objectReferenceValue = bindings[i].Sampler;
            }
        }
    }

    /// <summary>A single bound resource discovered by DXIL reflection.</summary>
    internal sealed class ShaderBindingEntry
    {
        public string Name;
        public string Type;     // SRV / UAV / CBV / Sampler / TLAS
        public int    Space;
        public int    Reg;
        public int    Size;     // CBV byte size (0 when not reflected / not a CBV)
        public string Dim;      // e.g. "Texture2D", "Buffer", "ByteAddressBuffer"
        public string RetType;  // e.g. "float", "uint", ""

        /// <summary>Human-readable HLSL-like type, e.g. "Texture2D&lt;float4&gt;" or "StructuredBuffer".</summary>
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

    /// <summary>A shader entry point (raygen / miss / closesthit / ...) discovered in a DXIL library.</summary>
    internal sealed class ShaderEntry
    {
        public string Name;
        public string Kind;
    }

    /// <summary>
    /// Parsed reflection data shared by every native-shader importer. Compute shaders populate
    /// <see cref="HasNumThreads"/> + the thread-group size; ray / hit-group libraries populate
    /// <see cref="Shaders"/>. Resource bindings are common to all.
    /// </summary>
    internal sealed class ShaderReflectionInfo
    {
        public bool HasNumThreads;
        public int  NumThreadsX, NumThreadsY, NumThreadsZ;

        public readonly List<ShaderEntry>        Shaders = new();
        public readonly List<ShaderBindingEntry> SRV     = new();
        public readonly List<ShaderBindingEntry> UAV     = new();
        public readonly List<ShaderBindingEntry> CBV     = new();
        public readonly List<ShaderBindingEntry> Sampler = new();
        public readonly List<ShaderBindingEntry> TLAS    = new();

        public static ShaderReflectionInfo Parse(string json)
        {
            if (string.IsNullOrEmpty(json)) return null;
            try
            {
                var result = new ShaderReflectionInfo();

                // numthreads: [X, Y, Z]  (compute only)
                string numThreads = SliceArrayBody(json, "numthreads");
                if (numThreads != null)
                {
                    var parts = numThreads.Split(',');
                    if (parts.Length >= 3)
                    {
                        int.TryParse(parts[0].Trim(), out result.NumThreadsX);
                        int.TryParse(parts[1].Trim(), out result.NumThreadsY);
                        int.TryParse(parts[2].Trim(), out result.NumThreadsZ);
                        result.HasNumThreads = true;
                    }
                }

                // shaders: [ { name, kind }, ... ]  (ray / hit-group libraries)
                string shadersBody = SliceArrayBody(json, "shaders");
                if (shadersBody != null)
                {
                    foreach (var obj in Objects(shadersBody))
                        result.Shaders.Add(new ShaderEntry
                        {
                            Name = ExtractString(obj, "name"),
                            Kind = ExtractString(obj, "kind"),
                        });
                }

                // bindings: [ { name, type, space, reg, dim, retType }, ... ]
                string bindingsBody = SliceArrayBody(json, "bindings");
                if (bindingsBody != null)
                {
                    foreach (var obj in Objects(bindingsBody))
                    {
                        var entry = new ShaderBindingEntry
                        {
                            Name    = ExtractString(obj, "name"),
                            Type    = ExtractString(obj, "type"),
                            Space   = ExtractInt   (obj, "space"),
                            Reg     = ExtractInt   (obj, "reg"),
                            Size    = ExtractInt   (obj, "size"),
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
                    }
                }

                return result;
            }
            catch
            {
                return null;
            }
        }

        // ── tiny JSON helpers (avoids a Newtonsoft / System.Text.Json dependency) ──

        /// <summary>Returns the text between the matching brackets of the array named <paramref name="key"/>, or null.</summary>
        private static string SliceArrayBody(string json, string key)
        {
            int ki = json.IndexOf("\"" + key + "\"", StringComparison.Ordinal);
            if (ki < 0) return null;
            int open = json.IndexOf('[', ki);
            if (open < 0) return null;

            int depth = 0;
            for (int i = open; i < json.Length; i++)
            {
                if (json[i] == '[') depth++;
                else if (json[i] == ']' && --depth == 0)
                    return json.Substring(open + 1, i - open - 1);
            }
            return null;
        }

        /// <summary>Yields the body of each top-level <c>{ ... }</c> object inside an array body.</summary>
        private static IEnumerable<string> Objects(string arrayBody)
        {
            int pos = 0;
            while (pos < arrayBody.Length)
            {
                int o1 = arrayBody.IndexOf('{', pos);
                if (o1 < 0) yield break;

                int depth = 0, o2 = -1;
                for (int i = o1; i < arrayBody.Length; i++)
                {
                    if (arrayBody[i] == '{') depth++;
                    else if (arrayBody[i] == '}' && --depth == 0) { o2 = i; break; }
                }
                if (o2 < 0) yield break;

                yield return arrayBody.Substring(o1 + 1, o2 - o1 - 1);
                pos = o2 + 1;
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

    /// <summary>Shared IMGUI building blocks for the native-shader importer inspectors.</summary>
    internal static class ShaderImporterGUI
    {
        private static readonly Color CompiledColor = new(0.4f, 0.8f, 0.4f);
        private static readonly Color StaleColor    = new(1f,   0.6f, 0.3f);

        /// <summary>Session-persisted foldout (survives selection changes, unlike a plain field).</summary>
        public static bool Foldout(string key, string label, bool defaultOpen, bool header = false)
        {
            bool cur = SessionState.GetBool(key, defaultOpen);
            bool now = EditorGUILayout.Foldout(cur, label, true,
                header ? EditorStyles.foldoutHeader : EditorStyles.foldout);
            if (now != cur) SessionState.SetBool(key, now);
            return now;
        }

        /// <summary>Coloured Compile / Recompile button. Returns true on click.</summary>
        public static bool DrawCompileButton(bool hasCompiledBytes)
        {
            EditorGUILayout.Space(6);
            GUI.backgroundColor = hasCompiledBytes ? CompiledColor : StaleColor;
            bool clicked = GUILayout.Button(hasCompiledBytes ? "Recompile" : "Compile", GUILayout.Height(28));
            GUI.backgroundColor = Color.white;
            return clicked;
        }

        public static void DrawCompileStatus(bool hasCompiledBytes, int byteCount)
        {
            EditorGUILayout.Space(2);
            if (hasCompiledBytes)
                EditorGUILayout.HelpBox($"DXIL cached ({byteCount:N0} bytes)", MessageType.Info);
            else
                EditorGUILayout.HelpBox("No compiled DXIL – click Compile to build.", MessageType.Warning);
        }

        /// <summary>
        /// Reimports the asset on the next editor tick. <see cref="AssetImporter.SaveAndReimport"/>
        /// rebuilds the inspector's UIElements hierarchy, which throws "Cannot modify VisualElement
        /// hierarchy during layout calculation" if called synchronously from OnInspectorGUI — so we
        /// always defer it out of the current GUI pass.
        /// </summary>
        public static void ScheduleReimport(ScriptedImporter importer)
        {
            if (importer == null) return;
            EditorApplication.delayCall += importer.SaveAndReimport;
        }

        /// <summary>
        /// Draws the collapsible reflection panel: thread-group size / entry points / resource
        /// bindings, and a "Print to Console" button.
        ///
        /// <paramref name="drawRow"/> lets a subclass render an interactive row for a binding
        /// (e.g. a "promote to root binding" checkbox); return true if it handled the row, false
        /// to fall back to the default read-only label. <paramref name="drawExtra"/> is an optional
        /// trailing block (e.g. cleanup of hints whose binding no longer exists).
        /// </summary>
        public static void DrawReflectionPanel(string keyPrefix, string assetPath, ShaderReflectionInfo info,
                                               Func<ShaderBindingEntry, bool> drawRow = null, Action drawExtra = null)
        {
            EditorGUILayout.Space(6);
            if (!Foldout(keyPrefix + ".reflection", "Shader Reflection", true, header: true))
                return;

            EditorGUI.indentLevel++;

            if (info.HasNumThreads)
                EditorGUILayout.LabelField("numthreads",
                    $"[{info.NumThreadsX}, {info.NumThreadsY}, {info.NumThreadsZ}]",
                    EditorStyles.boldLabel);

            DrawShaderGroup (keyPrefix + ".shaders", info.Shaders);
            DrawBindingGroup(keyPrefix + ".srv",     "SRV",     info.SRV,     drawRow);
            DrawBindingGroup(keyPrefix + ".uav",     "UAV",     info.UAV,     drawRow);
            DrawBindingGroup(keyPrefix + ".cbv",     "CBV",     info.CBV,     drawRow);
            DrawBindingGroup(keyPrefix + ".sampler", "Sampler", info.Sampler, drawRow);
            DrawBindingGroup(keyPrefix + ".tlas",    "TLAS",    info.TLAS,    drawRow);

            drawExtra?.Invoke();

            EditorGUILayout.Space(4);
            if (GUILayout.Button("Print to Console"))
                PrintToConsole(assetPath, info);

            EditorGUI.indentLevel--;
        }

        private static void DrawShaderGroup(string key, List<ShaderEntry> entries)
        {
            if (entries.Count == 0) return;
            if (!Foldout(key, $"Shaders  ({entries.Count})", true)) return;

            EditorGUI.indentLevel++;
            foreach (var s in entries)
                EditorGUILayout.LabelField(s.Name, s.Kind);
            EditorGUI.indentLevel--;
        }

        /// <summary>Read-only label for a single binding: "Name    HlslType    spaceN:xN".</summary>
        public static void DrawBindingLabel(string label, ShaderBindingEntry e)
            => EditorGUILayout.LabelField(e.Name, $"{e.HlslType}    space{e.Space}:{ResourcePrefix(label)}{e.Reg}");

        private static void DrawBindingGroup(string key, string label, List<ShaderBindingEntry> entries,
                                             Func<ShaderBindingEntry, bool> drawRow)
        {
            if (entries.Count == 0) return;
            if (!Foldout(key, $"{label}  ({entries.Count})", true)) return;

            EditorGUI.indentLevel++;
            foreach (var e in entries)
                if (drawRow == null || !drawRow(e))
                    DrawBindingLabel(label, e);
            EditorGUI.indentLevel--;
        }

        public static string ResourcePrefix(string type) => type switch
        {
            "SRV"     => "t",
            "UAV"     => "u",
            "CBV"     => "b",
            "Sampler" => "s",
            "TLAS"    => "t",
            _         => ""
        };

        private static void PrintToConsole(string assetPath, ShaderReflectionInfo info)
        {
            var sb = new StringBuilder();
            sb.AppendLine($"=== Shader Reflection: {System.IO.Path.GetFileName(assetPath)} ===");

            if (info.HasNumThreads)
                sb.AppendLine($"numthreads  [{info.NumThreadsX}, {info.NumThreadsY}, {info.NumThreadsZ}]");
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
    }

    /// <summary>
    /// Shared "promote a binding to a root binding" UI for the native-shader importers.
    /// Operates on any <see cref="ScriptedImporter"/> exposing public <c>rootConstantsHints</c>
    /// (RootConstantsHint[]) and <c>rootSRVHints</c> (string[]) fields, entirely via
    /// <see cref="SerializedObject"/> — so it is importer-type-agnostic and shared by the
    /// compute and ray-tracing shader inspectors.
    ///
    /// The root 32-bit-constant count is derived from the reflected constant-buffer byte size
    /// (<see cref="ShaderBindingEntry.Size"/>); it falls back to a manual field only when the
    /// size is unavailable (e.g. before the reflection plugin DLL is rebuilt).
    /// </summary>
    internal static class RootBindingHintsGUI
    {
        /// <summary>Inline checkbox row for a promotable binding; returns true if it drew the row.</summary>
        public static bool TryDrawBindingRow(ScriptedImporter importer, ShaderBindingEntry e)
        {
            if (e.Type == "CBV")
                return DrawRootConstantRow(importer, e);
            if (e.Type == "TLAS" || (e.Type == "SRV" && IsBufferLike(e)))
                return DrawRootSrvRow(importer, e);
            if (e.Type == "Sampler")
                return DrawSamplerRow(importer, e);
            return false; // textures, UAVs → default read-only label
        }

        /// <summary>Trailing cleanup block for hints whose binding is no longer in the reflection.</summary>
        public static void DrawStaleHints(ScriptedImporter importer, ShaderReflectionInfo info)
        {
            var reflected = new HashSet<string>();
            foreach (var c in info.CBV)     reflected.Add(c.Name);
            foreach (var s in info.SRV)     reflected.Add(s.Name);
            foreach (var t in info.TLAS)    reflected.Add(t.Name);
            foreach (var s in info.Sampler) reflected.Add(s.Name);

            var so = new SerializedObject(importer);
            var rc = so.FindProperty("rootConstantsHints");
            var sv = so.FindProperty("rootSRVHints");
            var sh = so.FindProperty("samplerBindings");

            var staleConstants = new List<string>();
            if (rc != null)
                for (int i = 0; i < rc.arraySize; i++)
                {
                    string n = rc.GetArrayElementAtIndex(i).FindPropertyRelative("Name").stringValue;
                    if (!reflected.Contains(n)) staleConstants.Add(n);
                }

            var staleSrv = new List<string>();
            if (sv != null)
                for (int i = 0; i < sv.arraySize; i++)
                {
                    string n = sv.GetArrayElementAtIndex(i).stringValue;
                    if (!reflected.Contains(n)) staleSrv.Add(n);
                }

            var staleSamplers = new List<string>();
            if (sh != null)
                for (int i = 0; i < sh.arraySize; i++)
                {
                    string n = sh.GetArrayElementAtIndex(i).FindPropertyRelative("Name").stringValue;
                    if (!reflected.Contains(n)) staleSamplers.Add(n);
                }

            if (staleConstants.Count == 0 && staleSrv.Count == 0 && staleSamplers.Count == 0) return;

            EditorGUILayout.Space(4);
            EditorGUILayout.LabelField("Stale hints (binding not in current reflection)", EditorStyles.miniBoldLabel);
            EditorGUI.indentLevel++;
            foreach (var name in staleConstants)
                if (!EditorGUILayout.ToggleLeft($"{name}    (root constants)", true))
                    RemoveHint(importer, "rootConstantsHints", name, isStruct: true);
            foreach (var name in staleSrv)
                if (!EditorGUILayout.ToggleLeft($"{name}    (root SRV)", true))
                    RemoveHint(importer, "rootSRVHints", name, isStruct: false);
            foreach (var name in staleSamplers)
                if (!EditorGUILayout.ToggleLeft($"{name}    (sampler reference)", true))
                    RemoveHint(importer, "samplerBindings", name, isStruct: true);
            EditorGUI.indentLevel--;
        }

        // Only buffer SRVs / TLAS can become inline root descriptors — textures need a descriptor table.
        private static bool IsBufferLike(ShaderBindingEntry e)
            => string.IsNullOrEmpty(e.Dim) || !e.Dim.StartsWith("Texture", StringComparison.Ordinal);

        private static bool DrawRootConstantRow(ScriptedImporter importer, ShaderBindingEntry e)
        {
            var so   = new SerializedObject(importer);
            var prop = so.FindProperty("rootConstantsHints");
            if (prop == null) return false;

            int  idx      = FindHintIndex(prop, e.Name);
            bool isOn     = idx >= 0;
            int  curCount = isOn ? prop.GetArrayElementAtIndex(idx).FindPropertyRelative("Count").intValue : 0;

            // Derived from the reflected constant-buffer byte size (0 when unavailable).
            int  derivedCount = (e.Size + 3) / 4;
            bool autoCount    = derivedCount > 0;

            string suffix = autoCount ? $"    {derivedCount} × 32-bit" : "";
            var label = new GUIContent(
                $"{e.Name}    ConstantBuffer  space{e.Space}:b{e.Reg}{suffix}",
                "Promote to inline root 32-bit constants (Root32BitConstants) — no GPU buffer required.");

            bool newOn = EditorGUILayout.ToggleLeft(label, isOn);
            if (newOn != isOn)
            {
                if (newOn)
                {
                    int i = prop.arraySize;
                    prop.InsertArrayElementAtIndex(i);
                    var el = prop.GetArrayElementAtIndex(i);
                    el.FindPropertyRelative("Name").stringValue = e.Name;
                    el.FindPropertyRelative("Count").intValue   = Mathf.Max(0, derivedCount);
                }
                else
                {
                    prop.DeleteArrayElementAtIndex(idx);
                }
                so.ApplyModifiedProperties();
                ShaderImporterGUI.ScheduleReimport(importer);
                return true;
            }

            if (newOn)
            {
                if (autoCount)
                {
                    // Keep the stored count in sync with reflection (handles the buffer size
                    // changing after a shader edit). Converges in one reimport, then matches.
                    if (curCount != derivedCount)
                    {
                        prop.GetArrayElementAtIndex(idx).FindPropertyRelative("Count").intValue = derivedCount;
                        so.ApplyModifiedProperties();
                        ShaderImporterGUI.ScheduleReimport(importer);
                    }
                }
                else
                {
                    EditorGUI.indentLevel++;
                    int newCount = EditorGUILayout.DelayedIntField("32-bit value count", curCount);
                    EditorGUI.indentLevel--;
                    if (newCount != curCount)
                    {
                        prop.GetArrayElementAtIndex(idx).FindPropertyRelative("Count").intValue = Mathf.Max(0, newCount);
                        so.ApplyModifiedProperties();
                        ShaderImporterGUI.ScheduleReimport(importer);
                    }
                }
            }
            return true;
        }

        private static bool DrawRootSrvRow(ScriptedImporter importer, ShaderBindingEntry e)
        {
            var so   = new SerializedObject(importer);
            var prop = so.FindProperty("rootSRVHints");
            if (prop == null) return false;

            int  idx  = FindStringIndex(prop, e.Name);
            bool isOn = idx >= 0;

            string typeName = e.Type == "TLAS" ? "RaytracingAccelerationStructure" : e.HlslType;
            var label = new GUIContent(
                $"{e.Name}    {typeName}  space{e.Space}:t{e.Reg}",
                "Promote to an inline root descriptor (Root SRV) instead of a descriptor-table entry.");

            bool newOn = EditorGUILayout.ToggleLeft(label, isOn);
            if (newOn != isOn)
            {
                if (newOn)
                {
                    int i = prop.arraySize;
                    prop.InsertArrayElementAtIndex(i);
                    prop.GetArrayElementAtIndex(i).stringValue = e.Name;
                }
                else
                {
                    prop.DeleteArrayElementAtIndex(idx);
                }
                so.ApplyModifiedProperties();
                ShaderImporterGUI.ScheduleReimport(importer);
            }
            return true;
        }

        // Sampler reference row: an object field on the reflected sampler binding that points at a
        // shared NativeSampler asset. The name comes from reflection, so there is nothing to type.
        // When left empty, the native plugin infers the attributes from the sampler name.
        private static bool DrawSamplerRow(ScriptedImporter importer, ShaderBindingEntry e)
        {
            var so   = new SerializedObject(importer);
            var prop = so.FindProperty("samplerBindings");
            if (prop == null) return false;

            int idx = FindHintIndex(prop, e.Name);
            var current = idx >= 0
                ? prop.GetArrayElementAtIndex(idx).FindPropertyRelative("Sampler").objectReferenceValue as NativeSampler
                : null;

            var label = new GUIContent(
                $"{e.Name}    SamplerState  space{e.Space}:s{e.Reg}",
                "Assign a shared NativeSampler asset (Create > Native Render > Sampler). Editing that "
                + "asset updates every shader that references it. When left empty, the native plugin "
                + "infers the attributes from the sampler name (e.g. sampler_LinearClamp).");

            var newSampler = (NativeSampler)EditorGUILayout.ObjectField(label, current, typeof(NativeSampler), false);
            if (newSampler == current) return true;

            if (newSampler != null)
            {
                if (idx < 0)
                {
                    idx = prop.arraySize;
                    prop.InsertArrayElementAtIndex(idx);
                    prop.GetArrayElementAtIndex(idx).FindPropertyRelative("Name").stringValue = e.Name;
                }
                prop.GetArrayElementAtIndex(idx).FindPropertyRelative("Sampler").objectReferenceValue = newSampler;
            }
            else if (idx >= 0)
            {
                // Element is a struct (not a bare object-reference array), so a single delete removes it.
                prop.DeleteArrayElementAtIndex(idx);
            }

            so.ApplyModifiedProperties();
            ShaderImporterGUI.ScheduleReimport(importer);
            return true;
        }

        private static void RemoveHint(ScriptedImporter importer, string propName, string name, bool isStruct)
        {
            var so = new SerializedObject(importer);
            var p  = so.FindProperty(propName);
            if (p == null) return;
            int idx = isStruct ? FindHintIndex(p, name) : FindStringIndex(p, name);
            if (idx < 0) return;
            p.DeleteArrayElementAtIndex(idx);
            so.ApplyModifiedProperties();
            ShaderImporterGUI.ScheduleReimport(importer);
        }

        private static int FindHintIndex(SerializedProperty arrayProp, string name)
        {
            for (int i = 0; i < arrayProp.arraySize; i++)
                if (arrayProp.GetArrayElementAtIndex(i).FindPropertyRelative("Name").stringValue == name)
                    return i;
            return -1;
        }

        private static int FindStringIndex(SerializedProperty arrayProp, string name)
        {
            for (int i = 0; i < arrayProp.arraySize; i++)
                if (arrayProp.GetArrayElementAtIndex(i).stringValue == name)
                    return i;
            return -1;
        }
    }

    /// <summary>
    /// Common inspector skeleton for the native-shader importers. Draws a collapsible
    /// "Compile Settings" section (subclass-defined), the standard Apply/Revert bar, a
    /// Compile button + status, and the shared reflection panel. Subclasses only supply
    /// their per-importer fields and a way to read the compiled-asset status.
    /// </summary>
    internal abstract class NativeShaderImporterEditorBase : ScriptedImporterEditor
    {
        /// <summary>Stable session key namespace for this editor type's foldouts.</summary>
        private string KeyPrefix => "NativeShader." + GetType().Name;

        /// <summary>Draw the importer's own serialized fields (excluding anything shown elsewhere).</summary>
        protected abstract void DrawImportSettings();

        /// <summary>
        /// Fetch the compiled-asset status for the given asset path. Return false if the asset
        /// could not be loaded (in which case the compile/reflection UI is skipped).
        /// </summary>
        protected abstract bool TryGetStatus(string assetPath,
            out bool hasCompiledBytes, out int byteCount, out string reflectionJson);

        /// <summary>
        /// Optionally render an interactive row for a reflected binding (e.g. a "promote to root
        /// binding" checkbox). Return true if the row was drawn; false to use the default label.
        /// </summary>
        protected virtual bool TryDrawBindingRow(ShaderBindingEntry entry) => false;

        /// <summary>Optional trailing UI inside the reflection panel (e.g. stale-hint cleanup).</summary>
        protected virtual void DrawExtraReflection(ShaderReflectionInfo info) { }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            if (ShaderImporterGUI.Foldout(KeyPrefix + ".settings", "Compile Settings", true, header: true))
            {
                EditorGUI.indentLevel++;
                DrawImportSettings();
                EditorGUI.indentLevel--;
            }

            serializedObject.ApplyModifiedProperties();
            ApplyRevertGUI();

            // Compile button + reflection only make sense for a single selected asset.
            if (targets.Length != 1) return;

            var importer = (ScriptedImporter)target;
            if (!TryGetStatus(importer.assetPath, out bool hasBytes, out int byteCount, out string json))
                return;

            if (ShaderImporterGUI.DrawCompileButton(hasBytes))
                ShaderImporterGUI.ScheduleReimport(importer);
            ShaderImporterGUI.DrawCompileStatus(hasBytes, byteCount);

            var info = ShaderReflectionInfo.Parse(json);
            if (info != null)
                ShaderImporterGUI.DrawReflectionPanel(KeyPrefix, importer.assetPath, info,
                    TryDrawBindingRow, () => DrawExtraReflection(info));
        }
    }
}
