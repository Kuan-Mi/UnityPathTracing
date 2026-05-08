using System;
using System.Collections.Generic;
using System.Reflection;
using NativeRender;
using Rtxdi.ReGIR;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

namespace PathTracing
{
    [CustomEditor(typeof(NativeRtxdiFeature))]
    public class NativeRtxdiFeatureEditor : Editor
    {
        private string GetKey(string headerName) => $"PT_NativeRtxdi_Foldout_{target.GetInstanceID()}_{headerName}";

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            var feature = (NativeRtxdiFeature)target;

            // Settings (use default property field — RtxdiSetting has its own property drawers).
            var settingsProp = serializedObject.FindProperty("setting");
            if (settingsProp != null)
                EditorGUILayout.PropertyField(settingsProp, includeChildren: true);

            EditorGUILayout.Space(4);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("renderPassEvent"));

            EditorGUILayout.Space(4);
            DrawGroupedAssetFields();

            EditorGUILayout.Space(8);
            if (GUILayout.Button("Auto Fill Shaders & Materials"))
            {
                Undo.RecordObject(feature, "Auto Fill Shaders");
                feature.AutoFillShaders();
            }

            if (GUILayout.Button("InitializeBuffers"))
            {
                feature.InitializeBuffers();
            }

            EditorGUILayout.Space(4);

            if (GUILayout.Button("TestPrepareLight — Readback LightDataBuffer"))
            {
                feature.TestPrepareLight();
            }

            if (GUILayout.Button("Dump resamplingConstants"))
            {
                var skip = new HashSet<string> { "view", "prevView", "prevPrevView" };
                var sb   = new System.Text.StringBuilder();
                sb.AppendLine("[0]\t{ g_Const={...} }\t");
                DumpStruct(sb, "g_Const", feature.resamplingConstants, typeof(NativeResamplingConstants), 1, skip);
                Debug.Log(sb.ToString());
            }

            if (GUILayout.Button("Dump compositingConstants"))
            {
                var skip = new HashSet<string> { "view", "prevView", "prevPrevView" };
                var sb   = new System.Text.StringBuilder();
                sb.AppendLine("[0]\t{ g_Const={...} }\t");
                DumpStruct(sb, "g_Const", feature.compositingConstants, typeof(NativeCompositingConstants), 1, skip);
                Debug.Log(sb.ToString());
            }

            EditorGUILayout.HelpBox(
                "Reads back LightDataBuffer from GPU and logs every non-black PolymorphicLightInfo " +
                "entry to the Console + draws normals in the Scene view. " +
                "Run the scene for a few frames before clicking.",
                MessageType.Info);

            EditorGUILayout.Space(8);
            var globalConstProp = serializedObject.FindProperty("globalConstants");
            if (globalConstProp != null)
                EditorGUILayout.PropertyField(globalConstProp, includeChildren: true);
            var resamplingConstProp = serializedObject.FindProperty("resamplingConstants");
            if (resamplingConstProp != null)
                EditorGUILayout.PropertyField(resamplingConstProp, includeChildren: true);

            serializedObject.ApplyModifiedProperties();
        }

        /// <summary>
        /// Reflection-driven asset grouping (mirrors RtxdiFeatureEditor) with NativeComputeShader added.
        /// </summary>
        private void DrawGroupedAssetFields()
        {
            var skip = new HashSet<string>
            {
                "globalConstants", "resamplingConstants", "renderPassEvent", "setting"
            };

            var groupLabels = new Dictionary<Type, string>
            {
                { typeof(Material), "Materials" },
                { typeof(NativeComputeShader), "Native Compute Shaders" },
                { typeof(ComputeShader), "Compute Shaders" },
                { typeof(Texture), "Textures" },
                { typeof(Texture2D), "Textures" },
                { typeof(Texture3D), "Textures" },
                { typeof(RenderTexture), "Textures" },
                { typeof(Cubemap), "Textures" },
            };

            var groups = new Dictionary<string, List<string>>();

            FieldInfo[] fields = typeof(NativeRtxdiFeature)
                .GetFields(BindingFlags.Public | BindingFlags.Instance);

            foreach (var field in fields)
            {
                if (skip.Contains(field.Name)) continue;

                string groupName = null;
                foreach (var kv in groupLabels)
                {
                    if (kv.Key.IsAssignableFrom(field.FieldType))
                    {
                        groupName = kv.Value;
                        break;
                    }
                }

                if (groupName == null) groupName = "Other";

                if (!groups.ContainsKey(groupName))
                    groups[groupName] = new List<string>();
                groups[groupName].Add(field.Name);
            }

            var order = new[] { "Materials", "Native Compute Shaders", "Compute Shaders", "Textures", "Other" };
            foreach (var groupName in order)
            {
                if (!groups.TryGetValue(groupName, out var fieldNames) || fieldNames.Count == 0)
                    continue;

                string foldoutKey  = GetKey("AssetGroup_" + groupName);
                bool   isExpanded  = SessionState.GetBool(foldoutKey, true);
                bool   newExpanded = EditorGUILayout.BeginFoldoutHeaderGroup(isExpanded, groupName);
                if (newExpanded != isExpanded)
                    SessionState.SetBool(foldoutKey, newExpanded);

                if (newExpanded)
                {
                    EditorGUI.indentLevel++;
                    foreach (var name in fieldNames)
                    {
                        SerializedProperty prop = serializedObject.FindProperty(name);
                        if (prop != null)
                            EditorGUILayout.PropertyField(prop);
                    }

                    EditorGUI.indentLevel--;
                }

                EditorGUILayout.EndFoldoutHeaderGroup();
            }
        }

        // -----------------------------------------------------------------------
        // Reflection-based struct dumper — produces the same tree format as 01.txt
        // -----------------------------------------------------------------------

        private static string Indent(int depth) => new string(' ', depth * 4);

        /// <summary>
        /// Appends a single field (or array element) to <paramref name="sb"/> in the
        /// Visual Studio watch / PIX style used in 01.txt.
        /// </summary>
        private static void DumpValue(System.Text.StringBuilder sb, string name, object value, System.Type type, int depth,
            System.Collections.Generic.HashSet<string> skipFields = null)
        {
            string indent   = Indent(depth);
            string typeName = GetShaderTypeName(type);

            if (value == null)
            {
                sb.AppendLine($"{indent}{name}\tnull\t{typeName}");
                return;
            }

            if (IsPrimitive(type))
            {
                sb.AppendLine($"{indent}{name}\t{FormatPrimitive(value)}\t{typeName}");
                return;
            }

            if (type.IsArray)
            {
                var    arr     = (System.Array)value;
                string summary = arr.Length > 0 ? $"[{arr.Length}]" : "[]";
                sb.AppendLine($"{indent}{name}\t{summary}\t{typeName}");
                for (int i = 0; i < arr.Length; i++)
                    DumpValue(sb, $"[{i}]", arr.GetValue(i), type.GetElementType(), depth + 1);
                return;
            }

            // Special case: ReGIR_OnionParameters — fixed byte[] fields need accessor methods
            if (type == typeof(ReGIR_OnionParameters))
            {
                DumpOnionParameters(sb, name, (ReGIR_OnionParameters)value, depth);
                return;
            }

            // Struct / class — build summary like { f1=val, f2=val, ... }
            var    fields   = type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            string summary2 = BuildSummary(value, fields);
            sb.AppendLine($"{indent}{name}\t{summary2}\t{typeName}");
            bool isVector  = IsVectorType(type);
            int  compIndex = 0;
            foreach (var f in fields)
            {
                if (skipFields != null && skipFields.Contains(f.Name)) continue;
                object child     = f.GetValue(value);
                string childName = isVector ? $"C{compIndex++}" : f.Name;
                DumpValue(sb, childName, child, f.FieldType, depth + 1);
            }
        }

        private static void DumpStruct(System.Text.StringBuilder sb, string name, object value, System.Type type, int depth,
            System.Collections.Generic.HashSet<string> skipFields = null)
        {
            DumpValue(sb, name, value, type, depth, skipFields);
        }

        private static void DumpOnionParameters(System.Text.StringBuilder sb, string name,
            ReGIR_OnionParameters onion, int depth)
        {
            string indent    = Indent(depth);
            int    maxLayers = ReGIRConstants.RTXDI_ONION_MAX_LAYER_GROUPS;
            int    maxRings  = ReGIRConstants.RTXDI_ONION_MAX_RINGS;

            // Top-level summary — mimic PIX: show layers/rings as {...}, then scalar fields
            string summary = $"{{ layers={{...}}, rings={{...}}, numLayerGroups={onion.numLayerGroups}," +
                             $" cubicRootFactor={FormatPrimitive(onion.cubicRootFactor)}," +
                             $" linearFactor={FormatPrimitive(onion.linearFactor)}, pad1={FormatPrimitive(onion.pad1)} }}";
            sb.AppendLine($"{indent}{name}\t{summary}\tReGIR_OnionParameters");

            string indent2 = Indent(depth + 2);

            // --- layers ---
            var layer0 = onion.GetLayer(0);
            string layerSummary0 = BuildSummary(layer0,
                typeof(ReGIR_OnionLayerGroup).GetFields(BindingFlags.Public | BindingFlags.Instance));
            string arrayHeader = $"{{ {layerSummary0}, ... }}";
            sb.AppendLine($"{Indent(depth + 1)}layers\t{arrayHeader}\tReGIR_OnionLayerGroup[{maxLayers}]");
            for (int i = 0; i < maxLayers; i++)
            {
                var layer = onion.GetLayer(i);
                string lsummary = BuildSummary(layer,
                    typeof(ReGIR_OnionLayerGroup).GetFields(BindingFlags.Public | BindingFlags.Instance));
                sb.AppendLine($"{indent2}[{i}]\t{lsummary}\tReGIR_OnionLayerGroup");
                DumpValue(sb, "innerRadius", layer.innerRadius, typeof(float), depth + 3);
                DumpValue(sb, "outerRadius", layer.outerRadius, typeof(float), depth + 3);
                DumpValue(sb, "invLogLayerScale", layer.invLogLayerScale, typeof(float), depth + 3);
                DumpValue(sb, "layerCount", layer.layerCount, typeof(int), depth + 3);
                DumpValue(sb, "invEquatorialCellAngle", layer.invEquatorialCellAngle, typeof(float), depth + 3);
                DumpValue(sb, "cellsPerLayer", layer.cellsPerLayer, typeof(int), depth + 3);
                DumpValue(sb, "ringOffset", layer.ringOffset, typeof(int), depth + 3);
                DumpValue(sb, "ringCount", layer.ringCount, typeof(int), depth + 3);
                DumpValue(sb, "equatorialCellAngle", layer.equatorialCellAngle, typeof(float), depth + 3);
                DumpValue(sb, "layerScale", layer.layerScale, typeof(float), depth + 3);
                DumpValue(sb, "layerCellOffset", layer.layerCellOffset, typeof(int), depth + 3);
                DumpValue(sb, "pad1", layer.pad1, typeof(int), depth + 3);
            }

            // --- rings ---
            var ring0 = onion.GetRing(0);
            string ring0summary = BuildSummary(ring0,
                typeof(ReGIR_OnionRing).GetFields(BindingFlags.Public | BindingFlags.Instance));
            sb.AppendLine($"{Indent(depth + 1)}rings\t{{ {ring0summary}, ... }}\tReGIR_OnionRing[{maxRings}]");
            for (int i = 0; i < maxRings; i++)
            {
                var ring = onion.GetRing(i);
                string rsummary = BuildSummary(ring,
                    typeof(ReGIR_OnionRing).GetFields(BindingFlags.Public | BindingFlags.Instance));
                sb.AppendLine($"{indent2}[{i}]\t{rsummary}\tReGIR_OnionRing");
                DumpValue(sb, "cellAngle", ring.cellAngle, typeof(float), depth + 3);
                DumpValue(sb, "invCellAngle", ring.invCellAngle, typeof(float), depth + 3);
                DumpValue(sb, "cellOffset", ring.cellOffset, typeof(int), depth + 3);
                DumpValue(sb, "cellCount", ring.cellCount, typeof(int), depth + 3);
            }

            // --- scalar fields ---
            DumpValue(sb, "numLayerGroups", onion.numLayerGroups, typeof(uint), depth + 1);
            DumpValue(sb, "cubicRootFactor", onion.cubicRootFactor, typeof(float), depth + 1);
            DumpValue(sb, "linearFactor", onion.linearFactor, typeof(float), depth + 1);
            DumpValue(sb, "pad1", onion.pad1, typeof(float), depth + 1);
        }

        private static bool IsVectorType(System.Type t)
        {
            // Unity.Mathematics vector types: float2/3/4, uint2/3/4, int2/3/4, bool2/3/4
            string n = t.Name;
            return (n.StartsWith("float") || n.StartsWith("uint") || n.StartsWith("int") || n.StartsWith("bool"))
                   && n.Length > 0 && char.IsDigit(n[n.Length - 1]);
        }

        // PIX summary: add fields while current length + '... }'
        private const int SummaryCharLimit = 135;

        private static string BuildSummary(object value, FieldInfo[] fields)
        {
            if (fields.Length == 0) return "{}";
            var  parts  = new System.Text.StringBuilder("{ ");
            int  shown  = 0;
            bool vector = IsVectorType(value.GetType());
            foreach (var f in fields)
            {
                string fieldStr;
                object child = f.GetValue(value);
                if (IsPrimitive(f.FieldType))
                    fieldStr = vector ? $"{FormatPrimitive(child)}, " : $"{f.Name}={FormatPrimitive(child)}, ";
                else if (IsVectorType(f.FieldType))
                {
                    var vfields = f.FieldType.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                    var vparts  = new System.Text.StringBuilder("{ ");
                    foreach (var vf in vfields) vparts.Append($"{FormatPrimitive(vf.GetValue(child))}, ");
                    int vlen = vparts.Length;
                    if (vlen > 2 && vparts[vlen - 2] == ',') vparts.Remove(vlen - 2, 2);
                    vparts.Append(" }");
                    fieldStr = $"{f.Name}={vparts}, ";
                }
                else
                    fieldStr = $"{f.Name}={{...}}, ";

                // Check if adding this field + '... }' would exceed the limit
                bool isLast       = (shown + 1 == fields.Length);
                int  projectedLen = parts.Length + fieldStr.Length + (isLast ? 2 : 5); // ' }' or '... }'
                if (!isLast && projectedLen > SummaryCharLimit)
                {
                    parts.Append("...");
                    break;
                }

                parts.Append(fieldStr);
                shown++;
                if (isLast) break;
            }

            // trim trailing ", "
            int len = parts.Length;
            if (len > 2 && parts[len - 2] == ',') parts.Remove(len - 2, 2);
            parts.Append(" }");
            return parts.ToString();
        }

        private static bool IsPrimitive(System.Type t) =>
            t.IsPrimitive || t == typeof(float) || t == typeof(double) ||
            t == typeof(int) || t == typeof(uint) || t == typeof(bool) ||
            t == typeof(long) || t == typeof(ulong) || t == typeof(short) || t == typeof(ushort) ||
            t == typeof(byte) || t == typeof(sbyte) ||
            t.IsEnum;

        private static string FormatPrimitive(object v)
        {
            if (v is float f) return f.ToString("G6");
            if (v is double d) return d.ToString("G6");
            if (v != null && v.GetType().IsEnum)
                return System.Convert.ToUInt32(v).ToString();
            return v?.ToString() ?? "null";
        }

        private static string GetShaderTypeName(System.Type t)
        {
            if (t == typeof(float)) return "float";
            if (t == typeof(int)) return "int";
            if (t == typeof(uint)) return "uint";
            if (t == typeof(bool)) return "bool";
            if (t.IsEnum) return "uint";
            // Unity.Mathematics types
            string n = t.Name;
            if (n.StartsWith("float") || n.StartsWith("int") || n.StartsWith("uint") || n.StartsWith("bool"))
                return n;
            // Strip "Native" prefix to match HLSL/PIX type names
            if (n.StartsWith("Native"))
                return n.Substring(6);
            return n;
        }
    }
}