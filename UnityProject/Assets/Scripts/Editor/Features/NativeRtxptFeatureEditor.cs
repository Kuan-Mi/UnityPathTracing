using System.Collections.Generic;
using System.Reflection;
using NativeRender;
using UnityEditor;
using UnityEngine;

namespace PathTracing
{
    [CustomEditor(typeof(NativeRtxptFeature))]
    public class NativeRtxptFeatureEditor : Editor
    {
        private string GetKey(string headerName) =>
            $"PT_NativeRtxpt_Foldout_{target.GetInstanceID()}_{headerName}";

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            var feature = (NativeRtxptFeature)target;

            // Setting
            var settingsProp = serializedObject.FindProperty("setting");
            if (settingsProp != null)
                EditorGUILayout.PropertyField(settingsProp, includeChildren: true);

            // Flush setting edits so the macro-sync check below sees the current values.
            serializedObject.ApplyModifiedProperties();

            EditorGUILayout.Space(4);
            DrawShaderMacroSync(feature);

            EditorGUILayout.Space(4);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("renderPassEvent"));

            EditorGUILayout.Space(4);
            DrawGroupedAssetFields();

            EditorGUILayout.Space(8);
            if (GUILayout.Button("Auto Fill Shaders"))
            {
                Undo.RecordObject(feature, "Auto Fill Shaders");
                feature.AutoFillShaders();
            }

            EditorGUILayout.Space(4);
            if (GUILayout.Button("Test Emissive Triangles — Readback LightBuffer"))
            {
                feature.TestEmissiveTriangles();
            }
            EditorGUILayout.HelpBox(
                "Reads back the emissive-triangle range of LightBuffer from GPU and prints center, " +
                "intensity, and type of the first non-zero entries to the Console. " +
                "Run the scene for a few frames before clicking.",
                MessageType.Info);

            EditorGUILayout.Space(4);
            if (GUILayout.Button("Test NEE-AT - Readback Buffers"))
            {
                feature.TestNeeAtReadback();
            }
            EditorGUILayout.HelpBox(
                "Reads back LightingControl, weights, proxy counters, proxy lists, local sampling, " +
                "and feedback textures. Set useNEE=true and neeType=NEEAT, then run for a few frames before clicking.",
                MessageType.Info);

            EditorGUILayout.Space(10);

            DrawObjectHelper.Draw(target.GetInstanceID(), "Sample Constants", feature.sampleConstants);


            serializedObject.ApplyModifiedProperties();
        }

        /// <summary>
        /// Settings ↔ shader-macro sync (mirrors Sample.cpp SetGlobalShaderMacros). Several
        /// NativeRtxptSetting fields are shader compile-time macros, not constants — they only
        /// take effect when baked into the .rayshader/.hitgroupshader importer defines and the
        /// shaders are reimported. Shows a warning while the importers are out of sync.
        /// </summary>
        private void DrawShaderMacroSync(NativeRtxptFeature feature)
        {
            var stale = NativeRtxptShaderMacroSync.FindOutOfSync(feature);
            if (stale.Count > 0)
            {
                EditorGUILayout.HelpBox(
                    "Shader macros are out of sync with the settings for: " + string.Join(", ", stale) +
                    ".\nThese settings (RR, MIS, NEE sample counts, FP16, nested dielectrics, LD sampler, " +
                    "firefly filter, stable-plane count, NEE discard debug) are compile-time macros and " +
                    "only take effect after reimport.",
                    MessageType.Warning);
            }

            using (new EditorGUI.DisabledScope(stale.Count == 0))
            {
                string label = stale.Count > 0
                    ? $"Apply Shader Macros (reimports {stale.Count} shader{(stale.Count > 1 ? "s" : "")})"
                    : "Shader Macros In Sync";
                if (GUILayout.Button(label))
                {
                    int n = NativeRtxptShaderMacroSync.Apply(feature);
                    Debug.Log($"[NativeRtxptFeature] Shader macros applied; {n} shader asset(s) reimported.");
                }
            }
        }

        private void DrawGroupedAssetFields()
        {
            var skip = new HashSet<string> { "renderPassEvent", "setting" };

            var groupLabels = new Dictionary<System.Type, string>
            {
                { typeof(NativeComputeShader), "Native Compute Shaders" },
                { typeof(RayTraceShader), "Ray Trace Shaders" },
                { typeof(ComputeShader), "Compute Shaders" },
            };

            var groups = new Dictionary<string, List<string>>();

            var fields = typeof(NativeRtxptFeature)
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

            var order = new[] { "Ray Trace Shaders", "Native Compute Shaders", "Compute Shaders", "Other" };
            foreach (var groupName in order)
            {
                if (!groups.TryGetValue(groupName, out var fieldNames) || fieldNames.Count == 0)
                    continue;

                string foldoutKey  = GetKey("AssetGroup_" + groupName);
                bool   isExpanded  = SessionState.GetBool(foldoutKey, true);
                bool   newExpanded = EditorGUILayout.Foldout(isExpanded, groupName, toggleOnLabelClick: true, EditorStyles.foldoutHeader);
                if (newExpanded != isExpanded)
                    SessionState.SetBool(foldoutKey, newExpanded);

                if (newExpanded)
                {
                    EditorGUI.indentLevel++;
                    foreach (var name in fieldNames)
                    {
                        var prop = serializedObject.FindProperty(name);
                        if (prop != null)
                            EditorGUILayout.PropertyField(prop);
                    }

                    EditorGUI.indentLevel--;
                }
            }
        }
    }
}
