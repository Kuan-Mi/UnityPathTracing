using System.Collections.Generic;
using System.Reflection;
using NativeRender;
using UnityEditor;
using UnityEngine;

namespace PathTracing
{
    /// <summary>
    /// Custom inspector for <see cref="NativeRtxptFeature"/>. The settings page mirrors the RTXPT
    /// C++ ImGui layout (SampleUI.cpp buildUI): same section structure, ordering, labels, ranges
    /// and realtime/reference conditional visibility. Fields marked with " *" are shader
    /// compile-time macros — they only take effect via the "Apply Shader Macros" button below.
    /// Settings with no C++ counterpart are grouped under "Unity-specific". Inert legacy fields
    /// (NRD leftovers, ReSTIR toggles) are intentionally not drawn but kept serialized.
    /// </summary>
    [CustomEditor(typeof(NativeRtxptFeature))]
    public class NativeRtxptFeatureEditor : Editor
    {
        // Mirror of shader-side limits (PathTracerConfig is internal to the runtime assembly).
        private const int   kStablePlaneCount          = 3;   // PathTracerConfig.cStablePlaneCount
        private const int   kStablePlaneMaxVertexIndex = 15;  // PathTracerConfig.cStablePlaneMaxVertexIndex
        private const int   kMaxBounceCount            = 96;  // Config.h MAX_BOUNCE_COUNT
        private const int   kMaxLightSamples           = 63;  // LightingConfig.h RTXPT_LIGHTING_MAX_SAMPLE_COUNT
        private const int   LightingConfigMaxLights    = 512 * 1024; // LightingConfig.RTXPT_LIGHTING_MAX_LIGHTS

        private string GetKey(string headerName) =>
            $"PT_NativeRtxpt_Foldout_{target.GetInstanceID()}_{headerName}";

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            var feature = (NativeRtxptFeature)target;

            DrawSettings(feature);

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

        // ═════════════════════════════════════════════════════════════════════
        // Settings page (mirrors SampleUI.cpp buildUI)
        // ═════════════════════════════════════════════════════════════════════

        private void DrawSettings(NativeRtxptFeature feature)
        {
            var s = feature.setting;
            if (s == null)
            {
                EditorGUILayout.HelpBox("No settings instance.", MessageType.Warning);
                return;
            }

            EditorGUI.BeginChangeCheck();
            Undo.RecordObject(feature, "RTXPT Settings");

            DrawEnvironmentMapSection(s);     // SampleUI.cpp:586  (Scene -> Environment Map)
            DrawCameraSection(s);             // SampleUI.cpp:669
            DrawLightPreprocessingSection(s); // SampleUI.cpp:722
            DrawPathTracerSection(s);         // SampleUI.cpp:794  (DefaultOpen in C++)
            DrawDlssSection(s);               // SampleUI.cpp:1104
            DrawStablePlanesSection(s);       // SampleUI.cpp:1207
            DrawPostProcessSection(s);        // SampleUI.cpp:1273
            DrawDebuggingSection(s);          // SampleUI.cpp:1371
            DrawUnitySpecificSection(s);

            EditorGUILayout.Space(2);
            EditorGUILayout.LabelField("Fields marked * are compile-time shader macros — use 'Apply Shader Macros' below.",
                                       EditorStyles.miniLabel);

            if (EditorGUI.EndChangeCheck())
                EditorUtility.SetDirty(feature);
        }

        // ── Scene -> Environment Map (SampleUI.cpp:586) ───────────────────────
        private void DrawEnvironmentMapSection(NativeRtxptSetting s)
        {
            if (!Foldout("EnvMap", "Environment Map")) return;
            using var _ = new EditorGUI.IndentLevelScope();

            s.environmentMapEnabled = EditorGUILayout.Toggle("Enabled", s.environmentMapEnabled);
            // C++ has a media-list Override combo; Unity overrides via an explicit texture
            // (the env cube otherwise bakes directional lights only).
            s.environmentMap = (Texture)EditorGUILayout.ObjectField(
                new GUIContent("Override", "HDR environment map baked into the env cube: an equirectangular Texture2D or a Cubemap. None = directional lights only."),
                s.environmentMap, typeof(Texture), false);

            EditorGUILayout.Space(2);
            s.environmentMapTint      = EditorGUILayout.ColorField("Tint Color", s.environmentMapTint);
            s.environmentMapIntensity = EditorGUILayout.FloatField("Intensity", s.environmentMapIntensity);
            s.environmentMapRotationY = EditorGUILayout.FloatField("Rotation", s.environmentMapRotationY);
        }

        // ── Camera (SampleUI.cpp:669) ─────────────────────────────────────────
        private void DrawCameraSection(NativeRtxptSetting s)
        {
            if (!Foldout("Camera", "Camera")) return;
            using var _ = new EditorGUI.IndentLevelScope();

            s.cameraAperture = Mathf.Clamp(EditorGUILayout.FloatField("Aperture", s.cameraAperture), 0f, 1f);
            s.cameraFocalDistance = Mathf.Clamp(
                EditorGUILayout.FloatField("Focal Distance", s.cameraFocalDistance), 0.001f, 1e16f);
            s.cameraJitter = EditorGUILayout.Toggle(
                new GUIContent("Camera jitter (Unity)", "Viewport jitter for AA/upscalers (C++ derives this from the AA mode)."),
                s.cameraJitter);
        }

        // ── Light pre-processing and sampling (SampleUI.cpp:722) ──────────────
        private void DrawLightPreprocessingSection(NativeRtxptSetting s)
        {
            if (!Foldout("Lighting", "Light pre-processing and sampling")) return;
            using var _ = new EditorGUI.IndentLevelScope();

            if (!s.useNEE)
                EditorGUILayout.HelpBox("NOTE: NEE inactive (enable in `Path Tracer -> Light sampling`).", MessageType.Warning);

            // LightsBaker::InfoGUI (LightsBaker.cpp:1420) — live light statistics.
            Category("Info and statistics:");
            using (new EditorGUI.IndentLevelScope())
            {
                // Counts are filled by the LightingUpdateBegin pass — valid once the RTXPT
                // renderer has drawn at least one frame (game OR scene view, edit mode included).
                var lub = (target as NativeRtxptFeature)?.LightingUpdateBeginPass;
                if (lub != null && lub.TotalLightCount > 0)
                {
                    string[] modes = { "Uniform", "Power+", "NEE-AT" };
                    EditorGUILayout.LabelField($"Current mode:  {modes[Mathf.Clamp((int)s.neeType, 0, 2)]}", EditorStyles.miniLabel);
                    EditorGUILayout.LabelField("Scene lights by type:", EditorStyles.miniLabel);
                    EditorGUILayout.LabelField($"   envmap quads:  {lub.EnvmapQuadNodeCount}", EditorStyles.miniLabel);
                    EditorGUILayout.LabelField($"   emissive tris: {lub.EmissiveTriangleCount}", EditorStyles.miniLabel);
                    EditorGUILayout.LabelField($"   analytic:      {lub.AnalyticLightCount}", EditorStyles.miniLabel);
                    EditorGUILayout.LabelField($"   TOTAL:         {lub.TotalLightCount}", EditorStyles.miniLabel);
                    EditorGUILayout.LabelField($"(used: {lub.TotalLightCount * 100f / LightingConfigMaxLights:0.00}% of max {LightingConfigMaxLights})", EditorStyles.miniLabel);
                    if (lub.HasControlReadback)
                        EditorGUILayout.LabelField($"(proxies: {lub.SamplingProxyCount}, weightsum: {lub.WeightsSum:0.00000})", EditorStyles.miniLabel);
                    if (lub.TotalLightCount > LightingConfigMaxLights)
                        EditorGUILayout.HelpBox("!!WARNING - scene light count overflow!! increase RTXPT_LIGHTING_MAX_LIGHTS", MessageType.Error);
                }
                else
                {
                    EditorGUILayout.LabelField("(no RTXPT frame rendered yet)", EditorStyles.miniLabel);
                }
            }

            Category("Importance sampling:");
            using (new EditorGUI.IndentLevelScope())
            {
                if (s.neeType != NativeRtxptNeeType.NEEAT)
                {
                    EditorGUILayout.LabelField("NOTE: NEE-AT inactive (enable in `Path Tracer -> NEE settings`).", EditorStyles.wordWrappedMiniLabel);
                }
                else
                {
                    Category("NEE-AT settings:");
                    using (new EditorGUI.IndentLevelScope())
                    {
                        s.neeatGlobalTemporalFeedbackWeight = EditorGUILayout.Slider(
                            new GUIContent("Global feedback weight", "How much to rely on last frame's usage statistic as opposed to simple power based sampling."),
                            s.neeatGlobalTemporalFeedbackWeight, 0.0f, 0.95f);
                        s.neeatLocalToGlobalSampleRatio = EditorGUILayout.Slider(
                            new GUIContent("Local to global sampler ratio", "When drawing new light candidate samples, how many to draw from Global versus Local (tile) samplers."),
                            s.neeatLocalToGlobalSampleRatio, 0.0f, 0.95f);

                        // LightingTypes.hlsli:148 candidate split, shown as a tooltip in C++.
                        int total = Mathf.Max(1, s.neeCandidateSamples);
                        int local = (int)((total - 1) * s.neeatLocalToGlobalSampleRatio + 0.75f);
                        EditorGUILayout.LabelField($"Candidate split: {local} local / {total - local} global (of {total}) *", EditorStyles.miniLabel);

                        s.neeatDistantVsLocalImportance = EditorGUILayout.Slider(
                            new GUIContent("Distant vs Local initial importance", "Higher = more initial importance to environment map / sunlight vs local scene lights."),
                            s.neeatDistantVsLocalImportance, 0.01f, 100.0f);
                    }
                }
            }

            // LightsBaker::DebugGUI (LightsBaker.cpp:1452).
            Category("Debugging:");
            using (new EditorGUI.IndentLevelScope())
            {
                using (new EditorGUI.DisabledScope(!s.enableShaderDebug))
                {
                    s.neeatDbgDrawLights = EditorGUILayout.Toggle(
                        new GUIContent("Debug draw all lights", "Wireframe colour indicates type: red - environment map; green - emissive triangles; blue - analytic. Requires 'Enable shader debug'."),
                        s.neeatDbgDrawLights);
                    s.neeatDbgDrawTileLightConnections = EditorGUILayout.Toggle(
                        new GUIContent("Debug draw NEE-AT tile light connections", "Shows lights sampled by the debug pixel's tile local sampling pdf (C++ uses the mouse cursor; Unity uses 'Debug pixel')."),
                        s.neeatDbgDrawTileLightConnections);
                }

                s.neeatDbgFreezeUpdates = EditorGUILayout.Toggle(
                    new GUIContent("Freeze NEE-AT feedback updates", "Feedback from the path tracer remains frozen while enabled."),
                    s.neeatDbgFreezeUpdates);
                s.neeatDbgViewType = (RtxptLightingDebugViewType)EditorGUILayout.EnumPopup(
                    new GUIContent("NEE-AT debug view", "Show various NEE-AT buffers (via the debug-viz overlay)."),
                    s.neeatDbgViewType);
                s.neeatDbgDisableJitter = EditorGUILayout.Toggle(
                    new GUIContent("Debug disable local tile jitter", "Pixel→tile mapping jitter avoids denoising artifacts and helps spatial sharing; disable for debugging."),
                    s.neeatDbgDisableJitter);
                s.neeatDbgDisableLastFrameFeedback = EditorGUILayout.Toggle(
                    new GUIContent("Debug disable last frame feedback", "Quality reverts to slightly worse than power-based sampling."),
                    s.neeatDbgDisableLastFrameFeedback);
                s.neeatDbgFreezeFrustumUpdates = EditorGUILayout.Toggle("Debug freeze frustum updates", s.neeatDbgFreezeFrustumUpdates);

                if (Foldout("LightingAdv", "Advanced settings"))
                {
                    using var __ = new EditorGUI.IndentLevelScope();
                    s.neeatScreenSpaceVsWorldSpaceThreshold = EditorGUILayout.FloatField("ScreenSpace vs WorldSpace threshold", s.neeatScreenSpaceVsWorldSpaceThreshold);
                    s.neeatDepthDisocclusionThreshold       = EditorGUILayout.FloatField("Depth disocclusion threshold", s.neeatDepthDisocclusionThreshold);
                    s.neeatReservoirHistoryDropoff          = EditorGUILayout.FloatField("Reservoir history dropoff", s.neeatReservoirHistoryDropoff);
                    s.neeatEnableMotionReprojection         = EditorGUILayout.Toggle("Motion reprojection", s.neeatEnableMotionReprojection);
                    s.neeatSampleBakedEnvironment           = EditorGUILayout.Toggle(
                        new GUIContent("Sample environment proxy lights *", "Bake the env map into sampling proxies instead of direct NEE sampling. Biased, faster, blurrier shadows in some cases."),
                        s.neeatSampleBakedEnvironment);
                    EditorGUILayout.LabelField("Importance boosts:", EditorStyles.miniBoldLabel);
                    using (new EditorGUI.IndentLevelScope())
                    {
                        s.neeatImportanceBoostIntensityDelta    = EditorGUILayout.FloatField("...by light intensity change (mul)", s.neeatImportanceBoostIntensityDelta);
                        s.neeatImportanceBoostFrustumMul        = EditorGUILayout.FloatField("...by light frustum proximity (mul)", s.neeatImportanceBoostFrustumMul);
                        s.neeatImportanceBoostFrustumFadeDistance = EditorGUILayout.FloatField(
                            new GUIContent("fade distance", "How fast the boost fades outside of the frustum; bigger = slower fade."),
                            s.neeatImportanceBoostFrustumFadeDistance);
                        s.neeatImportanceBoostPreFilter         = EditorGUILayout.Toggle(
                            new GUIContent("...by pre-filter merge", "Stronger feedback in a 3x3 kernel can 'overwhelm' neighbors. EXPERIMENTAL - SUPER-SLOW."),
                            s.neeatImportanceBoostPreFilter);
                    }
                    s.neeatSceneAverageContentsDistance     = EditorGUILayout.FloatField("Scene average contents distance", s.neeatSceneAverageContentsDistance);
                }
            }
        }

        // ── Path Tracer (SampleUI.cpp:794, DefaultOpen) ───────────────────────
        private void DrawPathTracerSection(NativeRtxptSetting s)
        {
            if (!Foldout("PathTracer", "Path Tracer", defaultOpen: true)) return;
            using var _ = new EditorGUI.IndentLevelScope();

            int modeIndex = s.realtimeMode ? 1 : 0;
            modeIndex = EditorGUILayout.Popup("Mode", modeIndex, new[] { "Reference", "Realtime" });
            s.realtimeMode = modeIndex != 0;

            Category("Setup:");
            using (new EditorGUI.IndentLevelScope())
            {
                if (s.realtimeMode)
                {
                    s.realtimeSamplesPerPixel = EditorGUILayout.IntSlider("Samples per pixel", s.realtimeSamplesPerPixel, 1, 8);
                }
                else
                {
                    s.accumulationTarget = Mathf.Clamp(
                        EditorGUILayout.IntField(new GUIContent("Sample count", "Number of path samples per pixel to collect"), s.accumulationTarget),
                        1, 4 * 1024 * 1024);
                    s.accumulationPreWarmRealtimeCaches = EditorGUILayout.Toggle(
                        new GUIContent("Pre-warm real-time caches", "Pre-warm lighting and other temporal systems before sample 0 is accumulated; otherwise initial samples are lower quality."),
                        s.accumulationPreWarmRealtimeCaches);
                    s.accumulationAA = EditorGUILayout.Toggle(
                        new GUIContent("Jitter anti-aliasing", "Each sample gets a random per-pixel jitter emulating a box filter."),
                        s.accumulationAA);
                }

                s.bounceCount = Mathf.Clamp(
                    EditorGUILayout.IntField(new GUIContent("Max bounces", "Max number of all bounces (including NEE and diffuse bounces)"), s.bounceCount),
                    0, kMaxBounceCount);
                s.diffuseBounceCount = Mathf.Clamp(
                    EditorGUILayout.IntField(new GUIContent("Max diffuse bounces", "Max number of diffuse bounces"), s.diffuseBounceCount),
                    0, kMaxBounceCount);

                if (s.realtimeMode)
                {
                    s.realtimeFireflyFilterEnabled = EditorGUILayout.Toggle(
                        new GUIContent("FireflyFilter (realtime) *", "Smart firefly filter that clamps max radiance based on a probability heuristic."),
                        s.realtimeFireflyFilterEnabled);
                    if (s.realtimeFireflyFilterEnabled)
                    {
                        using var __ = new EditorGUI.IndentLevelScope();
                        s.realtimeFireflyFilterThreshold = Mathf.Clamp(
                            EditorGUILayout.FloatField("FF Threshold", s.realtimeFireflyFilterThreshold), 0.00001f, 1000.0f);
                    }
                }
                else
                {
                    s.referenceFireflyFilterEnabled = EditorGUILayout.Toggle(
                        new GUIContent("FireflyFilter (reference *)", "With both auto-exposure and firefly filter enabled in reference mode, results are no longer deterministic."),
                        s.referenceFireflyFilterEnabled);
                    if (s.referenceFireflyFilterEnabled)
                    {
                        using var __ = new EditorGUI.IndentLevelScope();
                        s.referenceFireflyFilterThreshold = Mathf.Clamp(
                            EditorGUILayout.FloatField("FF Threshold", s.referenceFireflyFilterThreshold), 0.01f, 1000.0f);
                    }
                }

                s.texLODBias = EditorGUILayout.FloatField("Texture MIP bias", s.texLODBias);

                s.environmentMapDiffuseSampleMIPLevel = Mathf.Clamp(
                    EditorGUILayout.IntField(new GUIContent("Diffuse sample envmap MIP level", "MIP level for env-map light sampling and diffuse path termination into sky. Only 0 is unbiased."),
                                             s.environmentMapDiffuseSampleMIPLevel),
                    0, 16);

                s.enableRussianRoulette = EditorGUILayout.Toggle(
                    new GUIContent("Use Russian Roulette early out *", "Stochastic path termination for low-throughput diffuse paths."),
                    s.enableRussianRoulette);
            }

            Category("Post processing:");
            using (new EditorGUI.IndentLevelScope())
            {
                if (s.realtimeMode)
                {
                    s.realtimeAA = EditorGUILayout.Popup(
                        new GUIContent("AA/SR/Denoising", "TAA — temporal AA\nDLSS — super sampling\nDLSS-RR — DLSS + Ray Reconstruction (denoise & upscale)"),
                        s.realtimeAA, Options("Disabled", "TAA", "DLSS", "DLSS-RR"));
                    if (s.realtimeAA == 1 || s.realtimeAA == 2)
                        EditorGUILayout.HelpBox("TAA and DLSS (SR-only) are not implemented in the Unity port yet — use Disabled or DLSS-RR.", MessageType.Warning);
                    if (s.realtimeAA >= 2)
                    {
                        using var __ = new EditorGUI.IndentLevelScope();
                        s.upscalerMode = (UpscalerMode)EditorGUILayout.EnumPopup("DLSS Mode", s.upscalerMode);
                    }
                }
                s.enableToneMapping = EditorGUILayout.Toggle(
                    new GUIContent("Enable tone mapping", "Full tone mapping settings available under `Post-process -> Tone Mapping`."),
                    s.enableToneMapping);
            }

            Category("Light sampling:");
            using (new EditorGUI.IndentLevelScope())
            {
                if (s.realtimeMode)
                {
                    // ReSTIR DI/GI exist in the original; this fork has no RTXDI (PT_USE_RESTIR_*=0).
                    using (new EditorGUI.DisabledScope(true))
                    {
                        EditorGUILayout.Toggle(new GUIContent("Use ReSTIR DI (RTXDI)", "Not available — RTXDI is excluded from this fork."), false);
                        EditorGUILayout.Toggle(new GUIContent("Use ReSTIR GI (RTXDI)", "Not available — RTXDI is excluded from this fork."), false);
                    }
                }

                s.useNEE = EditorGUILayout.Toggle(
                    new GUIContent("Use Next Event Estimation *", "Direct light importance sampling. Analytic lights only come out of NEE — they go missing when disabled."),
                    s.useNEE);

                if (s.useNEE)
                {
                    Category("NEE settings:");
                    using var __ = new EditorGUI.IndentLevelScope();
                    s.neeType = (NativeRtxptNeeType)EditorGUILayout.Popup(
                        new GUIContent("Sampling technique", "Light importance sampling technique for NEE. NEE-AT settings under `Light pre-processing and sampling`."),
                        (int)s.neeType, Options("Uniform", "Power+", "NEE-AT"));
                    s.neeCandidateSamples = Mathf.Clamp(
                        EditorGUILayout.IntField(new GUIContent("Candidate samples *", "Number of light samples weighted with BSDF used to pick each full sample."), s.neeCandidateSamples),
                        1, kMaxLightSamples);
                    s.neeFullSamples = Mathf.Clamp(
                        EditorGUILayout.IntField(new GUIContent("Full samples *", "Number of light samples to shadow test and integrate. Max total samples is 63."), s.neeFullSamples),
                        0, kMaxLightSamples);
                    s.neeMisType = EditorGUILayout.Popup(
                        new GUIContent("MIS Type *", "Path (BSDF) vs light sampler MIS approach. 'Approximate' is faster but more noisy — especially detrimental in reference accumulation."),
                        s.neeMisType, Options("Full", "ApproxInRealtime", "Approximate"));
                }
            }

            // PT: Advanced Settings (SampleUI.cpp:1016)
            if (Foldout("PTAdvanced", "PT: Advanced Settings"))
            {
                using var __ = new EditorGUI.IndentLevelScope();

                Category("Features:");
                s.nestedDielectricsQuality = EditorGUILayout.Popup(
                    new GUIContent("Nested Dielectrics *", "Priority-based nested dielectrics; 'Quality' allows more correct rejections, 'Fast' is faster."),
                    s.nestedDielectricsQuality, Options("Off", "Fast", "Quality"));
                if (s.realtimeMode && s.realtimeAA == 3)
                {
                    s.dlssrrBrightnessClampK = EditorGUILayout.FloatField(
                        new GUIContent("RR brightness clamp", "RR doesn't handle too-bright areas well; clamps brightness at the expense of bloom."),
                        s.dlssrrBrightnessClampK);
                }
                s.denoiserRadianceClampK = EditorGUILayout.FloatField(
                    new GUIContent("Denoiser radiance clamp", "Radiance clamp K applied in the denoiser front-end (ptConsts.denoiserRadianceClampK)."),
                    s.denoiserRadianceClampK);

                Category("Performance:");
                EditorGUILayout.LabelField("<NVAPI Hit Object Extension not supported>", EditorStyles.miniLabel);
                s.dxHitObjectExtension = EditorGUILayout.Toggle(
                    new GUIContent("dx::HitObject codepath *", "If disabled, the traditional TraceRay path is used; if enabled, TraceRayInline→MakeHit→MaybeReorderThread→InvokeHit (SM 6.9)."),
                    s.dxHitObjectExtension);
                if (s.dxHitObjectExtension)
                {
                    using var ___ = new EditorGUI.IndentLevelScope();
                    s.dxMaybeReorderThreads = EditorGUILayout.Toggle("dx::MaybeReorderThreads *", s.dxMaybeReorderThreads);
                }
                s.useFp16Types          = EditorGUILayout.Toggle("Use explicit fp16 types *", s.useFp16Types);
                s.enableLDSamplerForBSDF = EditorGUILayout.Toggle("Enable LD sampler for BSDF *", s.enableLDSamplerForBSDF);
            }
        }

        // ── DLSS settings (SampleUI.cpp:1104, shown when realtime && AA > 1) ──
        private void DrawDlssSection(NativeRtxptSetting s)
        {
            if (!(s.realtimeMode && s.realtimeAA > 1)) return;
            if (!Foldout("DLSS", "DLSS settings")) return;
            using var _ = new EditorGUI.IndentLevelScope();

            s.upscalerMode = (UpscalerMode)EditorGUILayout.EnumPopup("DLSS Mode", s.upscalerMode);
            s.dlssRRPreset =  (DlssRRPreset)EditorGUILayout.EnumPopup("DLSS-RR preset", s.dlssRRPreset);
            if (s.realtimeAA == 3)
            {
                s.dlssrrMicroJitter = EditorGUILayout.Slider("DLSS-RR micro jitter", s.dlssrrMicroJitter, 0.0f, 1.0f);
                s.tmpDisableDlssRR  = EditorGUILayout.Toggle(
                    new GUIContent("Temporarily disable DLSS-RR (Unity)", "Debug: skip the DLSS-RR dispatch and show the noisy input."),
                    s.tmpDisableDlssRR);
            }
        }

        // ── Stable Planes (denoising layers) (SampleUI.cpp:1207) ──────────────
        private void DrawStablePlanesSection(NativeRtxptSetting s)
        {
            if (!Foldout("StablePlanes", "Stable Planes (denoising layers)")) return;
            using var _ = new EditorGUI.IndentLevelScope();

            if (!s.realtimeMode)
            {
                EditorGUILayout.LabelField("Not available in reference mode", EditorStyles.miniLabel);
                return;
            }

            s.stablePlanesActiveCount = Mathf.Clamp(
                EditorGUILayout.IntField(new GUIContent("Active stable planes *", "How many stable planes to allow — 1 is just standard denoising"), s.stablePlanesActiveCount),
                1, kStablePlaneCount);
            s.stablePlanesMaxVertexDepth = Mathf.Clamp(
                EditorGUILayout.IntField(new GUIContent("Max stable plane vertex depth", "How deep the stable part of path tracing can go"), s.stablePlanesMaxVertexDepth),
                2, kStablePlaneMaxVertexIndex);
            s.stablePlanesSplitStopThreshold = EditorGUILayout.Slider(
                new GUIContent("Path split stop threshold", "Stops splitting if more than this throughput would be on a non-taken branch (divided by vertexIndex)."),
                s.stablePlanesSplitStopThreshold, 0.0f, 2.0f);
            s.allowPrimarySurfaceReplacement = EditorGUILayout.Toggle(
                new GUIContent("Primary Surface Replacement", "Whether PSR can be used for the first (base) plane"),
                s.allowPrimarySurfaceReplacement);
            s.stablePlanesSuppressPrimaryIndirectSpecular = EditorGUILayout.Toggle(
                new GUIContent("Suppress primary plane noisy specular", "Suppress noisy specular on the primary stable plane when at least one more plane is in use."),
                s.stablePlanesSuppressPrimaryIndirectSpecular);
            s.stablePlanesSuppressPrimaryIndirectSpecularK = EditorGUILayout.Slider(
                "Suppress primary plane noisy specular amount", s.stablePlanesSuppressPrimaryIndirectSpecularK, 0.0f, 1.0f);
            s.stablePlanesAntiAliasingFallthrough = EditorGUILayout.Slider(
                new GUIContent("Non-primary plane anti-aliasing fallthrough", "Divert some radiance on highly curved/edge areas from non-0 planes back to plane 0 to reduce aliasing."),
                s.stablePlanesAntiAliasingFallthrough, 0.0f, 1.0f);
        }

        // ── Post-process (SampleUI.cpp:1273) ──────────────────────────────────
        private void DrawPostProcessSection(NativeRtxptSetting s)
        {
            if (!Foldout("PostProcess", "Post-process")) return;
            using var _ = new EditorGUI.IndentLevelScope();

            if (Foldout("Bloom", "Bloom"))
            {
                using var __ = new EditorGUI.IndentLevelScope();
                s.enableBloom    = EditorGUILayout.Toggle("Enable Bloom", s.enableBloom);
                s.bloomRadius    = EditorGUILayout.Slider("Bloom Width (Pixels)", s.bloomRadius, 0f, 64f);
                s.bloomIntensity = EditorGUILayout.Slider("Bloom Intensity", s.bloomIntensity, 0f, 0.1f);
            }

            if (Foldout("ToneMapping", "Tone Mapping"))
            {
                using var __ = new EditorGUI.IndentLevelScope();
                s.enableToneMapping = EditorGUILayout.Toggle("Enable", s.enableToneMapping);
                s.toneMapOperator   = (NativeRtxptToneMapOperator)EditorGUILayout.EnumPopup("Operator", s.toneMapOperator);

                s.autoExposure = EditorGUILayout.Toggle("Auto Exposure", s.autoExposure);
                if (s.autoExposure)
                {
                    s.exposureValueMin = EditorGUILayout.FloatField("Auto Exposure Min", s.exposureValueMin);
                    s.exposureValueMin = Mathf.Min(s.exposureValueMax, s.exposureValueMin);
                    s.exposureValueMax = EditorGUILayout.FloatField("Auto Exposure Max", s.exposureValueMax);
                    s.exposureValueMax = Mathf.Max(s.exposureValueMin, s.exposureValueMax);
                }

                s.exposureCompensation = Mathf.Clamp(
                    EditorGUILayout.FloatField("Exposure Compensation", s.exposureCompensation), -12.0f, 12.0f);
                s.exposureValue = Mathf.Clamp(
                    EditorGUILayout.FloatField("Exposure Value", s.exposureValue),
                    Mathf.Log(0.001f, 2f), Mathf.Log(1e9f, 2f));
                s.filmSpeed = Mathf.Clamp(EditorGUILayout.FloatField("Film Speed", s.filmSpeed), 1.0f, 6400.0f);
                s.fNumber   = Mathf.Clamp(EditorGUILayout.FloatField("fNumber", s.fNumber), 0.1f, 100.0f);
                s.shutter   = Mathf.Clamp(EditorGUILayout.FloatField("Shutter", s.shutter), 0.1f, 10000.0f);

                s.toneMapWhiteBalance = EditorGUILayout.Toggle("Enable White Balance", s.toneMapWhiteBalance);
                s.toneMapWhitePoint   = Mathf.Clamp(EditorGUILayout.FloatField("White Point", s.toneMapWhitePoint), 1905.0f, 25000.0f);
                s.toneMapWhiteMaxLuminance = Mathf.Max(0.1f, EditorGUILayout.FloatField("White Max Luminance", s.toneMapWhiteMaxLuminance));
                s.toneMapWhiteScale   = Mathf.Clamp(EditorGUILayout.FloatField("White Scale", s.toneMapWhiteScale), 0f, 100f);
                s.toneMapClamped      = EditorGUILayout.Toggle("Enable Clamp", s.toneMapClamped);
            }
        }

        // ── Debugging (SampleUI.cpp:1371) ─────────────────────────────────────
        private void DrawDebuggingSection(NativeRtxptSetting s)
        {
            if (!Foldout("Debugging", "Debugging")) return;
            using var _ = new EditorGUI.IndentLevelScope();

            s.enableShaderDebug = EditorGUILayout.Toggle(
                new GUIContent("Enable shader debug", "ShaderDebug machinery: DebugPrint → console, DebugLine/DebugTriangle drawing, viz overlay. Allocates a ~100 MB GPU buffer (same as the original)."),
                s.enableShaderDebug);

            var feature = (NativeRtxptFeature)target;

            if (Foldout("DebugSwitches", "Debug switches"))
            {
                using var __ = new EditorGUI.IndentLevelScope();
                if (s.realtimeMode)
                {
                    s.dbgFreezeRealtimeNoiseSeed = EditorGUILayout.Toggle(
                        new GUIContent("Freeze realtime noise seed", "Global noise seed will not change per frame. Useful for debugging transient issues hidden by noise."),
                        s.dbgFreezeRealtimeNoiseSeed);
                }
                s.dbgDisableSERTerminationHint = EditorGUILayout.Toggle(
                    new GUIContent("Disable SER path termination hint *", "Disable the path-termination hint passed to SER reordering."),
                    s.dbgDisableSERTerminationHint);
                s.dbgDiscardNonNEELighting = EditorGUILayout.Toggle("Discard path (non-NEE) lighting *", s.dbgDiscardNonNEELighting);
                s.dbgDiscardNEELighting    = EditorGUILayout.Toggle("Discard NEE lighting *", s.dbgDiscardNEELighting);
            }

            s.debugViewType = (RtxptDebugViewType)EditorGUILayout.EnumPopup("Debug view *", s.debugViewType);
            if (s.debugViewType >= RtxptDebugViewType.StablePlane_VirtualRayLength &&
                s.debugViewType <= RtxptDebugViewType.StablePlane_DenoiserValidation)
            {
                using var __ = new EditorGUI.IndentLevelScope();
                s.debugViewStablePlaneIndex = EditorGUILayout.IntSlider(
                    new GUIContent("Stable Plane index", "-1 = all planes combined"),
                    s.debugViewStablePlaneIndex, -1, s.stablePlanesActiveCount - 1);
            }

            using (new EditorGUI.DisabledScope(!s.enableShaderDebug))
            {
                var px = EditorGUILayout.Vector2IntField(
                    new GUIContent("Debug pixel", "Render-resolution pixel whose path is debugged (pick feedback, DebugPrint slots, debug lines)."),
                    new Vector2Int(s.debugPixelX, s.debugPixelY));
                s.debugPixelX = px.x;
                s.debugPixelY = px.y;

                s.continuousDebugFeedback = EditorGUILayout.Toggle(
                    new GUIContent("Continuous feedback", "Pick the debug pixel every frame and read back the feedback struct."),
                    s.continuousDebugFeedback);

                s.showDebugLines = EditorGUILayout.Toggle(
                    new GUIContent("Show debug lines *", "Draw the picked pixel's path-trace debug lines. Compile-time macro ENABLE_DEBUG_LINES_VIZ — apply shader macros after toggling."),
                    s.showDebugLines);
                if (s.showDebugLines)
                {
                    using var __ = new EditorGUI.IndentLevelScope();
                    s.debugLineScale = EditorGUILayout.FloatField(
                        new GUIContent("Debug line scale", "DebugConstants::debugLineScale (C++ default 0.05; 0 disables)."),
                        s.debugLineScale);
                    if (!s.continuousDebugFeedback)
                        EditorGUILayout.HelpBox(
                            "Debug lines are only emitted for the picked pixel (IsDebugPixel gates every DrawLine) — " +
                            "enable 'Continuous feedback' and point 'Debug pixel' at geometry (render-resolution coords).",
                            MessageType.Info);
                }
            }

            // Live pick feedback (mirrors the C++ ImGui "debugPrint %d: ..." block). Valid whenever
            // frames are being rendered — game or scene view, edit mode included.
            var fb = NativeRtxptShaderDebug.LastFeedback;
            if (s.continuousDebugFeedback && fb.Valid)
            {
                EditorGUILayout.LabelField($"Debug line count: {fb.LineVertexCount / 2}   picked materialID: {fb.PickedMaterialID}", EditorStyles.miniLabel);
                for (int i = 0; i < fb.DebugPrintSlots.Length; i++)
                {
                    Vector4 v = fb.DebugPrintSlots[i];
                    if (v == new Vector4(-1, -1, -1, -1)) continue; // unwritten slot (DebugContext::Reset)
                    EditorGUILayout.LabelField($"debugPrint {i}: {v.x:0.####}, {v.y:0.####}, {v.z:0.####}, {v.w:0.####}", EditorStyles.miniLabel);
                }
                Repaint();
            }
        }

        // ── Unity-specific (no C++ counterpart) ───────────────────────────────
        private void DrawUnitySpecificSection(NativeRtxptSetting s)
        {
            if (!Foldout("UnitySpecific", "Unity-specific")) return;
            using var _ = new EditorGUI.IndentLevelScope();

            s.showMode = (NativeRtxptShowMode)EditorGUILayout.EnumPopup(
                new GUIContent("Show mode", "Which buffer the output blit pass displays."), s.showMode);
            s.showValidation   = EditorGUILayout.Toggle(
                new GUIContent("Show validation", "Show the DLSS validation overlay when available."), s.showValidation);
            s.skipRightEyeInVR = EditorGUILayout.Toggle("Skip right eye in VR", s.skipRightEyeInVR);
        }

        // ─────────────────────────────────────────────────────────────────────

        private bool Foldout(string key, string label, bool defaultOpen = false)
        {
            string k    = GetKey(key);
            bool   open = SessionState.GetBool(k, defaultOpen);
            bool   now  = EditorGUILayout.Foldout(open, label, toggleOnLabelClick: true, EditorStyles.foldoutHeader);
            if (now != open) SessionState.SetBool(k, now);
            return now;
        }

        private static void Category(string label) =>
            EditorGUILayout.LabelField(label, EditorStyles.boldLabel);

        // EditorGUILayout.Popup has no (GUIContent, int, string[]) overload on all Unity versions.
        private static GUIContent[] Options(params string[] items)
        {
            var result = new GUIContent[items.Length];
            for (int i = 0; i < items.Length; i++)
                result[i] = new GUIContent(items[i]);
            return result;
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
                    "firefly filter, stable-plane count, NEE/SER debug toggles, debug view, debug lines, " +
                    "DX hit-object, NEE-AT baked env) are compile-time macros and only take effect after reimport.",
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

        // Reduces a field type to the asset type it references: unwraps arrays and
        // LazyLoadReference<T> so grouping sees RayTraceShader/NativeComputeShader/etc.
        private static System.Type UnwrapAssetType(System.Type t)
        {
            if (t.IsArray)
                t = t.GetElementType();
            if (t != null && t.IsGenericType && t.GetGenericTypeDefinition() == typeof(LazyLoadReference<>))
                t = t.GetGenericArguments()[0];
            return t;
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

                // Asset fields are serialized as LazyLoadReference<T> (optionally arrayed); unwrap to
                // the underlying asset type so grouping matches the declared shader kind.
                var assetType = UnwrapAssetType(field.FieldType);

                string groupName = null;
                foreach (var kv in groupLabels)
                {
                    if (kv.Key.IsAssignableFrom(assetType))
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
