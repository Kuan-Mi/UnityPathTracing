using System.Collections.Generic;
using NativeRender;
using UnityEditor;
using UnityEngine;

namespace PathTracing
{
    /// <summary>
    /// Mirrors RTXPT's Sample::SetGlobalShaderMacros (Sample.cpp:990-1043): derives the path-tracer
    /// compile-time macros from <see cref="NativeRtxptSetting"/> and writes them into the
    /// .rayshader / .hitgroupshader importer defines, reimporting (= recompiling via DXC) only the
    /// assets whose values actually changed.
    ///
    /// The C++ app recompiles one pipeline set whenever realtime/reference mode flips; Unity keeps
    /// all three pipelines (BuildSP / FillSP / Reference) imported side by side, so the
    /// mode-dependent macros are resolved per pipeline instead:
    ///   RTXPT_USE_APPROXIMATE_MIS — ActualUseApproximateMIS() (SampleUI.h:132):
    ///       realtime pipelines: neeMisType != 0, reference pipeline: neeMisType == 2.
    ///   RTXPT_FIREFLY_FILTER     — ActualFireflyFilterEnabled() (SampleUI.h:125):
    ///       realtime pipelines: realtimeFireflyFilterEnabled, reference: referenceFireflyFilterEnabled.
    ///
    /// Unmanaged defines on the assets (PATH_TRACER_MODE, permutation names, hit-object extension
    /// selection, debug viz toggles, …) are preserved untouched.
    /// </summary>
    internal static class NativeRtxptShaderMacroSync
    {
        /// <summary>
        /// Computes the settings-derived macros for one pipeline.
        /// Order and names follow Sample.cpp:1004-1038.
        /// </summary>
        public static Dictionary<string, string> ComputeManagedMacros(NativeRtxptSetting s, bool referencePipeline)
        {
            static string B(bool b) => b ? "1" : "0";

            // ActualNEEAT_LocalToGlobalSampleRatio (SampleUI.h:124): no local samples unless NEE-AT.
            float localToGlobalRatio = s.neeType == NativeRtxptNeeType.NEEAT ? s.neeatLocalToGlobalSampleRatio : 0f;

            // ComputeCandidateSampleLocalCount / GlobalCount (LightingTypes.hlsli:148-163):
            // local = (total-1)*ratio + 0.75 — always allow at least 1 global, then even out.
            uint totalCandidates  = (uint)Mathf.Max(1, s.neeCandidateSamples);
            uint localCandidates  = (uint)((totalCandidates - 1) * localToGlobalRatio + 0.75f);
            uint globalCandidates = totalCandidates - localCandidates;

            bool approximateMis = referencePipeline ? s.neeMisType == 2 : s.neeMisType != 0;
            bool fireflyFilter  = referencePipeline ? s.referenceFireflyFilterEnabled : s.realtimeFireflyFilterEnabled;

            return new Dictionary<string, string>
            {
                ["PT_ENABLE_RUSSIAN_ROULETTE"]                    = B(s.enableRussianRoulette),
                ["PT_NEE_ENABLED"]                                = B(s.useNEE),
                ["RTXPT_USE_APPROXIMATE_MIS"]                     = B(approximateMis),
                ["RTXPT_NEE_FULL_SAMPLE_COUNT"]                   = s.neeFullSamples.ToString(),
                ["RTXPT_NEE_LOCAL_CANDIDATE_SAMPLE_COUNT"]        = localCandidates.ToString(),
                ["RTXPT_NEE_GLOBAL_CANDIDATE_SAMPLE_COUNT"]       = globalCandidates.ToString(),
                ["RTXPT_NEE_TOTAL_CANDIDATE_SAMPLE_COUNT"]        = totalCandidates.ToString(),
                ["RTXPT_DISCARD_NON_NEE_LIGHTING"]                = B(s.dbgDiscardNonNEELighting),
                ["RTXPT_DISCARD_NEE_LIGHTING"]                    = B(s.dbgDiscardNEELighting),
                ["RTXPT_FIREFLY_FILTER"]                          = B(fireflyFilter),
                ["RTXPT_ACTIVE_STABLE_PLANE_COUNT"]               = s.stablePlanesActiveCount.ToString(),
                ["RTXPT_NESTED_DIELECTRICS_QUALITY"]              = s.nestedDielectricsQuality.ToString(),
                ["RTXPT_LP_TYPES_USE_16BIT_PRECISION"]            = B(s.useFp16Types),
                ["RTXPT_ENABLE_LOW_DISCREPANCY_SAMPLER_FOR_BSDF"] = B(s.enableLDSamplerForBSDF),
            };
        }

        /// <summary>Asset names whose importer defines do not match the current settings.</summary>
        public static List<string> FindOutOfSync(NativeRtxptFeature feature)
        {
            var stale = new List<string>();
            if (feature == null || feature.setting == null)
                return stale;

            foreach (var (asset, referencePipeline) in EnumerateShaderAssets(feature))
            {
                var importer = GetImporter(asset, out _);
                if (importer == null)
                    continue;

                var managed = ComputeManagedMacros(feature.setting, referencePipeline);
                if (Merge(GetDefines(importer), managed) != null)
                    stale.Add(asset.name);
            }
            return stale;
        }

        /// <summary>
        /// Writes the settings-derived macros into all assigned PT shader importers and reimports
        /// the changed ones. Returns the number of reimported assets.
        /// </summary>
        public static int Apply(NativeRtxptFeature feature)
        {
            if (feature == null || feature.setting == null)
                return 0;

            int changedCount = 0;
            AssetDatabase.StartAssetEditing();
            try
            {
                foreach (var (asset, referencePipeline) in EnumerateShaderAssets(feature))
                {
                    var importer = GetImporter(asset, out string path);
                    if (importer == null)
                        continue;

                    var managed = ComputeManagedMacros(feature.setting, referencePipeline);
                    var merged  = Merge(GetDefines(importer), managed);
                    if (merged == null)
                        continue; // already in sync

                    SetDefines(importer, merged);
                    EditorUtility.SetDirty(importer);
                    importer.SaveAndReimport();
                    changedCount++;
                    Debug.Log($"[NativeRtxptShaderMacroSync] Updated shader macros: {path}");
                }
            }
            finally
            {
                AssetDatabase.StopAssetEditing();
            }
            return changedCount;
        }

        // ─────────────────────────────────────────────────────────────────────

        private static IEnumerable<(Object asset, bool referencePipeline)> EnumerateShaderAssets(NativeRtxptFeature f)
        {
            if (f.buildStablePlanesShader != null) yield return (f.buildStablePlanesShader, false);
            if (f.fillStablePlanesShader  != null) yield return (f.fillStablePlanesShader,  false);
            if (f.referenceShader         != null) yield return (f.referenceShader,         true);

            foreach (var hg in f.buildHitGroups     ?? System.Array.Empty<HitGroupShader>())
                if (hg != null) yield return (hg, false);
            foreach (var hg in f.fillHitGroups      ?? System.Array.Empty<HitGroupShader>())
                if (hg != null) yield return (hg, false);
            foreach (var hg in f.referenceHitGroups ?? System.Array.Empty<HitGroupShader>())
                if (hg != null) yield return (hg, true);
        }

        private static AssetImporter GetImporter(Object asset, out string path)
        {
            path = AssetDatabase.GetAssetPath(asset);
            return string.IsNullOrEmpty(path) ? null : AssetImporter.GetAtPath(path);
        }

        private static string[] GetDefines(AssetImporter importer) => importer switch
        {
            RayTraceShaderImporter r => r.defines,
            HitGroupShaderImporter h => h.defines,
            _                        => null,
        };

        private static void SetDefines(AssetImporter importer, string[] defines)
        {
            switch (importer)
            {
                case RayTraceShaderImporter r: r.defines = defines; break;
                case HitGroupShaderImporter h: h.defines = defines; break;
            }
        }

        /// <summary>
        /// Replaces the managed "KEY=VALUE" entries in <paramref name="current"/> in place
        /// (appending any missing key) and returns the result, or null when nothing changed.
        /// </summary>
        private static string[] Merge(string[] current, Dictionary<string, string> managed)
        {
            if (current == null)
                return null;

            var result  = new List<string>(current);
            var missing = new Dictionary<string, string>(managed);
            bool changed = false;

            for (int i = 0; i < result.Count; i++)
            {
                int    eq  = result[i].IndexOf('=');
                string key = (eq < 0 ? result[i] : result[i].Substring(0, eq)).Trim();
                if (!missing.TryGetValue(key, out string value))
                    continue;

                missing.Remove(key);
                string entry = $"{key}={value}";
                if (result[i] != entry)
                {
                    result[i] = entry;
                    changed   = true;
                }
            }

            foreach (var kv in missing)
            {
                result.Add($"{kv.Key}={kv.Value}");
                changed = true;
            }

            return changed ? result.ToArray() : null;
        }
    }
}
