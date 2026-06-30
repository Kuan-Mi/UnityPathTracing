#if UNITY_EDITOR
using System.Collections.Generic;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace PathTracing
{
    /// <summary>
    /// Creates (or refreshes) the bistro StringLights blink driver, reproducing the RTXPT scene's six
    /// per-colour <c>StringLights-*</c> material animations (bistro-programmer-art.scene.json): each
    /// Paris_StringLights colour lights for a staggered 2 s window on a 4 s loop.
    ///
    /// Finds the six <see cref="RtxptMaterial"/> assets by their LMBR id and fills a single
    /// <see cref="RtxptStringLightBlink"/> on a "StringLightBlink" scene object with the source schedule.
    ///
    /// Run via  RTXPT ▸ Setup String Light Blink.
    /// </summary>
    public static class RtxptStringLightBlinkSetup
    {
        // LMBR id → (on-intensity, onStart, onEnd) within the 4 s loop, transcribed from the source
        // step keyframes (the bright "on" value, off = 0).
        private static readonly (string lmbr, string colour, float onIntensity, float onStart, float onEnd)[] kSchedule =
        {
            ("000019c", "Yellow", 10.0f, 0.0f, 2.0f),
            ("000019a", "Pink", 12.8f, 0.4f, 2.4f),
            ("0000197", "Red", 8.0f, 0.8f, 2.8f),
            ("0000199", "Green", 11.4f, 1.2f, 3.2f),
            ("000019b", "Orange", 20.0f, 1.6f, 3.6f),
            ("0000198", "Blue", 9.0f, 1.6f, 3.6f),
        };

        [MenuItem("RTXPT/Setup String Light Blink")]
        public static void SetupStringLightBlink()
        {
            // Index all RtxptMaterial assets by their (lowercased) path for LMBR-id lookup.
            var byPath = new List<(string path, RtxptMaterial mat)>();
            foreach (var guid in AssetDatabase.FindAssets("t:RtxptMaterial"))
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                var    mat  = AssetDatabase.LoadAssetAtPath<RtxptMaterial>(path);
                if (mat != null) byPath.Add((path.ToLowerInvariant(), mat));
            }

            var entries = new List<RtxptStringLightBlink.Entry>();
            var missing = new List<string>();
            foreach (var (lmbr, colour, onIntensity, onStart, onEnd) in kSchedule)
            {
                RtxptMaterial found = null;
                string        key   = lmbr.ToLowerInvariant();
                foreach (var (path, mat) in byPath)
                    if (path.Contains(key) && path.Contains("stringlights"))
                    {
                        found = mat;
                        break;
                    }

                if (found == null)
                {
                    missing.Add($"{colour} (LMBR_{lmbr})");
                    continue;
                }

                entries.Add(new RtxptStringLightBlink.Entry
                {
                    material = found, onIntensity = onIntensity, onStart = onStart, onEnd = onEnd
                });
            }

            var go = GameObject.Find("StringLightBlink");
            if (go == null)
            {
                go = new GameObject("StringLightBlink");
                Undo.RegisterCreatedObjectUndo(go, "Create StringLightBlink");
            }

            var blink                = go.GetComponent<RtxptStringLightBlink>();
            if (blink == null) blink = Undo.AddComponent<RtxptStringLightBlink>(go);

            blink.loopDuration = 4f;
            blink.entries      = entries;
            EditorUtility.SetDirty(go);
            EditorSceneManager.MarkAllScenesDirty();

            string warn = missing.Count > 0 ? $"  (missing: {string.Join(", ", missing)})" : "";
            Debug.Log($"[RTXPT] Setup String Light Blink: wired {entries.Count}/{kSchedule.Length} colours.{warn}");
        }
    }
}
#endif