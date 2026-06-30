using System;
using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace PathTracing
{
    /// <summary>
    /// Drives the bistro "StringLights" emissive blink, ported from the RTXPT scene animations
    /// (bistro-programmer-art.scene.json): six per-colour material channels animate
    /// <c>emissiveIntensity</c> in <b>step</b> mode on a looping timeline, lighting each colour for a
    /// staggered window — a chasing twinkle.
    ///
    /// Each <see cref="Entry"/> drives one <see cref="RtxptMaterial"/>'s
    /// <see cref="RtxptMaterial.EmissiveRuntimeIntensity"/> to <see cref="Entry.onIntensity"/> (the
    /// source's absolute step value) over its [<see cref="Entry.onStart"/>, <see cref="Entry.onEnd"/>)
    /// window within <see cref="loopDuration"/>, and to 0 otherwise. The override is a
    /// non-classification value, so the toggle is the engine's cheap per-frame material re-upload (no
    /// scene-topology rebuild). The material is a shared asset, so every string-light mesh of that
    /// colour blinks together — matching the material-targeted source animation. State is only pushed
    /// on change to avoid per-frame churn.
    /// </summary>
    [ExecuteAlways]
    [DisallowMultipleComponent]
    public class RtxptStringLightBlink : MonoBehaviour
    {
        [Serializable]
        public class Entry
        {
            public RtxptMaterial material;

            [Tooltip("Emitted intensity while on (the source's step value, e.g. Red=8, Orange=20).")]
            public float onIntensity = 1f;

            [Tooltip("Seconds within the loop when this colour turns on (step).")]
            public float onStart;

            [Tooltip("Seconds within the loop when this colour turns off (step). May wrap past onEnd<onStart.")]
            public float onEnd;

            [NonSerialized]
            public int lastState; // -1 = unknown, 0 = off, 1 = on
        }

        [Tooltip("Loop length in seconds (RTXPT StringLights = 4s).")]
        public float loopDuration = 4f;

        public List<Entry> entries = new();

        [Tooltip("Also animate in the editor so the twinkle previews while the path tracer renders.")]
        public bool playInEditMode = true;

        private void OnEnable()
        {
            foreach (var e in entries)
                if (e != null)
                    e.lastState = -1;
#if UNITY_EDITOR
            if (!Application.isPlaying)
                EditorApplication.update += EditorTick;
#endif
        }

        private void OnDisable()
        {
#if UNITY_EDITOR
            EditorApplication.update -= EditorTick;
#endif
            // Drop the override so every channel reverts to its authored brightness when not running.
            foreach (var e in entries)
            {
                if (e?.material == null) continue;
                if (e.material.EmissiveRuntimeIntensity >= 0f)
                {
                    e.material.EmissiveRuntimeIntensity = -1f;
                    e.material.MarkModified();
                }
            }
        }

        private void Update()
        {
            if (Application.isPlaying) Tick();
        }

#if UNITY_EDITOR
        private void EditorTick()
        {
            if (!Application.isPlaying && playInEditMode) Tick();
        }
#endif

        private void Tick()
        {
            if (loopDuration <= 0f || entries == null) return;
            float phase = (float)(GameTime() % loopDuration);

            foreach (var e in entries)
            {
                if (e?.material == null) continue;

                int state = IsOn(phase, e.onStart, e.onEnd) ? 1 : 0;
                if (state == e.lastState) continue; // only push on change
                e.lastState = state;

                e.material.EmissiveRuntimeIntensity = state == 1 ? e.onIntensity : 0f;
                e.material.MarkModified(); // → cheap per-frame material re-upload
            }
        }

        // On for phase in [onStart, onEnd); supports a window that wraps across the loop boundary.
        private static bool IsOn(float phase, float onStart, float onEnd)
        {
            if (onStart <= onEnd) return phase >= onStart && phase < onEnd;
            return phase >= onStart || phase < onEnd; // wrapped
        }

        private static double GameTime()
        {
#if UNITY_EDITOR
            if (!Application.isPlaying) return EditorApplication.timeSinceStartup;
#endif
            return Time.timeAsDouble;
        }
    }
}