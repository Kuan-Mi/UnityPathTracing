using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace PathTracing
{
    /// <summary>
    /// Auto on/off ("blinking") analytic light, ported from RTXPT SampleGame
    /// <c>GameModel.cpp::ModelInstance::UpdateLightFromControllers</c>.
    ///
    /// The light is on for the first <see cref="autoOffTime"/> seconds of every
    /// (<see cref="autoOffTime"/> + <see cref="autoOnTime"/>) period and off for the rest
    /// (the RTXPT field names are kept verbatim even though "off"/"on" read inverted — the
    /// reference logic gates the light <c>on</c> while <c>remainder &lt; autoOffTime/period</c>).
    /// <see cref="autoOnOffTimeOffset"/> phase-shifts the cycle so many lights can be desynced.
    ///
    /// Time uses the global game clock (real seconds, unaffected by per-prop playback speed),
    /// matching the reference which feeds <c>gameTime</c> straight into the controller. The path
    /// tracer re-reads <see cref="Light.intensity"/> every frame, so simply driving it animates
    /// the blink in both play and edit mode.
    /// </summary>
    [ExecuteAlways]
    [RequireComponent(typeof(Light))]
    [DisallowMultipleComponent]
    public class RtxptBlinkingLight : MonoBehaviour
    {
        [Tooltip("Intensity when the light is on (the override/model.json resolved intensity).")]
        public float baseIntensity = 1f;

        [Tooltip("Master enable; when false the light stays off regardless of the blink cycle.")]
        public bool baseEnabled = true;

        [Tooltip("RTXPT autoOffTime — the light is ON for this many seconds at the start of each period.")]
        public float autoOffTime = 0f;

        [Tooltip("RTXPT autoOnTime — the light is OFF for this many seconds after the on-phase.")]
        public float autoOnTime = 0f;

        [Tooltip("RTXPT autoOnOffTimeOffset — phase shift (seconds) to desync multiple blinking lights.")]
        public float autoOnOffTimeOffset = 0f;

        private Light _light;

        private void OnEnable()
        {
            _light = GetComponent<Light>();
#if UNITY_EDITOR
            if (!Application.isPlaying)
                EditorApplication.update += EditorTick;
#endif
            Apply();
        }

        private void OnDisable()
        {
#if UNITY_EDITOR
            EditorApplication.update -= EditorTick;
#endif
        }

        private void Update()
        {
            if (Application.isPlaying) Apply();
        }

#if UNITY_EDITOR
        private void EditorTick()
        {
            if (!Application.isPlaying) Apply();
        }
#endif

        private void Apply()
        {
            if (_light == null) _light = GetComponent<Light>();
            if (_light == null) return;

            bool enabled = baseEnabled;
            if (autoOffTime != 0f && autoOnTime != 0f)
            {
                double period   = (double)autoOffTime + autoOnTime;
                double scaled   = (GameTime() + autoOnOffTimeOffset) / period;
                double remainder = Saturate(scaled - System.Math.Floor(scaled));
                enabled = enabled && remainder < (autoOffTime / period);
            }
            _light.intensity = enabled ? baseIntensity : 0f;
        }

        private static double GameTime()
        {
#if UNITY_EDITOR
            if (!Application.isPlaying) return EditorApplication.timeSinceStartup;
#endif
            return Time.timeAsDouble;
        }

        private static double Saturate(double v) => v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
    }
}
