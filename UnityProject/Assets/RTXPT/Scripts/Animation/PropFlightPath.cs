using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace PathTracing
{
    /// <summary>
    /// Drives this GameObject's local transform along a keyframed position+rotation path,
    /// looping at <see cref="duration"/>.
    ///
    /// Ported from RTXPT SampleGame <c>*.prop.json</c> flight paths (the RX6 ship fleet). Keyframes
    /// are pre-converted to Unity space by <c>RtxptPropImporter</c> (glTFast negate-X). The path
    /// tracer reads <c>transform.localToWorldMatrix</c> every frame, so simply moving the transform
    /// animates both the mesh and any child analytic lights — no extra plumbing required.
    /// </summary>
    [ExecuteAlways]
    [DisallowMultipleComponent]
    public class PropFlightPath : MonoBehaviour
    {
        [Tooltip("Keyframe times in seconds, ascending. Parallel to positions/rotations.")]
        public float[] times;

        [Tooltip("Keyframe local positions (already in Unity space). Parallel to times.")]
        public Vector3[] positions;

        [Tooltip("Keyframe local rotations (already in Unity space). Parallel to times.")]
        public Quaternion[] rotations;

        [Tooltip("Path length in seconds (normally the last keytime). Loop/clamp boundary.")]
        public float duration = 0f;

        [Tooltip("Playback speed multiplier (animPlaybackSpeed from the prop JSON).")]
        public float playbackSpeed = 1f;

        [Tooltip("Wrap back to the start at 'duration' instead of stopping on the last key.")]
        public bool loop = true;

        [Tooltip("Also advance the path in the editor, so the flight previews while the path tracer " +
                 "renders in edit mode. Note: this continuously moves the transform (marks the scene dirty).")]
        public bool playInEditMode = true;

        // Cursor in seconds along the (unscaled) path timeline. Not serialized.
        private float _cursor;

        // Wrapped time within [0, duration] last applied — mirrors RTXPT's per-prop animTime,
        // consumed by prop components such as RtxptPoliceLights.
        private float _animationTime;

        /// <summary>The current wrapped animation time (seconds), matching RTXPT's <c>animTime</c>.</summary>
        public float AnimationTime => _animationTime;

#if UNITY_EDITOR
        private double _lastEditorTime;
#endif

        private void OnEnable()
        {
            _cursor = 0f;
#if UNITY_EDITOR
            _lastEditorTime = EditorApplication.timeSinceStartup;
            if (!Application.isPlaying)
                EditorApplication.update += EditorTick;
#endif
            Sample(0f); // snap to the start pose
        }

        private void OnDisable()
        {
#if UNITY_EDITOR
            EditorApplication.update -= EditorTick;
#endif
        }

        private void Update()
        {
            if (Application.isPlaying)
                Advance(Time.deltaTime);
        }

#if UNITY_EDITOR
        private void EditorTick()
        {
            if (Application.isPlaying) return;
            double now = EditorApplication.timeSinceStartup;
            float  dt  = (float)(now - _lastEditorTime);
            _lastEditorTime = now;
            if (playInEditMode)
                Advance(dt);
        }
#endif

        private void Advance(float dt)
        {
            if (!HasKeys()) return;
            _cursor += dt * playbackSpeed;
            Sample(_cursor);
        }

        private bool HasKeys()
            => times != null && times.Length > 0 &&
               positions != null && positions.Length == times.Length &&
               rotations != null && rotations.Length == times.Length;

        private void Sample(float t)
        {
            if (!HasKeys()) return;
            int n = times.Length;

            if (duration > 0f)
                t = loop ? Mathf.Repeat(t, duration) : Mathf.Clamp(t, 0f, duration);
            _animationTime = t;

            if (t <= times[0])
            {
                Apply(0);
                return;
            }

            if (t >= times[n - 1])
            {
                Apply(n - 1);
                return;
            }

            // Binary search for the segment [lo, hi] with times[lo] <= t < times[hi].
            int lo = 0, hi = n - 1;
            while (lo + 1 < hi)
            {
                int mid = (lo + hi) >> 1;
                if (times[mid] <= t) lo = mid;
                else hi                 = mid;
            }

            float span = times[hi] - times[lo];
            float a    = span > 1e-6f ? (t - times[lo]) / span : 0f;

            transform.localPosition = Vector3.Lerp(positions[lo], positions[hi], a);
            transform.localRotation = Quaternion.Slerp(rotations[lo], rotations[hi], a);
        }

        private void Apply(int i)
        {
            transform.localPosition = positions[i];
            transform.localRotation = rotations[i];
        }
    }
}