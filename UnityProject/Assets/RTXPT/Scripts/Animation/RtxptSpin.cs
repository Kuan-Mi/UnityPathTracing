using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace PathTracing
{
    /// <summary>
    /// Continuous constant-velocity spin about a local axis, ported from the RTXPT scene's
    /// <c>RotatingSphere</c> animation (bistro-programmer-art.scene.json) — a full 360° about the
    /// node's local Z every <see cref="secondsPerRevolution"/> seconds. The sphere_with_holes lamp
    /// spins its <c>body</c> (bulb assembly) inside the static holed shell; the same component also
    /// suits other constant spinners (e.g. RotatingFans).
    ///
    /// The spin is layered on top of <see cref="baseRotation"/> (the node's authored local rotation,
    /// captured when the component is attached) and restored on disable, so it never bakes drift into
    /// the saved scene. The path tracer reads <c>localToWorldMatrix</c> each frame, so driving the
    /// transform animates it in both play and edit mode.
    /// </summary>
    [ExecuteAlways]
    [DisallowMultipleComponent]
    public class RtxptSpin : MonoBehaviour
    {
        [Tooltip("Local axis to spin around. RTXPT RotatingSphere spins about local Z (forward).")]
        public Vector3 axis = Vector3.forward;

        [Tooltip("Seconds for one full 360° revolution (RTXPT RotatingSphere = 4s).")]
        public float secondsPerRevolution = 4f;

        [Tooltip("Reverse the spin direction.")]
        public bool reverse = false;

        [Tooltip("Also spin in the editor so it previews while the path tracer renders in edit mode.")]
        public bool playInEditMode = true;

        [Tooltip("Authored local rotation the spin is layered on (set on attach; restored on disable).")]
        public Quaternion baseRotation = Quaternion.identity;

        [HideInInspector]
        public bool baseCaptured = false;

        private void OnEnable()
        {
            CaptureBase();
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
            if (baseCaptured) transform.localRotation = baseRotation; // leave the node un-spun
        }

        private void Update()
        {
            if (Application.isPlaying) Apply();
        }

#if UNITY_EDITOR
        private void EditorTick()
        {
            if (!Application.isPlaying && playInEditMode) Apply();
        }
#endif

        /// <summary>Records the current local rotation as the base. Call when (re)attaching.</summary>
        public void CaptureBase()
        {
            if (!baseCaptured)
            {
                baseRotation = transform.localRotation;
                baseCaptured = true;
            }
        }

        private void Apply()
        {
            if (secondsPerRevolution <= 0f) return;
            float frac = (float)((GameTime() / secondsPerRevolution) % 1.0);
            float deg  = frac * 360f * (reverse ? -1f : 1f);
            transform.localRotation = baseRotation * Quaternion.AngleAxis(deg, axis.normalized);
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