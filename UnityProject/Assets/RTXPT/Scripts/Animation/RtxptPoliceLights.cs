using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace PathTracing
{
    /// <summary>
    /// "PoliceLightingOnRX6" prop component, ported from RTXPT SampleGame
    /// <c>GamePropComponent.cpp::PropComponentPoliceLights</c> (used by SHIP_police_squad).
    ///
    /// Drives the four RX6 ship lights off the prop's wrapped animation time (see
    /// <see cref="PropFlightPath.AnimationTime"/>) between <see cref="animTimeStart"/> and
    /// <see cref="animTimeStop"/>: an intro/outro dim ramp, alternating left/right spots on a
    /// 0.9 s beat, and pulsing/spinning blob lights on a 1.1 s beat. The blob nodes are rotated
    /// about their local X axis so the (override-promoted) beams sweep.
    /// </summary>
    [ExecuteAlways]
    [DisallowMultipleComponent]
    public class RtxptPoliceLights : MonoBehaviour
    {
        [Tooltip("Animation time (seconds, wrapped) at which the police lights switch on.")]
        public float animTimeStart = 0f;

        [Tooltip("Animation time (seconds, wrapped) at which the police lights switch off.")]
        public float animTimeStop = 0f;

        public Light spotLeft, spotRight, blobLeft, blobRight;

        [Tooltip("On-intensities captured from the resolved (override-applied) light defs.")]
        public float spotLeftIntensity, spotRightIntensity, blobLeftIntensity, blobRightIntensity;

        [Tooltip("Base local rotations of the blob nodes (the spin is applied on top of these).")]
        public Quaternion blobLeftRot = Quaternion.identity, blobRightRot = Quaternion.identity;

        private PropFlightPath _path;

        private void OnEnable()
        {
            _path = GetComponent<PropFlightPath>();
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
        }

        private void Update()
        {
            if (Application.isPlaying) Tick();
        }

#if UNITY_EDITOR
        private void EditorTick()
        {
            if (!Application.isPlaying) Tick();
        }
#endif

        private void Tick()
        {
            if (_path == null) _path = GetComponent<PropFlightPath>();
            float t = _path != null ? _path.AnimationTime : 0f;

            float blobRightI = 0f;
            float blobRotAngle = 0f;   // radians

            if (t > animTimeStart && t < animTimeStop)
            {
                float spotLeftI = 0f, spotRightI = 0f, blobLeftI = 0f;

                if (t < animTimeStart + 2f || t > animTimeStop - 2f) // intro / outro
                {
                    spotLeftI  = spotLeftIntensity  * 0.1f;
                    spotRightI = spotRightIntensity * 0.1f;
                    blobLeftI  = blobLeftIntensity  * 0.01f;
                    blobRightI = blobRightIntensity * 0.01f;
                }
                else
                {
                    float spotPeriod = Saturate(Repeat(t, 0.9f) / 0.9f);
                    if (spotPeriod < 0.5f) spotLeftI = spotLeftIntensity;
                    else                   spotRightI = spotRightIntensity;

                    float blobPeriod = Saturate(Repeat(t, 1.1f) / 1.1f);
                    blobRotAngle = (blobPeriod - 0.5f) * Mathf.PI * 2f;
                    blobLeftI  = blobLeftIntensity;
                    blobRightI = blobRightIntensity;
                }

                SetIntensity(spotLeft,  spotLeftI);
                SetIntensity(spotRight, spotRightI);
                SetIntensity(blobLeft,  blobLeftI);
                SetIntensity(blobRight, blobRightI);
            }
            else
            {
                SetIntensity(spotLeft,  0f);
                SetIntensity(spotRight, 0f);
                SetIntensity(blobLeft,  0f);
                SetIntensity(blobRight, 0f);
            }

            var spin = Quaternion.AngleAxis(blobRotAngle * Mathf.Rad2Deg, Vector3.right);
            if (blobLeft  != null) blobLeft.transform.localRotation  = blobLeftRot  * spin;
            if (blobRight != null) blobRight.transform.localRotation = blobRightRot * spin;
        }

        private static void SetIntensity(Light l, float i) { if (l != null) l.intensity = i; }

        // C fmodf(t, p) for t >= 0; Mathf.Repeat matches it on the non-negative animation timeline.
        private static float Repeat(float t, float p) => Mathf.Repeat(t, p);

        private static float Saturate(float v) => v < 0f ? 0f : (v > 1f ? 1f : v);
    }
}
