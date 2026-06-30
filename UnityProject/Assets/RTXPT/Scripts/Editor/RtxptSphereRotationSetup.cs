#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace PathTracing
{
    /// <summary>
    /// Attaches <see cref="RtxptSpin"/> to every sphere_with_holes <c>body</c> node in the open scene,
    /// reproducing the RTXPT <c>RotatingSphere</c> animations (one per "Sphere"/"Sphere1" instance):
    /// the bulb assembly ("body") spins 360° about local Z every 4 s while the holed shell ("base")
    /// stays put. The body lives nested in the glTFast prefab instance, so this menu wires it reliably
    /// instead of hand-attaching. Idempotent — re-running only fixes up params on existing components.
    ///
    /// Run via  RTXPT ▸ Setup Sphere Rotations.
    /// </summary>
    public static class RtxptSphereRotationSetup
    {
        [MenuItem("RTXPT/Setup Sphere Rotations")]
        public static void SetupSphereRotations()
        {
            int count = 0;
            foreach (var t in Object.FindObjectsByType<Transform>(FindObjectsSortMode.None))
            {
                // The sphere_with_holes "body" is the unique node named "body" that parents a "bulb".
                if (t.name != "body" || t.Find("bulb") == null) continue;

                var  spin       = t.GetComponent<RtxptSpin>();
                bool fresh      = spin == null;
                if (fresh) spin = Undo.AddComponent<RtxptSpin>(t.gameObject);

                spin.axis                 = Vector3.forward; // local Z, per RotatingSphere
                spin.secondsPerRevolution = 4f;
                spin.reverse              = false;
                // Capture the authored base only on a fresh attach — re-running mustn't latch a
                // mid-spin rotation as the base.
                if (fresh) spin.CaptureBase();

                EditorUtility.SetDirty(t.gameObject);
                count++;
            }

            EditorSceneManager.MarkAllScenesDirty();
            Debug.Log($"[RTXPT] Setup Sphere Rotations: RtxptSpin on {count} sphere_with_holes 'body' node(s).");
        }
    }
}
#endif