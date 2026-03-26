// PathTracingAdditionalLightDataEditor.cs
// Two responsibilities:
//   1. NRDLightDataAutoAttach  — Automatically adds PathTracingAdditionalLightData whenever
//      a Light component is added to a GameObject in the Editor (supports Ctrl+Z undo).
//   2. PathTracingAdditionalLightDataEditor — Custom Inspector + Scene-view gizmo
//      for the PathTracingAdditionalLightData component.

using UnityEditor;
using UnityEngine;

// ---------------------------------------------------------------------------
// Auto-attach: mirrors how URP auto-adds UniversalAdditionalLightData,
// but without touching the built-in Light editor.
// ObjectFactory.componentWasAdded fires whenever a component is added via
// the Editor (menu, Add Component button, drag-drop, etc.).
// ---------------------------------------------------------------------------
[InitializeOnLoad]
static class PathTracingLightDataAutoAttach
{
    static PathTracingLightDataAutoAttach()
    {
        ObjectFactory.componentWasAdded += OnComponentAdded;
    }

    private static void OnComponentAdded(Component component)
    {
        if (component is not Light light) return;

        var go = light.gameObject;

        // Skip prefab assets themselves (only handle scene instances).
        if (!go.scene.IsValid()) return;

        // Already present — nothing to do.
        if (go.TryGetComponent<PathTracingAdditionalLightData>(out _)) return;

        // Undo.AddComponent registers the addition in the undo stack so that
        // Ctrl+Z removes both the Light and the additional data together.
        Undo.AddComponent<PathTracingAdditionalLightData>(go);
    }
}

// ---------------------------------------------------------------------------
// Custom Inspector + Scene gizmo.
// Supports multi-object editing: gizmos are drawn per selected light;
// radius handle delta is applied to all selected objects simultaneously.
// ---------------------------------------------------------------------------
[CustomEditor(typeof(PathTracingAdditionalLightData))]
[CanEditMultipleObjects]
public class PathTracingAdditionalLightDataEditor : Editor
{
    private SerializedProperty m_RadiusProp;
    private PathTracingAdditionalLightData[] m_Targets;

    private void OnEnable()
    {
        m_RadiusProp = serializedObject.FindProperty("radius");
        m_Targets = System.Array.ConvertAll(
            targets, t => (PathTracingAdditionalLightData)t);
    }

    public override void OnInspectorGUI()
    {
        serializedObject.Update();

        EditorGUI.BeginChangeCheck();
        EditorGUILayout.Slider(
            m_RadiusProp,
            0f, 2f,
            new GUIContent("Radius",
                "Light source radius (world units).\n" +
                "0  = ideal hard light (delta, no area).\n" +
                "> 0 = sphere area light (soft shadows via stochastic sampling)."));

        if (EditorGUI.EndChangeCheck())
        {
            if (m_RadiusProp.floatValue < 0f)
                m_RadiusProp.floatValue = 0f;
        }

        serializedObject.ApplyModifiedProperties();

        if (m_Targets.Length == 1)
        {
            float r = m_RadiusProp.floatValue;
            if (r <= 0.0001f)
                EditorGUILayout.HelpBox(
                    "Radius = 0: treated as an ideal hard light (no sphere sampling).",
                    MessageType.Info);
            else
                EditorGUILayout.HelpBox(
                    $"Sphere radius: {r:F4} m — produces soft shadows via stochastic sampling.",
                    MessageType.Info);
        }
        else
        {
            EditorGUILayout.HelpBox($"{m_Targets.Length} objects selected.", MessageType.Info);
        }
    }

    private void OnSceneGUI()
    {
        var data  = (PathTracingAdditionalLightData)target;
        var light = data.GetComponent<Light>();
        if (light == null) return;
        if (light.type != LightType.Point && light.type != LightType.Spot) return;

        float r = data.radius;
        if (r <= 0.0001f) return;

        var pos = data.transform.position;

        Color prevCol = Handles.color;
        Color c = light.color;
        c.a = 0.55f;
        Handles.color = c;

        if (light.type == LightType.Point)
        {
            // Three orthogonal wire discs visualise the sphere.
            Handles.DrawWireDisc(pos, Vector3.up,      r);
            Handles.DrawWireDisc(pos, Vector3.right,   r);
            Handles.DrawWireDisc(pos, Vector3.forward, r);
        }
        else // Spot — draw disc on the light's local XY plane at the origin.
        {
            Handles.DrawWireDisc(pos, data.transform.forward, r);
        }

        Handles.color = prevCol;

        // Radius drag handle — delta applied uniformly to all selected targets.
        EditorGUI.BeginChangeCheck();
        float newR = Handles.RadiusHandle(Quaternion.identity, pos, r);
        if (EditorGUI.EndChangeCheck())
        {
            float delta = newR - r;
            foreach (var t in m_Targets)
            {
                Undo.RecordObject(t, "Change PathTracing Light Radius");
                t.radius = Mathf.Max(0f, t.radius + delta);
            }
        }
    }
}
