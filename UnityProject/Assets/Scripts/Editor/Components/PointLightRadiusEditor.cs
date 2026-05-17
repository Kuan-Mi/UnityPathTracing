// PointLightRadiusEditor.cs
// Custom Inspector and Scene-view gizmo for the PointLightRadius component.
// Supports multi-object editing: gizmos are drawn for every selected light,
// and dragging the radius handle on the primary selection applies the same
// delta to all selected objects.

using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(PointLightRadius))]
[CanEditMultipleObjects]
public class PointLightRadiusEditor : Editor
{
    private SerializedProperty m_RadiusProp;
    // Cache targets here; 'targets' property must NOT be accessed inside OnSceneGUI.
    private PointLightRadius[] m_Targets;

    private void OnEnable()
    {
        m_RadiusProp = serializedObject.FindProperty("radius");
        m_Targets    = System.Array.ConvertAll(targets, t => (PointLightRadius)t);
    }

    public override void OnInspectorGUI()
    {
        serializedObject.Update();

        EditorGUI.BeginChangeCheck();
        EditorGUILayout.Slider(
            m_RadiusProp,
            0f, 1f,
            new GUIContent("Radius",
                "Sphere light radius (world units).\n" +
                "0 = ideal point light (hard shadows).\n" +
                "> 0 = sphere area light (soft shadows via stochastic sampling)."));

        if (EditorGUI.EndChangeCheck())
        {
            // Clamp all targets — SerializedProperty already handles multi-edit write.
            if (m_RadiusProp.floatValue < 0f)
                m_RadiusProp.floatValue = 0f;
        }

        serializedObject.ApplyModifiedProperties();

        // Info box — summarise across all selected targets.
        if (m_Targets.Length == 1)
        {
            float r = m_RadiusProp.floatValue;
            if (r <= 0.0001f)
                EditorGUILayout.HelpBox("Radius = 0: ideal hard point light (no sphere sampling).", MessageType.Info);
            else
                EditorGUILayout.HelpBox($"Sphere radius: {r:F4} m — soft shadows accumulated over time.", MessageType.Info);
        }
        else
        {
            EditorGUILayout.HelpBox($"{m_Targets.Length} objects selected.", MessageType.Info);
        }
    }

    // Draw gizmos and radius handles for every selected object.
    private void OnSceneGUI()
    {
        // OnSceneGUI is called once per selected target by Unity when
        // [CanEditMultipleObjects] is present, so 'target' always refers to
        // the current object in the iteration.
        var plr   = (PointLightRadius)target;
        var light = plr.GetComponent<Light>();
        if (light == null || light.type != LightType.Point) return;

        float r = plr.radius;
        if (r <= 0.0001f) return;

        Color prevCol = Handles.color;
        Color c = light.color;
        c.a = 0.6f;
        Handles.color = c;

        Vector3 pos = plr.transform.position;
        Handles.DrawWireDisc(pos, Vector3.up,      r);
        Handles.DrawWireDisc(pos, Vector3.right,   r);
        Handles.DrawWireDisc(pos, Vector3.forward, r);

        Handles.color = prevCol;

        // Radius drag handle — apply delta to all selected targets so they
        // scale together when multiple lights are selected.
        EditorGUI.BeginChangeCheck();
        float newR = Handles.RadiusHandle(Quaternion.identity, pos, r);
        if (EditorGUI.EndChangeCheck())
        {
            float delta = newR - r;
            foreach (var otherPlr in m_Targets)
            {
                Undo.RecordObject(otherPlr, "Change Point Light Radius");
                otherPlr.radius = Mathf.Max(0f, otherPlr.radius + delta);
            }
        }
    }
}
