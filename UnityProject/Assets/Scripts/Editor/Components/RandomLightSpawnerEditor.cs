// RandomLightSpawnerEditor.cs
// RandomLightSpawner 的自定义 Inspector 和 Scene 视图句柄。
//   • Inspector 的"生成灯光"/"清除灯光"按钮支持 Undo。
//   • Scene 视图中显示可拖拽的 BoxBoundsHandle，调整生成范围。

using UnityEditor;
using UnityEditor.IMGUI.Controls;
using UnityEngine;

[CustomEditor(typeof(RandomLightSpawner))]
public class RandomLightSpawnerEditor : Editor
{
    private BoxBoundsHandle m_BoundsHandle;

    private void OnEnable()
    {
        m_BoundsHandle = new BoxBoundsHandle();
    }

    // -----------------------------------------------------------------------
    // Inspector GUI
    // -----------------------------------------------------------------------
    public override void OnInspectorGUI()
    {
        var spawner = (RandomLightSpawner)target;

        // --- 参数区域（排除 intensityMultiplier，单独绘制带实时响应的版本）---
        EditorGUI.BeginChangeCheck();
        DrawDefaultInspector();
        bool paramChanged = EditorGUI.EndChangeCheck();
        if (paramChanged && spawner.spawnedLights.Count > 0)
        {
            Undo.RecordObject(spawner, "Adjust Intensity Multiplier");
            spawner.ApplyIntensityMultiplier();
            EditorUtility.SetDirty(spawner);
        }

        EditorGUILayout.Space(8f);

        // 已生成数量提示
        EditorGUILayout.BeginHorizontal();
        EditorGUILayout.PrefixLabel("已生成灯光数");
        EditorGUILayout.LabelField(spawner.spawnedLights.Count.ToString(),
            EditorStyles.boldLabel);
        EditorGUILayout.EndHorizontal();

        EditorGUILayout.Space(4f);

        // 生成按钮
        Color prevBg = GUI.backgroundColor;
        GUI.backgroundColor = new Color(0.45f, 0.85f, 0.45f);
        if (GUILayout.Button("生成灯光", GUILayout.Height(32f)))
        {
            Undo.RegisterFullObjectHierarchyUndo(spawner.gameObject, "Generate Random Lights");
            spawner.GenerateLights();
            EditorUtility.SetDirty(spawner);
        }

        GUI.backgroundColor = new Color(0.95f, 0.45f, 0.45f);
        if (GUILayout.Button("清除灯光", GUILayout.Height(24f)))
        {
            Undo.RegisterFullObjectHierarchyUndo(spawner.gameObject, "Clear Random Lights");
            spawner.ClearLights();
            EditorUtility.SetDirty(spawner);
        }

        GUI.backgroundColor = prevBg;

        EditorGUILayout.Space(4f);
        EditorGUILayout.HelpBox(
            "在 Scene 视图中选中本物体后，可拖拽黄色方框的控制点来调整生成范围。",
            MessageType.Info);
    }

    // -----------------------------------------------------------------------
    // Scene GUI — BoxBoundsHandle
    // -----------------------------------------------------------------------
    private void OnSceneGUI()
    {
        var spawner = (RandomLightSpawner)target;

        // 在物体本地坐标系下绘制句柄（handle 中心始终为原点）
        m_BoundsHandle.center = Vector3.zero;
        m_BoundsHandle.size   = spawner.boundsSize;

        using (new Handles.DrawingScope(
            new Color(1f, 0.92f, 0.016f, 0.9f),
            spawner.transform.localToWorldMatrix))
        {
            EditorGUI.BeginChangeCheck();
            m_BoundsHandle.DrawHandle();
            if (EditorGUI.EndChangeCheck())
            {
                Undo.RecordObject(spawner, "Resize Light Spawner Bounds");

                // 确保尺寸不为负
                Vector3 newSize = m_BoundsHandle.size;
                newSize.x = Mathf.Max(0f, newSize.x);
                newSize.y = Mathf.Max(0f, newSize.y);
                newSize.z = Mathf.Max(0f, newSize.z);
                spawner.boundsSize = newSize;

                EditorUtility.SetDirty(spawner);
            }
        }

        // 在 Scene 视图中显示灯光数量标签
        Handles.color = new Color(1f, 0.92f, 0.016f, 1f);
        Vector3 labelPos = spawner.transform.position
                           + spawner.transform.up * (spawner.boundsSize.y * 0.5f + 0.3f);
        Handles.Label(labelPos,
            $"灯光范围  {spawner.boundsSize.x:F1} × {spawner.boundsSize.y:F1} × {spawner.boundsSize.z:F1}\n" +
            $"数量: {spawner.lightCount}  已生成: {spawner.spawnedLights.Count}",
            EditorStyles.boldLabel);
    }
}
