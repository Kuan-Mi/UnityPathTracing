using SER;
using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(HelloSERFeature))]
public sealed class HelloSERFeatureEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        EditorGUILayout.Space();
        if (GUILayout.Button("Auto Fill Hello SER Assets"))
        {
            foreach (var targetObject in targets)
            {
                if (targetObject is not HelloSERFeature feature)
                    continue;

                Undo.RecordObject(feature, "Auto Fill Hello SER Assets");
                feature.AutoFillAssets();
            }
        }
    }
}
