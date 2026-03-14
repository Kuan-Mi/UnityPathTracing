using DefaultNamespace;
using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(DemoRenderFeature))]
public class DemoRenderFeatureEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        DemoRenderFeature demoRenderFeature = (DemoRenderFeature)target;
        if (GUILayout.Button("Test"))
        { 
            demoRenderFeature.Test(); 
        }         
        if (GUILayout.Button("Rebuild"))
        { 
            demoRenderFeature.gpuScene.Build();
        }           
         
    }
}