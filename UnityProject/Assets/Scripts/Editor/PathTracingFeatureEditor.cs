using UnityEditor;
using UnityEngine;

namespace PathTracing
{
    [CustomEditor(typeof(PathTracingFeature))]
    public class PathTracingFeatureEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            PathTracingFeature ptFeature = (PathTracingFeature)target;
            if (GUILayout.Button("ReBuild"))
            {
                ptFeature.ReBuild();
            }   
            if (GUILayout.Button("InitializeBuffers"))
            {
                ptFeature.InitializeBuffers();
            }        
             
        }
    }
}