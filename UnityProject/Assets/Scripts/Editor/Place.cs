using UnityEditor;
using UnityEngine;

namespace PathTracing
{
    public class Place : EditorWindow
    {
        public GameObject prefab;

        public string objName;
        
        // 将所有包含这个名字的物体替换成Prefab
        
        [MenuItem("Tools/物体替换器")]
        public static void ShowWindow()
        {
            GetWindow<Place>("材质替换器");
        }

        private void OnGUI()
        {
            GUILayout.Label("将场景中包含指定名字的物体替换为预设", EditorStyles.boldLabel);
            prefab = (GameObject)EditorGUILayout.ObjectField("预设体", prefab, typeof(GameObject), false);
            objName = EditorGUILayout.TextField("物体名称包含", objName);

            if (GUILayout.Button("替换物体"))
            {
                ReplaceObjects();
            }
        }

        private void ReplaceObjects()
        {
            if (prefab == null || string.IsNullOrEmpty(objName))
            {
                Debug.LogError("请指定预设体和物体名称");
                return;
            }

            var allObjects = FindObjectsOfType<GameObject>();
            int replaceCount = 0;

            foreach (var obj in allObjects)
            {
                if (obj.name.Contains(objName))
                {
                    var meshRenderer = obj.GetComponent<MeshRenderer>();
                    
                    Vector3 position = obj.transform.position; 
                    
                    if (meshRenderer != null)
                    {
                         var bounds =  meshRenderer.bounds;
                         position = bounds.center - new Vector3(0, bounds.extents.y, 0);
                    }
                    

                    GameObject newObj = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
                    newObj.transform.position = position; 
 

                    replaceCount++;
                }
            }

            Debug.Log($"替换完成，共替换了 {replaceCount} 个物体。");
        }
    }
}