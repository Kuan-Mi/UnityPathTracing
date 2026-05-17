using UnityEditor;
using UnityEngine;

namespace PathTracing
{
    public class MatTextFinder : EditorWindow
    {
        [MenuItem("Tools/材质贴图补全")]
        public static void ShowWindow()
        {
            GetWindow<MatTextFinder>("贴图补全");
        }


        private void OnGUI()
        {
            if (GUILayout.Button("点击补全 (选中材质)"))
            {
                FillMaterials();
            }
        }

        private void FillMaterials()
        {
            var allMaterials = Selection.GetFiltered<Material>(SelectionMode.Editable | SelectionMode.Assets);
            
            Debug.Log($"选中材质数量: {allMaterials.Length}");
            int count = 0;
            foreach (var mat in allMaterials)
            {
                var baseTex = mat.GetTexture("_BaseMap");
                
                if(baseTex == null)
                {
                    Debug.LogWarning($"材质 {mat.name} 没有_BaseMap贴图，跳过");
                    continue;
                }
                var texName = baseTex.name;
                 
                Debug.Log($"检查材质 {mat.name} 的贴图 {texName}");
                if (texName.Contains("_BaseColor"))
                {
                    texName = texName.Replace("_BaseColor", "");
                    var maskTexName = texName + "_Specular";
                    
                    var path = AssetDatabase.GetAssetPath(baseTex);
                    var dir = System.IO.Path.GetDirectoryName(path);
                    var maskTexPath = System.IO.Path.Combine(dir, maskTexName + ".dds");
                    Debug.Log($"尝试加载贴图 {maskTexPath}");
                    
                    var maskTex = AssetDatabase.LoadAssetAtPath<Texture2D>(maskTexPath);
                    if (maskTex != null)
                    {
                        mat.SetTexture("_MetallicGlossMap", maskTex);
                        Debug.Log($"材质 {mat.name} 补全贴图 {maskTexName}");
                        count++;
                    }
                    else
                    {
                        Debug.LogWarning($"未找到贴图 {maskTexPath}，无法补全材质 {mat.name}");
                    }
                    
                }
            }
        }
    }
}