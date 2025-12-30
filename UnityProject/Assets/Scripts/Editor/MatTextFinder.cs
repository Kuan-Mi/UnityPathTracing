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
            int count = 0;
            foreach (var mat in allMaterials)
            {
                var baseTex = mat.GetTexture("_BaseMap");
                var texName = baseTex.name;
                if (texName.Contains("_BaseColor"))
                {
                    texName = texName.Replace("_BaseColor", "");
                    var maskTexName = texName + "_Specular";
                    
                    var path = AssetDatabase.GetAssetPath(baseTex);
                    var dir = System.IO.Path.GetDirectoryName(path);
                    var maskTexPath = System.IO.Path.Combine(dir, maskTexName + ".png");
                    var maskTex = AssetDatabase.LoadAssetAtPath<Texture2D>(maskTexPath);
                    if (maskTex != null)
                    {
                        mat.SetTexture("_SpecGlossMap", maskTex);
                        Debug.Log($"材质 {mat.name} 补全贴图 {maskTexName}");
                        count++;
                    }
                    
                }
            }
        }
    }
}