using UnityEditor;
using UnityEngine;

namespace PathTracing
{
    public class MatHelperWindow : EditorWindow
    { 
        public Material targetMaterial;

        [MenuItem("Tools/材质替换器")]
        public static void ShowWindow()
        {
            GetWindow<MatHelperWindow>("材质替换器");
        }

        private void OnGUI()
        {
            GUILayout.Label("批量替换材质", EditorStyles.boldLabel);
        
            // 目标材质选择框
            targetMaterial = (Material)EditorGUILayout.ObjectField("目标材质", targetMaterial, typeof(Material), false);

            EditorGUILayout.Space();

            if (GUILayout.Button("点击替换 (选中物体中名为 'lit' 的材质)"))
            {
                ReplaceMaterials();
            }
        }

        private void ReplaceMaterials()
        {
            if (targetMaterial == null)
            {
                EditorUtility.DisplayDialog("提示", "请先选择目标材质！", "确定");
                return;
            }

            // 获取选中的所有物体
            GameObject[] selectedObjects = Selection.gameObjects;
            int count = 0;

            foreach (GameObject obj in selectedObjects)
            {
                MeshRenderer renderer = obj.GetComponent<MeshRenderer>();
                if (renderer != null)
                {
                    // 记录撤销操作
                    Undo.RecordObject(renderer, "Replace Materials");

                    // 在编辑器脚本中，必须先获取 sharedMaterials 数组
                    // 修改数组后再重新赋值回去，否则修改不会生效
                    Material[] materials = renderer.sharedMaterials;
                    bool changed = false;

                    for (int i = 0; i < materials.Length; i++)
                    {
                        if (materials[i] != null && materials[i].name == "Lit")
                        {
                            materials[i] = targetMaterial;
                            changed = true;
                        }
                    }

                    if (changed)
                    {
                        renderer.sharedMaterials = materials;
                        count++;
                        // 标记场景已修改，确保保存
                        EditorUtility.SetDirty(obj);
                    }
                }
            }

            Debug.Log($"替换完成！共修改了 {count} 个物体的材质。");
        }
    }
}