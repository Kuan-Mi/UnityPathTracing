using UnityEngine;
using UnityEditor;
using System.Collections.Generic;

public class TerrainTreeConverter : EditorWindow
{
    [MenuItem("Tools/Convert Terrain Trees to GameObjects")]
    static void Init()
    {
        Terrain terrain = Terrain.activeTerrain;
        if (terrain == null)
        {
            Debug.LogError("没有找到活动的地形 (Active Terrain)！请在场景中选择一个地形或确保地形处于活动状态。");
            return;
        }

        TerrainData data = terrain.terrainData;
        
        // 创建一个父物体来存放所有的树，保持层级整洁
        GameObject parent = new GameObject("Converted Trees");
        parent.transform.position = terrain.transform.position; // 确保父物体位置与地形一致

        // 获取所有的树木实例
        TreeInstance[] trees = data.treeInstances;
        TreePrototype[] prototypes = data.treePrototypes;

        // 用于记录每种树转换了多少个
        int count = 0;

        foreach (TreeInstance tree in trees)
        {
            // 获取对应的预制体
            GameObject prefab = prototypes[tree.prototypeIndex].prefab;
            if (prefab == null) continue;

            // 计算世界坐标位置
            // tree.position 是 0-1 的归一化坐标，需要乘以地形尺寸
            Vector3 worldPos = Vector3.Scale(tree.position, data.size) + terrain.transform.position;

            // 创建物体（使用PrefabUtility保持与Prefab的关联，方便后续修改）
            GameObject treeObj = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
            
            if (treeObj != null)
            {
                treeObj.transform.position = worldPos;
                treeObj.transform.parent = parent.transform;

                // 设置缩放
                Vector3 scale = new Vector3(tree.widthScale, tree.heightScale, tree.widthScale);
                treeObj.transform.localScale = scale;

                // 设置旋转 (TreeInstance的rotation是弧度制)
                float rotationY = tree.rotation * Mathf.Rad2Deg;
                treeObj.transform.rotation = Quaternion.Euler(0, rotationY, 0);

                // 注册撤销操作，防止误操作无法回退
                Undo.RegisterCreatedObjectUndo(treeObj, "Convert Tree");
                count++;
            }
        }

        Debug.Log($"成功转换了 {count} 棵树！");
        
        // 可选：转换后是否清空地形上的树？
        // 如果想清空，取消下面这行的注释
        // data.treeInstances = new TreeInstance[0];
    }
}