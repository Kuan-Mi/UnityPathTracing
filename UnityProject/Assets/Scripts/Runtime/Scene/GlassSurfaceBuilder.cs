using System.Collections.Generic;
using NativeRender;
using UnityEngine;

public class GlassSurfaceBuilder : MonoBehaviour
{
    // 对应 C++ 中的 GLASS_THICKNESS 常量（单位：米）
    private const float GlassThickness = 0.002f;

    [Tooltip("场景中所有透明玻璃物体的根节点列表")]
    public List<GameObject> transparentObjects = new List<GameObject>();

    /// <summary>
    /// 对场景中所有透明且足够厚的物体，在其内部创建一个缩小的内表面副本。
    /// 对应 C++ 的 Sample::AddInnerGlassSurfaces()。
    /// </summary>
    [ContextMenu("Add Inner Glass Surfaces")]
    public void AddInnerGlassSurfaces()
    {
        // 快照列表，避免遍历时修改集合
        var snapshot = new List<GameObject>(transparentObjects);

        foreach (var go in snapshot)
        {
            if (go == null) continue;

            var rends = go.GetComponentsInChildren<Renderer>();

            foreach (var rend in rends)
            {
                if (rend == null) continue;

                // 跳过非透明材质
                if (!RayTracingMaterialHelper.IsMaterialTransparent(rend.sharedMaterial))
                {
                    Debug.LogWarning($"Skipping '{go.name}' because its material is not transparent.");
                    continue;
                }

                // 用世界空间缩放后的 AABB 计算 size
                Bounds  aabb = rend.bounds; // 世界空间轴对齐包围盒
                Vector3 size = aabb.size; // = vMax - vMin，已含 lossyScale

                // 跳过过薄的物体
                float minSize = Mathf.Min(size.x, Mathf.Min(size.y, size.z));
                if (minSize < GlassThickness * 2.0f) continue;

                // 计算内表面相对于当前物体的缩放比例
                // scale = (size - GLASS_THICKNESS) / size
                Vector3 innerScale = new Vector3(
                    (size.x - GlassThickness) / (size.x + 1e-15f),
                    (size.y - GlassThickness) / (size.y + 1e-15f),
                    (size.z - GlassThickness) / (size.z + 1e-15f)
                );

                // 创建内表面：复制物体，作为其子节点，仅改变缩放
                GameObject inner = Instantiate(rend.gameObject, transform);
                inner.name = go.name + "_InnerSurface";

                // 以 AABB 中心为缩放原点：pivot 随缩放平移
                Vector3 pivotOffset = rend.transform.position - aabb.center;
                Vector3 scaledWorldPos = aabb.center + Vector3.Scale(innerScale, pivotOffset);
                inner.transform.SetPositionAndRotation(scaledWorldPos, go.transform.rotation);

                // // 将局部 lossyScale 换算成相对父节点的 localScale
                // Vector3 parentScale = go.transform.parent != null
                //     ? go.transform.parent.lossyScale
                //     : Vector3.one;

                inner.transform.localScale = new Vector3(
                    rend.transform.localScale.x * innerScale.x,
                    rend.transform.localScale.y * innerScale.y,
                    rend.transform.localScale.z * innerScale.z
                );

                Debug.Log($"Added inner surface for '{rend.gameObject.name}' with scale {inner.transform.localScale}");
            }
        }
    }
}