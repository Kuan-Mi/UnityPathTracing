using UnityEngine;

[RequireComponent(typeof(Camera))]
[ExecuteAlways]
public class FroxelFrustumVisualizer : MonoBehaviour
{
    [Header("Froxel 参数（需与 Shader 一致）")]
    public float fogFar = 200f;
    public int sliceCount = 64;
    public int froxelW = 16;
    public int froxelH = 16;

    [Header("可视化选项")]
    [Range(1, 64)]
    public int displaySliceStep = 4;          // 每隔几个 Slice 画一次
    public bool showFrustumEdges = true;      // 显示视锥体四条边
    public bool showSlicePlanes = true;       // 显示各 Slice 平面
    public bool showFroxelCells = false;      // 显示 Froxel 单元格（性能消耗大）
    public bool showSliceLabels = true;       // 显示 Slice 深度标签

    [Header("颜色设置")]
    public Color frustumEdgeColor = new Color(1f, 1f, 0f, 0.8f);
    public Color nearSliceColor   = new Color(0f, 1f, 0f, 0.5f);
    public Color farSliceColor    = new Color(1f, 0f, 0f, 0.5f);
    public Color cellColor        = new Color(0.5f, 0.8f, 1f, 0.15f);

    private Camera _cam;

    private Camera Cam => _cam != null ? _cam : (_cam = GetComponent<Camera>());

    // ──────────────────────────────────────────────────────────────
    //  核心公式：与 Shader 中 SliceToViewZ 完全一致
    // ──────────────────────────────────────────────────────────────
    private float SliceToViewZ(float k)
    {
        float nearZ = -Cam.nearClipPlane;  // gNearZ（负值）
        // nearZ * pow(fogFar / -nearZ, k / sliceCount)
        return nearZ * Mathf.Pow(fogFar / (-nearZ), k / (float)sliceCount);
    }

    /// <summary>
    /// 将 View Space 深度 + UV(0-1) 转换为世界坐标（透视）
    /// </summary>
    private Vector3 ViewZUVToWorld(float viewZ, float u, float v)
    {
        float ndcX = u * 2f - 1f;
        float ndcY = v * 2f - 1f;

        Matrix4x4 proj = Cam.projectionMatrix;
        float viewX = viewZ * (ndcX - proj[0, 2]) / proj[0, 0];
        float viewY = viewZ * (ndcY - proj[1, 2]) / proj[1, 1];

        Vector3 viewPos = new Vector3(viewX, viewY, viewZ);

        // 使用 worldToCameraMatrix 的逆矩阵：View Space -> World Space
        return Cam.cameraToWorldMatrix.MultiplyPoint3x4(viewPos);
    }

    // ──────────────────────────────────────────────────────────────
    //  获取某个 Slice 平面的四个角点（世界坐标）
    // ──────────────────────────────────────────────────────────────
    private void GetSliceCorners(float k, out Vector3 tl, out Vector3 tr,
                                           out Vector3 bl, out Vector3 br)
    {
        float vz = SliceToViewZ(k);
        tl = ViewZUVToWorld(vz, 0f, 1f);
        tr = ViewZUVToWorld(vz, 1f, 1f);
        bl = ViewZUVToWorld(vz, 0f, 0f);
        br = ViewZUVToWorld(vz, 1f, 0f);
    }

    // ──────────────────────────────────────────────────────────────
    //  Gizmos 绘制
    // ──────────────────────────────────────────────────────────────
    private void OnDrawGizmos()
    {
        if (Cam == null || sliceCount <= 0) return;

        // 1. 绘制视锥体四条边（从 Near 到 Far）
        if (showFrustumEdges)
        {
            Gizmos.color = frustumEdgeColor;
            GetSliceCorners(0f,               out var n_tl, out var n_tr, out var n_bl, out var n_br);
            GetSliceCorners((float)sliceCount, out var f_tl, out var f_tr, out var f_bl, out var f_br);

            Gizmos.DrawLine(n_tl, f_tl);
            Gizmos.DrawLine(n_tr, f_tr);
            Gizmos.DrawLine(n_bl, f_bl);
            Gizmos.DrawLine(n_br, f_br);
        }

        // 2. 绘制 Slice 平面
        if (showSlicePlanes)
        {
            for (int i = 0; i <= sliceCount; i += displaySliceStep)
            {
                // 颜色从绿色（近）渐变到红色（远）
                float t = (float)i / sliceCount;
                Gizmos.color = Color.Lerp(nearSliceColor, farSliceColor, t);

                GetSliceCorners((float)i, out var tl, out var tr, out var bl, out var br);

                // 绘制平面四条边
                Gizmos.DrawLine(tl, tr);
                Gizmos.DrawLine(tr, br);
                Gizmos.DrawLine(br, bl);
                Gizmos.DrawLine(bl, tl);

                // 绘制对角线，表示这是个平面
                // Gizmos.DrawLine(tl, br);
                // Gizmos.DrawLine(tr, bl);
            }
        }

        // 3. 显示 Slice 深度标签
#if UNITY_EDITOR
        if (showSliceLabels)
        {
            for (int i = 0; i <= sliceCount; i += displaySliceStep)
            {
                float vz = SliceToViewZ(i);
                float t  = (float)i / sliceCount;
                GetSliceCorners((float)i, out var tl, out var tr, out var bl, out var br);
                Vector3 center = (tl + tr + bl + br) * 0.25f;

                UnityEditor.Handles.color = Color.Lerp(nearSliceColor, farSliceColor, t);
                UnityEditor.Handles.Label(center,
                    $"Slice {i}\nViewZ: {vz:F2}m");
            }
        }
#endif

        // 4. 绘制 Froxel 单元格（可选，开销较大）
        if (showFroxelCells)
        {
            DrawFroxelCells();
        }
    }

    private void DrawFroxelCells()
    {
        Gizmos.color = cellColor;

        for (int zi = 0; zi < sliceCount; zi += displaySliceStep)
        {
            float vzNear = SliceToViewZ(zi);
            float vzFar  = SliceToViewZ(zi + 1);

            // 限制绘制列数/行数，避免卡顿
            int drawW = Mathf.Min(froxelW, 8);
            int drawH = Mathf.Min(froxelH, 8);

            for (int xi = 0; xi < drawW; xi++)
            for (int yi = 0; yi < drawH; yi++)
            {
                float u0 = (float) xi      / froxelW;
                float u1 = (float)(xi + 1) / froxelW;
                float v0 = (float) yi      / froxelH;
                float v1 = (float)(yi + 1) / froxelH;

                // 近面四点
                Vector3 p000 = ViewZUVToWorld(vzNear, u0, v0);
                Vector3 p100 = ViewZUVToWorld(vzNear, u1, v0);
                Vector3 p010 = ViewZUVToWorld(vzNear, u0, v1);
                Vector3 p110 = ViewZUVToWorld(vzNear, u1, v1);
                // 远面四点
                Vector3 p001 = ViewZUVToWorld(vzFar,  u0, v0);
                Vector3 p101 = ViewZUVToWorld(vzFar,  u1, v0);
                Vector3 p011 = ViewZUVToWorld(vzFar,  u0, v1);
                Vector3 p111 = ViewZUVToWorld(vzFar,  u1, v1);

                // 近面
                Gizmos.DrawLine(p000, p100); Gizmos.DrawLine(p100, p110);
                Gizmos.DrawLine(p110, p010); Gizmos.DrawLine(p010, p000);
                // 远面
                Gizmos.DrawLine(p001, p101); Gizmos.DrawLine(p101, p111);
                Gizmos.DrawLine(p111, p011); Gizmos.DrawLine(p011, p001);
                // 侧边
                Gizmos.DrawLine(p000, p001); Gizmos.DrawLine(p100, p101);
                Gizmos.DrawLine(p010, p011); Gizmos.DrawLine(p110, p111);
            }
        }
    }
}