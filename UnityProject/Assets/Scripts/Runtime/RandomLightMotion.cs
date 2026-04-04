// RandomLightMotion.cs
// 附加到每盏随机灯光上，使其在指定包围盒内平滑漫游。
// 由 RandomLightSpawner.GenerateLights() 自动挂载并初始化。

using UnityEngine;

[ExecuteAlways]
public class RandomLightMotion : MonoBehaviour
{
    // 由 Spawner 初始化——不在 Inspector 中直接编辑
    [HideInInspector] public Transform  spawnerTransform;   // 生成器 Transform（包围盒中心）
    [HideInInspector] public Vector3    boundsSize;         // 包围盒尺寸（本地空间）
    [HideInInspector] public float      moveSpeed   = 1f;   // 移动速度（世界单位/秒）
    [HideInInspector] public float      rotSpeed    = 30f;  // 朝向变化速度（度/秒），仅 Spot/Directional

    private Vector3    m_Target;
    private Quaternion m_TargetRot;
    private bool       m_HasLight;
    private LightType  m_LightType;

    private const float k_ArrivalThreshold = 0.15f;

    private void Start()
    {
        if (spawnerTransform == null)
            spawnerTransform = transform.parent;

        if (TryGetComponent<Light>(out var light))
        {
            m_HasLight  = true;
            m_LightType = light.type;
        }

        PickNewTarget();
    }
 
    private void Update()
    {
        if (spawnerTransform == null) return;

        // 位置漫游
        transform.position = Vector3.MoveTowards(
            transform.position, m_Target, moveSpeed * Time.deltaTime);

        if (Vector3.Distance(transform.position, m_Target) < k_ArrivalThreshold)
            PickNewTarget();

        // 朝向漫游（仅对有方向意义的灯光）
        if (m_HasLight && (m_LightType == LightType.Spot      ||
                           m_LightType == LightType.Directional ||
                           m_LightType == LightType.Rectangle  ||
                           m_LightType == LightType.Disc))
        {
            transform.rotation = Quaternion.RotateTowards(
                transform.rotation, m_TargetRot, rotSpeed * Time.deltaTime);

            if (Quaternion.Angle(transform.rotation, m_TargetRot) < 1f)
                PickNewTargetRot();
        }
    }

    // -----------------------------------------------------------------------
    // 私有辅助
    // -----------------------------------------------------------------------

    private void PickNewTarget()
    {
        m_Target = RandomPointInBounds();
    }

    private void PickNewTargetRot()
    {
        m_TargetRot = RandomDownwardRotation();
    }

    /// <summary>在包围盒（Spawner 本地空间）内随机取一个世界坐标点。</summary>
    private Vector3 RandomPointInBounds()
    {
        if (spawnerTransform == null)
            return transform.position;

        Vector3 half = boundsSize * 0.5f;
        Vector3 localPt = new Vector3(
            Random.Range(-half.x, half.x),
            Random.Range(-half.y, half.y),
            Random.Range(-half.z, half.z));

        return spawnerTransform.TransformPoint(localPt);
    }

    /// <summary>大致朝下的随机旋转（供 Spot / Area 灯光使用）。</summary>
    private static Quaternion RandomDownwardRotation()
    {
        return Quaternion.Euler(
            Random.Range(60f, 120f),
            Random.Range(0f, 360f),
            Random.Range(0f, 360f));
    }
}
