using UnityEngine;

[ExecuteInEditMode] // 允许在编辑器不运行游戏时也能实时看到效果
public class RealisticSunController : MonoBehaviour
{
    [Header("时间控制")]
    [Tooltip("当前时间 (0-24小时制)")]
    [Range(0f, 24f)]
    public float timeOfDay = 12f;

    [Tooltip("一年中的第几天 (1-365)，决定季节")]
    [Range(1, 365)]
    public int dayOfYear = 80; // 约3月21日 (春分)

    [Header("地理位置")]
    [Tooltip("纬度 (-90 南极 到 90 北极)")]
    [Range(-90f, 90f)]
    public float latitude = 35.68f; // 默认东京/北京附近纬度

    [Tooltip("经度 (-180 西经 到 180 东经)")]
    [Range(-180f, 180f)]
    public float longitude = 139.69f; // 默认东京

    [Tooltip("时区 (UTC偏移量)，例如北京是+8，东京是+9")]
    [Range(-12, 14)]
    public int utcOffset = 9;

    [Header("设置")]
    [Tooltip("是否在场景视图中绘制太阳轨迹")]
    public bool drawGizmos = true;

    // 常量
    private const float Deg2Rad = Mathf.Deg2Rad;
    private const float Rad2Deg = Mathf.Rad2Deg;

    void Update()
    {
        UpdateSunPosition();
    }

    // 当你在Inspector中修改数值时触发
    void OnValidate()
    {
        UpdateSunPosition();
    }

    void UpdateSunPosition()
    {
        // 1. 计算太阳赤纬角 (Solar Declination) - δ
        // 这个角度决定了季节。近似公式：23.44 * sin(360/365 * (day - 81))
        // 81 是春分日左右，此时赤纬为0。
        float declination = 23.44f * Mathf.Sin(2f * Mathf.PI * (dayOfYear - 81f) / 365f) * Deg2Rad;

        // 2. 计算太阳时角 (Hour Angle) - H
        // 首先计算地方太阳时 (Local Solar Time)
        // 太阳时 = 标准时间 + (经度差带来的时间偏移) + (真太阳时差Equation of Time，此处为简化忽略EOT)
        
        // 经度修正：地球每小时转15度。
        // 标准子午线经度 = 时区 * 15
        float standardMeridian = utcOffset * 15f;
        float longitudeCorrection = (longitude - standardMeridian) / 15f; // 小时为单位的修正
        
        float solarTime = timeOfDay + longitudeCorrection;
        
        // 时角：正午12点时角为0，每小时15度。
        // 早上是负数，下午是正数。
        float hourAngle = (solarTime - 12f) * 15f * Deg2Rad;

        // 3. 将经纬度转换为弧度
        float latRad = latitude * Deg2Rad;

        // 4. 核心天文学公式：计算太阳在天空中的坐标
        // Unity坐标系：Y是上(Up)，Z是北(North)，X是东(East)
        
        // 计算高度角 (Elevation) 的正弦值 (sinAlt)
        float sinAltitude = Mathf.Sin(latRad) * Mathf.Sin(declination) + 
                            Mathf.Cos(latRad) * Mathf.Cos(declination) * Mathf.Cos(hourAngle);
        
        // 计算方位角 (Azimuth)
        // 在Unity坐标系中，我们需要算出光线来自的方向向量
        
        // Y分量 (Up) = sin(Altitude)
        float y = sinAltitude;

        // 辅助计算水平面投影
        // x (East) = -cos(δ) * sin(H) 
        // 负号是因为时角增加是向西运动，而X轴正方向是东
        float x = -Mathf.Cos(declination) * Mathf.Sin(hourAngle);

        // z (North) = sin(δ)cos(φ) - cos(δ)sin(φ)cos(H)
        float z = Mathf.Sin(declination) * Mathf.Cos(latRad) - 
                  Mathf.Cos(declination) * Mathf.Sin(latRad) * Mathf.Cos(hourAngle);

        // 得到太阳在天球上的位置向量（指向太阳的向量）
        Vector3 sunDirection = new Vector3(x, y, z).normalized;

        // 5. 应用到灯光
        // Directional Light的旋转代表光线“射向”哪里，所以方向是 -sunDirection
        transform.rotation = Quaternion.LookRotation(-sunDirection);
    }
    
    // 可视化辅助线
    void OnDrawGizmos()
    {
        if (!drawGizmos) return;

        // 简单的绘制当前太阳方向
        Gizmos.color = Color.yellow;
        Gizmos.DrawRay(transform.position, -transform.forward * 5f);
        Gizmos.DrawWireSphere(transform.position - transform.forward * 5f, 0.5f);
    }
}