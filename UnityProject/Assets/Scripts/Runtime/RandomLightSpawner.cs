// RandomLightSpawner.cs
// 测试灯光用工具组件。
// 在指定范围内随机生成若干个灯光，灯光类型随机，位置随机。
// 范围（Bounds）可在 Scene 视图中通过拖拽句柄调整（见配套 Editor 脚本）。

using System.Collections.Generic;
using UnityEngine;

[DisallowMultipleComponent]
[ExecuteAlways]
public class RandomLightSpawner : MonoBehaviour
{
    [Header("生成数量")]
    [Min(1)]
    public int lightCount = 10;

    [Header("生成范围（以本物体为中心）")]
    public Vector3 boundsSize = new Vector3(10f, 5f, 10f);

    [Header("灯光类型（至少勾选一项）")]
    public bool includePointLights       = true;
    public bool includeSpotLights        = true;
    public bool includeDirectionalLights = false;
    public bool includeRectangleLights   = true;
    public bool includeDiscLights        = true;

    [Header("面光尺寸（Rectangle / Disc 有效）")]
    [Min(0.01f)] public float minAreaSize = 0.5f;
    [Min(0.01f)] public float maxAreaSize = 2.0f;

    [Header("强度")]
    [Min(0f)] public float minIntensity = 1f;
    [Min(0f)] public float maxIntensity = 5f;

    [Header("范围（Point / Spot 有效）")]
    [Min(0f)] public float minRange = 3f;
    [Min(0f)] public float maxRange = 10f;

    [Header("PathTracing 光源半径（Point / Spot）")]
    [Min(0f)] public float minLightRadius = 0f;
    [Min(0f)] public float maxLightRadius = 0.1f;

    [Header("颜色")]
    public bool  randomColor = true;
    public Color fixedColor  = Color.white;

    [Header("亮度统一调整（生成后可实时修改）")]
    [Min(0f)] public float intensityMultiplier = 1f;

    [Header("随机运动")]
    public bool  enableMotion    = true;
    [Min(0f)] public float minMoveSpeed  = 0.5f;
    [Min(0f)] public float maxMoveSpeed  = 2.0f;
    [Min(0f)] public float minRotSpeed   = 15f;
    [Min(0f)] public float maxRotSpeed   = 60f;

    // 已生成的灯光列表，供 Editor 脚本管理和显示
    [HideInInspector]
    public List<GameObject> spawnedLights    = new List<GameObject>();
    // 生成时记录的基础强度，用于 multiplier 缩放
    [HideInInspector]
    public List<float>      baseIntensities  = new List<float>();

    // -----------------------------------------------------------------------
    // 公开方法
    // -----------------------------------------------------------------------

    /// <summary>清除所有已生成的灯光，然后随机生成新的灯光。</summary>
    public void GenerateLights()
    {
        ClearLights();

        List<LightType> types = BuildTypeList();
        if (types.Count == 0)
        {
            Debug.LogWarning("[RandomLightSpawner] 未勾选任何灯光类型，无法生成。", this);
            return;
        }

        for (int i = 0; i < lightCount; i++)
        {
            // 在本地空间内随机偏移，再转世界空间
            Vector3 localPos = new Vector3(
                Random.Range(-boundsSize.x * 0.5f, boundsSize.x * 0.5f),
                Random.Range(-boundsSize.y * 0.5f, boundsSize.y * 0.5f),
                Random.Range(-boundsSize.z * 0.5f, boundsSize.z * 0.5f));
            Vector3 worldPos = transform.TransformPoint(localPos);

            LightType type = types[Random.Range(0, types.Count)];

            GameObject go = new GameObject($"RandomLight_{i:D3}_{type}");
            go.transform.SetPositionAndRotation(worldPos, RandomLightRotation(type));
            go.transform.SetParent(transform);

            Light light       = go.AddComponent<Light>();
            light.type        = type;
            float baseIntensity = Random.Range(minIntensity, maxIntensity);
            light.intensity   = baseIntensity * intensityMultiplier;
            light.color       = randomColor ? Random.ColorHSV(0f, 1f, 0.6f, 1f, 0.8f, 1f) : fixedColor;
            baseIntensities.Add(baseIntensity);

            if (type == LightType.Rectangle || type == LightType.Disc)
            {
                float w = Random.Range(minAreaSize, maxAreaSize);
                float h = Random.Range(minAreaSize, maxAreaSize);
                light.areaSize = new Vector2(w, h);
            }
            else if (type != LightType.Directional)
            {
                light.range = Random.Range(minRange, maxRange);
            }

            if (type == LightType.Point || type == LightType.Spot)
            {
                light.innerSpotAngle = 50;
                light.spotAngle      = 60;
                var ptData = go.AddComponent<PathTracingAdditionalLightData>();
                ptData.radius = Random.Range(minLightRadius, maxLightRadius);
            }

            if (enableMotion)
            {
                var motion              = go.AddComponent<RandomLightMotion>();
                motion.spawnerTransform = transform;
                motion.boundsSize       = boundsSize;
                motion.moveSpeed        = Random.Range(minMoveSpeed, maxMoveSpeed);
                motion.rotSpeed         = Random.Range(minRotSpeed,  maxRotSpeed);
            }

            spawnedLights.Add(go);
        }
    }

    /// <summary>销毁所有已生成的灯光。</summary>
    public void ClearLights()
    {
        foreach (GameObject go in spawnedLights)
        {
            if (go == null) continue;
#if UNITY_EDITOR
            DestroyImmediate(go);
#else
            Destroy(go);
#endif
        }
        spawnedLights.Clear();
        baseIntensities.Clear();
    }

    /// <summary>将 intensityMultiplier 应用到所有已生成灯光（不改变基础强度）。</summary>
    public void ApplyIntensityMultiplier()
    {
        for (int i = 0; i < spawnedLights.Count; i++)
        {
            if (spawnedLights[i] == null) continue;
            if (!spawnedLights[i].TryGetComponent<Light>(out var light)) continue;
            float baseVal = (i < baseIntensities.Count) ? baseIntensities[i] : 1f;
            light.intensity = baseVal * intensityMultiplier;
        }
    }

    // -----------------------------------------------------------------------
    // 私有辅助
    // -----------------------------------------------------------------------

    private List<LightType> BuildTypeList()
    {
        var list = new List<LightType>();
        if (includePointLights)       list.Add(LightType.Point);
        if (includeSpotLights)        list.Add(LightType.Spot);
        if (includeDirectionalLights) list.Add(LightType.Directional);
        if (includeRectangleLights)   list.Add(LightType.Rectangle);
        if (includeDiscLights)        list.Add(LightType.Disc);
        return list;
    }

    /// <summary>
    /// 根据灯光类型给出合理的随机朝向：
    /// Directional / Spot 偏向朝下；Point 旋转无意义但仍随机（方便扩展）。
    /// </summary>
    private static Quaternion RandomLightRotation(LightType type)
    {
        switch (type)
        {
            case LightType.Directional:
            case LightType.Spot:
                // 随机偏转角度，基座朝下（-Y），左右偏 ±60°
                return Quaternion.Euler(
                    Random.Range(60f, 120f),
                    Random.Range(0f, 360f),
                    0f);
            case LightType.Rectangle:
            case LightType.Disc:
                // 面光朝下，随机水平旋转，轻微随机倾斜
                return Quaternion.Euler(
                    Random.Range(75f, 105f),
                    Random.Range(0f, 360f),
                    Random.Range(0f, 360f));
            default:
                return Quaternion.identity;
        }
    }

    // -----------------------------------------------------------------------
    // Gizmo（仅在选中时显示）
    // -----------------------------------------------------------------------
#if UNITY_EDITOR
    private void OnDrawGizmosSelected()
    {
        Gizmos.matrix = transform.localToWorldMatrix;

        Gizmos.color = new Color(1f, 0.92f, 0.016f, 0.12f);
        Gizmos.DrawCube(Vector3.zero, boundsSize);

        Gizmos.color = new Color(1f, 0.92f, 0.016f, 0.85f);
        Gizmos.DrawWireCube(Vector3.zero, boundsSize);
    }
#endif
}
