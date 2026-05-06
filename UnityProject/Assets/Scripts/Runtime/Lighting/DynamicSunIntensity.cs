using UnityEngine;

[ExecuteAlways]
[RequireComponent(typeof(Light))]
public class DynamicSunIntensity : MonoBehaviour
{
    [Header("Settings")]
    public float sunIntensityMultiplier = 1.0f;
    [Range(0.005f, 0.1f)]
    public float gTanSunAngularRadius = 0.01f; // 太阳角半径的切线值

    private Light _light;

    void Start()
    {
        _light = GetComponent<Light>();
    }

    void Update()
    {
        if (_light == null) return;

        // 获取当前灯光的方向 (对应 HLSL 中的 gSunDirection)
        Vector3 sunDir = -transform.forward; 
        
        // 在该逻辑中，计算太阳自身的颜色通常是评估它中心点的强度
        // 所以我们传入 v = sunDir 或者是观察方向。
        // 为了得到光照颜色，我们假设观察者正视太阳：
        Vector3 viewDir = sunDir; 

        Vector3 finalColor = GetSunIntensity(viewDir, sunDir);

        // 将结果应用到灯光
        // 注意：Color 在 Unity 中通常是 0-1 范围，强度由 Intensity 控制
        // 或者直接叠加到颜色上（HDR模式）
        _light.color = new Color(finalColor.x, finalColor.y, finalColor.z).gamma;
        _light.intensity = sunIntensityMultiplier;
    }

    Vector3 GetSunIntensity(Vector3 v, Vector3 sunDir)
    {
        // HLSL: dot(v, gSunDirection.xyz)
        float b = Vector3.Dot(v, sunDir);
        
        // HLSL: length(v - gSunDirection.xyz * b)
        float d = (v - sunDir * b).magnitude;

        // HLSL: saturate(1.015 - d)
        float glow = Mathf.Clamp01(1.015f - d);
        glow *= b * 0.5f + 0.5f;
        glow *= 0.6f;

        // HLSL: Math::Sqrt01(1.0 - b * b) / b
        // 避免除以 0
        float bClamped = Mathf.Max(b, 0.0001f);
        float a = Mathf.Sqrt(Mathf.Clamp01(1.0f - b * b)) / bClamped;

        // HLSL: Math::SmoothStep(edge0, edge1, x)
        float sun = 1.0f - SmoothStep(gTanSunAngularRadius * 0.9f, gTanSunAngularRadius * 1.66f + 0.01f, a);
        
        sun *= (b > 0.0f ? 1.0f : 0.0f);
        
        // HLSL: 1.0 - Math::Pow01(1.0 - v.y, 4.85)
        sun *= 1.0f - Mathf.Pow(Mathf.Clamp01(1.0f - v.y), 4.85f);
        
        // HLSL: Math::SmoothStep(0.0, 0.1, gSunDirection.y)
        sun *= SmoothStep(0.0f, 0.1f, sunDir.y);
        
        sun += glow;

        // HLSL: lerp(float3(1.0, 0.6, 0.3), float3(1.0, 0.9, 0.7), Math::Sqrt01(gSunDirection.y))
        Vector3 colorA = new Vector3(1.0f, 0.6f, 0.3f);
        Vector3 colorB = new Vector3(1.0f, 0.9f, 0.7f);
        Vector3 sunColor = Vector3.Lerp(colorA, colorB, Mathf.Sqrt(Mathf.Clamp01(sunDir.y)));
        
        sunColor *= sun;

        // HLSL: Math::SmoothStep(-0.01, 0.05, gSunDirection.y)
        sunColor *= SmoothStep(-0.01f, 0.05f, sunDir.y);

        // SUN_INTENSITY 这里作为外部乘数处理
        return sunColor; 
    }

    // HLSL SmoothStep 的 C# 实现
    float SmoothStep(float edge0, float edge1, float x)
    {
        float t = Mathf.Clamp01((x - edge0) / (edge1 - edge0));
        return t * t * (3.0f - 2.0f * t);
    }
}