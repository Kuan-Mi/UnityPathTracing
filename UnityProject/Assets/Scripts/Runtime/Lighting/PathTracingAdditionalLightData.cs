// PathTracingAdditionalLightData.cs
// Attach to any GameObject that also has a Light (Point or Spot).
// Provides per-light properties used by the GPU path-tracing pipeline.
// Currently exposes 'radius' (sphere/disk light radius for soft shadows).
// Future properties (e.g. IES profile, light temperature) can be added here.

using UnityEngine;

[DisallowMultipleComponent]
[RequireComponent(typeof(Light))]
public class PathTracingAdditionalLightData : MonoBehaviour
{
    [Min(0f)]
    [Tooltip("Light source radius in world units.\n" +
             "0  = ideal hard point/spot light (delta light, no area).\n" +
             "> 0 = sphere area light (soft shadows via stochastic sampling).")]
    public float radius = 0f;

#if UNITY_EDITOR
    private void OnValidate()
    {
        if (radius < 0f) radius = 0f;

        var light = GetComponent<Light>();
        if (light != null &&
            light.type != LightType.Point &&
            light.type != LightType.Spot)
        {
            Debug.LogWarning(
                $"[PathTracingAdditionalLightData] '{name}': radius is only meaningful for " +
                $"Point and Spot lights. Current type: {light.type}", this);
        }
    }
#endif
}
