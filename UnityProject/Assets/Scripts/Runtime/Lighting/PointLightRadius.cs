// PointLightRadius.cs
// Attach this component to any GameObject that also has a Point Light.
// The 'radius' value is read by PathTracingPass and uploaded to the GPU
// so that the sphere-area-light sampling path in PointLights.hlsl is used.
// When radius == 0 the light behaves as an ideal hard point light.

using UnityEngine;

[RequireComponent(typeof(Light))]
[DisallowMultipleComponent]
public class PointLightRadius : MonoBehaviour
{
    [Min(0f)]
    [Tooltip("Radius of the sphere light (world units). " +
             "0 = ideal point light (hard shadows). " +
             "> 0 = sphere area light (soft shadows via stochastic sampling).")]
    public float radius = 0.1f;

#if UNITY_EDITOR
    private void OnValidate()
    {
        // Clamp at runtime in the editor as well
        if (radius < 0f) radius = 0f;

        // Warn if the attached light is not a point light
        var light = GetComponent<Light>();
        if (light != null && light.type != LightType.Point)
        {
            Debug.LogWarning(
                $"[PointLightRadius] '{name}': PointLightRadius should only be used on Point lights. " +
                $"Current light type: {light.type}", this);
        }
    }
#endif
}
