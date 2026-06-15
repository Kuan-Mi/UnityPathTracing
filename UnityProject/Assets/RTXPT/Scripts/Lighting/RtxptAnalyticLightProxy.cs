using UnityEngine;

namespace PathTracing
{
    /// <summary>
    /// Marks a <see cref="MeshRenderer"/> as a stand-in ("proxy") for an analytic Spot/Point light.
    ///
    /// When the renderer's <see cref="RtxptMaterial"/> has <c>EnableAsAnalyticLightProxy</c> set, the
    /// path tracer evaluates the referenced analytic light on hit instead of treating the mesh as an
    /// emissive area light. This mirrors RTXPT's ProxiedAnalyticLight / parent-light-node mechanism
    /// (LightsBaker.cpp), where a mesh under a Spot/Point light node redirects its NEE to that light.
    ///
    /// If <see cref="TargetLight"/> is left null, <see cref="NativeRtxptGPUScene"/> falls back to the
    /// first Spot/Point <see cref="Light"/> found on this GameObject or an ancestor — so simply
    /// parenting the proxy mesh under the light (the donut convention) is enough.
    /// </summary>
    [RequireComponent(typeof(MeshRenderer))]
    [DisallowMultipleComponent]
    public class RtxptAnalyticLightProxy : MonoBehaviour
    {
        [Tooltip("The analytic Spot/Point light this mesh stands in for. If null, the first Spot/Point " +
                 "Light on this GameObject or an ancestor is used.")]
        public Light TargetLight;
    }
}
