using System.Collections.Generic;
using UnityEngine;

namespace NativeRender
{
    /// <summary>
    /// Attach this component to any GameObject with a MeshFilter to register it
    /// as a ray tracing geometry target. Multiple targets can be active simultaneously.
    /// </summary>
    [RequireComponent(typeof(MeshFilter))]
    [ExecuteAlways]
    public class NativeRayTracingTarget : MonoBehaviour
    {
        private static readonly List<NativeRayTracingTarget> s_All = new List<NativeRayTracingTarget>();

        /// <summary>All currently active ray tracing targets.</summary>
        public static IReadOnlyList<NativeRayTracingTarget> All => s_All;

        /// <summary>Per-submesh OMM caches. Index matches the submesh index on the Mesh.</summary>
        public OMMCache[] ommCaches;

        private void OnEnable()
        {
            if (!s_All.Contains(this))
                s_All.Add(this);
        }

        private void OnDisable()
        {
            s_All.Remove(this);
        }
    }
}
