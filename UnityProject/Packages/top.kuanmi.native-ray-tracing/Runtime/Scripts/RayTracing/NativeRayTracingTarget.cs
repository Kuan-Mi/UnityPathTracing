using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace NativeRender
{
    // -------------------------------------------------------------------------
    // Change Queue types
    // -------------------------------------------------------------------------

    /// <summary>
    /// Pushed to <see cref="NativeRayTracingTarget.AddQueue"/> when a target becomes active.
    /// All Unity object references are valid at enqueue time (object still alive in OnEnable).
    /// </summary>
    public readonly struct TargetAddEvent
    {
        /// <summary>The target component that was just enabled.</summary>
        public readonly NativeRayTracingTarget Target;

        /// <summary>The MeshRenderer on the same GameObject.</summary>
        public readonly MeshRenderer Renderer;

        /// <summary>Stable integer key for <c>_perTargetBlas</c> lookups.</summary>
        public readonly int RendererInstanceId;

        public TargetAddEvent(NativeRayTracingTarget target, MeshRenderer renderer, int rendererInstanceId)
        {
            Target             = target;
            Renderer           = renderer;
            RendererInstanceId = rendererInstanceId;
        }
    }

    /// <summary>
    /// Pushed to <see cref="NativeRayTracingTarget.RemoveQueue"/> when a target is disabled or
    /// destroyed.  Only pre-computed plain-data fields are stored — no Unity object references —
    /// because by the time the consumer processes this event the GameObject may already be null.
    /// </summary>
    public readonly struct TargetRemoveEvent
    {
        /// <summary>
        /// Stable integer key matching the <c>MeshRenderer.GetInstanceID()</c> captured while
        /// the object was still alive.  Used to look up the entry in <c>_perTargetBlas</c>.
        /// </summary>
        public readonly int RendererInstanceId;

        public TargetRemoveEvent(int rendererInstanceId)
        {
            RendererInstanceId = rendererInstanceId;
        }
    }

    // -------------------------------------------------------------------------

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

        /// <summary>Pending add events to be consumed by the resource manager each frame.</summary>
        public static readonly Queue<TargetAddEvent>    AddQueue    = new Queue<TargetAddEvent>();

        /// <summary>
        /// Pending remove events to be consumed by the resource manager each frame.
        /// Contains only pre-computed plain data — no Unity object references.
        /// </summary>
        public static readonly Queue<TargetRemoveEvent> RemoveQueue = new Queue<TargetRemoveEvent>();

        /// <summary>Per-submesh OMM caches. Index matches the submesh index on the Mesh.</summary>
        public OMMCache[] ommCaches;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        private static void ResetStatics()
        {
            s_All.Clear();
            AddQueue.Clear();
            RemoveQueue.Clear();
        }

        private void OnEnable()
        {
            if (s_All.Count == 0)
                SceneManager.sceneUnloaded += OnSceneUnloaded;

            if (!s_All.Contains(this))
                s_All.Add(this);

            // Object is alive here — safe to capture component references.
            var mr = GetComponent<MeshRenderer>();
            AddQueue.Enqueue(new TargetAddEvent(this, mr, mr != null ? mr.GetInstanceID() : 0));
        }

        private void OnDisable()
        {
            // Capture RendererInstanceId BEFORE the object is potentially destroyed.
            // No Unity object references are stored in TargetRemoveEvent.
            var mr   = GetComponent<MeshRenderer>();
            int mrId = mr != null ? mr.GetInstanceID() : 0;

            RemoveQueue.Enqueue(new TargetRemoveEvent(mrId));

            s_All.Remove(this);

            if (s_All.Count == 0)
                SceneManager.sceneUnloaded -= OnSceneUnloaded;
        }

        private static void OnSceneUnloaded(Scene scene)
        {
            s_All.RemoveAll(t => t == null);
        }
    }
}
