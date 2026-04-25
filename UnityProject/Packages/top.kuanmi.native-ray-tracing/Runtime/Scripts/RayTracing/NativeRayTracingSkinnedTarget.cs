using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace NativeRender
{
    // -------------------------------------------------------------------------
    // Change Queue types for SkinnedMeshRenderer
    // -------------------------------------------------------------------------

    public readonly struct SkinnedTargetAddEvent
    {
        public readonly NativeRayTracingSkinnedTarget Target;
        public readonly SkinnedMeshRenderer           Renderer;
        public readonly int                           RendererInstanceId;

        public SkinnedTargetAddEvent(NativeRayTracingSkinnedTarget target, SkinnedMeshRenderer renderer, int id)
        {
            Target             = target;
            Renderer           = renderer;
            RendererInstanceId = id;
        }
    }

    public readonly struct SkinnedTargetRemoveEvent
    {
        public readonly int RendererInstanceId;

        public SkinnedTargetRemoveEvent(int rendererInstanceId)
        {
            RendererInstanceId = rendererInstanceId;
        }
    }

    // -------------------------------------------------------------------------

    /// <summary>
    /// Attach this component to any GameObject with a <see cref="SkinnedMeshRenderer"/> to
    /// register it as a dynamic (skinned) ray tracing geometry target.
    /// <para>
    /// <see cref="NRDSampleResource"/> consumes <see cref="AddQueue"/> / <see cref="RemoveQueue"/>
    /// automatically each frame — no manual <c>AddSkinnedInstance</c> calls are needed.
    /// </para>
    /// </summary>
    [RequireComponent(typeof(SkinnedMeshRenderer))]
    [ExecuteAlways]
    public class NativeRayTracingSkinnedTarget : MonoBehaviour
    {
        private static readonly List<NativeRayTracingSkinnedTarget> s_All =
            new List<NativeRayTracingSkinnedTarget>();

        /// <summary>All currently active skinned ray tracing targets.</summary>
        public static IReadOnlyList<NativeRayTracingSkinnedTarget> All => s_All;

        /// <summary>Pending add events — consumed by <see cref="NRDSampleResource"/> each frame.</summary>
        public static readonly Queue<SkinnedTargetAddEvent>    AddQueue    = new();

        /// <summary>Pending remove events — consumed by <see cref="NRDSampleResource"/> each frame.</summary>
        public static readonly Queue<SkinnedTargetRemoveEvent> RemoveQueue = new();

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

            var smr = GetComponent<SkinnedMeshRenderer>();
            if (smr != null)
            {
                // CRITICAL: Set vertexBufferTarget BEFORE the first render so Unity creates
                // the double-buffered vertex buffers needed for GetPreviousVertexBuffer().
                smr.vertexBufferTarget |= GraphicsBuffer.Target.Raw;
                smr.skinnedMotionVectors = true;
            }

            AddQueue.Enqueue(new SkinnedTargetAddEvent(this, smr, smr != null ? smr.GetInstanceID() : 0));
        }

        private void OnDisable()
        {
            var smr  = GetComponent<SkinnedMeshRenderer>();
            int smrId = smr != null ? smr.GetInstanceID() : 0;

            RemoveQueue.Enqueue(new SkinnedTargetRemoveEvent(smrId));

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
