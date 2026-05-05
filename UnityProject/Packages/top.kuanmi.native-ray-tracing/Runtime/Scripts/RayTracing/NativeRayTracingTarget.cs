using System;
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

        /// <summary>
        /// Cached static flag — serialized so it survives a build (GameObject.isStatic is editor-only).
        /// </summary>
        [SerializeField] private bool _isStaticObject;

        /// <summary>True when this target should be treated as a static (immovable) object.</summary>
        public bool IsStatic => _isStaticObject;

        public MeshRenderer meshRenderer;
        [NonSerialized]
        public int instanceId;

        /// <summary>Pre-computed per-submesh material data. Rebuilt in OnEnable and OnValidate.</summary>
        public SubmeshMaterialData[] SubmeshMaterialInfos;

        /// <summary>Submeshes grouped by (isTransparent, isEmissive) pair. Rebuilt alongside SubmeshMaterialInfos.</summary>
        public SubmeshGroupDesc[] SubmeshGroups;

        /// <summary>
        /// Reads sharedMaterials from the attached MeshRenderer, pre-computes material flags,
        /// texture native pointers, scalar properties, and classifies submeshes into groups.
        /// Must be called on the main thread.
        /// </summary>
        
        [ContextMenu("Rebuild Material Data")]
        public void RebuildMaterialData()
        {
            var mr = meshRenderer != null ? meshRenderer : GetComponent<MeshRenderer>();
            if (mr == null)
            {
                SubmeshMaterialInfos = Array.Empty<SubmeshMaterialData>();
                SubmeshGroups        = Array.Empty<SubmeshGroupDesc>();
                return;
            }

            var mf = GetComponent<MeshFilter>();
            if (mf == null || mf.sharedMesh == null)
            {
                SubmeshMaterialInfos = Array.Empty<SubmeshMaterialData>();
                SubmeshGroups        = Array.Empty<SubmeshGroupDesc>();
                return;
            }

            Mesh       mesh   = mf.sharedMesh;
            Material[] mats   = mr.sharedMaterials;
            int        subCnt = mesh.subMeshCount;

            var infos = new SubmeshMaterialData[subCnt];
            for (int s = 0; s < subCnt; s++)
            {
                Material mat = s < mats.Length ? mats[s] : null;
                if (mat == null)
                    Debug.LogError($"[NativeRayTracingTarget] Submesh {s} of '{name}' has no material assigned.");
                infos[s] = RayTracingMaterialHelper.BuildSubmeshMaterialData(mat);
            }

            SubmeshMaterialInfos = infos;
            SubmeshGroups        = RayTracingMaterialHelper.BuildSubmeshGroupDescs(infos);
        }

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
            meshRenderer = GetComponent<MeshRenderer>();
            instanceId = meshRenderer != null ? meshRenderer.GetInstanceID() : 0;
            // RebuildMaterialData();
            AddQueue.Enqueue(new TargetAddEvent(this, meshRenderer, instanceId));
        }

// #if UNITY_EDITOR
//         private void OnValidate()
//         {
//             // Bake the static flag into the serialized field while in the editor
//             // so the correct value survives a build (gameObject.isStatic is editor-only).
//             _isStaticObject = gameObject.isStatic;
//             RebuildMaterialData();
//         }
//
//         private void Reset()
//         {
//             _isStaticObject = gameObject.isStatic;
//         }
// #endif

        private void OnDisable()
        {
            RemoveQueue.Enqueue(new TargetRemoveEvent(instanceId));

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
