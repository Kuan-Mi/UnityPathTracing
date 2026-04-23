using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using Debug = UnityEngine.Debug;

namespace NativeRender
{
    /// <summary>
    /// Scene-resident resources mirroring NRDSample.cpp's resource taxonomy.
    ///
    /// Two modes (selected at construction time via <c>mergeBlas</c>):
    ///
    /// <b>Merged mode</b> (mergeBlas = true, default):
    ///     BLAS_MergedOpaque / MergedTransparent / MergedEmissive – one merged
    ///     pre-transformed VB/IB per category, identity TLAS transform.
    ///
    /// <b>Separate mode</b> (mergeBlas = false):
    ///     One BLAS per MeshRenderer, using the mesh's native VB/IB.
    ///     TLAS transform = localToWorldMatrix; InstanceID set so that
    ///     InstanceID() + GeometryIndex() indexes into _instanceDataBuf.
    ///     Transform-only changes skip BLAS rebuild (only TLAS transform +
    ///     mOverloadedMatrix in the instance buffer are updated).
    ///
    /// Binding names (<c>BindToShader</c>) exactly match NRDSample.cpp.
    /// </summary>
    public sealed class NRDSampleResource : IDisposable
    {
        // Matches "#define SHARC_CAPACITY ( 1 << 23 )" in Shared.hlsl.
        public const int SharcCapacity = 1 << 23;

        // Matches "constexpr uint32_t TEXTURES_PER_MATERIAL = 4;" in NRDSample.cpp.
        public const int TexturesPerMaterial = 4;

        // Matches "#define FLAG_FIRST_BIT 24" in Shared.hlsl.
        public const int FlagFirstBit = 24;

        // Scene flag constants (Shared.hlsl).
        public const uint FLAG_NON_TRANSPARENT = 0x01;
        public const uint FLAG_TRANSPARENT     = 0x02;
        public const uint FLAG_EMISSIVE        = 0x04;
        public const uint FLAG_STATIC          = 0x08;

        private const uint kHandleOpaque      = 0xFFFF0001u;
        private const uint kHandleTransparent = 0xFFFF0002u;
        private const uint kHandleEmissive    = 0xFFFF0003u;

        // ----- Mode -----
        private bool _mergeBlas;

        /// <summary>
        /// Switches between merged-BLAS and separate-BLAS mode at runtime.
        /// Setting this triggers a full scene rebuild on the next <see cref="UpdateForFrame"/> call.
        /// </summary>
        public bool MergeBlas
        {
            get => _mergeBlas;
            set
            {
                if (_mergeBlas == value) return;
                _mergeBlas  = value;
                _sceneDirty = true;
            }
        }

        // Identity 3x4 row-major transform (12 floats).
        private static readonly float[] kIdentity3x4 =
        {
            1f, 0f, 0f, 0f,
            0f, 1f, 0f, 0f,
            0f, 0f, 1f, 0f,
        };

        // ----- Acceleration structures -----
        private RayTracingAccelerationStructure _worldAS; // gWorldTlas
        private RayTracingAccelerationStructure _lightAS; // gLightTlas

        // ----- Merged BLAS resources -----
        private sealed class MergedBlas : IDisposable
        {
            public GraphicsBuffer vb; // float3 world-space positions, stride = 12
            public GraphicsBuffer ib; // uint32 indices,               stride = 4

            public uint vertexCount;

            // Per-submesh records (one entry per submesh of every target in the BLAS).
            public NativeRenderPlugin.SubmeshDesc[] submeshDescs;

            public void Dispose()
            {
                vb?.Release();
                vb = null;
                ib?.Release();
                ib = null;
            }
        }

        private MergedBlas _blasOpaque;
        private MergedBlas _blasTransparent;
        private MergedBlas _blasEmissive;

        // ----- Scene structured buffers -----
        private GraphicsBuffer _instanceDataBuf; // gIn_InstanceData
        private GraphicsBuffer _primitiveDataBuf; // gIn_PrimitiveData
        private GraphicsBuffer _morphPrimitivePositionsPrevBuf; // gIn_MorphPrimitivePositionsPrev (stub)

        // ----- SHARC UAV ring buffers -----
        private GraphicsBuffer _sharcHashEntries; // gInOut_SharcHashEntriesBuffer
        private GraphicsBuffer _sharcAccumulated; // gInOut_SharcAccumulated
        private GraphicsBuffer _sharcResolved; // gInOut_SharcResolved

        // ----- Material texture array (gIn_Textures) -----
        private BindlessTexture _textures;

        // ----- CPU mirrors -----
        private InstanceDataNRD[]             _instanceCpu;
        private NativeArray<PrimitiveDataNRD> _primitiveCpu;

        // ----- Separate-BLAS per-target tracking -----

        /// <summary>Tracks per-renderer state when running in separate (non-merged) BLAS mode.</summary>
        private sealed class PerTargetBlas
        {
            /// <summary>Which TLAS this renderer belongs to (worldAS or lightAS).</summary>
            public RayTracingAccelerationStructure tlas;

            /// <summary>Index of this renderer's first submesh in _instanceDataBuf (= first contiguous slot).</summary>
            public uint firstInstanceDataIndex;

            /// <summary>Number of submeshes (= number of InstanceDataNRD entries).</summary>
            public int submeshCount;

            /// <summary>Cached transform to detect changes.</summary>
            public Matrix4x4 lastTransform;

            /// <summary>Starting element index in _primitiveCpu for each submesh.</summary>
            public uint[] primitiveOffsets;

            /// <summary>Triangle count for each submesh (matches primitiveOffsets length).</summary>
            public int[] primitiveCounts;

            /// <summary>Material for each submesh — used to release material refs when the target is removed.</summary>
            public Material[] submeshMaterials;
        }

        // Keyed by MeshRenderer.GetInstanceID()
        private readonly Dictionary<int, PerTargetBlas> _perTargetBlas = new();

        // ----- Separate-BLAS incremental slot management -----
        // One allocator per buffer; both track element-level free/used ranges.
        private readonly PrimitiveSlotAllocator _instanceAlloc = new PrimitiveSlotAllocator { Name = "InstanceAlloc" };
        private readonly PrimitiveSlotAllocator _primAlloc     = new PrimitiveSlotAllocator { Name = "PrimitiveAlloc" };

        // Fragmentation thresholds: trigger MarkRebuildDirty when both conditions are met.
        private const float kFragThreshold    = 0.5f;
        private const uint  kFragMinFreeCount = 10_000;

        // ----- Tracking for dirty-detection -----
        private readonly Dictionary<Material, int> _materialSlots = new();

        // Reference counts per material slot (how many submeshes reference it).
        private readonly Dictionary<Material, int> _materialRefCounts = new();

        // Freed material slot indices available for reuse (each slot = TexturesPerMaterial descriptors).
        private readonly Queue<int> _freeMatSlots = new();

        private bool _sceneDirty = true;
        private bool _disposed;

        public RayTracingAccelerationStructure WorldAS => _worldAS;
        public RayTracingAccelerationStructure LightAS => _lightAS;

        public GraphicsBuffer InstanceDataBuf => _instanceDataBuf;
        public GraphicsBuffer PrimitiveDataBuf => _primitiveDataBuf;
        public GraphicsBuffer MorphPrimitivePositionsPrevBuf => _morphPrimitivePositionsPrevBuf;
        public BindlessTexture Textures => _textures;

        public NRDSampleResource(bool mergeBlas = true)
        {
            _mergeBlas = mergeBlas;
            _worldAS   = new RayTracingAccelerationStructure();
            _lightAS   = new RayTracingAccelerationStructure();
            AllocateStaticResources();
        }

        public void MarkRebuildDirty() => _sceneDirty = true;

        /// <summary>Dirty detection + full scene rebuild when needed.</summary>
        public void UpdateForFrame()
        {
            var targets = NativeRayTracingTarget.All;

            if (_mergeBlas)
            {
                // Merged mode: any structural change needs a full rebuild.
                // Drain both queues without inspecting contents.
                if (NativeRayTracingTarget.RemoveQueue.Count > 0 || NativeRayTracingTarget.AddQueue.Count > 0)
                {
                    NativeRayTracingTarget.RemoveQueue.Clear();
                    NativeRayTracingTarget.AddQueue.Clear();
                    _sceneDirty = true;
                }
            }

            if (_sceneDirty)
            {
                RebuildScene(targets);
                foreach (var t in targets)
                    if (t != null)
                        t.transform.hasChanged = false;
                _sceneDirty = false;
                
                NativeRayTracingTarget.RemoveQueue.Clear();
                NativeRayTracingTarget.AddQueue.Clear();
                return;
            }

            // Consume all pending Add/Remove events submitted by NativeRayTracingTarget lifecycle callbacks.
            DrainChangeQueue();

            if (AnyTransformChanged(targets))
            {
                if (_mergeBlas)
                {
                    // Merged mode: vertices are pre-transformed → must rebuild geometry.
                    // But textures/materials are unchanged, so we can preserve them.
                    RebuildScene(targets, preserveTextures: true);
                }
                else
                {
                    // Separate mode: only update TLAS transforms + mOverloadedMatrix.
                    UpdateTransformsOnly(targets);
                }

                foreach (var t in targets)
                    if (t != null)
                        t.transform.hasChanged = false;
            }
        }

        /// <summary>
        /// Drains all pending <see cref="TargetChange"/> events from
        /// <see cref="NativeRayTracingTarget.ChangeQueue"/> and applies them.
        /// <para>In merged mode, any structural change sets <see cref="_sceneDirty"/>.</para>
        /// <para>In separate mode, additions and removals are applied incrementally.</para>
        /// </summary>
        private void DrainChangeQueue()
        {
            // Separate mode — process removals first so freed slots can be reused by additions.
            while (NativeRayTracingTarget.RemoveQueue.Count > 0)
            {
                var ev = NativeRayTracingTarget.RemoveQueue.Dequeue();
                RemoveTargetIncremental(ev.RendererInstanceId);
            }

            while (NativeRayTracingTarget.AddQueue.Count > 0)
            {
                var ev = NativeRayTracingTarget.AddQueue.Dequeue();

                // Target or Renderer may be null if the object was destroyed before we consumed the event.
                if (ev.Target == null || ev.Renderer == null) continue;

                // New materials are handled incrementally by GetOrAddMaterial (called inside
                // AddTargetIncremental), which grows _textures in-place if needed — no full rebuild.

                Material rep           = GetRepresentativeMaterial(ev.Renderer);
                bool     isTransparent = IsMaterialTransparent(rep);
                bool     isEmissive    = IsMaterialEmissive(rep);

                uint worldFlags = FLAG_STATIC | (isTransparent ? FLAG_TRANSPARENT : FLAG_NON_TRANSPARENT);
                AddTargetIncremental(ev.Target, _worldAS, worldFlags);

                if (isEmissive)
                    AddTargetIncremental(ev.Target, _lightAS, FLAG_STATIC | FLAG_NON_TRANSPARENT | FLAG_EMISSIVE);
            }

            // Auto-schedule full rebuild when fragmentation becomes excessive.
            if (_primAlloc.FragmentationRatio > kFragThreshold &&
                _primAlloc.TotalFreeCount > kFragMinFreeCount)
            {
                Debug.Log($"[NRDSampleResource] Primitive buffer fragmentation {_primAlloc.FragmentationRatio:P0} — scheduling full rebuild.");
                _sceneDirty = true;
            }
        }

        /// <summary>Build / update both TLASes (call inside a CommandBuffer).</summary>
        public void BuildAccelerationStructures(CommandBuffer cmd)
        {
            _worldAS.BuildOrUpdate(cmd);
            _lightAS.BuildOrUpdate(cmd);
        }

        /// <summary>Bind all scene GPU resources to a ray tracing pipeline using NRDSample names.</summary>
        public void BindToShader(RayTracePipeline pipeline)
        {
            if (pipeline == null || !pipeline.IsValid) return;

            pipeline.SetAccelerationStructure("gWorldTlas", _worldAS);
            pipeline.SetAccelerationStructure("gLightTlas", _lightAS);

            pipeline.SetStructuredBuffer("gIn_InstanceData", _instanceDataBuf);
            pipeline.SetStructuredBuffer("gIn_PrimitiveData", _primitiveDataBuf);
            pipeline.SetStructuredBuffer("gIn_MorphPrimitivePositionsPrev", _morphPrimitivePositionsPrevBuf);

            pipeline.SetRWStructuredBuffer("gInOut_SharcHashEntriesBuffer", _sharcHashEntries);
            pipeline.SetRWStructuredBuffer("gInOut_SharcAccumulated", _sharcAccumulated);
            pipeline.SetRWStructuredBuffer("gInOut_SharcResolved", _sharcResolved);

            pipeline.SetBindlessTexture("gIn_Textures", _textures);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            DisposeSceneGpuBuffers();
            DisposeStaticResources();

            _worldAS?.Dispose();
            _worldAS = null;
            _lightAS?.Dispose();
            _lightAS = null;
        }

        // =====================================================================
        // Static resources
        // =====================================================================

        private void AllocateStaticResources()
        {
            const GraphicsBuffer.Target t = GraphicsBuffer.Target.Structured;
            _sharcHashEntries = new GraphicsBuffer(t, SharcCapacity, sizeof(ulong));
            _sharcAccumulated = new GraphicsBuffer(t, SharcCapacity, sizeof(uint) * 4);
            _sharcResolved    = new GraphicsBuffer(t, SharcCapacity, sizeof(uint) * 4);

            _morphPrimitivePositionsPrevBuf = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured, 1, Marshal.SizeOf<MorphPrimitivePositionsNRD>());
            _morphPrimitivePositionsPrevBuf.SetData(new MorphPrimitivePositionsNRD[1]);
        }

        private void DisposeStaticResources()
        {
            _sharcHashEntries?.Release();
            _sharcHashEntries = null;
            _sharcAccumulated?.Release();
            _sharcAccumulated = null;
            _sharcResolved?.Release();
            _sharcResolved = null;
            _morphPrimitivePositionsPrevBuf?.Release();
            _morphPrimitivePositionsPrevBuf = null;
        }

        // =====================================================================
        // Dynamic scene GPU data + merged BLAS construction
        // =====================================================================

        private void DisposeSceneGpuBuffers(bool preserveTextures = false)
        {
            _instanceDataBuf?.Release();
            _instanceDataBuf = null;
            _primitiveDataBuf?.Release();
            _primitiveDataBuf = null;

            if (!preserveTextures)
            {
                _textures?.Dispose();
                _textures = null;
            }

            _blasOpaque?.Dispose();
            _blasOpaque = null;
            _blasTransparent?.Dispose();
            _blasTransparent = null;
            _blasEmissive?.Dispose();
            _blasEmissive = null;

            _perTargetBlas.Clear();

            _instanceCpu = null;
            if (_primitiveCpu.IsCreated)
            {
                _primitiveCpu.Dispose();
                _primitiveCpu = default;
            }

            _instanceAlloc.Reset(0);
            _primAlloc.Reset(0);

            if (!preserveTextures)
            {
                _materialSlots.Clear();
                _materialRefCounts.Clear();
                _freeMatSlots.Clear();
            }
        }

        private static bool AnyTransformChanged(IReadOnlyList<NativeRayTracingTarget> targets)
        {
            for (int i = 0; i < targets.Count; i++)
            {
                var t = targets[i];
                if (t == null) continue;
                if (t.transform.hasChanged) return true;
            }

            return false;
        }

        /// <summary>
        /// Classifies targets into opaque / transparent / emissive, builds one
        /// merged pre-transformed VB/IB per category, uploads instance &amp;
        /// primitive data, and registers each merged BLAS as a single TLAS
        /// instance with identity transform (matching NRDSample).
        /// </summary>
        private void RebuildScene(IReadOnlyList<NativeRayTracingTarget> targets, bool preserveTextures = false)
        {
            DisposeSceneGpuBuffers(preserveTextures);

            _worldAS?.Clear();
            _lightAS?.Clear();

            // Bucket targets by category.
            var opaque      = new List<NativeRayTracingTarget>();
            var transparent = new List<NativeRayTracingTarget>();
            var emissive    = new List<NativeRayTracingTarget>();

            foreach (var t in targets)
            {
                if (t == null) continue;
                var mr = t.GetComponent<MeshRenderer>();
                if (mr == null) continue;
                var mf = mr.GetComponent<MeshFilter>();
                if (mf == null || mf.sharedMesh == null) continue;

                Material rep           = GetRepresentativeMaterial(mr);
                bool     isTransparent = IsMaterialTransparent(rep);
                bool     isEmissive    = IsMaterialEmissive(rep);

                if (isTransparent) transparent.Add(t);
                else opaque.Add(t);

                if (isEmissive) emissive.Add(t);
            }

            // Running primitive / instance offsets (NRDSample orders opaque → transparent → emissive).
            uint primitiveCursor = 0;
            uint instanceCursor  = 0;


            if (_mergeBlas)
            {
                var instList = new List<InstanceDataNRD>();
                var texPtrs  = new List<IntPtr>();

                // Pre-allocate _primitiveCpu so Burst jobs write directly into it.
                int totalPrimsMerged = CountGroupTriangles(opaque)
                                       + CountGroupTriangles(transparent)
                                       + CountGroupTriangles(emissive);
                _primitiveCpu = new NativeArray<PrimitiveDataNRD>(
                    Mathf.Max(totalPrimsMerged, 1), Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

                uint opaqueFirstInstance = instanceCursor;
                _blasOpaque = BuildMergedBlas(opaque, ref instanceCursor, ref primitiveCursor,
                    instList, _primitiveCpu, texPtrs,
                    FLAG_STATIC | FLAG_NON_TRANSPARENT);

                uint transparentFirstInstance = instanceCursor;
                _blasTransparent = BuildMergedBlas(transparent, ref instanceCursor, ref primitiveCursor,
                    instList, _primitiveCpu, texPtrs,
                    FLAG_STATIC | FLAG_TRANSPARENT);

                uint emissiveFirstInstance = instanceCursor;
                _blasEmissive = BuildMergedBlas(emissive, ref instanceCursor, ref primitiveCursor,
                    instList, _primitiveCpu, texPtrs,
                    FLAG_STATIC | FLAG_NON_TRANSPARENT | FLAG_EMISSIVE);

                // Texture array — skip when preserving existing textures (transform-only rebuild).
                if (!preserveTextures)
                {
                    int texCount = Mathf.Max(texPtrs.Count, 1);
                    _textures = new BindlessTexture(texCount);
                    for (int i = 0; i < texPtrs.Count; i++)
                        _textures.SetNativePtr(i, texPtrs[i]);
                }

                // Instance / primitive GPU buffers.
                if (instList.Count == 0) instList.Add(default);
                _instanceCpu = instList.ToArray();

                _instanceDataBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                    _instanceCpu.Length, Marshal.SizeOf<InstanceDataNRD>());
                _instanceDataBuf.SetData(_instanceCpu);

                _primitiveDataBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                    _primitiveCpu.Length, Marshal.SizeOf<PrimitiveDataNRD>());
                _primitiveDataBuf.SetData(_primitiveCpu);

                // Register each merged BLAS with the appropriate TLAS.
                if (_blasOpaque != null)
                    RegisterMergedBlas(_worldAS, _blasOpaque, kHandleOpaque,
                        opaqueFirstInstance, (byte)FLAG_NON_TRANSPARENT);
                if (_blasTransparent != null)
                    RegisterMergedBlas(_worldAS, _blasTransparent, kHandleTransparent,
                        transparentFirstInstance, (byte)FLAG_TRANSPARENT);
                if (_blasEmissive != null)
                    RegisterMergedBlas(_lightAS, _blasEmissive, kHandleEmissive,
                        emissiveFirstInstance, (byte)FLAG_NON_TRANSPARENT);
            }
            else
            {
                BuildSeparateBlas(opaque, transparent, emissive);
            }
        }

        /// <summary>
        /// Non-merged path: one BLAS per MeshRenderer (using the mesh's native VB/IB).
        /// TLAS transform = localToWorldMatrix. InstanceID = firstSubmeshIndex in _instanceDataBuf.
        /// </summary>
        // Count triangles for a group without building full validPairs — used to pre-allocate _primitiveCpu.
        private static int CountGroupTriangles(List<NativeRayTracingTarget> group)
        {
            int count = 0;
            foreach (var t in group)
            {
                if (t == null) continue;
                var mf = t.GetComponent<MeshFilter>();
                if (mf == null || mf.sharedMesh == null) continue;
                Mesh mesh = mf.sharedMesh;
                for (int s = 0; s < mesh.subMeshCount; s++)
                    count += (int)mesh.GetIndexCount(s) / 3;
            }

            return count;
        }

        private void BuildSeparateBlas(
            List<NativeRayTracingTarget> opaque,
            List<NativeRayTracingTarget> transparent,
            List<NativeRayTracingTarget> emissive
        )
        {
            uint                  instanceCursor  = 0;
            uint                  primitiveCursor = 0;
            List<InstanceDataNRD> instList        = new List<InstanceDataNRD>();
            List<IntPtr>          texPtrs         = new List<IntPtr>();

            // Pre-allocate _primitiveCpu so jobs can write directly into it.
            int totalPrims = CountGroupTriangles(opaque)
                             + CountGroupTriangles(transparent)
                             + CountGroupTriangles(emissive);
            _primitiveCpu = new NativeArray<PrimitiveDataNRD>(
                Mathf.Max(totalPrims, 1), Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

            // Process ordered opaque → transparent → emissive (emissive targets were also added to
            // opaque already; they get a second separate entry in the light TLAS).
            ProcessSeparateGroup(opaque, _worldAS, FLAG_STATIC | FLAG_NON_TRANSPARENT,
                ref instanceCursor, ref primitiveCursor, instList, texPtrs);
            ProcessSeparateGroup(transparent, _worldAS, FLAG_STATIC | FLAG_TRANSPARENT,
                ref instanceCursor, ref primitiveCursor, instList, texPtrs);
            ProcessSeparateGroup(emissive, _lightAS, FLAG_STATIC | FLAG_NON_TRANSPARENT | FLAG_EMISSIVE,
                ref instanceCursor, ref primitiveCursor, instList, texPtrs);

            // Texture array.
            int texCount = Mathf.Max(texPtrs.Count, 1);
            _textures = new BindlessTexture(texCount);
            for (int i = 0; i < texPtrs.Count; i++)
                _textures.SetNativePtr(i, texPtrs[i]);

            // Instance / primitive GPU buffers.
            if (instList.Count == 0) instList.Add(default);
            _instanceCpu = instList.ToArray();

            _instanceDataBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                _instanceCpu.Length, Marshal.SizeOf<InstanceDataNRD>());
            _instanceDataBuf.SetData(_instanceCpu);

            _primitiveDataBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                _primitiveCpu.Length, Marshal.SizeOf<PrimitiveDataNRD>());
            _primitiveDataBuf.SetData(_primitiveCpu);

            // Initialize slot allocators to reflect the just-built fully-packed layout.
            _instanceAlloc.ResetFullyAllocated(_instanceCpu.Length);
            _primAlloc.ResetFullyAllocated(Mathf.Max(totalPrims, 1));
        }

        private void ProcessSeparateGroup(
            List<NativeRayTracingTarget> group,
            RayTracingAccelerationStructure tlas,
            uint baseFlags,
            ref uint instanceCursor,
            ref uint primitiveCursor,
            List<InstanceDataNRD> instList,
            List<IntPtr> texPtrs)
        {
            var swTotal = Stopwatch.StartNew();

            // First pass: collect valid pairs and build mesh list.
            var validPairs    = new List<(NativeRayTracingTarget target, MeshRenderer mr, Mesh mesh, int meshIndex)>(group.Count);
            var meshList      = new List<Mesh>(group.Count);
            int totalTriCount = 0;

            foreach (var t in group)
            {
                if (t == null) continue;
                var mr = t.GetComponent<MeshRenderer>();
                if (mr == null) continue;
                var mf = mr.GetComponent<MeshFilter>();
                if (mf == null || mf.sharedMesh == null) continue;

                Mesh mesh = mf.sharedMesh;
                for (int s = 0; s < mesh.subMeshCount; s++)
                    totalTriCount += (int)mesh.GetIndexCount(s) / 3;

                validPairs.Add((t, mr, mesh, meshList.Count));
                meshList.Add(mesh);
            }

            if (validPairs.Count == 0)
            {
                swTotal.Stop();
                Debug.Log($"[ProcessSeparateGroup] Total: {swTotal.Elapsed.TotalMilliseconds:F2} ms  (0 renderers, flags=0x{baseFlags:X})");
                return;
            }

            // Single AcquireReadOnlyMeshData call for all meshes.
            var meshDataArr = Mesh.AcquireReadOnlyMeshData(meshList);

            // Per-job tracking — kept alive until all jobs Complete().
            var jobHandles = new List<JobHandle>(validPairs.Count * 4);
            var tempArrays = new List<System.IDisposable>(validPairs.Count * 5);

            foreach (var (target, mr, mesh, meshIndex) in validPairs)
            {
                Matrix4x4 xform = target.transform.localToWorldMatrix;

                // TLAS calls must happen on the main thread.
                if (!tlas.AddInstance(mr))
                {
                    Debug.LogWarning($"[NRDSampleResource] AddInstance failed for '{mr.name}' — skipping");
                    continue;
                }

                uint firstInstanceDataIndex = instanceCursor;
                tlas.SetInstanceID(mr, firstInstanceDataIndex);
                tlas.SetInstanceTransform(mr, xform);
                tlas.SetInstanceMask(mr, GetMaskForFlags(baseFlags));

                bool leftHanded = xform.determinant < 0f;
                Vector3 sc = new Vector3(
                    new Vector3(xform.m00, xform.m10, xform.m20).magnitude,
                    new Vector3(xform.m01, xform.m11, xform.m21).magnitude,
                    new Vector3(xform.m02, xform.m12, xform.m22).magnitude);
                float scaleMax = Mathf.Max(sc.x, Mathf.Max(sc.y, sc.z));

                int vertCount = mesh.vertexCount;
                int subCnt    = mesh.subMeshCount;
                var meshData  = meshDataArr[meshIndex];

                // Vertex attribute arrays — TempJob so they stay valid across job scheduling.
                var nativePos = new NativeArray<Vector3>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
                var nativeN   = new NativeArray<Vector3>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
                var nativeT   = new NativeArray<Vector4>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
                var nativeUV  = new NativeArray<Vector2>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

                meshData.GetVertices(nativePos);

                bool hasN  = meshData.HasVertexAttribute(VertexAttribute.Normal);
                bool hasT  = meshData.HasVertexAttribute(VertexAttribute.Tangent);
                bool hasUV = meshData.HasVertexAttribute(VertexAttribute.TexCoord0);

                if (hasN) meshData.GetNormals(nativeN);
                if (hasT) meshData.GetTangents(nativeT);
                if (hasUV) meshData.GetUVs(0, nativeUV);

                // Reinterpret vertex buffers as Unity.Mathematics types for the Burst job.
                var posF3 = nativePos.Reinterpret<float3>(sizeof(float) * 3);
                var norF3 = nativeN.Reinterpret<float3>(sizeof(float) * 3);
                var tanF4 = nativeT.Reinterpret<float4>(sizeof(float) * 4);
                var uvF2  = nativeUV.Reinterpret<float2>(sizeof(float) * 2);

                // Track for disposal after Complete().
                tempArrays.Add(nativePos);
                tempArrays.Add(nativeN);
                tempArrays.Add(nativeT);
                tempArrays.Add(nativeUV);

                Material[] sharedMaterials = mr.sharedMaterials;
                var        subPrimOffsets  = new uint[subCnt];
                var        subPrimCounts   = new int[subCnt];
                var        subMats         = new Material[subCnt];

                for (int sub = 0; sub < subCnt; sub++)
                {
                    uint primitiveOffsetForSubMesh = primitiveCursor;

                    Material subMat = (sub < sharedMaterials.Length) ? sharedMaterials[sub] : GetRepresentativeMaterial(mr);
                    subMats[sub] = subMat;
                    int subMatIdx = GetOrAddMaterial(subMat, texPtrs);

                    int indexCount = (int)mesh.GetIndexCount(sub);
                    int triCount   = indexCount / 3;

                    subPrimOffsets[sub] = primitiveOffsetForSubMesh;
                    subPrimCounts[sub]  = triCount;

                    var nativeTris = new NativeArray<int>(indexCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
                    meshData.GetIndices(nativeTris, sub);
                    tempArrays.Add(nativeTris);

                    // Schedule Burst job for this submesh's triangle range.
                    // primitiveOffsetForSubMesh is the absolute offset into _primitiveCpu.
                    var job = new BuildPrimitivesJob
                    {
                        Indices   = nativeTris,
                        Positions = posF3,
                        Normals   = norF3,
                        Tangents  = tanF4,
                        UVs       = uvF2,
                        HasN      = hasN,
                        HasT      = hasT,
                        HasUV     = hasUV,
                        Output    = _primitiveCpu.GetSubArray((int)primitiveOffsetForSubMesh, triCount),
                    };
                    jobHandles.Add(job.Schedule(triCount, 64));

                    primitiveCursor += (uint)triCount;

                    // InstanceDataNRD is built on the main thread (depends on material + transform).
                    uint baseTextureIndex = (uint)(subMatIdx * TexturesPerMaterial);
                    var inst = new InstanceDataNRD
                    {
                        mOverloadedMatrix0 = new Vector4(xform.m00, xform.m01, xform.m02, xform.m03),
                        mOverloadedMatrix1 = new Vector4(xform.m10, xform.m11, xform.m12, xform.m13),
                        mOverloadedMatrix2 = new Vector4(xform.m20, xform.m21, xform.m22, xform.m23),

                        textureOffsetAndFlags = baseTextureIndex | (baseFlags << FlagFirstBit),
                        primitiveOffset       = primitiveOffsetForSubMesh,
                        scale                 = (leftHanded ? -1f : 1f) * scaleMax,
                        morphPrimitiveOffset  = 0,
                    };
                    EncodeMaterial(subMat, ref inst);
                    instList.Add(inst);
                    instanceCursor++;
                }

                // Record per-target state for transform-only updates.
                _perTargetBlas[(int)(uint)mr.GetInstanceID()] = new PerTargetBlas
                {
                    tlas                   = tlas,
                    firstInstanceDataIndex = firstInstanceDataIndex,
                    submeshCount           = subCnt,
                    lastTransform          = xform,
                    primitiveOffsets       = subPrimOffsets,
                    primitiveCounts        = subPrimCounts,
                    submeshMaterials       = subMats,
                };
            }

            // Wait for all Burst jobs to finish.
            var allHandles = new NativeArray<JobHandle>(jobHandles.Count, Allocator.Temp);
            for (int i = 0; i < jobHandles.Count; i++)
                allHandles[i] = jobHandles[i];
            JobHandle.CombineDependencies(allHandles).Complete();
            allHandles.Dispose();

            // Dispose all temporary vertex / index arrays.
            foreach (var arr in tempArrays)
                arr.Dispose();

            meshDataArr.Dispose();

            swTotal.Stop();
            Debug.Log($"[ProcessSeparateGroup] Total: {swTotal.Elapsed.TotalMilliseconds:F2} ms  ({validPairs.Count} renderers, {totalTriCount} tris, flags=0x{baseFlags:X})");
        }

        /// <summary>
        /// Transform-only update for separate-BLAS mode.
        /// Updates TLAS transforms and mOverloadedMatrix in the instance buffer;
        /// does not rebuild any BLAS or re-upload primitive data.
        /// </summary>
        private void UpdateTransformsOnly(IReadOnlyList<NativeRayTracingTarget> targets)
        {
            if (_instanceCpu == null) return;

            bool anyChanged = false;
            foreach (var target in targets)
            {
                if (target == null) continue;
                var mr = target.GetComponent<MeshRenderer>();
                if (mr == null) continue;

                int key = mr.GetInstanceID();
                if (!_perTargetBlas.TryGetValue(key, out var info)) continue;

                Matrix4x4 xform = target.transform.localToWorldMatrix;
                if (xform == info.lastTransform) continue;

                // Update TLAS transform.
                info.tlas.SetInstanceTransform(mr, xform);

                // Recompute per-submesh scale/sign for mOverloadedMatrix.
                Vector3 s = new Vector3(
                    new Vector3(xform.m00, xform.m10, xform.m20).magnitude,
                    new Vector3(xform.m01, xform.m11, xform.m21).magnitude,
                    new Vector3(xform.m02, xform.m12, xform.m22).magnitude);
                float scaleMax = Mathf.Max(s.x, Mathf.Max(s.y, s.z));

                // Patch mOverloadedMatrix and scale in every submesh InstanceDataNRD entry.
                for (int sub = 0; sub < info.submeshCount; sub++)
                {
                    int idx = (int)info.firstInstanceDataIndex + sub;
                    if (idx >= _instanceCpu.Length) break;

                    _instanceCpu[idx].mOverloadedMatrix0 = new Vector4(xform.m00, xform.m01, xform.m02, xform.m03);
                    _instanceCpu[idx].mOverloadedMatrix1 = new Vector4(xform.m10, xform.m11, xform.m12, xform.m13);
                    _instanceCpu[idx].mOverloadedMatrix2 = new Vector4(xform.m20, xform.m21, xform.m22, xform.m23);
                    _instanceCpu[idx].scale              = scaleMax;
                }

                info.lastTransform = xform;
                anyChanged         = true;
            }

            if (anyChanged)
                _instanceDataBuf?.SetData(_instanceCpu);
        }

        // =====================================================================
        // Incremental add / remove (separate-BLAS mode only)
        // =====================================================================

        /// <summary>
        /// Diffs <paramref name="current"/> against <see cref="_registeredTargets"/>, removes
        /// <summary>
        /// Removes a single target's TLAS instance and returns its GPU slots to the free pools.
        /// </summary>
        private void RemoveTargetIncremental(int rendererInstanceId)
        {
            if (!_perTargetBlas.TryGetValue(rendererInstanceId, out var info)) return;

            // Use the MeshRenderer stored at registration time.
            info.tlas.RemoveInstance(rendererInstanceId);

            // Return contiguous instance slots.
            _instanceAlloc.Free(info.firstInstanceDataIndex, info.submeshCount);
            for (int sub = 0; sub < info.submeshCount; sub++)
            {
                int slotIdx = (int)info.firstInstanceDataIndex + sub;
                if (slotIdx < _instanceCpu.Length)
                    _instanceCpu[slotIdx] = default; // zero-out tombstone
            }

            // Return per-submesh primitive slots.
            if (info.primitiveOffsets != null)
                for (int sub = 0; sub < info.primitiveOffsets.Length; sub++)
                    _primAlloc.Free(info.primitiveOffsets[sub], info.primitiveCounts[sub]);

            // Release material reference counts; free slots when count reaches zero.
            if (info.submeshMaterials != null)
                foreach (var mat in info.submeshMaterials)
                    ReleaseMaterial(mat);

            _perTargetBlas.Remove(rendererInstanceId);

            // Upload zeroed instance slots. Partial upload over the contiguous block.
            _instanceDataBuf?.SetData(_instanceCpu,
                (int)info.firstInstanceDataIndex,
                (int)info.firstInstanceDataIndex,
                info.submeshCount);
        }

        /// <summary>
        /// Adds a single target to <paramref name="tlas"/> without touching any other renderer.
        /// Allocates contiguous instance and primitive slots, grows backing buffers if needed,
        /// schedules a Burst job for primitive data, then partial-uploads changed ranges to the GPU.
        /// </summary>
        private void AddTargetIncremental(
            NativeRayTracingTarget target,
            RayTracingAccelerationStructure tlas,
            uint baseFlags)
        {
            if (target == null) return;
            var mr = target.GetComponent<MeshRenderer>();
            if (mr == null) return;
            var mf = mr.GetComponent<MeshFilter>();
            if (mf == null || mf.sharedMesh == null) return;

            Mesh mesh   = mf.sharedMesh;
            int  subCnt = mesh.subMeshCount;

            if (!tlas.AddInstance(mr))
            {
                Debug.LogWarning($"[NRDSampleResource] AddTargetIncremental: AddInstance failed for '{mr.name}'");
                return;
            }

            Matrix4x4 xform = target.transform.localToWorldMatrix;
            tlas.SetInstanceTransform(mr, xform);
            tlas.SetInstanceMask(mr, GetMaskForFlags(baseFlags));

            bool leftHanded = xform.determinant < 0f;
            Vector3 sc = new Vector3(
                new Vector3(xform.m00, xform.m10, xform.m20).magnitude,
                new Vector3(xform.m01, xform.m11, xform.m21).magnitude,
                new Vector3(xform.m02, xform.m12, xform.m22).magnitude);
            float scaleMax = Mathf.Max(sc.x, Mathf.Max(sc.y, sc.z));

            // ------------------------------------------------------------------
            // Allocate a contiguous block of instance slots.
            // ------------------------------------------------------------------
            uint instBase = _instanceAlloc.Allocate(subCnt);
            if (instBase == PrimitiveSlotAllocator.InvalidOffset)
            {
                int newInstCap = Mathf.Max((int)_instanceAlloc.Capacity * 2,
                    (int)_instanceAlloc.Capacity + subCnt);
                EnsureInstanceCapacity(newInstCap);
                instBase = _instanceAlloc.Allocate(subCnt);
            }

            tlas.SetInstanceID(mr, instBase);

            // ------------------------------------------------------------------
            // Build per-submesh primitive + instance data.
            // ------------------------------------------------------------------
            var subPrimOffsets = new uint[subCnt];
            var subPrimCounts  = new int[subCnt];

            var meshDataArr = Mesh.AcquireReadOnlyMeshData(mesh);
            var meshData    = meshDataArr[0];

            int vertCount = mesh.vertexCount;
            var nativePos = new NativeArray<Vector3>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
            var nativeN   = new NativeArray<Vector3>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
            var nativeT   = new NativeArray<Vector4>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
            var nativeUV  = new NativeArray<Vector2>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

            meshData.GetVertices(nativePos);
            bool hasN  = meshData.HasVertexAttribute(VertexAttribute.Normal);
            bool hasT  = meshData.HasVertexAttribute(VertexAttribute.Tangent);
            bool hasUV = meshData.HasVertexAttribute(VertexAttribute.TexCoord0);
            if (hasN) meshData.GetNormals(nativeN);
            if (hasT) meshData.GetTangents(nativeT);
            if (hasUV) meshData.GetUVs(0, nativeUV);

            var posF3 = nativePos.Reinterpret<float3>(sizeof(float) * 3);
            var norF3 = nativeN.Reinterpret<float3>(sizeof(float) * 3);
            var tanF4 = nativeT.Reinterpret<float4>(sizeof(float) * 4);
            var uvF2  = nativeUV.Reinterpret<float2>(sizeof(float) * 2);

            var jobHandles = new List<JobHandle>(subCnt);
            var tempTris   = new List<NativeArray<int>>(subCnt);

            Material[] sharedMaterials = mr.sharedMaterials;
            var        subMaterials    = new Material[subCnt];

            // Pre-calculate total tri count across all submeshes and grow _primitiveCpu
            // once before scheduling any Burst jobs. Growing inside the loop would
            // Dispose the NativeArray while already-scheduled jobs still hold pointers
            // into it, causing NullReferenceException on worker threads.
            {
                int totalTriCount = 0;
                for (int s = 0; s < subCnt; s++)
                    totalTriCount += (int)(mesh.GetIndexCount(s) / 3);
                if (_primAlloc.TotalFreeCount < (uint)totalTriCount)
                {
                    int newPrimCap = Mathf.Max((int)_primAlloc.Capacity * 2,
                        (int)_primAlloc.Capacity + totalTriCount);
                    EnsurePrimitiveCapacity(newPrimCap);
                }
            }

            for (int sub = 0; sub < subCnt; sub++)
            {
                Material subMat = (sub < sharedMaterials.Length) ? sharedMaterials[sub] : GetRepresentativeMaterial(mr);
                subMaterials[sub] = subMat;
                // GetOrAddMaterial(null texPtrs) = incremental path: grows _textures in-place if needed.
                int subMatIdx = GetOrAddMaterial(subMat, null);

                int indexCount = (int)mesh.GetIndexCount(sub);
                int triCount   = indexCount / 3;

                // Allocate primitive slot — capacity is guaranteed above, should never fail.
                uint primOffset = _primAlloc.Allocate(triCount);

                subPrimOffsets[sub] = primOffset;
                subPrimCounts[sub]  = triCount;

                var nativeTris = new NativeArray<int>(indexCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
                meshData.GetIndices(nativeTris, sub);
                tempTris.Add(nativeTris);

                jobHandles.Add(new BuildPrimitivesJob
                {
                    Indices   = nativeTris,
                    Positions = posF3,
                    Normals   = norF3,
                    Tangents  = tanF4,
                    UVs       = uvF2,
                    HasN      = hasN,
                    HasT      = hasT,
                    HasUV     = hasUV,
                    Output    = _primitiveCpu.GetSubArray((int)primOffset, triCount),
                }.Schedule(triCount, 64));

                uint instSlot         = instBase + (uint)sub;
                uint baseTextureIndex = (uint)(subMatIdx * TexturesPerMaterial);
                _instanceCpu[instSlot] = new InstanceDataNRD
                {
                    mOverloadedMatrix0    = new Vector4(xform.m00, xform.m01, xform.m02, xform.m03),
                    mOverloadedMatrix1    = new Vector4(xform.m10, xform.m11, xform.m12, xform.m13),
                    mOverloadedMatrix2    = new Vector4(xform.m20, xform.m21, xform.m22, xform.m23),
                    textureOffsetAndFlags = baseTextureIndex | (baseFlags << FlagFirstBit),
                    primitiveOffset       = primOffset,
                    scale                 = (leftHanded ? -1f : 1f) * scaleMax,
                    morphPrimitiveOffset  = 0,
                };
                EncodeMaterial(subMat, ref _instanceCpu[instSlot]);
            }

            // Wait for all Burst jobs.
            var allHandles                                           = new NativeArray<JobHandle>(jobHandles.Count, Allocator.Temp);
            for (int i = 0; i < jobHandles.Count; i++) allHandles[i] = jobHandles[i];
            JobHandle.CombineDependencies(allHandles).Complete();
            allHandles.Dispose();

            foreach (var arr in tempTris) arr.Dispose();
            nativePos.Dispose();
            nativeN.Dispose();
            nativeT.Dispose();
            nativeUV.Dispose();
            meshDataArr.Dispose();

            // Partial GPU uploads — only the ranges we touched.
            _instanceDataBuf.SetData(_instanceCpu, (int)instBase, (int)instBase, subCnt);
            for (int sub = 0; sub < subCnt; sub++)
                _primitiveDataBuf.SetData(_primitiveCpu,
                    (int)subPrimOffsets[sub], (int)subPrimOffsets[sub], subPrimCounts[sub]);

            _perTargetBlas[mr.GetInstanceID()] = new PerTargetBlas
            {
                tlas                   = tlas,
                firstInstanceDataIndex = instBase,
                submeshCount           = subCnt,
                lastTransform          = xform,
                primitiveOffsets       = subPrimOffsets,
                primitiveCounts        = subPrimCounts,
                submeshMaterials       = subMaterials,
            };
        }

        /// <summary>
        /// Grows <see cref="_instanceCpu"/> and <see cref="_instanceDataBuf"/> to at least
        /// <paramref name="newCapacity"/> elements, and updates <see cref="_instanceAlloc"/>.
        /// </summary>
        private void EnsureInstanceCapacity(int newCapacity)
        {
            if (_instanceCpu != null && _instanceCpu.Length >= newCapacity) return;

            int cap = Mathf.Max(newCapacity, _instanceCpu != null ? _instanceCpu.Length * 2 : newCapacity);

            var newArr = new InstanceDataNRD[cap];
            if (_instanceCpu != null) Array.Copy(_instanceCpu, newArr, _instanceCpu.Length);
            _instanceCpu = newArr;

            _instanceAlloc.GrowTo(cap);

            _instanceDataBuf?.Release();
            _instanceDataBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                cap, Marshal.SizeOf<InstanceDataNRD>());
            _instanceDataBuf.SetData(_instanceCpu);
        }

        /// <summary>
        /// Grows <see cref="_primitiveCpu"/> and <see cref="_primitiveDataBuf"/> to at least
        /// <paramref name="newCapacity"/> elements, and updates <see cref="_primAlloc"/>.
        /// </summary>
        private void EnsurePrimitiveCapacity(int newCapacity)
        {
            if (_primitiveCpu.IsCreated && _primitiveCpu.Length >= newCapacity) return;

            int cap = Mathf.Max(newCapacity, _primitiveCpu.IsCreated ? _primitiveCpu.Length * 2 : newCapacity);

            var newArr = new NativeArray<PrimitiveDataNRD>(cap, Allocator.Persistent, NativeArrayOptions.ClearMemory);
            if (_primitiveCpu.IsCreated)
            {
                NativeArray<PrimitiveDataNRD>.Copy(_primitiveCpu, newArr, _primitiveCpu.Length);
                _primitiveCpu.Dispose();
            }

            _primitiveCpu = newArr;

            _primAlloc.GrowTo(cap);

            _primitiveDataBuf?.Release();
            _primitiveDataBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                cap, Marshal.SizeOf<PrimitiveDataNRD>());
            _primitiveDataBuf.SetData(_primitiveCpu);
        }

        // =====================================================================
        // Merged BLAS construction
        // =====================================================================

        /// <summary>
        /// Builds one merged BLAS for the given list of targets. Returns null when the list is empty.
        /// Writes one <see cref="InstanceDataNRD"/> per sub-instance and one
        /// <see cref="PrimitiveDataNRD"/> per triangle into <paramref name="instList"/> /
        /// <paramref name="primList"/>. Advances <paramref name="instanceCursor"/> and
        /// <paramref name="primitiveCursor"/> by the number of entries appended.
        /// </summary>
        private MergedBlas BuildMergedBlas(
            List<NativeRayTracingTarget> group,
            ref uint instanceCursor,
            ref uint primitiveCursor,
            List<InstanceDataNRD> instList,
            NativeArray<PrimitiveDataNRD> primOutput,
            List<IntPtr> texPtrs,
            uint baseFlags)
        {
            if (group.Count == 0) return null;

            // First pass – sum sizes and build mesh list for AcquireReadOnlyMeshData.
            var meshList     = new List<Mesh>(group.Count);
            var validTargets = new List<(NativeRayTracingTarget target, MeshRenderer mr)>(group.Count);
            int totalVerts   = 0, totalIndices = 0;
            foreach (var t in group)
            {
                if (t == null) continue;
                var mr = t.GetComponent<MeshRenderer>();
                if (mr == null) continue;
                var mf = mr.GetComponent<MeshFilter>();
                if (mf == null || mf.sharedMesh == null) continue;
                var mesh = mf.sharedMesh;
                totalVerts += mesh.vertexCount;
                for (int s = 0; s < mesh.subMeshCount; s++)
                    totalIndices += (int)mesh.GetIndexCount(s);
                validTargets.Add((t, mr));
                meshList.Add(mesh);
            }

            if (totalVerts == 0 || totalIndices == 0) return null;

            // Allocate merged VB/IB as NativeArrays so Burst jobs can write into them.
            var mergedPos = new NativeArray<float3>(totalVerts, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
            var mergedIdx = new NativeArray<uint>(totalIndices, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

            var meshDataArr  = Mesh.AcquireReadOnlyMeshData(meshList);
            var submeshDescs = new List<NativeRenderPlugin.SubmeshDesc>();
            var jobHandles   = new List<JobHandle>(validTargets.Count * 4);
            var tempArrays   = new List<System.IDisposable>(validTargets.Count * 5);

            int vertBase = 0, iBase = 0;

            for (int mi = 0; mi < validTargets.Count; mi++)
            {
                var (target, mr) = validTargets[mi];
                var mesh     = meshList[mi];
                var meshData = meshDataArr[mi];

                Matrix4x4 xform        = target.transform.localToWorldMatrix;
                Matrix4x4 normalMatrix = xform.inverse.transpose;

                int vertCount = mesh.vertexCount;

                // Local vertex attribute arrays (TempJob lifetime spans job scheduling).
                var localPos = new NativeArray<Vector3>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
                var localN   = new NativeArray<Vector3>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
                var localT   = new NativeArray<Vector4>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
                var localUV  = new NativeArray<Vector2>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

                meshData.GetVertices(localPos);
                bool hasN  = meshData.HasVertexAttribute(VertexAttribute.Normal);
                bool hasT  = meshData.HasVertexAttribute(VertexAttribute.Tangent);
                bool hasUV = meshData.HasVertexAttribute(VertexAttribute.TexCoord0);
                if (hasN) meshData.GetNormals(localN);
                if (hasT) meshData.GetTangents(localT);
                if (hasUV) meshData.GetUVs(0, localUV);

                var posF3 = localPos.Reinterpret<float3>(sizeof(float) * 3);
                var norF3 = localN.Reinterpret<float3>(sizeof(float) * 3);
                var tanF4 = localT.Reinterpret<float4>(sizeof(float) * 4);
                var uvF2  = localUV.Reinterpret<float2>(sizeof(float) * 2);

                tempArrays.Add(localPos);
                tempArrays.Add(localN);
                tempArrays.Add(localT);
                tempArrays.Add(localUV);

                // Transform positions into world space → merged VB slice.
                jobHandles.Add(new TransformVerticesJob
                {
                    LocalPositions = posF3,
                    LocalToWorld   = xform,
                    Output         = mergedPos.GetSubArray(vertBase, vertCount),
                }.Schedule(vertCount, 64));

                Material[] sharedMaterials = mr.sharedMaterials;
                int        subCnt          = mesh.subMeshCount;

                for (int sub = 0; sub < subCnt; sub++)
                {
                    uint primitiveOffsetForSubMesh = primitiveCursor;

                    Material subMat    = (sub < sharedMaterials.Length) ? sharedMaterials[sub] : GetRepresentativeMaterial(mr);
                    int      subMatIdx = GetOrAddMaterial(subMat, texPtrs);

                    int indexCount = (int)mesh.GetIndexCount(sub);
                    int triCount   = indexCount / 3;

                    var localTris = new NativeArray<int>(indexCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
                    meshData.GetIndices(localTris, sub);
                    tempArrays.Add(localTris);

                    // Record submesh descriptor (IB offset inside merged IB, before remapping).
                    submeshDescs.Add(new NativeRenderPlugin.SubmeshDesc
                    {
                        indexCount      = (uint)indexCount,
                        indexByteOffset = (uint)(iBase * sizeof(uint)),
                    });

                    // Remap local indices to global merged-VB space.
                    jobHandles.Add(new RemapIndicesJob
                    {
                        LocalIndices = localTris,
                        VertBase     = vertBase,
                        Output       = mergedIdx.GetSubArray(iBase, indexCount),
                    }.Schedule(indexCount, 128));

                    // Compute world-space PrimitiveDataNRD for this submesh.
                    jobHandles.Add(new BuildMergedPrimitivesJob
                    {
                        Indices        = localTris,
                        LocalPositions = posF3,
                        LocalNormals   = norF3,
                        LocalTangents  = tanF4,
                        UVs            = uvF2,
                        HasN           = hasN,
                        HasT           = hasT,
                        HasUV          = hasUV,
                        LocalToWorld   = xform,
                        NormalMatrix   = normalMatrix,
                        Output         = primOutput.GetSubArray((int)primitiveOffsetForSubMesh, triCount),
                    }.Schedule(triCount, 64));

                    iBase           += indexCount;
                    primitiveCursor += (uint)triCount;

                    // InstanceDataNRD is built on the main thread.
                    uint baseTextureIndex = (uint)(subMatIdx * TexturesPerMaterial);
                    var inst = new InstanceDataNRD
                    {
                        // Vertices are already world-space → mOverloadedMatrix encodes identity.
                        mOverloadedMatrix0 = new Vector4(1f, 0f, 0f, 0f),
                        mOverloadedMatrix1 = new Vector4(0f, 1f, 0f, 0f),
                        mOverloadedMatrix2 = new Vector4(0f, 0f, 1f, 0f),

                        textureOffsetAndFlags = baseTextureIndex | (baseFlags << FlagFirstBit),
                        primitiveOffset       = primitiveOffsetForSubMesh,
                        scale                 = 1f,
                        morphPrimitiveOffset  = 0,
                    };
                    EncodeMaterial(subMat, ref inst);
                    instList.Add(inst);
                    instanceCursor++;
                }

                vertBase += vertCount;
            }

            // Wait for all jobs.
            var allHandles                                           = new NativeArray<JobHandle>(jobHandles.Count, Allocator.Temp);
            for (int i = 0; i < jobHandles.Count; i++) allHandles[i] = jobHandles[i];
            JobHandle.CombineDependencies(allHandles).Complete();
            allHandles.Dispose();

            foreach (var arr in tempArrays) arr.Dispose();
            meshDataArr.Dispose();

            // Upload merged VB/IB to GPU buffers.
            var blas = new MergedBlas
            {
                vertexCount  = (uint)totalVerts,
                submeshDescs = submeshDescs.ToArray(),
                vb           = new GraphicsBuffer(GraphicsBuffer.Target.Structured, totalVerts, sizeof(float) * 3),
                ib           = new GraphicsBuffer(GraphicsBuffer.Target.Structured, totalIndices, sizeof(uint)),
            };
            blas.vb.SetData(mergedPos);
            blas.ib.SetData(mergedIdx);

            mergedPos.Dispose();
            mergedIdx.Dispose();

            return blas;
        }

        /// <summary>Passes the merged BLAS's VB/IB pointers to the native AS as a single instance.</summary>
        private unsafe void RegisterMergedBlas(RayTracingAccelerationStructure dstAS,
            MergedBlas blas, uint handle, uint firstInstanceDataIndex, byte mask)
        {
            if (dstAS == null || blas == null) return;
            if (blas.submeshDescs == null || blas.submeshDescs.Length == 0) return;

            fixed (NativeRenderPlugin.SubmeshDesc* pDescs = blas.submeshDescs)
            {
                var desc = new NativeRenderPlugin.AddInstanceDesc
                {
                    vertexBufferNativePtr = blas.vb.GetNativeBufferPtr(),
                    indexBufferNativePtr  = blas.ib.GetNativeBufferPtr(),
                    submeshDescs          = (IntPtr)pDescs,
                    ommDescs              = IntPtr.Zero,
                    instanceHandle        = handle,
                    vertexCount           = blas.vertexCount,
                    vertexStride          = sizeof(float) * 3,
                    indexStride           = sizeof(uint),
                    submeshCount          = (uint)blas.submeshDescs.Length,
                };

                if (!NativeRenderPlugin.NR_AS_AddInstance(dstAS.Handle, ref desc))
                {
                    Debug.LogError("[NRDSampleResource] NR_AS_AddInstance failed for merged BLAS");
                    return;
                }
            }

            // Identity transform – vertices already in world space.
            var handles = GCHandle.Alloc(kIdentity3x4, GCHandleType.Pinned);
            try
            {
                NativeRenderPlugin.NR_AS_SetInstanceTransform(dstAS.Handle, handle, handles.AddrOfPinnedObject());
            }
            finally
            {
                handles.Free();
            }

            NativeRenderPlugin.NR_AS_SetInstanceMask(dstAS.Handle, handle, mask);
            NativeRenderPlugin.NR_AS_SetInstanceID(dstAS.Handle, handle, firstInstanceDataIndex);
        }

        // =====================================================================
        // Material / texture helpers
        // =====================================================================

        private static byte GetMaskForFlags(uint flags)
        {
            // Use FLAG bits 0–2 as the visibility mask (matches merged-BLAS usage).
            return (byte)(flags & 0xFF);
        }

        private static Material GetRepresentativeMaterial(MeshRenderer mr)
        {
            var mats = mr.sharedMaterials;
            return (mats != null && mats.Length > 0) ? mats[0] : null;
        }

        /// <summary>
        /// Returns the material slot index for <paramref name="mat"/>, registering it if new.
        /// <para>
        /// <b>Bulk path</b> (<paramref name="texPtrs"/> != null): appends 4 native texture pointers
        /// to <paramref name="texPtrs"/>; the caller creates <see cref="_textures"/> afterwards.
        /// </para>
        /// <para>
        /// <b>Incremental path</b> (<paramref name="texPtrs"/> == null): writes directly into
        /// <see cref="_textures"/>, growing it via <see cref="BindlessTexture.Resize"/> as needed.
        /// Reuses freed slots from <see cref="_freeMatSlots"/> before appending.
        /// </para>
        /// Always increments the material reference count.
        /// </summary>
        private int GetOrAddMaterial(Material mat, List<IntPtr> texPtrs)
        {
            // Already registered — bump ref count and return existing slot.
            if (mat != null && _materialSlots.TryGetValue(mat, out int existing))
            {
                _materialRefCounts[mat] = (_materialRefCounts.TryGetValue(mat, out int rc) ? rc : 0) + 1;
                return existing;
            }

            // Determine slot index: reuse a freed slot if one is available.
            int idx;
            if (_freeMatSlots.Count > 0)
            {
                idx = _freeMatSlots.Dequeue();
            }
            else if (texPtrs != null)
            {
                // Bulk build path: next sequential slot based on how many materials are already known.
                idx = _materialSlots.Count;
            }
            else
            {
                // Incremental path: derive from current _textures capacity.
                idx = _textures != null ? _textures.Capacity / TexturesPerMaterial : _materialSlots.Count;
            }

            if (mat != null)
            {
                _materialSlots[mat]     = idx;
                _materialRefCounts[mat] = 1;
            }

            if (texPtrs != null)
            {
                // Bulk build path: append raw pointers; caller will create _textures.
                AppendTexture(TryGetTex(mat, "_BaseMap"), PlaceholderKind.White, texPtrs);
                AppendTexture(TryGetTex(mat, "_MetallicGlossMap"), PlaceholderKind.Black, texPtrs);
                AppendTexture(TryGetTex(mat, "_BumpMap"), PlaceholderKind.FlatNormal, texPtrs);
                AppendTexture(TryGetTex(mat, "_EmissionMap"), PlaceholderKind.Black, texPtrs);
            }
            else if (_textures != null)
            {
                // Incremental path: write directly into the live descriptor array.
                int base4 = idx * TexturesPerMaterial;
                int need  = base4 + TexturesPerMaterial;
                if (need > _textures.Capacity)
                {
                    Debug.Log($"[NRDSampleResource] Growing _textures from capacity {_textures.Capacity} to {need} to accommodate new material '{mat.name}'");
                    _textures.Resize(need);
                }

                _textures.SetNativePtr(base4 + 0, NativePtrOf(TryGetTex(mat, "_BaseMap"), PlaceholderKind.White));
                _textures.SetNativePtr(base4 + 1, NativePtrOf(TryGetTex(mat, "_MetallicGlossMap"), PlaceholderKind.Black));
                _textures.SetNativePtr(base4 + 2, NativePtrOf(TryGetTex(mat, "_BumpMap"), PlaceholderKind.FlatNormal));
                _textures.SetNativePtr(base4 + 3, NativePtrOf(TryGetTex(mat, "_EmissionMap"), PlaceholderKind.Black));
            }

            return idx;
        }

        /// <summary>
        /// Decrements the reference count for <paramref name="mat"/>.
        /// When the count reaches zero, the material slot is freed:
        /// its 4 descriptor entries are cleared to null SRVs and the slot index
        /// is enqueued in <see cref="_freeMatSlots"/> for future reuse.
        /// </summary>
        private void ReleaseMaterial(Material mat)
        {
            if (mat == null || !_materialSlots.TryGetValue(mat, out int slotIdx)) return;

            int newRc = _materialRefCounts.GetValueOrDefault(mat, 1) - 1;
            if (newRc > 0)
            {
                _materialRefCounts[mat] = newRc;
                return;
            }

            // Reference count hit zero — free the slot.
            _materialSlots.Remove(mat);
            _materialRefCounts.Remove(mat);
            _freeMatSlots.Enqueue(slotIdx);

            // Write null SRVs so stale GPU resources don't linger.
            if (_textures != null)
            {
                int base4 = slotIdx * TexturesPerMaterial;
                for (int i = 0; i < TexturesPerMaterial; i++)
                    _textures.SetNativePtr(base4 + i, IntPtr.Zero);
            }
        }

        /// <summary>Returns the native texture pointer for <paramref name="tex"/>, or the placeholder if null.</summary>
        private static IntPtr NativePtrOf(Texture tex, PlaceholderKind fallback)
        {
            var t = tex != null ? tex : GetPlaceholder(fallback);
            return t.GetNativeTexturePtr();
        }

        // 1x1 placeholder textures so missing material slots never bind null to gIn_Textures.
        // Order matches GetOrAddMaterial: BaseMap, BumpMap, MetallicGlossMap, EmissionMap.
        private enum PlaceholderKind
        {
            White,
            FlatNormal,
            Black,
            Black2
        }

        private static Texture2D _phWhite;
        private static Texture2D _phFlatNormal;
        private static Texture2D _phBlack;

        private static Texture2D GetPlaceholder(PlaceholderKind kind)
        {
            switch (kind)
            {
                case PlaceholderKind.White:
                    if (_phWhite == null)
                    {
                        _phWhite = new Texture2D(1, 1, TextureFormat.RGBA32, false, true) { name = "NRD_Placeholder_White", hideFlags = HideFlags.HideAndDontSave };
                        _phWhite.SetPixel(0, 0, Color.white);
                        _phWhite.Apply(false, true);
                    }

                    return _phWhite;
                case PlaceholderKind.FlatNormal:
                    if (_phFlatNormal == null)
                    {
                        _phFlatNormal = new Texture2D(1, 1, TextureFormat.RGBA32, false, true) { name = "NRD_Placeholder_FlatNormal", hideFlags = HideFlags.HideAndDontSave };
                        _phFlatNormal.SetPixel(0, 0, new Color(0.5f, 0.5f, 1f, 1f));
                        _phFlatNormal.Apply(false, true);
                    }

                    return _phFlatNormal;
                default: // Black / Black2
                    if (_phBlack == null)
                    {
                        _phBlack = new Texture2D(1, 1, TextureFormat.RGBA32, false, true) { name = "NRD_Placeholder_Black", hideFlags = HideFlags.HideAndDontSave };
                        _phBlack.SetPixel(0, 0, new Color(0f, 0f, 0f, 1f));
                        _phBlack.Apply(false, true);
                    }

                    return _phBlack;
            }
        }

        private static void AppendTexture(Texture tex, PlaceholderKind fallback, List<IntPtr> texPtrs)
        {
            var t = tex != null ? tex : GetPlaceholder(fallback);
            texPtrs.Add(t.GetNativeTexturePtr());
        }

        private static void EncodeMaterial(Material mat, ref InstanceDataNRD inst)
        {
            Color baseColor = TryGetColor(mat, "_BaseColor", Color.white);
            Color emission  = TryGetColor(mat, "_EmissionColor", Color.black);
            float metal     = TryGetFloat(mat, "_Metallic", 0f);
            float smooth    = TryGetFloat(mat, "_Smoothness", 0.5f);

            float normScale = TryGetFloat(mat, "_BumpScale", 1f);

            inst.baseColorAndMetalnessScale.x = new half(baseColor.r);
            inst.baseColorAndMetalnessScale.y = new half(baseColor.g);
            inst.baseColorAndMetalnessScale.z = new half(baseColor.b);
            inst.baseColorAndMetalnessScale.w = new half(metal);

            inst.emissionAndRoughnessScale.x = new half(emission.r);
            inst.emissionAndRoughnessScale.y = new half(emission.g);
            inst.emissionAndRoughnessScale.z = new half(emission.b);
            // 这里实际传入的是光滑度
            inst.emissionAndRoughnessScale.w = new half(smooth);

            inst.normalUvScale.x = new half(normScale);
            inst.normalUvScale.y = new half(normScale);
        }

        private static bool IsMaterialTransparent(Material mat)
        {
            if (mat == null) return false;
            // URP lit: _Surface = 1 → Transparent.
            if (mat.HasProperty("_Surface") && mat.GetFloat("_Surface") > 0.5f) return true;
            return false;
        }

        private static bool IsMaterialEmissive(Material mat)
        {
            if (mat == null) return false;
            if (mat.HasProperty("_EmissionColor"))
            {
                Color e = mat.GetColor("_EmissionColor").linear;
                if (e.r > 0f || e.g > 0f || e.b > 0f) return true;
            }

            if (mat.HasProperty("_EmissionMap") && mat.GetTexture("_EmissionMap") != null) return true;
            return false;
        }

        // =====================================================================
        // Geometry helpers
        // =====================================================================

        private static Vector3 TransformNormal(Matrix4x4 normalMatrix, Vector3[] arr, int idx)
        {
            Vector3 n  = (arr != null && idx < arr.Length) ? arr[idx] : Vector3.up;
            Vector3 tn = normalMatrix.MultiplyVector(n);
            return tn.sqrMagnitude > 1e-12f ? tn.normalized : Vector3.up;
        }

        private static Vector4 TransformTangent(Matrix4x4 xform, Vector4[] arr, int idx)
        {
            Vector4 tt = (arr != null && idx < arr.Length) ? arr[idx] : new Vector4(1f, 0f, 0f, 1f);
            Vector3 td = xform.MultiplyVector(new Vector3(tt.x, tt.y, tt.z));
            Vector3 n  = td.sqrMagnitude > 1e-12f ? td.normalized : Vector3.right;
            return new Vector4(n.x, n.y, n.z, tt.w);
        }

        // Raw object-space accessors for the non-merged path (shader transforms via mOverloadedMatrix).
        private static Vector3 GetNormal(Vector3[] arr, int idx)
        {
            Vector3 n = (arr != null && idx < arr.Length) ? arr[idx] : Vector3.up;
            return n.sqrMagnitude > 1e-12f ? n.normalized : Vector3.up;
        }

        private static Vector4 GetTangent(Vector4[] arr, int idx)
        {
            return (arr != null && idx < arr.Length) ? arr[idx] : new Vector4(1f, 0f, 0f, 1f);
        }

        /// <summary>Signed octahedral unit-vector encoding, matching MathLib Packing::EncodeUnitVector(v, true).</summary>
        private static half2 EncodeUnitVectorSigned(Vector3 v)
        {
            return EncodeUnitVector(v, true);
        }

        static float SafeSign(float x)
        {
            return x >= 0.0f ? 1.0f : -1.0f;
        }

        static float2 SafeSign(float2 v)
        {
            return new float2(SafeSign(v.x), SafeSign(v.y));
        }


        static half2 EncodeUnitVector(float3 v, bool bSigned = false)
        {
            v /= math.dot(math.abs(v), 1.0f);

            float2 octWrap = (1.0f - math.abs(v.yx)) * SafeSign(v.xy);
            v.xy = v.z >= 0.0f ? v.xy : octWrap;

            return new half2(bSigned ? v.xy : 0.5f * v.xy + 0.5f);
        }


        private static Texture TryGetTex(Material mat, string prop) =>
            mat != null && mat.HasProperty(prop) ? mat.GetTexture(prop) : null;

        private static Color TryGetColor(Material mat, string prop, Color def) =>
            mat != null && mat.HasProperty(prop) ? mat.GetColor(prop).linear : def;

        private static float TryGetFloat(Material mat, string prop, float def) =>
            mat != null && mat.HasProperty(prop) ? mat.GetFloat(prop) : def;
    }
}