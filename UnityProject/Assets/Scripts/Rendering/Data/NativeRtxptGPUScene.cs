using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using NativeRender;
using UnityEngine;
using UnityEngine.Rendering;
using RayTracingAccelerationStructure = NativeRender.RayTracingAccelerationStructure;

namespace PathTracing
{
    // =========================================================================
    // NativeRtxptGPUScene
    // =========================================================================

    /// <summary>
    /// Self-contained GPU scene for the RTXPT native passes. Owns the TLAS and provides
    /// donut-compatible structured buffers (<c>t_SubInstanceData</c>, <c>t_InstanceData</c>,
    /// <c>t_GeometryData</c>, <c>t_PTMaterialData</c>, bindless VB/IB, bindless textures).
    /// Struct layouts exactly mirror <c>donut/shaders/bindless.h</c> and RTXPT's
    /// <c>SubInstanceData</c>/<c>PTMaterialData</c>, so shaders that include
    /// <c>SceneGeometry.hlsli</c> / <c>PathTracerBridgeDonut.hlsli</c> work without any
    /// layout mismatch.
    ///
    /// Update model (mirrors original RTXPT): on a topology change a single traversal builds
    /// the flat <see cref="RtxptInstanceRecord"/> list (<see cref="RtxptSceneLayout"/>), the
    /// acceleration structure is diffed incrementally against it (<see cref="RtxptAccelRegistry"/>,
    /// persistent BLASes), and the GPU-side arrays are rebuilt from the same list — so the TLAS
    /// instance order and every CPU/GPU array can never drift apart.
    /// </summary>
    public sealed class NativeRtxptGPUScene : IDisposable
    {
        // Acceleration structure + incremental TLAS registration
        private RayTracingAccelerationStructure _accelStructure;
        private readonly RtxptAccelRegistry     _accelRegistry = new();

        // Scene layout of the last topology change — the single source of truth that both the
        // AS registration and the GPU-buffer build derive from.
        private List<RtxptInstanceRecord> _layout = new();

        // Per-mesh donut SoA buffers, persistent across rebuilds.
        private readonly RtxptGeometryCache _geometryCache = new();

        // Structured buffers (donut-compatible)
        // t_InstanceData (t2): transforms are re-uploaded every frame, so this uses
        // UploadBuffer — its NativePtr is stable across SetData (unlike
        // GraphicsBuffer.GetNativeBufferPtr), so we fetch the pointer once at creation.
        private UploadBuffer   _instanceGpuBuf;
        private GraphicsBuffer _geometryGpuBuf; // t_GeometryData  (t3)

        // RTXPT-specific structured buffers
        // t_SubInstanceData (t1): emissive-light mapping offsets are recomputed per frame, so
        // this uses UploadBuffer for a stable NativePtr. SRV-only — never bound as a UAV.
        private UploadBuffer   _subInstanceGpuBuf; // t_SubInstanceData    (t1)
        private GraphicsBuffer _ptMaterialGpuBuf;  // t_PTMaterialData     (t5)
        private GraphicsBuffer _geomDebugGpuBuf;   // t_GeometryDebugData  (t4)

        // _instanceGpuBuf / _subInstanceGpuBuf are UploadBuffers — bound by handle (no cached ptr).
        private IntPtr _geometryGpuBufPtr;
        private IntPtr _ptMaterialGpuBufPtr;
        private IntPtr _geomDebugGpuBufPtr;

        // Bindless
        private BindlessBuffer  _sceneBuffers;
        private BindlessTexture _sceneTextures;

        // CPU-side mirrors
        private DonutInstanceData[] _instanceCpu;
        private DonutGeometryData[] _geometryCpu;
        private SubInstanceData[]   _subInstanceCpu;
        private PTMaterialData[]    _ptMaterialCpu;
        private GeometryDebugData[] _geomDebugCpu;

        // True when any _subInstanceCpu field changed since the last upload (emissive mapping
        // offsets / analytic-proxy light indices). Lets the per-frame path skip the full-array
        // re-upload when nothing moved.
        private bool _subInstanceCpuDirty;

        // Native TLAS handle per instance, parallel to _instanceCpu (for transform updates).
        private readonly List<uint> _instanceHandles = new();

        // Per-renderer tracking for transform updates (one entry per renderer; its TLAS
        // instances occupy the contiguous range [firstInstanceIdx, firstInstanceIdx+instanceCount)).
        private sealed class RendererEntry
        {
            public Transform transform; // skinned: the root bone (vertices are in root-bone space)
            public bool wasMoving = true; // start true so first frame always syncs prev = current
            public int firstInstanceIdx;
            public int instanceCount;

            // Skinned instances deform every frame: transforms update unconditionally from the
            // root bone, with last frame's root kept for prevTransform (motion vectors).
            public bool      isSkinned;
            public Matrix4x4 lastRoot;
            public bool      hasLastRoot;
        }

        private readonly Dictionary<int, RendererEntry>    _rendererEntries = new();
        private readonly Dictionary<int, (int vb, int ib)> _meshBufferSlots = new();

        // Skinned instances: bindless slot pair per renderer (own SoA VB + shared donut IB),
        // and the set of mesh ids referenced this rebuild (for geometry-cache eviction).
        private readonly Dictionary<int, (int vb, int ib)> _skinnedBufferSlots = new();
        private readonly HashSet<int>                      _usedMeshIds        = new();

        // One repack dispatch per skinned renderer, rebuilt on topology change and consumed by
        // NativeRtxptBuildTlasPass every frame before the TLAS/BLAS build.
        private readonly List<RtxptSkinnedDispatch> _skinnedDispatches   = new();
        private readonly HashSet<int>               _skinnedSeenScratch  = new();

        internal IReadOnlyList<RtxptSkinnedDispatch> SkinnedDispatches => _skinnedDispatches;

        private readonly List<RtxptRenderer> _registeredTargets = new();
        private int _lastTopologyVersion = -1;

        public bool ShaderTableDirty => _accelRegistry.ShaderTableDirty;

        private bool _sceneGpuDirty = true;
        private bool _forceRebuild;
        private bool _disposed;

        // ---- Emissive triangle light tracking --------------------------------
        // (instanceIndex, geometrySubIndex) → last-frame DestinationBufferOffset. Two maps are
        // swapped each frame to avoid a per-frame dictionary allocation.
        private Dictionary<(int, int), uint> _emissiveHistoricOffsets = new();
        private Dictionary<(int, int), uint> _emissiveHistoricScratch = new();

        // Emissive geometry entries, cached at rebuild time: membership only changes on a
        // topology change (emissive flips alter SubmeshGroups, which bumps TopologyVersion),
        // so the per-frame emissive pass never has to re-scan all instances × geometries.
        private readonly List<EmissiveGeometryEntry> _emissiveCache = new();

        // ---- Analytic-light-proxy tracking -----------------------------------
        // Sub-instances whose material has EnableAsAnalyticLightProxy, recorded at scene rebuild
        // as (subInstanceIndex, targetLightInstanceID). Resolved each frame by
        // ResolveAnalyticProxyLights (the global light index can change frame to frame).
        private readonly List<(int subIdx, int lightId)> _proxySubInstances = new();

        // Max task count: MaxLights / LLB_MAX_TRIANGLES_PER_TASK * 2
        private const int MaxEmissiveProcTasks = NativeRtxptBufferResources.MaxLights / 32 * 2;
        private static readonly RtxptEmissiveTrianglesProcTask[] s_emissiveTaskStaging =
            new RtxptEmissiveTrianglesProcTask[MaxEmissiveProcTasks];

        /// <summary>Number of tasks produced by the last <see cref="PrepareEmissiveTriangleTasks"/> call.</summary>
        public int  LastEmissiveTaskCount     { get; private set; }
        /// <summary>Total triangle-light count produced by the last <see cref="PrepareEmissiveTriangleTasks"/> call.</summary>
        public uint LastEmissiveTriangleCount { get; private set; }

        // Maps MeshRenderer.GetInstanceID() → per-submesh material indices in _ptMaterialCpu.
        // Used for lightweight material-only updates (CheckAndUpdateMaterialOverrides).
        private readonly Dictionary<int, int[]> _overrideMaterialIndices = new();

        // Cached list of (component, matIndices), rebuilt during RebuildSceneGpuData so
        // CheckAndUpdateMaterialOverrides never calls GetComponent.
        private readonly List<(RtxptRenderer comp, int[] matIndices)> _overrideCache = new();

        // Optional equirectangular environment map appended to the bindless texture array.
        private Texture _pendingEnvMap;
        private int     _environmentMapTextureIndex = -1;

        /// <summary>
        /// Index of the environment map texture in the bindless texture array, or -1 if none.
        /// Valid after the scene has been rebuilt (after <see cref="UpdateForFrame"/>).
        /// </summary>
        public int EnvironmentMapTextureIndex => _environmentMapTextureIndex;

        /// <summary>
        /// Registers an equirectangular environment map to include in the bindless texture array.
        /// Call before <see cref="UpdateForFrame"/> each frame; only triggers a scene rebuild when
        /// the texture instance changes.
        /// </summary>
        public void SetEnvironmentMap(Texture envMap)
        {
            if (_pendingEnvMap == envMap) return;
            _pendingEnvMap = envMap;
            _sceneGpuDirty = true;
        }

        public RayTracingAccelerationStructure AccelerationStructure => _accelStructure;

        /// <summary>
        /// Number of geometry entries in the flat geometry array (length of t_GeometryData buffer).
        /// Used by PrepareLightsPass to size the GeometryInstanceToLight mapping array.
        /// </summary>
        public int TotalGeometryInstanceCount => _geometryCpu != null ? _geometryCpu.Length : 0;

        /// <summary>
        /// Number of PT material entries (length of t_PTMaterialData). Mirrors
        /// <c>m_materialsBaker->GetMaterialDataCount()</c> (Sample.cpp:2095) and feeds
        /// <c>SampleConstants.MaterialCount</c>, which shaders use to bounds-check material indices
        /// in <c>Bridge::loadIoR</c> / <c>loadHomogeneousVolumeData</c>.
        /// </summary>
        public uint MaterialDataCount => _ptMaterialCpu != null ? (uint)_ptMaterialCpu.Length : 0u;

        /// <summary>
        /// One <see cref="EmissiveGeometryEntry"/> for every sub-mesh whose material has a
        /// non-zero emissiveColor. Cached at rebuild time (emissive membership changes always
        /// bump the topology version via SubmeshGroups). Valid after <see cref="UpdateForFrame"/>.
        /// </summary>
        public IReadOnlyList<EmissiveGeometryEntry> GetEmissiveGeometries() => _emissiveCache;

        public NativeRtxptGPUScene()
        {
            _accelStructure = new RayTracingAccelerationStructure();
        }

        public void MarkRebuildDirty() => _forceRebuild = true;

        // -----------------------------------------------------------------------
        // Per-frame entry point
        // -----------------------------------------------------------------------

        /// <summary>
        /// Call once per frame before <see cref="BuildAccelerationStructure"/>.
        /// Handles dirty detection, AS diff, GPU data rebuild, and transform updates.
        /// </summary>
        public void UpdateForFrame()
        {
            var targets = RtxptRenderer.All;

            if (_forceRebuild || _lastTopologyVersion != RtxptRenderer.TopologyVersion || TargetSetChanged(targets))
            {
                _layout = RtxptSceneLayout.Build(targets);
                PrepareSkinnedRecords(_layout);
                _accelRegistry.Sync(_accelStructure, _layout);

                _registeredTargets.Clear();
                _registeredTargets.AddRange(targets);
                // Captured AFTER Build: it may invoke RebuildGroups, which can bump the version.
                _lastTopologyVersion = RtxptRenderer.TopologyVersion;
                _forceRebuild  = false;
                _sceneGpuDirty = true;
            }

            if (_sceneGpuDirty)
                RebuildSceneGpuData();
            else
                CheckAndUpdateMaterialOverrides();

            UpdateInstanceTransforms();
        }

        private bool TargetSetChanged(IReadOnlyList<RtxptRenderer> current)
        {
            if (current.Count != _registeredTargets.Count) return true;
            for (int i = 0; i < current.Count; i++)
                if (current[i] != _registeredTargets[i])
                    return true;
            return false;
        }

        /// <summary>
        /// Ensures every skinned record has its per-instance SoA geometry (created/validated via
        /// the geometry cache) BEFORE the AS sync registers it as a dynamic BLAS, and rebuilds
        /// the per-renderer repack dispatch list consumed each frame by the TLAS build pass.
        /// </summary>
        private void PrepareSkinnedRecords(List<RtxptInstanceRecord> layout)
        {
            _skinnedDispatches.Clear();
            _skinnedSeenScratch.Clear();

            foreach (var rec in layout)
            {
                if (!rec.IsSkinned) continue;

                var geo = _geometryCache.GetOrCreateSkinned(rec.RendererId, rec.Mesh);
                rec.SkinnedVb          = geo.Vb;
                rec.SkinnedIb          = geo.Ib;
                rec.SkinnedVertexCount = geo.VertexCount;

                if (!_skinnedSeenScratch.Add(rec.RendererId)) continue; // one dispatch per renderer

                Mesh mesh = rec.Mesh;
                // Unity's GPU-skinned vertex buffer mirrors stream 0 of the shared mesh; the
                // skinned attributes (position/normal/tangent) are float32 there. Attributes
                // outside stream 0 are not present in the skinned buffer.
                if (mesh.GetVertexAttributeStream(VertexAttribute.Position) != 0 ||
                    mesh.GetVertexAttributeFormat(VertexAttribute.Position) != VertexAttributeFormat.Float32)
                {
                    Debug.LogError($"[NativeRtxptGPUScene] Skinned mesh '{mesh.name}': position is not a float32 stream-0 attribute — skinned repack skipped (geometry will stay in rest pose).");
                    continue;
                }

                int stride  = mesh.GetVertexBufferStride(0);
                int posOff  = mesh.GetVertexAttributeOffset(VertexAttribute.Position);
                int normOff = mesh.HasVertexAttribute(VertexAttribute.Normal) &&
                              mesh.GetVertexAttributeStream(VertexAttribute.Normal) == 0 &&
                              mesh.GetVertexAttributeFormat(VertexAttribute.Normal) == VertexAttributeFormat.Float32
                    ? mesh.GetVertexAttributeOffset(VertexAttribute.Normal) : -1;
                int tanOff  = mesh.HasVertexAttribute(VertexAttribute.Tangent) &&
                              mesh.GetVertexAttributeStream(VertexAttribute.Tangent) == 0 &&
                              mesh.GetVertexAttributeFormat(VertexAttribute.Tangent) == VertexAttributeFormat.Float32
                    ? mesh.GetVertexAttributeOffset(VertexAttribute.Tangent) : -1;

                var streams = new RtxptMeshStreamOffsets(mesh, withPrevPosition: true);
                uint flags = 0;
                if (normOff >= 0 && streams.HasNormal)  flags |= RtxptSkinnedDispatch.FlagHasNormal;
                if (tanOff  >= 0 && streams.HasTangent) flags |= RtxptSkinnedDispatch.FlagHasTangent;


                _skinnedDispatches.Add(new RtxptSkinnedDispatch
                {
                    Smr              = rec.Skinned,
                    Geometry         = geo,
                    VertexCount      = geo.VertexCount,
                    SrcStride        = (uint)stride,
                    SrcPosOffset     = (uint)posOff,
                    SrcNormalOffset  = (uint)Mathf.Max(normOff, 0),
                    SrcTangentOffset = (uint)Mathf.Max(tanOff, 0),
                    Streams          = streams,
                    BaseFlags        = flags,
                });
            }
        }

        // -----------------------------------------------------------------------
        // Binding / command-buffer hooks
        // -----------------------------------------------------------------------

        /// <summary>
        /// Binds all RTXPT scene buffers to a native descriptor set (compute or ray-trace).
        /// Binds: t_SubInstanceData(t1), t_InstanceData(t2), t_GeometryData(t3),
        ///        t_GeometryDebugData(t4), t_PTMaterialData(t5),
        ///        t_BindlessBuffers(space1), t_BindlessTextures(space2).
        /// </summary>
        public void BindToShader(NativeDescriptorSetBase ds)
        {
            if (ds == null) return;
            ds.SetStructuredBuffer("t_SubInstanceData", _subInstanceGpuBuf, _subInstanceGpuBuf.count, _subInstanceGpuBuf.stride);
            ds.SetStructuredBuffer("t_InstanceData", _instanceGpuBuf, _instanceGpuBuf.count, _instanceGpuBuf.stride);
            ds.SetStructuredBuffer("t_GeometryData", _geometryGpuBufPtr, _geometryGpuBuf.count, _geometryGpuBuf.stride);
            ds.SetStructuredBuffer("t_GeometryDebugData", _geomDebugGpuBufPtr, _geomDebugGpuBuf.count, _geomDebugGpuBuf.stride);
            ds.SetStructuredBuffer("t_PTMaterialData", _ptMaterialGpuBufPtr, _ptMaterialGpuBuf.count, _ptMaterialGpuBuf.stride);
            ds.SetBindlessBuffer("t_BindlessBuffers", _sceneBuffers);
            ds.SetBindlessTexture("t_BindlessTextures", _sceneTextures);
        }

        /// <summary>Builds / updates the TLAS. Call inside a CommandBuffer recording.</summary>
        public void BuildAccelerationStructure(CommandBuffer cmd)
        {
            _accelStructure.BuildOrUpdate(cmd);
        }

        /// <summary>
        /// Records the deferred GPU copy for any t_InstanceData writes accumulated this frame
        /// by <see cref="UpdateInstanceTransforms"/>. A no-op when nothing changed. Must run in
        /// the same CommandBuffer as the TLAS build, before any pass that reads t_InstanceData.
        /// </summary>
        public void FlushInstanceBuffer(CommandBuffer cmd)
        {
            _instanceGpuBuf?.Flush(cmd);
        }

        /// <summary>
        /// Records the deferred GPU copy for any t_SubInstanceData writes accumulated this frame
        /// (emissive-light mapping offsets). A no-op when nothing changed. Must run in the same
        /// CommandBuffer as, and before, the first pass that reads t_SubInstanceData.
        /// </summary>
        public void FlushSubInstanceBuffer(CommandBuffer cmd)
        {
            _subInstanceGpuBuf?.Flush(cmd);
        }

        /// <summary>
        /// Issues a render event to rebuild <paramref name="pipeline"/>'s hit-group table from the
        /// scene's flat per-geometry variant array. Only does work while <see cref="ShaderTableDirty"/>
        /// is set (i.e. after a scene topology change); call once per pipeline, then
        /// <see cref="MarkShaderTableClean"/>. Issue in the same CommandBuffer as the TLAS build.
        /// </summary>
        public void RebuildShaderTable(CommandBuffer cmd, RayTracePipeline pipeline)
            => _accelRegistry.RebuildShaderTable(cmd, pipeline);

        /// <summary>Clears the dirty flag after every pipeline's hit-group table has been rebuilt.</summary>
        public void MarkShaderTableClean() => _accelRegistry.MarkShaderTableClean();

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            DisposeGpuBuffers();
            _geometryCache.Dispose();
            _accelRegistry.Dispose();
            _accelStructure?.Dispose();
            _accelStructure = null;
        }

        // -----------------------------------------------------------------------
        // GPU data rebuild (consumes _layout)
        // -----------------------------------------------------------------------

        private void DisposeGpuBuffers()
        {
            _instanceGpuBuf?.Dispose();
            _instanceGpuBuf = null;
            _geometryGpuBuf?.Release();
            _geometryGpuBuf = null;
            _geometryGpuBufPtr = IntPtr.Zero;
            _subInstanceGpuBuf?.Dispose();
            _subInstanceGpuBuf = null;
            _ptMaterialGpuBuf?.Release();
            _ptMaterialGpuBuf = null;
            _ptMaterialGpuBufPtr = IntPtr.Zero;
            _geomDebugGpuBuf?.Release();
            _geomDebugGpuBuf = null;
            _geomDebugGpuBufPtr = IntPtr.Zero;
            _sceneBuffers?.Dispose();
            _sceneBuffers = null;
            _sceneTextures?.Dispose();
            _sceneTextures  = null;
            _instanceCpu    = null;
            _geometryCpu    = null;
            _subInstanceCpu = null;
            _ptMaterialCpu  = null;
            _geomDebugCpu   = null;
            _instanceHandles.Clear();
            _rendererEntries.Clear();
            _meshBufferSlots.Clear();
            _skinnedBufferSlots.Clear();
            _usedMeshIds.Clear();
            _overrideMaterialIndices.Clear();
            _overrideCache.Clear();
            _environmentMapTextureIndex = -1;

            // The donut buffer cache (_geometryCache) is intentionally NOT released here: the
            // per-mesh SoA buffers are persistent across rebuilds (unused meshes are evicted at
            // the end of RebuildSceneGpuData, everything is released in Dispose).
        }

        private void RebuildSceneGpuData()
        {
            DisposeGpuBuffers();

            // Historic emissive offsets are keyed by (instanceIndex, geometrySubIndex), which
            // remap when the scene topology changes — stale entries would hand another object's
            // light history to the baker. Drop them; the first frame after a rebuild treats all
            // emissive geometry as new (HistoricBufferOffset = invalid).
            _emissiveHistoricOffsets.Clear();
            _proxySubInstances.Clear();

            var matTable    = new RtxptMaterialTable();
            var instList    = new List<DonutInstanceData>(_layout.Count);
            var geomList    = new List<DonutGeometryData>();
            var subInstList = new List<SubInstanceData>();
            var geomDbgList = new List<GeometryDebugData>();
            var bufPtrs     = new List<IntPtr>();

            foreach (var rec in _layout)
            {
                Mesh mesh   = rec.Mesh;
                int  meshId = mesh.GetInstanceID();
                _usedMeshIds.Add(meshId);

                (int vb, int ib) slots;
                if (rec.IsSkinned)
                {
                    // Per-instance SoA VB (rewritten each frame by the repack compute) + shared
                    // per-mesh donut IB, registered as their own bindless slot pair per renderer.
                    if (!_skinnedBufferSlots.TryGetValue(rec.RendererId, out slots))
                    {
                        slots = (bufPtrs.Count, bufPtrs.Count + 1);
                        if (bufPtrs.Count > 0xFFFF)
                            Debug.LogError($"[NativeRtxptGPUScene] Bindless buffer slot overflow: VB slot index {bufPtrs.Count} exceeds 16-bit limit (65535). Rendering will be corrupted. Mesh='{mesh.name}'.");
                        bufPtrs.Add(rec.SkinnedVb.GetNativeBufferPtr());
                        bufPtrs.Add(rec.SkinnedIb.GetNativeBufferPtr());
                        _skinnedBufferSlots[rec.RendererId] = slots;
                    }
                }
                else if (!_meshBufferSlots.TryGetValue(meshId, out slots))
                {
                    var (donutVb, donutIb) = _geometryCache.GetOrCreate(mesh);

                    slots = (bufPtrs.Count, bufPtrs.Count + 1);
                    // SubInstanceData.IndexBufferIndex_VertexBufferIndex packs both as 16-bit.
                    if (bufPtrs.Count > 0xFFFF)
                        Debug.LogError($"[NativeRtxptGPUScene] Bindless buffer slot overflow: VB slot index {bufPtrs.Count} exceeds 16-bit limit (65535). Rendering will be corrupted. Mesh='{mesh.name}'.");
                    bufPtrs.Add(donutVb.GetNativeBufferPtr());
                    bufPtrs.Add(donutIb.GetNativeBufferPtr());
                    _meshBufferSlots[meshId] = slots;

                    if (!mesh.HasVertexAttribute(VertexAttribute.Normal) ||
                        !mesh.HasVertexAttribute(VertexAttribute.Tangent))
                        Debug.LogWarning($"[NativeRtxptGPUScene] '{mesh.name}': missing normal or tangent stream");
                }

                var streams = new RtxptMeshStreamOffsets(mesh, withPrevPosition: rec.IsSkinned);

                // Per-renderer material-index array for the lightweight material-edit path.
                // A renderer's records are consecutive in the layout; the array spans all of
                // its sub-meshes, so create it on the renderer's first record.
                if (!_overrideMaterialIndices.TryGetValue(rec.RendererId, out int[] overrideMatIndices))
                {
                    overrideMatIndices = new int[mesh.subMeshCount];
                    _overrideMaterialIndices[rec.RendererId] = overrideMatIndices;
                    _overrideCache.Add((rec.Renderer, overrideMatIndices));
                }

                int firstGeom = geomList.Count;
                foreach (int s in rec.SubmeshIndices)
                    AddSubmeshGeometry(rec, s, slots, streams, matTable, geomList, subInstList, geomDbgList, overrideMatIndices);

                // TLAS InstanceID must be the SubInstanceData base offset (firstGeometryInstanceIndex),
                // NOT the instance index: the any-hit alpha test indexes t_SubInstanceData[InstanceID()
                // + geometryIndex] directly (PathTracerBridgeDonut.hlsli AlphaTest), mirroring RTXPT's
                // instanceDesc.instanceID = instance->GetGeometryInstanceIndex() (Sample.cpp). The
                // closest-hit path is unaffected — it uses the automatic InstanceIndex() to fetch the
                // instance, then adds firstGeometryInstanceIndex itself.
                _accelStructure.SetInstanceID(rec.Handle, (uint)firstGeom);

                Matrix4x4 m    = RtxptSceneLayout.GetRootTransform(rec);
                var       row0 = new Vector4(m.m00, m.m01, m.m02, m.m03);
                var       row1 = new Vector4(m.m10, m.m11, m.m12, m.m13);
                var       row2 = new Vector4(m.m20, m.m21, m.m22, m.m23);

                instList.Add(new DonutInstanceData
                {
                    flags                      = 0u,
                    firstGeometryInstanceIndex = (uint)firstGeom,
                    firstGeometryIndex         = (uint)firstGeom,
                    numGeometries              = (uint)(geomList.Count - firstGeom),
                    transformRow0              = row0,
                    transformRow1              = row1,
                    transformRow2              = row2,
                    prevTransformRow0          = row0,
                    prevTransformRow1          = row1,
                    prevTransformRow2          = row2,
                });
                _instanceHandles.Add(rec.Handle);

                if (_rendererEntries.TryGetValue(rec.RendererId, out var entry))
                    entry.instanceCount++; // records of one renderer are consecutive
                else
                    _rendererEntries[rec.RendererId] = new RendererEntry
                    {
                        // Skinned vertices are in root-bone space, so track the root bone.
                        transform = rec.IsSkinned && rec.Skinned.rootBone != null
                            ? rec.Skinned.rootBone
                            : rec.TargetRenderer.transform,
                        isSkinned        = rec.IsSkinned,
                        firstInstanceIdx = instList.Count - 1,
                        instanceCount    = 1,
                    };
            }

            // Append environment map to bindless texture array (if registered)
            var texPtrs = matTable.TexturePtrs;
            if (_pendingEnvMap != null)
            {
                _environmentMapTextureIndex = texPtrs.Count;
                texPtrs.Add(_pendingEnvMap.GetNativeTexturePtr());
            }

            _sceneBuffers = new BindlessBuffer(Mathf.Max(bufPtrs.Count, 1));
            for (int i = 0; i < bufPtrs.Count; i++)
                _sceneBuffers.SetNativePtr(i, bufPtrs[i]);

            _sceneTextures = new BindlessTexture(Mathf.Max(texPtrs.Count, 1));
            for (int i = 0; i < texPtrs.Count; i++)
                _sceneTextures.SetNativePtr(i, texPtrs[i]);

            if (instList.Count == 0)
            {
                instList.Add(default);
                geomList.Add(default);
                subInstList.Add(default);
                matTable.Materials.Add(default);
                geomDbgList.Add(default);
            }

            _instanceCpu    = instList.ToArray();
            _geometryCpu    = geomList.ToArray();
            _subInstanceCpu = subInstList.ToArray();
            _ptMaterialCpu  = matTable.Materials.ToArray();
            _geomDebugCpu   = geomDbgList.ToArray();

            // Ranges mode: per-frame transform updates touch only the moved instances,
            // so we upload just those sub-ranges rather than a full-buffer span.
            // Debug names match the original RTXPT/donut debugName strings (donut Scene.cpp
            // "Instances"/"BindlessGeometry", Sample.cpp "Instances", MaterialsBaker.cpp
            // "PTMaterialDataStorage", OmmBaker.cpp "BindlessGeometryDebug") for PIX parity.
            _instanceGpuBuf = new UploadBuffer(_instanceCpu.Length, Marshal.SizeOf<DonutInstanceData>(), debugName: "Instances");
            _instanceGpuBuf.SetData(_instanceCpu, 0, _instanceCpu.Length);

            _geometryGpuBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _geometryCpu.Length, Marshal.SizeOf<DonutGeometryData>()) { name = "BindlessGeometry" };
            _geometryGpuBuf.SetData(_geometryCpu);
            _geometryGpuBufPtr = _geometryGpuBuf.GetNativeBufferPtr();

            _subInstanceGpuBuf = new UploadBuffer(_subInstanceCpu.Length, Marshal.SizeOf<SubInstanceData>(), debugName: "Instances");
            _subInstanceGpuBuf.SetData(_subInstanceCpu, 0, _subInstanceCpu.Length);
            _subInstanceCpuDirty = false;

            _ptMaterialGpuBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _ptMaterialCpu.Length, Marshal.SizeOf<PTMaterialData>()) { name = "PTMaterialDataStorage" };
            _ptMaterialGpuBuf.SetData(_ptMaterialCpu);
            _ptMaterialGpuBufPtr = _ptMaterialGpuBuf.GetNativeBufferPtr();

            _geomDebugGpuBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _geomDebugCpu.Length, Marshal.SizeOf<GeometryDebugData>()) { name = "BindlessGeometryDebug" };
            _geomDebugGpuBuf.SetData(_geomDebugCpu);
            _geomDebugGpuBufPtr = _geomDebugGpuBuf.GetNativeBufferPtr();

            RebuildEmissiveCache();

            // Drop donut buffers no longer referenced: per-mesh VB/IB by mesh usage (static or
            // skinned — skinned instances share the per-mesh IB), skinned VBs by renderer.
            _geometryCache.EvictUnused(_usedMeshIds.Contains, _skinnedBufferSlots.ContainsKey);

            _sceneGpuDirty = false;
        }

        // Appends geometry + sub-instance + debug data for one assigned sub-mesh of a record.
        // Only called for sub-meshes the layout has already confirmed assigned, so
        // rec.Renderer.Slots[s] is non-null here.
        private void AddSubmeshGeometry(RtxptInstanceRecord rec, int s, (int vb, int ib) slots,
            in RtxptMeshStreamOffsets streams, RtxptMaterialTable matTable,
            List<DonutGeometryData> geomList, List<SubInstanceData> subInstList,
            List<GeometryDebugData> geomDbgList, int[] overrideMatIndices)
        {
            Mesh              mesh = rec.Mesh;
            SubMeshDescriptor sub  = mesh.GetSubMesh(s);
            RtxptMaterial     slot = rec.Renderer.Slots[s];

            int matIdx = matTable.GetOrAdd(slot);
            if (s < overrideMatIndices.Length)
                overrideMatIndices[s] = matIdx;

            int globalGeomIdx = geomList.Count;

            // SubInstanceData.GlobalGeometryIndex_PTMaterialDataIndex packs both fields as 16-bit.
            if (globalGeomIdx > 0xFFFF)
                Debug.LogError($"[NativeRtxptGPUScene] GlobalGeometryIndex overflow: geomIndex={globalGeomIdx} exceeds 16-bit limit (65535). Rendering will be corrupted. Renderer='{rec.TargetRenderer.name}' subMesh={s}.");
            if (matIdx > 0xFFFF)
                Debug.LogError($"[NativeRtxptGPUScene] PTMaterialDataIndex overflow: matIndex={matIdx} exceeds 16-bit limit (65535). Rendering will be corrupted. Renderer='{rec.TargetRenderer.name}' subMesh={s}.");

            geomList.Add(new DonutGeometryData
            {
                numIndices         = (uint)sub.indexCount,
                numVertices        = (uint)mesh.vertexCount,
                indexBufferIndex   = slots.ib,
                indexOffset        = (uint)sub.indexStart * 4u, // donut IB is always uint32
                vertexBufferIndex  = slots.vb,
                positionOffset     = streams.Pos,
                // Skinned buffers carry a real PrevPosition stream (repack copies last frame's
                // positions there, donut SkinningPass model); static buffers alias positions.
                prevPositionOffset = streams.PrevPos,
                texCoord1Offset    = streams.Uv,
                texCoord2Offset    = 0xFFFFFFFFu,
                normalOffset       = streams.Normal,
                tangentOffset      = streams.Tangent,
                curveRadiusOffset  = 0xFFFFFFFFu,
                materialIndex      = (uint)matIdx,
            });

            // --- SubInstanceData (RTXPT t1) ---
            bool  isAlphaTested  = slot.EnableAlphaTesting;
            float aCutoff        = isAlphaTested ? slot.AlphaCutoff : 0f;
            uint  alphaU8        = (uint)Mathf.RoundToInt(Mathf.Clamp01(aCutoff) * 255f);
            // AlphaTextureIndex in lo16: use base texture slot if alpha-tested, else 0
            uint alphaTexIdx = isAlphaTested && matTable.Materials[matIdx].BaseOrDiffuseTextureIndex != 0xFFFFFFFFu
                ? matTable.Materials[matIdx].BaseOrDiffuseTextureIndex
                : 0u;
            uint siFlags = alphaTexIdx & 0xFFFFu;
            if (isAlphaTested)       siFlags |= SubInstanceFlags.AlphaTested;
            if (slot.ExcludeFromNEE) siFlags |= SubInstanceFlags.ExcludeFromNEE;
            siFlags |= (alphaU8 << SubInstanceFlags.AlphaOffsetOffset);

            // Record analytic-light-proxy sub-instances. The actual AnalyticProxyLightIndex is
            // filled per-frame by ResolveAnalyticProxyLights once the light global indices exist;
            // here we only capture (subInstanceIndex, targetLightInstanceID). subInstList.Count
            // equals the index this entry is about to occupy.
            if (slot.EnableAsAnalyticLightProxy)
            {
                Light targetLight = ResolveProxyTargetLight(rec.TargetRenderer);
                _proxySubInstances.Add((subInstList.Count, targetLight != null ? targetLight.GetInstanceID() : 0));
                if (targetLight == null)
                    Debug.LogWarning($"[NativeRtxptGPUScene] Renderer '{rec.TargetRenderer.name}' submesh {s} is flagged " +
                                     "EnableAsAnalyticLightProxy but has no Spot/Point target light " +
                                     "(add an RtxptAnalyticLightProxy component or parent it under a light).");
            }

            subInstList.Add(new SubInstanceData
            {
                FlagsAndAlphaInfo                       = siFlags,
                GlobalGeometryIndex_PTMaterialDataIndex = ((uint)globalGeomIdx << 16) | ((uint)matIdx & 0xFFFFu),
                EmissiveLightMappingOffset              = 0xFFFFFFFFu, // set per-frame by PrepareEmissiveTriangleTasks
                // Default to RTXPT_INVALID_LIGHT_INDEX (disables the proxy lookup, which the
                // shader only performs for EnableAsAnalyticLightProxy materials; 0u would alias
                // env-quad light 0). For proxy materials this is overwritten each frame by
                // ResolveAnalyticProxyLights once the analytic light global indices are known.
                AnalyticProxyLightIndex                 = 0xFFFFFFFFu,
                IndexBufferIndex_VertexBufferIndex      = ((uint)slots.ib << 16) | ((uint)slots.vb & 0xFFFFu),
                IndexOffset                             = (uint)sub.indexStart * 4u,
                TexCoord1Offset                         = streams.Uv,
                padding0                                = 0u,
            });

            // --- GeometryDebugData (RTXPT t4, all zero = no OMM) ---
            geomDbgList.Add(default);
        }

        private void RebuildEmissiveCache()
        {
            _emissiveCache.Clear();
            if (_instanceCpu == null || _geometryCpu == null)
                return;

            for (int i = 0; i < _instanceCpu.Length; i++)
            {
                var inst      = _instanceCpu[i];
                int firstGeom = (int)inst.firstGeometryIndex;
                int numGeoms  = (int)inst.numGeometries;

                for (int s = 0; s < numGeoms; s++)
                {
                    int geomIdx = firstGeom + s;
                    if (geomIdx >= _geometryCpu.Length) break;

                    var geom   = _geometryCpu[geomIdx];
                    int matIdx = (int)geom.materialIndex;
                    if (matIdx < 0 || matIdx >= _ptMaterialCpu.Length) continue;

                    var mat = _ptMaterialCpu[matIdx];
                    if (mat.EmissiveColor.x <= 0f && mat.EmissiveColor.y <= 0f && mat.EmissiveColor.z <= 0f)
                        continue;

                    _emissiveCache.Add(new EmissiveGeometryEntry
                    {
                        InstanceIndex              = i,
                        GeometrySubIndex           = s,
                        TriangleCount              = geom.numIndices / 3u,
                        FirstGeometryInstanceIndex = inst.firstGeometryInstanceIndex,
                    });
                }
            }
        }

        // -----------------------------------------------------------------------
        // Per-frame: material edits, transforms, lighting bridge
        // -----------------------------------------------------------------------

        /// <summary>
        /// Checks all registered targets for dirty <see cref="RtxptRenderer"/> components
        /// and, if any are found, refreshes only the affected entries in <c>_ptMaterialCpu</c> and
        /// re-uploads the material buffer.  Texture assignments are not changed; only scalar/color/flag
        /// parameters are updated.  If you also change texture assignments, call
        /// <see cref="MarkRebuildDirty"/> instead to trigger a full scene rebuild.
        /// </summary>
        private void CheckAndUpdateMaterialOverrides()
        {
            if (_ptMaterialCpu == null || _ptMaterialGpuBuf == null) return;

            int dirtyMin = int.MaxValue;
            int dirtyMax = int.MinValue;

            foreach (var (matOverride, matIndices) in _overrideCache)
            {
                if (matOverride == null || !matOverride.IsDirty) continue;

                for (int s = 0; s < matIndices.Length && s < matOverride.Slots.Count; s++)
                {
                    if (matOverride.Slots[s] == null) continue;
                    int idx = matIndices[s];
                    if (idx < 0 || idx >= _ptMaterialCpu.Length) continue;
                    RtxptMaterialTable.RefreshScalars(matOverride.Slots[s], ref _ptMaterialCpu[idx]);
                    if (idx < dirtyMin) dirtyMin = idx;
                    if (idx > dirtyMax) dirtyMax = idx;
                }

                matOverride.ClearDirty();
            }

            if (dirtyMin <= dirtyMax)
            {
                _ptMaterialGpuBuf.SetData(_ptMaterialCpu, dirtyMin, dirtyMin, dirtyMax - dirtyMin + 1);
                _ptMaterialGpuBufPtr = _ptMaterialGpuBuf.GetNativeBufferPtr();
            }
        }

        private void UpdateInstanceTransforms()
        {
            if (_instanceCpu == null || _instanceGpuBuf == null) return;

            foreach (var entry in _rendererEntries.Values)
            {
                if (entry.transform == null) continue;

                int start = entry.firstInstanceIdx;
                int count = Mathf.Min(entry.instanceCount, _instanceCpu.Length - start);
                if (count <= 0) continue;

                // Skinned instances deform every frame: update unconditionally from the root
                // bone, with last frame's root as the previous transform (motion vectors).
                if (entry.isSkinned)
                {
                    Matrix4x4 cur  = entry.transform.localToWorldMatrix;
                    Matrix4x4 prev = entry.hasLastRoot ? entry.lastRoot : cur;

                    var curRow0  = new Vector4(cur.m00, cur.m01, cur.m02, cur.m03);
                    var curRow1  = new Vector4(cur.m10, cur.m11, cur.m12, cur.m13);
                    var curRow2  = new Vector4(cur.m20, cur.m21, cur.m22, cur.m23);
                    var prevRow0 = new Vector4(prev.m00, prev.m01, prev.m02, prev.m03);
                    var prevRow1 = new Vector4(prev.m10, prev.m11, prev.m12, prev.m13);
                    var prevRow2 = new Vector4(prev.m20, prev.m21, prev.m22, prev.m23);

                    for (int i = start; i < start + count; i++)
                    {
                        _instanceCpu[i].prevTransformRow0 = prevRow0;
                        _instanceCpu[i].prevTransformRow1 = prevRow1;
                        _instanceCpu[i].prevTransformRow2 = prevRow2;
                        _instanceCpu[i].transformRow0     = curRow0;
                        _instanceCpu[i].transformRow1     = curRow1;
                        _instanceCpu[i].transformRow2     = curRow2;

                        _accelStructure.SetInstanceTransform(_instanceHandles[i], cur);
                    }

                    _instanceGpuBuf.SetData(_instanceCpu, start, count);
                    entry.lastRoot    = cur;
                    entry.hasLastRoot = true;
                    continue;
                }

                bool moved = entry.transform.hasChanged;
                if (moved) entry.transform.hasChanged = false;

                if (!moved && !entry.wasMoving) continue;

                if (moved)
                {
                    Matrix4x4 m    = entry.transform.localToWorldMatrix;
                    var       row0 = new Vector4(m.m00, m.m01, m.m02, m.m03);
                    var       row1 = new Vector4(m.m10, m.m11, m.m12, m.m13);
                    var       row2 = new Vector4(m.m20, m.m21, m.m22, m.m23);

                    for (int i = start; i < start + count; i++)
                    {
                        _instanceCpu[i].prevTransformRow0 = _instanceCpu[i].transformRow0;
                        _instanceCpu[i].prevTransformRow1 = _instanceCpu[i].transformRow1;
                        _instanceCpu[i].prevTransformRow2 = _instanceCpu[i].transformRow2;
                        _instanceCpu[i].transformRow0     = row0;
                        _instanceCpu[i].transformRow1     = row1;
                        _instanceCpu[i].transformRow2     = row2;

                        _accelStructure.SetInstanceTransform(_instanceHandles[i], m);
                    }
                }
                else
                {
                    // First stationary frame: sync prev = current so the shader sees no ghost motion.
                    for (int i = start; i < start + count; i++)
                    {
                        _instanceCpu[i].prevTransformRow0 = _instanceCpu[i].transformRow0;
                        _instanceCpu[i].prevTransformRow1 = _instanceCpu[i].transformRow1;
                        _instanceCpu[i].prevTransformRow2 = _instanceCpu[i].transformRow2;
                    }
                }

                // Pointer is stable; no GetNativeBufferPtr re-fetch needed. The GPU copy
                // is recorded later by FlushInstanceBuffer(cmd) in the TLAS build pass.
                _instanceGpuBuf.SetData(_instanceCpu, start, count);
                entry.wasMoving = moved;
            }
        }

        /// <summary>
        /// Resolves the target Spot/Point <see cref="Light"/> a proxy mesh stands in for: an explicit
        /// <see cref="RtxptAnalyticLightProxy"/> reference if present, otherwise the nearest Spot/Point
        /// light on this GameObject or an ancestor (mirrors RTXPT's ProxiedAnalyticLight / parent-node
        /// rules). Returns null if no eligible light is found.
        /// </summary>
        private static Light ResolveProxyTargetLight(Component renderer)
        {
            var proxy = renderer.GetComponent<RtxptAnalyticLightProxy>();
            if (proxy != null && proxy.TargetLight != null)
                return proxy.TargetLight;

            for (var t = renderer.transform; t != null; t = t.parent)
            {
                var light = t.GetComponent<Light>();
                if (light != null && (light.type == LightType.Spot || light.type == LightType.Point))
                    return light;
            }
            return null;
        }

        /// <summary>
        /// Per-frame: write each proxy sub-instance's <c>AnalyticProxyLightIndex</c> from the current
        /// frame's analytic-light map (Unity <see cref="Light"/> InstanceID → global light index).
        /// Lights absent from the map (disabled/culled, or never collected) fall back to
        /// RTXPT_INVALID_LIGHT_INDEX, which disables the proxy lookup for that frame.
        ///
        /// Must be called before the SubInstanceData re-upload performed by
        /// <see cref="PrepareEmissiveTriangleTasks"/>, which is what pushes these writes to the GPU.
        /// </summary>
        public void ResolveAnalyticProxyLights(IReadOnlyDictionary<int, uint> lightIndexMap)
        {
            if (_subInstanceCpu == null || _proxySubInstances.Count == 0) return;

            const uint Invalid = 0xFFFFFFFFu;
            foreach (var (subIdx, lightId) in _proxySubInstances)
            {
                if (subIdx < 0 || subIdx >= _subInstanceCpu.Length) continue;

                uint globalIndex = Invalid;
                if (lightId != 0 && lightIndexMap != null && lightIndexMap.TryGetValue(lightId, out var idx))
                    globalIndex = idx;

                if (_subInstanceCpu[subIdx].AnalyticProxyLightIndex != globalIndex)
                {
                    _subInstanceCpu[subIdx].AnalyticProxyLightIndex = globalIndex;
                    _subInstanceCpuDirty = true;
                }
            }
        }

        /// <summary>
        /// CPU-side emissive-triangle pass: mirrors <c>LightsBaker::ProcessEmissiveGeometry</c>.
        /// Generates <see cref="RtxptEmissiveTrianglesProcTask"/> entries, uploads them to
        /// <paramref name="scratchBuffer"/>, updates <c>SubInstanceData.EmissiveLightMappingOffset</c>
        /// for emissive geometries, and re-uploads the sub-instance GPU buffer when anything
        /// actually changed. Must be called on the main thread after <see cref="UpdateForFrame"/>
        /// and before command-buffer recording (i.e. from a pass <c>Setup</c> method).
        /// </summary>
        /// <param name="lightOffset">Base index in lightsBuffer where triangle lights start
        ///     (= EnvQtTotalNodeCount + analyticLightCount).</param>
        /// <param name="scratchBuffer">Raw native buffer bound as u_scratchBuffer.
        ///     Tasks are written at element offset 0 (each element = 4 bytes; tasks are 32 B each,
        ///     so stride-8 within the raw buffer).</param>
        public void PrepareEmissiveTriangleTasks(uint lightOffset, UploadBuffer scratchBuffer)
        {
            if (_instanceCpu == null || _subInstanceCpu == null || _ptMaterialCpu == null)
            {
                LastEmissiveTaskCount     = 0;
                LastEmissiveTriangleCount = 0u;
                return;
            }

            const uint Invalid       = 0xFFFFFFFFu;
            const uint MaxTriPerTask = 32u;

            var newHistoric = _emissiveHistoricScratch;
            newHistoric.Clear();
            int  taskIdx        = 0;
            uint accumTriangles = 0u;

            foreach (var e in _emissiveCache)
            {
                if (taskIdx >= MaxEmissiveProcTasks)
                {
                    Debug.LogWarning("[NativeRtxptGPUScene] EmissiveTrianglesProcTask overflow — some emissive geometry ignored.");
                    break;
                }

                uint triCount = e.TriangleCount;
                uint destBase = lightOffset + accumTriangles;

                // Overflow guard
                if (destBase + triCount > NativeRtxptBufferResources.MaxLights)
                {
                    Debug.LogWarning($"[NativeRtxptGPUScene] MaxLights overflow at emissive geometry (inst={e.InstanceIndex}, geom={e.GeometrySubIndex}) — skipping.");
                    break;
                }

                if (!_emissiveHistoricOffsets.TryGetValue((e.InstanceIndex, e.GeometrySubIndex), out uint historicBase))
                    historicBase = Invalid;

                // Update SubInstanceData.EmissiveLightMappingOffset
                int siIdx = (int)(e.FirstGeometryInstanceIndex + (uint)e.GeometrySubIndex);
                if (siIdx >= 0 && siIdx < _subInstanceCpu.Length &&
                    _subInstanceCpu[siIdx].EmissiveLightMappingOffset != destBase)
                {
                    _subInstanceCpu[siIdx].EmissiveLightMappingOffset = destBase;
                    _subInstanceCpuDirty = true;
                }

                // Split into tasks of at most MaxTriPerTask triangles.
                // Each task writes to DestinationBufferOffset + subIndex (0..31), so successive
                // tasks must each advance the offset by MaxTriPerTask to avoid aliasing.
                for (uint from = 0u; from < triCount && taskIdx < MaxEmissiveProcTasks; from += MaxTriPerTask)
                {
                    uint to = Math.Min(from + MaxTriPerTask, triCount);
                    s_emissiveTaskStaging[taskIdx++] = new RtxptEmissiveTrianglesProcTask
                    {
                        InstanceIndex              = (uint)e.InstanceIndex,
                        GeometryIndex              = (uint)e.GeometrySubIndex,
                        TriangleIndexFrom          = from,
                        TriangleIndexTo            = to,
                        DestinationBufferOffset    = destBase + from,   // each task owns its own 32-slot window
                        HistoricBufferOffset       = (historicBase != Invalid) ? historicBase + from : Invalid,
                        EmissiveLightMappingOffset = (uint)siIdx,
                        Padding0                   = 0u,
                    };
                }

                newHistoric[(e.InstanceIndex, e.GeometrySubIndex)] = destBase;
                accumTriangles += triCount;
            }

            // Swap historic offsets for next frame (no per-frame dictionary allocation).
            (_emissiveHistoricOffsets, _emissiveHistoricScratch) = (newHistoric, _emissiveHistoricOffsets);

            // Re-upload SubInstanceData only when an offset or proxy-light index actually
            // changed this frame. Pointer is stable; the GPU copy is recorded by
            // FlushSubInstanceBuffer(cmd) in the lighting pass.
            if (_subInstanceCpuDirty && _subInstanceGpuBuf != null)
            {
                _subInstanceGpuBuf.SetData(_subInstanceCpu, 0, _subInstanceCpu.Length);
                _subInstanceCpuDirty = false;
            }

            // Upload task array to scratch buffer (raw buffer, stride = 4 bytes, tasks = 8 uints each).
            if (taskIdx > 0 && scratchBuffer != null)
                scratchBuffer.SetRawData(s_emissiveTaskStaging, 0, 0, taskIdx);

            LastEmissiveTaskCount     = taskIdx;
            LastEmissiveTriangleCount = accumTriangles;
        }
    }

    /// <summary>
    /// One skinned-repack compute dispatch: converts a SkinnedMeshRenderer's GPU-skinned vertex
    /// buffer (interleaved, root-bone space) into the instance's donut SoA buffer each frame,
    /// maintaining the PrevPosition stream (donut skinning_cs.hlsl model). Built per topology
    /// change by <see cref="NativeRtxptGPUScene"/>; recorded each frame by NativeRtxptBuildTlasPass
    /// before the TLAS/BLAS build.
    /// </summary>
    internal sealed class RtxptSkinnedDispatch
    {
        public const uint FlagFirstFrame = 1u << 0; // prev = current (no history yet)
        public const uint FlagHasNormal  = 1u << 1;
        public const uint FlagHasTangent = 1u << 2;

        public SkinnedMeshRenderer  Smr;
        public RtxptSkinnedGeometry Geometry;
        public int                  VertexCount;

        // Unity skinned-VB layout (stream 0 of the shared mesh).
        public uint SrcStride;
        public uint SrcPosOffset;
        public uint SrcNormalOffset;
        public uint SrcTangentOffset;

        // Destination donut SoA stream offsets (includes the PrevPosition stream).
        public RtxptMeshStreamOffsets Streams;

        public uint BaseFlags; // FlagHasNormal / FlagHasTangent
    }
}
