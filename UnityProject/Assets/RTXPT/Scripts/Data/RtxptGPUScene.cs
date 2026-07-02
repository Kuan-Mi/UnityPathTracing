using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using NativeRender;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;
using RayTracingAccelerationStructure = NativeRender.RayTracingAccelerationStructure;

namespace PathTracing
{
    // =========================================================================
    // RtxptGPUScene
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
    public sealed class RtxptGPUScene : IDisposable
    {
        // Acceleration structure + incremental TLAS registration
        private          RayTracingAccelerationStructure _accelStructure;
        private readonly RtxptAccelRegistry              _accelRegistry = new();

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
        private GraphicsBuffer _ptMaterialGpuBuf; // t_PTMaterialData     (t5)
        private GraphicsBuffer _geomDebugGpuBuf; // t_GeometryDebugData  (t4)

        // _instanceGpuBuf / _subInstanceGpuBuf are UploadBuffers — bound by handle (no cached ptr).
        private IntPtr _geometryGpuBufPtr;
        private IntPtr _ptMaterialGpuBufPtr;
        private IntPtr _geomDebugGpuBufPtr;

        // Bindless
        private BindlessBuffer  _sceneBuffers;
        private BindlessTexture _sceneTextures;
        private GraphicsBuffer  _mergedStaticVertexBuffer;
        private GraphicsBuffer  _mergedStaticIndexBuffer;

        // CPU-side mirrors
        private DonutInstanceData[] _instanceCpu;
        private DonutGeometryData[] _geometryCpu;
        private SubInstanceData[]   _subInstanceCpu;
        private PTMaterialData[]    _ptMaterialCpu;
        private GeometryDebugData[] _geomDebugCpu;
        // Per-material "reserve emissive triangle-light slots" flag, parallel to _ptMaterialCpu.
        // Mirrors RTXPT PTMaterial::IsEmissive(): baked-emissive OR UseDonutEmissiveIntensity
        // (the latter reserves slots for materials whose emission can animate on at runtime).
        private bool[]              _ptMaterialReserveEmissive;

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
            public bool      wasMoving = true; // start true so first frame always syncs prev = current
            public int       firstInstanceIdx;
            public int       instanceCount;

            // Skinned instances deform every frame: transforms update unconditionally from the
            // root bone, with last frame's root kept for prevTransform (motion vectors).
            public bool      isSkinned;
            public Matrix4x4 lastRoot;
            public bool      hasLastRoot;
        }

        private readonly Dictionary<int, RendererEntry>    _rendererEntries = new();
        private readonly Dictionary<int, MergedMeshOffsets> _mergedMeshOffsets = new();

        // Skinned instances: bindless slot pair per renderer (own SoA VB + shared donut IB),
        // and the set of mesh ids referenced this rebuild (for geometry-cache eviction).
        private readonly Dictionary<int, (int vb, int ib)> _skinnedBufferSlots = new();
        private readonly HashSet<int>                      _usedMeshIds        = new();

        private readonly struct GeometryRangeKey : IEquatable<GeometryRangeKey>
        {
            private readonly int  _meshId;
            private readonly bool _skinned;
            private readonly int  _rendererId;
            private readonly int  _vb;
            private readonly int  _ib;
            private readonly uint _indexBaseBytes;
            private readonly uint _positionOffset;
            private readonly uint _prevPositionOffset;
            private readonly uint _normalOffset;
            private readonly uint _texCoord1Offset;
            private readonly uint _tangentOffset;
            private readonly int  _hash;

            public readonly int[] Submeshes;
            public readonly int[] Materials;

            public GeometryRangeKey(RtxptInstanceRecord rec, (int vb, int ib) slots,
                in MergedMeshOffsets offsets, int[] submeshes, int[] materials)
            {
                _meshId             = rec.Mesh.GetInstanceID();
                _skinned            = rec.IsSkinned;
                _rendererId         = rec.IsSkinned ? rec.RendererId : 0;
                _vb                 = slots.vb;
                _ib                 = slots.ib;
                _indexBaseBytes     = offsets.IndexBaseBytes;
                _positionOffset     = offsets.PositionOffset;
                _prevPositionOffset = offsets.PrevPositionOffset;
                _normalOffset       = offsets.NormalOffset;
                _texCoord1Offset    = offsets.TexCoord1Offset;
                _tangentOffset      = offsets.TangentOffset;
                Submeshes           = submeshes;
                Materials           = materials;

                unchecked
                {
                    int h = _meshId;
                    h = (h * 397) ^ _skinned.GetHashCode();
                    h = (h * 397) ^ _rendererId;
                    h = (h * 397) ^ _vb;
                    h = (h * 397) ^ _ib;
                    h = (h * 397) ^ (int)_indexBaseBytes;
                    h = (h * 397) ^ (int)_positionOffset;
                    h = (h * 397) ^ (int)_prevPositionOffset;
                    h = (h * 397) ^ (int)_normalOffset;
                    h = (h * 397) ^ (int)_texCoord1Offset;
                    h = (h * 397) ^ (int)_tangentOffset;
                    for (int i = 0; i < Submeshes.Length; i++)
                    {
                        h = (h * 397) ^ Submeshes[i];
                        h = (h * 397) ^ Materials[i];
                    }
                    _hash = h;
                }
            }

            public bool Equals(GeometryRangeKey other)
            {
                if (_meshId != other._meshId || _skinned != other._skinned ||
                    _rendererId != other._rendererId || _vb != other._vb || _ib != other._ib ||
                    _indexBaseBytes != other._indexBaseBytes || _positionOffset != other._positionOffset ||
                    _prevPositionOffset != other._prevPositionOffset || _normalOffset != other._normalOffset ||
                    _texCoord1Offset != other._texCoord1Offset || _tangentOffset != other._tangentOffset ||
                    Submeshes.Length != other.Submeshes.Length)
                    return false;

                for (int i = 0; i < Submeshes.Length; i++)
                {
                    if (Submeshes[i] != other.Submeshes[i] || Materials[i] != other.Materials[i])
                        return false;
                }
                return true;
            }

            public override bool Equals(object obj) => obj is GeometryRangeKey other && Equals(other);
            public override int GetHashCode() => _hash;
        }

        private readonly struct MergedMeshOffsets
        {
            public readonly uint IndexBaseBytes;
            public readonly uint PositionOffset;
            public readonly uint PrevPositionOffset;
            public readonly uint NormalOffset;
            public readonly uint TexCoord1Offset;
            public readonly uint TangentOffset;

            public MergedMeshOffsets(uint indexBaseBytes, uint positionOffset, uint prevPositionOffset,
                uint normalOffset, uint texCoord1Offset, uint tangentOffset)
            {
                IndexBaseBytes      = indexBaseBytes;
                PositionOffset      = positionOffset;
                PrevPositionOffset  = prevPositionOffset;
                NormalOffset        = normalOffset;
                TexCoord1Offset     = texCoord1Offset;
                TangentOffset       = tangentOffset;
            }
        }

        // One repack dispatch per skinned renderer, rebuilt on topology change and consumed by
        // RtxptBuildTlasPass every frame before the TLAS/BLAS build.
        private readonly List<RtxptSkinnedDispatch> _skinnedDispatches  = new();
        private readonly HashSet<int>               _skinnedSeenScratch = new();

        internal IReadOnlyList<RtxptSkinnedDispatch> SkinnedDispatches => _skinnedDispatches;

        private readonly List<RtxptRenderer> _registeredTargets   = new();
        private          int                 _lastTopologyVersion = -1;

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
        private const int MaxEmissiveProcTasks = RtxptBufferResources.MaxLights / 32 * 2;

        private static readonly RtxptEmissiveTrianglesProcTask[] s_emissiveTaskStaging =
            new RtxptEmissiveTrianglesProcTask[MaxEmissiveProcTasks];

        /// <summary>Number of tasks produced by the last <see cref="PrepareEmissiveTriangleTasks"/> call.</summary>
        public int LastEmissiveTaskCount { get; private set; }

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

        public RtxptGPUScene()
        {
            _accelStructure = new RayTracingAccelerationStructure(new RayTracingAccelerationStructureOptions
            {
                UseRtxmu      = false,
                UseCompaction = false
            });
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
                _forceRebuild        = false;
                _sceneGpuDirty       = true;
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
                    Debug.LogError($"[RtxptGPUScene] Skinned mesh '{mesh.name}': position is not a float32 stream-0 attribute — skinned repack skipped (geometry will stay in rest pose).");
                    continue;
                }

                int stride = mesh.GetVertexBufferStride(0);
                int posOff = mesh.GetVertexAttributeOffset(VertexAttribute.Position);
                int normOff = mesh.HasVertexAttribute(VertexAttribute.Normal) &&
                              mesh.GetVertexAttributeStream(VertexAttribute.Normal) == 0 &&
                              mesh.GetVertexAttributeFormat(VertexAttribute.Normal) == VertexAttributeFormat.Float32
                    ? mesh.GetVertexAttributeOffset(VertexAttribute.Normal)
                    : -1;
                int tanOff = mesh.HasVertexAttribute(VertexAttribute.Tangent) &&
                             mesh.GetVertexAttributeStream(VertexAttribute.Tangent) == 0 &&
                             mesh.GetVertexAttributeFormat(VertexAttribute.Tangent) == VertexAttributeFormat.Float32
                    ? mesh.GetVertexAttributeOffset(VertexAttribute.Tangent)
                    : -1;

                var  streams                                 = new RtxptMeshStreamOffsets(mesh, withPrevPosition: true);
                uint flags                                   = 0;
                if (normOff >= 0 && streams.HasNormal) flags |= RtxptSkinnedDispatch.FlagHasNormal;
                if (tanOff >= 0 && streams.HasTangent) flags |= RtxptSkinnedDispatch.FlagHasTangent;


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
            _geometryGpuBuf    = null;
            _geometryGpuBufPtr = IntPtr.Zero;
            _subInstanceGpuBuf?.Dispose();
            _subInstanceGpuBuf = null;
            _ptMaterialGpuBuf?.Release();
            _ptMaterialGpuBuf    = null;
            _ptMaterialGpuBufPtr = IntPtr.Zero;
            _geomDebugGpuBuf?.Release();
            _geomDebugGpuBuf    = null;
            _geomDebugGpuBufPtr = IntPtr.Zero;
            _mergedStaticVertexBuffer?.Release();
            _mergedStaticVertexBuffer = null;
            _mergedStaticIndexBuffer?.Release();
            _mergedStaticIndexBuffer = null;
            _sceneBuffers?.Dispose();
            _sceneBuffers = null;
            _sceneTextures?.Dispose();
            _sceneTextures  = null;
            _instanceCpu    = null;
            _geometryCpu    = null;
            _subInstanceCpu = null;
            _ptMaterialCpu  = null;
            _ptMaterialReserveEmissive = null;
            _geomDebugCpu   = null;
            _instanceHandles.Clear();
            _rendererEntries.Clear();
            _mergedMeshOffsets.Clear();
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
            var geometryRangeCache = new Dictionary<GeometryRangeKey, int>();

            BuildMergedStaticGeometryBuffers();
            (int vb, int ib) staticSlots = (-1, -1);
            if (_mergedStaticIndexBuffer != null && _mergedStaticVertexBuffer != null)
            {
                // Donut creates the index descriptor before the vertex descriptor.
                staticSlots = (vb: 1, ib: 0);
                bufPtrs.Add(_mergedStaticIndexBuffer.GetNativeBufferPtr());
                bufPtrs.Add(_mergedStaticVertexBuffer.GetNativeBufferPtr());
            }

            foreach (var rec in _layout)
            {
                Mesh mesh   = rec.Mesh;
                int  meshId = mesh.GetInstanceID();
                _usedMeshIds.Add(meshId);

                (int vb, int ib) slots;
                MergedMeshOffsets merged = default;
                if (rec.IsSkinned)
                {
                    // Per-instance SoA VB (rewritten each frame by the repack compute) + shared
                    // per-mesh donut IB, registered as their own bindless slot pair per renderer.
                    if (!_skinnedBufferSlots.TryGetValue(rec.RendererId, out slots))
                    {
                        slots = (vb: bufPtrs.Count + 1, ib: bufPtrs.Count);
                        if (slots.vb > 0xFFFF)
                            Debug.LogError($"[RtxptGPUScene] Bindless buffer slot overflow: VB slot index {slots.vb} exceeds 16-bit limit (65535). Rendering will be corrupted. Mesh='{mesh.name}'.");
                        bufPtrs.Add(rec.SkinnedIb.GetNativeBufferPtr());
                        bufPtrs.Add(rec.SkinnedVb.GetNativeBufferPtr());
                        _skinnedBufferSlots[rec.RendererId] = slots;
                    }
                }
                else
                {
                    slots = staticSlots;
                    if (!_mergedMeshOffsets.TryGetValue(meshId, out merged))
                    {
                        Debug.LogError($"[RtxptGPUScene] Missing merged static geometry offsets for mesh '{mesh.name}'.");
                        merged = new MergedMeshOffsets(0u, 0u, 0u,
                            RtxptMeshStreamOffsets.Absent, RtxptMeshStreamOffsets.Absent, RtxptMeshStreamOffsets.Absent);
                    }

                    // if (!mesh.HasVertexAttribute(VertexAttribute.Normal) ||
                    //     !mesh.HasVertexAttribute(VertexAttribute.Tangent))
                    //     Debug.LogWarning($"[RtxptGPUScene] '{mesh.name}': missing normal or tangent stream");
                }

                var streams = rec.IsSkinned ? new RtxptMeshStreamOffsets(mesh, withPrevPosition: true) : default;
                var offsets = rec.IsSkinned
                    ? new MergedMeshOffsets(0u, streams.Pos, streams.PrevPos, streams.Normal, streams.Uv, streams.Tangent)
                    : merged;

                // Per-renderer material-index array for the lightweight material-edit path.
                // A renderer's records are consecutive in the layout; the array spans all of
                // its sub-meshes, so create it on the renderer's first record.
                if (!_overrideMaterialIndices.TryGetValue(rec.RendererId, out int[] overrideMatIndices))
                {
                    overrideMatIndices                       = new int[mesh.subMeshCount];
                    _overrideMaterialIndices[rec.RendererId] = overrideMatIndices;
                    _overrideCache.Add((rec.Renderer, overrideMatIndices));
                }

                int[] materialIndices = ResolveMaterialIndices(rec, matTable, overrideMatIndices);
                var   rangeKey        = new GeometryRangeKey(rec, slots, offsets, rec.SubmeshIndices, materialIndices);
                if (!geometryRangeCache.TryGetValue(rangeKey, out int firstGeometryIndex))
                {
                    firstGeometryIndex = geomList.Count;
                    geometryRangeCache.Add(rangeKey, firstGeometryIndex);
                    for (int i = 0; i < rec.SubmeshIndices.Length; i++)
                        AddGeometryData(rec, rec.SubmeshIndices[i], materialIndices[i], slots, offsets, geomList, geomDbgList);
                }

                int firstSubInstance = subInstList.Count;
                for (int i = 0; i < rec.SubmeshIndices.Length; i++)
                    AddSubInstanceData(rec, rec.SubmeshIndices[i], materialIndices[i],
                        firstGeometryIndex + i, slots, offsets, matTable, subInstList);

                // TLAS InstanceID must be the SubInstanceData base offset (firstGeometryInstanceIndex),
                // NOT the instance index: the any-hit alpha test indexes t_SubInstanceData[InstanceID()
                // + geometryIndex] directly (PathTracerBridgeDonut.hlsli AlphaTest), mirroring RTXPT's
                // instanceDesc.instanceID = instance->GetGeometryInstanceIndex() (Sample.cpp). The
                // closest-hit path is unaffected — it uses the automatic InstanceIndex() to fetch the
                // instance, then adds firstGeometryInstanceIndex itself.
                _accelStructure.SetInstanceID(rec.Handle, (uint)firstSubInstance);

                Matrix4x4 m    = RtxptSceneLayout.GetRootTransform(rec);
                var       row0 = new Vector4(m.m00, m.m01, m.m02, m.m03);
                var       row1 = new Vector4(m.m10, m.m11, m.m12, m.m13);
                var       row2 = new Vector4(m.m20, m.m21, m.m22, m.m23);

                instList.Add(new DonutInstanceData
                {
                    flags                      = 0u,
                    firstGeometryInstanceIndex = (uint)firstSubInstance,
                    firstGeometryIndex         = (uint)firstGeometryIndex,
                    numGeometries              = (uint)rec.SubmeshIndices.Length,
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
                matTable.ReserveEmissiveSlots.Add(false);
                geomDbgList.Add(default);
            }

            _instanceCpu    = instList.ToArray();
            _geometryCpu    = geomList.ToArray();
            _subInstanceCpu = subInstList.ToArray();
            _ptMaterialCpu  = matTable.Materials.ToArray();
            _geomDebugCpu   = geomDbgList.ToArray();
            _ptMaterialReserveEmissive = matTable.ReserveEmissiveSlots.ToArray();

            // Whole mode: one CopyBufferRegion spanning the dirty [min,max) element span per
            // flush, matching donut's WriteInstanceBuffer single full-array writeBuffer (one
            // copy per frame) instead of one copy per moved renderer. At 112 B/instance the
            // extra bytes in the span are cheaper than dozens of tiny copy commands.
            // Debug names match the original RTXPT/donut debugName strings (donut Scene.cpp
            // "Instances"/"BindlessGeometry", Sample.cpp "Instances", MaterialsBaker.cpp
            // "PTMaterialDataStorage", OmmBaker.cpp "BindlessGeometryDebug") for PIX parity.
            _instanceGpuBuf = new UploadBuffer(_instanceCpu.Length, Marshal.SizeOf<DonutInstanceData>(), UploadBuffer.UploadMode.Whole, debugName: "Instances");
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

        private void BuildMergedStaticGeometryBuffers()
        {
            var meshes     = new List<(Mesh mesh, int meshId, RtxptMeshStreamOffsets streams, NativeArray<uint> vb, uint vertexBase, uint indexBaseBytes)>();
            var indexWords = new List<uint>();
            var seenMeshes = new HashSet<int>();

            uint totalVertices = 0u;
            bool hasNormal     = false;
            bool hasUv         = false;
            bool hasTangent    = false;

            foreach (var rec in _layout)
            {
                if (rec.IsSkinned) continue;

                Mesh mesh = rec.Mesh;
                if (mesh == null) continue;

                int meshId = mesh.GetInstanceID();
                if (!seenMeshes.Add(meshId)) continue;

                var streams = new RtxptMeshStreamOffsets(mesh);
                using NativeArray<uint> vb = RtxptGeometryCache.BuildSoAVertexData(mesh, streams);
                using NativeArray<uint> ib = RtxptGeometryCache.BuildIndexData(mesh);

                uint vertexBase    = totalVertices;
                uint indexBaseBytes = (uint)indexWords.Count * 4u;

                hasNormal  |= streams.HasNormal;
                hasUv      |= streams.HasUv;
                hasTangent |= streams.HasTangent;

                for (int i = 0; i < ib.Length; i++)
                    indexWords.Add(ib[i]);

                var vbCopy = new NativeArray<uint>(vb.Length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                NativeArray<uint>.Copy(vb, vbCopy);
                meshes.Add((mesh, meshId, streams, vbCopy, vertexBase, indexBaseBytes));
                totalVertices += (uint)mesh.vertexCount;
            }

            if (seenMeshes.Count == 0)
                return;

            uint posBase     = 0u;
            uint writeCursor = Align16(totalVertices * 12u);
            uint normalBase  = RtxptMeshStreamOffsets.Absent;
            uint tangentBase = RtxptMeshStreamOffsets.Absent;
            uint uvBase      = RtxptMeshStreamOffsets.Absent;

            if (hasNormal)
            {
                normalBase  = writeCursor;
                writeCursor = Align16(writeCursor + totalVertices * 4u);
            }

            if (hasTangent)
            {
                tangentBase = writeCursor;
                writeCursor = Align16(writeCursor + totalVertices * 4u);
            }

            if (hasUv)
            {
                uvBase      = writeCursor;
                writeCursor = Align16(writeCursor + totalVertices * 8u);
            }

            var vertexWords = new uint[Mathf.Max((int)(writeCursor / 4u), 1)];

            foreach (var entry in meshes)
            {
                int  vc                 = entry.mesh.vertexCount;
                uint meshPositionOffset = posBase + entry.vertexBase * 12u;
                uint meshNormalOffset   = normalBase != RtxptMeshStreamOffsets.Absent
                    ? normalBase + entry.vertexBase * 4u
                    : RtxptMeshStreamOffsets.Absent;
                uint meshTangentOffset = tangentBase != RtxptMeshStreamOffsets.Absent
                    ? tangentBase + entry.vertexBase * 4u
                    : RtxptMeshStreamOffsets.Absent;
                uint meshUvOffset = uvBase != RtxptMeshStreamOffsets.Absent
                    ? uvBase + entry.vertexBase * 8u
                    : RtxptMeshStreamOffsets.Absent;

                CopyWords(entry.vb, entry.streams.Pos, vertexWords, meshPositionOffset, (uint)vc * 12u);
                if (entry.streams.HasNormal && meshNormalOffset != RtxptMeshStreamOffsets.Absent)
                    CopyWords(entry.vb, entry.streams.Normal, vertexWords, meshNormalOffset, (uint)vc * 4u);
                if (entry.streams.HasTangent && meshTangentOffset != RtxptMeshStreamOffsets.Absent)
                    CopyWords(entry.vb, entry.streams.Tangent, vertexWords, meshTangentOffset, (uint)vc * 4u);
                if (entry.streams.HasUv && meshUvOffset != RtxptMeshStreamOffsets.Absent)
                    CopyWords(entry.vb, entry.streams.Uv, vertexWords, meshUvOffset, (uint)vc * 8u);

                _mergedMeshOffsets[entry.meshId] = new MergedMeshOffsets(
                    indexBaseBytes: 0u, // overwritten below
                    positionOffset: meshPositionOffset,
                    prevPositionOffset: meshPositionOffset,
                    normalOffset: meshNormalOffset,
                    texCoord1Offset: meshUvOffset,
                    tangentOffset: meshTangentOffset);
            }

            foreach (var entry in meshes)
            {
                var existing = _mergedMeshOffsets[entry.meshId];
                _mergedMeshOffsets[entry.meshId] = new MergedMeshOffsets(entry.indexBaseBytes,
                    existing.PositionOffset, existing.PrevPositionOffset, existing.NormalOffset,
                    existing.TexCoord1Offset, existing.TangentOffset);
                entry.vb.Dispose();
            }

            _mergedStaticIndexBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Raw, Mathf.Max(indexWords.Count, 1), 4)
            {
                name = "IndexBuffer"
            };
            _mergedStaticIndexBuffer.SetData(indexWords.ToArray());

            _mergedStaticVertexBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Raw, Mathf.Max(vertexWords.Length, 1), 4)
            {
                name = "VertexBuffer"
            };
            _mergedStaticVertexBuffer.SetData(vertexWords);
        }

        private static uint Align16(uint value) => (value + 15u) & ~15u;

        private static void CopyWords(NativeArray<uint> src, uint srcByteOffset, uint[] dst, uint dstByteOffset, uint byteCount)
        {
            int srcWord = (int)(srcByteOffset / 4u);
            int dstWord = (int)(dstByteOffset / 4u);
            int words   = (int)(byteCount / 4u);
            for (int i = 0; i < words; i++)
                dst[dstWord + i] = src[srcWord + i];
        }

        private static int[] ResolveMaterialIndices(RtxptInstanceRecord rec,
            RtxptMaterialTable matTable, int[] overrideMatIndices)
        {
            var materialIndices = new int[rec.SubmeshIndices.Length];
            for (int i = 0; i < rec.SubmeshIndices.Length; i++)
            {
                int           s    = rec.SubmeshIndices[i];
                RtxptMaterial slot = rec.Renderer.Slots[s];
                int           idx  = matTable.GetOrAdd(slot);
                materialIndices[i] = idx;
                if (s < overrideMatIndices.Length)
                    overrideMatIndices[s] = idx;
            }
            return materialIndices;
        }

        // Appends shared GeometryData for one assigned sub-mesh. Static mesh instances with the
        // same mesh/material range reuse this record, mirroring Donut's MeshGeometry table.
        private static void AddGeometryData(RtxptInstanceRecord rec, int s, int matIdx,
            (int vb, int ib) slots, in MergedMeshOffsets offsets,
            List<DonutGeometryData> geomList, List<GeometryDebugData> geomDbgList)
        {
            Mesh              mesh = rec.Mesh;
            SubMeshDescriptor sub  = mesh.GetSubMesh(s);

            int globalGeomIdx = geomList.Count;

            if (globalGeomIdx > 0xFFFF)
                Debug.LogError($"[RtxptGPUScene] GlobalGeometryIndex overflow: geomIndex={globalGeomIdx} exceeds 16-bit limit (65535). Rendering will be corrupted. Renderer='{rec.TargetRenderer.name}' subMesh={s}.");

            geomList.Add(new DonutGeometryData
            {
                numIndices        = (uint)sub.indexCount,
                numVertices       = (uint)mesh.vertexCount,
                indexBufferIndex  = slots.ib,
                indexOffset       = offsets.IndexBaseBytes + (uint)sub.indexStart * 4u, // donut IB is always uint32
                vertexBufferIndex = slots.vb,
                positionOffset    = offsets.PositionOffset,
                // Skinned buffers carry a real PrevPosition stream (repack copies last frame's
                // positions there, donut SkinningPass model); static buffers alias positions.
                prevPositionOffset = offsets.PrevPositionOffset,
                texCoord1Offset    = offsets.TexCoord1Offset,
                texCoord2Offset    = 0xFFFFFFFFu,
                normalOffset       = offsets.NormalOffset,
                tangentOffset      = offsets.TangentOffset,
                curveRadiusOffset  = 0xFFFFFFFFu,
                materialIndex      = (uint)matIdx,
            });

            // --- GeometryDebugData (RTXPT t4, all zero = no OMM) ---
            geomDbgList.Add(default);
        }

        // Appends per-instance SubInstanceData for one assigned sub-mesh. Unlike GeometryData,
        // this is not shared: TLAS InstanceID() indexes this flat per-instance table directly.
        private void AddSubInstanceData(RtxptInstanceRecord rec, int s, int matIdx, int globalGeomIdx,
            (int vb, int ib) slots, in MergedMeshOffsets offsets, RtxptMaterialTable matTable,
            List<SubInstanceData> subInstList)
        {
            Mesh              mesh = rec.Mesh;
            SubMeshDescriptor sub  = mesh.GetSubMesh(s);
            RtxptMaterial     slot = rec.Renderer.Slots[s];

            // SubInstanceData.GlobalGeometryIndex_PTMaterialDataIndex packs both fields as 16-bit.
            if (globalGeomIdx > 0xFFFF)
                Debug.LogError($"[RtxptGPUScene] GlobalGeometryIndex overflow: geomIndex={globalGeomIdx} exceeds 16-bit limit (65535). Rendering will be corrupted. Renderer='{rec.TargetRenderer.name}' subMesh={s}.");
            if (matIdx > 0xFFFF)
                Debug.LogError($"[RtxptGPUScene] PTMaterialDataIndex overflow: matIndex={matIdx} exceeds 16-bit limit (65535). Rendering will be corrupted. Renderer='{rec.TargetRenderer.name}' subMesh={s}.");

            // --- SubInstanceData (RTXPT t1) ---
            bool  isAlphaTested = slot.EnableAlphaTesting;
            float aCutoff       = isAlphaTested ? slot.AlphaCutoff : 0f;
            uint  alphaU8       = (uint)Mathf.RoundToInt(Mathf.Clamp01(aCutoff) * 255f);
            // AlphaTextureIndex in lo16: use base texture slot if alpha-tested, else 0
            uint alphaTexIdx = isAlphaTested && matTable.Materials[matIdx].BaseOrDiffuseTextureIndex != 0xFFFFFFFFu
                ? matTable.Materials[matIdx].BaseOrDiffuseTextureIndex
                : 0u;
            uint siFlags                     = alphaTexIdx & 0xFFFFu;
            if (isAlphaTested) siFlags       |= SubInstanceFlags.AlphaTested;
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
                    Debug.LogWarning($"[RtxptGPUScene] Renderer '{rec.TargetRenderer.name}' submesh {s} is flagged " +
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
                AnalyticProxyLightIndex            = 0xFFFFFFFFu,
                IndexBufferIndex_VertexBufferIndex = ((uint)slots.ib << 16) | ((uint)slots.vb & 0xFFFFu),
                IndexOffset                        = offsets.IndexBaseBytes + (uint)sub.indexStart * 4u,
                TexCoord1Offset                    = offsets.TexCoord1Offset,
                padding0                           = 0u,
            });
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

                    // Mirror RTXPT PTMaterial::IsEmissive(): reserve triangle-light slots when the
                    // material is currently emissive OR is flagged UseDonutEmissiveIntensity (its
                    // emission can animate on at runtime, so RTXPT reserves slots up-front — see
                    // LightsBaker.cpp ProcessEmissiveGeometry). Without the second term, a material
                    // that lights up at runtime would emit no NEE light until a full rebuild.
                    bool reserve = _ptMaterialReserveEmissive != null && matIdx < _ptMaterialReserveEmissive.Length
                        ? _ptMaterialReserveEmissive[matIdx]
                        : (_ptMaterialCpu[matIdx].EmissiveColor.x > 0f || _ptMaterialCpu[matIdx].EmissiveColor.y > 0f || _ptMaterialCpu[matIdx].EmissiveColor.z > 0f);
                    if (!reserve)
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
                // Rigid frame only — bone scale is baked into the GPU-skinned positions
                // (see RtxptSceneLayout.GetRootTransform).
                if (entry.isSkinned)
                {
                    Matrix4x4 cur  = Matrix4x4.TRS(entry.transform.position, entry.transform.rotation, Vector3.one);
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

                bool moved                            = entry.transform.hasChanged;
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
                    _subInstanceCpuDirty                            = true;
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
                    Debug.LogWarning("[RtxptGPUScene] EmissiveTrianglesProcTask overflow — some emissive geometry ignored.");
                    break;
                }

                uint triCount = e.TriangleCount;
                uint destBase = lightOffset + accumTriangles;

                // Overflow guard
                if (destBase + triCount > RtxptBufferResources.MaxLights)
                {
                    Debug.LogWarning($"[RtxptGPUScene] MaxLights overflow at emissive geometry (inst={e.InstanceIndex}, geom={e.GeometrySubIndex}) — skipping.");
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
                    _subInstanceCpuDirty                              = true;
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
                        DestinationBufferOffset    = destBase + from, // each task owns its own 32-slot window
                        HistoricBufferOffset       = (historicBase != Invalid) ? historicBase + from : Invalid,
                        EmissiveLightMappingOffset = (uint)siIdx,
                        Padding0                   = 0u,
                    };
                }

                newHistoric[(e.InstanceIndex, e.GeometrySubIndex)] =  destBase;
                accumTriangles                                     += triCount;
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
    /// change by <see cref="RtxptGPUScene"/>; recorded each frame by RtxptBuildTlasPass
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
