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
    // NativeRtxdiGPUScene
    // =========================================================================

    /// <summary>
    /// Self-contained GPU scene for RTXDI-native compute passes.
    /// Owns the TLAS and provides donut-compatible structured buffers
    /// (<c>t_InstanceData</c>, <c>t_GeometryData</c>, <c>t_MaterialConstants</c>,
    /// bindless VB/IB, bindless textures).
    ///
    /// Struct layouts exactly mirror <c>donut/shaders/bindless.h</c> and
    /// <c>donut/shaders/material_cb.h</c> so all RTXDI shaders that include
    /// <c>SceneGeometry.hlsli</c> work without any layout mismatch.
    /// </summary>
    public sealed class NativeRtxptGPUScene : IDisposable
    {
        // Acceleration structure
        private RayTracingAccelerationStructure _accelStructure;

        // Structured buffers (donut-compatible)
        // t_InstanceData (t2): transforms are re-uploaded every frame, so this uses
        // UploadBuffer — its NativePtr is stable across SetData (unlike
        // GraphicsBuffer.GetNativeBufferPtr), so we fetch the pointer once at creation.
        private UploadBuffer _instanceGpuBuf;
        private GraphicsBuffer _geometryGpuBuf; // t_GeometryData  (t3)

        // RTXPT-specific structured buffers
        // t_SubInstanceData (t1): emissive-light mapping offsets are recomputed and re-uploaded
        // every frame, so this uses UploadBuffer for a stable NativePtr (no per-frame
        // GetNativeBufferPtr re-fetch). SRV-only — never bound as a UAV.
        private UploadBuffer _subInstanceGpuBuf; // t_SubInstanceData    (t1)
        private GraphicsBuffer _ptMaterialGpuBuf; // t_PTMaterialData     (t5)
        private GraphicsBuffer _geomDebugGpuBuf; // t_GeometryDebugData  (t4)

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

        // Per-instance tracking for transforms
        private struct SceneInstance
        {
            public MeshRenderer renderer;

            // Non-zero for TLAS instances added via AddInstanceGroup (SubmeshGroups path).
            // When non-zero, use SetInstanceTransform(groupHandle, ...) instead of SetInstanceTransform(renderer, ...).
            public uint groupHandle;
        }

        // Per-renderer tracking for transform updates (NRD-style: one entry per renderer, not per TLAS instance).
        private sealed class RendererEntry
        {
            public Transform transform;
            public bool wasMoving = true; // start true so first frame always syncs prev = current
            public int firstInstanceIdx;
            public int instanceCount;
        }

        private readonly List<SceneInstance>                                     _sceneInstances   = new();
        private readonly Dictionary<int, RendererEntry>                          _rendererEntries  = new();
        private readonly Dictionary<int, (int vb, int ib)>                       _meshBufferSlots  = new();
        // Dedup for RtxptMaterial slots, keyed on the shared RtxptMaterial InstanceID. Material
        // assets are explicitly shareable across renderers/sub-meshes, so identical references
        // collapse to a single PTMaterialData entry.
        private readonly Dictionary<int, int>                                    _overrideSlots    = new();
        private readonly Dictionary<int, int>                                    _textureSlots     = new();
        private readonly Dictionary<int, (GraphicsBuffer vb, GraphicsBuffer ib)> _donutBufferCache = new();
        private readonly List<GraphicsBuffer>                                    _ownedGfxBuffers  = new();

        private readonly List<NativeRayTracingTarget> _registeredTargets = new();

        // Maps MeshRenderer.GetInstanceID() → list of per-group TLAS handles registered in _accelStructure.
        private readonly Dictionary<int, List<uint>> _perTargetGroupHandles = new();

        // Flat per-geometry hit-group variant index (one entry per TLAS geometry, in insertion
        // order), used to rebuild each ray-trace pipeline's hit-group shader table. Owned here
        // (not by the AS) so RayTraceShader and RayTracingAccelerationStructure stay decoupled.
        // Rebuilt only when the scene topology changes (in RegisterScene); the SBT rebuild is
        // then issued once per affected pipeline instead of every frame.
        private NativeArray<uint> _variantIndexArray;
        private bool              _shaderTableDirty;

        /// <summary>
        /// True when the per-geometry hit-group variant layout changed since the last
        /// <see cref="MarkShaderTableClean"/>. While set, callers should rebuild the hit-group
        /// table of every ray-trace pipeline via <see cref="RebuildShaderTable"/>.
        /// </summary>
        public bool ShaderTableDirty => _shaderTableDirty;

        private bool _sceneGpuDirty = true;
        private bool _forceRebuild  = false;
        private bool _disposed;

        // ---- Emissive triangle light tracking --------------------------------
        // Maps (globalInstanceIndex, geometrySubIndex) → last-frame DestinationBufferOffset
        private readonly Dictionary<(int, int), uint> _emissiveHistoricOffsets = new();

        // Max task count: MaxLights / LLB_MAX_TRIANGLES_PER_TASK * 2
        private const int MaxEmissiveProcTasks = NativeRtxptBufferResources.MaxLights / 32 * 2;
        private static readonly RtxptEmissiveTrianglesProcTask[] s_emissiveTaskStaging =
            new RtxptEmissiveTrianglesProcTask[MaxEmissiveProcTasks];

        /// <summary>Number of tasks produced by the last <see cref="PrepareEmissiveTriangleTasks"/> call.</summary>
        public int  LastEmissiveTaskCount     { get; private set; }
        /// <summary>Total triangle-light count produced by the last <see cref="PrepareEmissiveTriangleTasks"/> call.</summary>
        public uint LastEmissiveTriangleCount { get; private set; }

        // Maps MeshRenderer.GetInstanceID() → per-submesh material indices in _ptMaterialCpu
        // for renderers that have a RtxptRenderer. Used for lightweight material-only updates.
        private readonly Dictionary<int, int[]> _overrideMaterialIndices = new();

        // Cached list of (component, matIndices) for all renderers with a RtxptRenderer.
        // Rebuilt during RebuildSceneGpuData so CheckAndUpdateMaterialOverrides never calls GetComponent.
        private readonly List<(RtxptRenderer comp, int[] matIndices)> _overrideCache = new();

        // Optional equirectangular environment map for RTXDI environment light.
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
        /// Returns one <see cref="EmissiveGeometryEntry"/> for every sub-mesh whose material has
        /// a non-zero emissiveColor.  Must be called after <see cref="UpdateForFrame"/>.
        /// </summary>
        public List<EmissiveGeometryEntry> GetEmissiveGeometries()
        {
            var result = new List<EmissiveGeometryEntry>();
            if (_instanceCpu == null || _geometryCpu == null)
                return result;

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

                    result.Add(new EmissiveGeometryEntry
                    {
                        InstanceIndex              = i,
                        GeometrySubIndex           = s,
                        TriangleCount              = geom.numIndices / 3u,
                        FirstGeometryInstanceIndex = inst.firstGeometryInstanceIndex,
                    });
                }
            }

            return result;
        }

        public NativeRtxptGPUScene()
        {
            _accelStructure = new RayTracingAccelerationStructure();
        }

        public void MarkRebuildDirty() => _forceRebuild = true;

        /// <summary>
        /// CPU-side emissive-triangle pass: mirrors <c>LightsBaker::ProcessEmissiveGeometry</c>.
        /// Generates <see cref="RtxptEmissiveTrianglesProcTask"/> entries, uploads them to
        /// <paramref name="scratchBuffer"/>, updates <c>SubInstanceData.EmissiveLightMappingOffset</c>
        /// for emissive geometries, and re-uploads the sub-instance GPU buffer.
        /// Must be called on the main thread after <see cref="UpdateForFrame"/> and before
        /// command-buffer recording (i.e. from a pass <c>Setup</c> method).
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

            const uint Invalid = 0xFFFFFFFFu;
            const uint MaxTriPerTask = 32u;

            var emissive = GetEmissiveGeometries();

            var newHistoric = new Dictionary<(int, int), uint>(emissive.Count);
            int taskIdx        = 0;
            uint accumTriangles = 0u;

            foreach (var e in emissive)
            {
                if (taskIdx >= MaxEmissiveProcTasks)
                {
                    Debug.LogWarning("[NativeRtxptGPUScene] EmissiveTrianglesProcTask overflow — some emissive geometry ignored.");
                    break;
                }

                uint triCount  = e.TriangleCount;
                uint destBase  = lightOffset + accumTriangles;

                // Overflow guard
                if (destBase + triCount > NativeRtxptBufferResources.MaxLights)
                {
                    Debug.LogWarning($"[NativeRtxptGPUScene] MaxLights overflow at emissive geometry (inst={e.InstanceIndex}, geom={e.GeometrySubIndex}) — skipping.");
                    break;
                }

                _emissiveHistoricOffsets.TryGetValue((e.InstanceIndex, e.GeometrySubIndex), out uint historicBase);
                if (!_emissiveHistoricOffsets.ContainsKey((e.InstanceIndex, e.GeometrySubIndex)))
                    historicBase = Invalid;

                // Update SubInstanceData.EmissiveLightMappingOffset
                int siIdx = (int)(e.FirstGeometryInstanceIndex + (uint)e.GeometrySubIndex);
                if (siIdx >= 0 && siIdx < _subInstanceCpu.Length)
                    _subInstanceCpu[siIdx].EmissiveLightMappingOffset = destBase;

                // Split into tasks of at most MaxTriPerTask triangles.
                // Each task writes to DestinationBufferOffset + subIndex (0..31), so successive
                // tasks must each advance the offset by MaxTriPerTask to avoid aliasing.
                for (uint from = 0u; from < triCount && taskIdx < MaxEmissiveProcTasks; from += MaxTriPerTask)
                {
                    uint to = System.Math.Min(from + MaxTriPerTask, triCount);
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

            // Swap historic offsets for next frame
            _emissiveHistoricOffsets.Clear();
            foreach (var kv in newHistoric)
                _emissiveHistoricOffsets[kv.Key] = kv.Value;

            // Re-upload SubInstanceData (EmissiveLightMappingOffset fields updated). Pointer is
            // stable; the GPU copy is recorded by FlushSubInstanceBuffer(cmd) in the lighting pass.
            if (_subInstanceGpuBuf != null && _subInstanceCpu != null)
                _subInstanceGpuBuf.SetData(_subInstanceCpu, 0, _subInstanceCpu.Length);

            // Upload task array to scratch buffer (raw buffer, stride = 4 bytes, tasks = 8 uints each).
            if (taskIdx > 0 && scratchBuffer != null)
                scratchBuffer.SetRawData(s_emissiveTaskStaging, 0, 0, taskIdx);

            LastEmissiveTaskCount     = taskIdx;
            LastEmissiveTriangleCount = accumTriangles;
        }

        /// <summary>
        /// Call once per frame before <see cref="BuildAccelerationStructure"/>.
        /// Handles dirty detection, GPU data rebuild, and transform updates.
        /// </summary>
        public void UpdateForFrame()
        {
            var targets = NativeRayTracingTarget.All;

            if (_forceRebuild || TargetSetChanged(targets))
            {
                RegisterScene(targets);
                _registeredTargets.Clear();
                _registeredTargets.AddRange(targets);
                _forceRebuild  = false;
                _sceneGpuDirty = true;
            }

            if (_sceneGpuDirty)
                RebuildSceneGpuData(targets);
            else
                CheckAndUpdateMaterialOverrides();

            UpdateInstanceTransforms();
        }

        /// <summary>
        /// Binds all RTXPT scene buffers to a native compute descriptor set.
        /// Binds: t_SubInstanceData(t1), t_InstanceData(t2), t_GeometryData(t3),
        ///        t_GeometryDebugData(t4), t_PTMaterialData(t5),
        ///        t_MaterialConstants (donut compat), t_BindlessBuffers(space1), t_BindlessTextures(space2).
        /// </summary>
        public void BindToShader(NativeComputeDescriptorSet ds)
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

        public void BindToShader(NativeRayTraceDescriptorSet ds)
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
        {
            if (!_shaderTableDirty || pipeline == null || !_variantIndexArray.IsCreated) return;
            pipeline.RebuildHitGroupTable(cmd, _variantIndexArray);
        }

        /// <summary>Clears the dirty flag after every pipeline's hit-group table has been rebuilt.</summary>
        public void MarkShaderTableClean() => _shaderTableDirty = false;

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            DisposeGpuBuffers();
            if (_variantIndexArray.IsCreated) _variantIndexArray.Dispose();
            _accelStructure?.Dispose();
            _accelStructure = null;
        }

        // -----------------------------------------------------------------------

        private void RegisterScene(IReadOnlyList<NativeRayTracingTarget> targets)
        {
            // Full teardown + rebuild
            if (_registeredTargets.Count > 0)
            {
                foreach (var t in _registeredTargets)
                {
                    if (t == null) continue;
                    var mr = t.GetComponent<MeshRenderer>();
                    if (mr == null) continue;
                    int mrId = mr.GetInstanceID();
                    if (_perTargetGroupHandles.TryGetValue(mrId, out var oldHandles))
                    {
                        foreach (var h in oldHandles)
                            _accelStructure.RemoveInstance(h);
                        _perTargetGroupHandles.Remove(mrId);
                    }
                    else
                    {
                        _accelStructure.RemoveInstance(mr);
                    }
                }
            }

            // Flat per-geometry hit-group variant array, rebuilt in lockstep with the TLAS
            // insertion order. runningContribution is each group's InstanceContributionToHitGroupIndex
            // (the base offset of its geometries in the flat shader table) — the value the AS used to
            // compute internally, now pre-calculated here so the AS stays free of hit-group concerns.
            var  variantList         = new List<uint>();
            uint runningContribution = 0;

            foreach (var t in targets)
            {
                if (t == null) continue;
                var mr = t.GetComponent<MeshRenderer>();
                if (mr == null) continue;

                var groups = t.SubmeshGroups;
                if (groups != null && groups.Length > 0)
                {
                    var mesh = mr.GetComponent<MeshFilter>()?.sharedMesh;
                    if (mesh == null) continue;

                    uint indexStride = mesh.indexFormat == UnityEngine.Rendering.IndexFormat.UInt16 ? 2u : 4u;
                    int  mrId        = mr.GetInstanceID();
                    var  rr          = mr.GetComponent<RtxptRenderer>();
                    var  handles     = new List<uint>(groups.Length);

                    for (int gi = 0; gi < groups.Length; gi++)
                    {
                        var grp = groups[gi];

                        // Only sub-meshes with a pre-baked RtxptMaterial are added to the BLAS. This
                        // must match RebuildSceneGpuData's grouped filter exactly so the shader's flat
                        // per-geometry arrays (variantList / geomList) stay aligned with the BLAS.
                        var descsList = new List<NativeRenderPlugin.SubmeshDesc>(grp.submeshIndices.Length);
                        foreach (int sIdx in grp.submeshIndices)
                        {
                            if (!SubmeshHasMaterial(rr, sIdx)) continue;
                            var sub = mesh.GetSubMesh(sIdx);
                            descsList.Add(new NativeRenderPlugin.SubmeshDesc
                            {
                                indexCount      = (uint)sub.indexCount,
                                indexByteOffset = (uint)sub.indexStart * indexStride,
                                baseVertex      = (uint)sub.baseVertex,
                                flags           = grp.isAlphaClip ? 0u : NativeRenderPlugin.SUBMESH_FLAG_GEOMETRY_OPAQUE,
                            });
                        }
                        if (descsList.Count == 0) continue; // every sub-mesh in this group is unassigned
                        var descs = descsList.ToArray();

                        uint handle          = MakeGroupHandle(mrId, gi);
                        uint hitGroupVariant = grp.isEmissive ? 0u : 1u;
                        if (_accelStructure.AddInstanceGroup(mesh, descs, handle, hitGroupContribution: runningContribution))
                        {
                            _accelStructure.SetInstanceTransform(handle, mr.transform.localToWorldMatrix);
                            handles.Add(handle);
                            for (int k = 0; k < descs.Length; k++)
                                variantList.Add(hitGroupVariant);
                            runningContribution += (uint)descs.Length;
                        }
                        else
                        {
                            Debug.LogWarning($"[NativeRtxptGPUScene] AddInstanceGroup failed for '{mr.name}' gi={gi}");
                        }
                    }

                    if (handles.Count > 0)
                        _perTargetGroupHandles[mrId] = handles;
                }
                else
                {
                    // Fallback: no group info. The native AddInstance(mr) path adds every sub-mesh to
                    // the BLAS and cannot disable individual ones, so per-sub-mesh skipping is not
                    // supported here — register the renderer only when all sub-meshes are assigned.
                    var fbMesh = mr.GetComponent<MeshFilter>()?.sharedMesh;
                    var fbRr   = mr.GetComponent<RtxptRenderer>();
                    if (fbMesh != null && AllSubmeshesAssigned(fbRr, fbMesh.subMeshCount))
                        _accelStructure.AddInstance(mr);
                    else
                        Debug.LogWarning($"[NativeRtxptGPUScene] '{mr.name}' has no SubmeshGroups and not all sub-meshes have an RtxptMaterial assigned — skipping the whole renderer (per-sub-mesh skip requires the grouped path).");
                }
            }

            // Publish the new per-geometry variant array and flag pipelines for a one-time SBT rebuild.
            if (_variantIndexArray.IsCreated) _variantIndexArray.Dispose();
            _variantIndexArray = new NativeArray<uint>(variantList.Count, Allocator.Persistent);
            for (int i = 0; i < variantList.Count; i++)
                _variantIndexArray[i] = variantList[i];
            _shaderTableDirty = true;
        }

        private static uint MakeGroupHandle(int mrInstanceId, int groupIndex)
            => (uint)(mrInstanceId & 0x0FFFFFFF) | ((uint)groupIndex << 28);

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
            _sceneInstances.Clear();
            _rendererEntries.Clear();
            _meshBufferSlots.Clear();
            _overrideSlots.Clear();
            _textureSlots.Clear();
            _perTargetGroupHandles.Clear();
            _overrideMaterialIndices.Clear();
            _overrideCache.Clear();
            _environmentMapTextureIndex = -1;

            foreach (var buf in _ownedGfxBuffers)
                buf?.Release();
            _ownedGfxBuffers.Clear();
            _donutBufferCache.Clear();
        }

        private void RebuildSceneGpuData(IReadOnlyList<NativeRayTracingTarget> targets)
        {
            DisposeGpuBuffers();

            var instList    = new List<DonutInstanceData>();
            var geomList    = new List<DonutGeometryData>();
            var subInstList = new List<SubInstanceData>();
            var ptMatList   = new List<PTMaterialData>();
            var geomDbgList = new List<GeometryDebugData>();
            var bufPtrs     = new List<IntPtr>();
            var texPtrs     = new List<IntPtr>();

            foreach (var target in targets)
            {
                if (target == null) continue;
                var mr = target.GetComponent<MeshRenderer>();
                if (mr == null) continue;
                var mf = mr.GetComponent<MeshFilter>();
                if (mf == null || mf.sharedMesh == null) continue;

                Mesh mesh = mf.sharedMesh;
                if (mesh == null) continue;
                int meshKey = mesh.GetInstanceID();

                if (!_meshBufferSlots.TryGetValue(meshKey, out var slots))
                {
                    var (donutVb, donutIb) = GetOrCreateDonutBuffers(mesh);
                    if (donutVb == null || donutIb == null)
                    {
                        Debug.LogWarning($"[NativeRtxdiGPUScene] '{mesh.name}': failed to build donut buffers — skipping");
                        continue;
                    }

                    slots = (bufPtrs.Count, bufPtrs.Count + 1);
                    // SubInstanceData.IndexBufferIndex_VertexBufferIndex packs both as 16-bit.
                    if (bufPtrs.Count > 0xFFFF)
                        Debug.LogError($"[NativeRtxptGPUScene] Bindless buffer slot overflow: VB slot index {bufPtrs.Count} exceeds 16-bit limit (65535). Rendering will be corrupted. Mesh='{mesh.name}'.");
                    bufPtrs.Add(donutVb.GetNativeBufferPtr());
                    bufPtrs.Add(donutIb.GetNativeBufferPtr());
                    _meshBufferSlots[meshKey] = slots;
                }

                // SoA offsets — must match GetOrCreateDonutBuffers stream order exactly:
                //   [positions: vc*12] [normals?: vc*4] [uvs?: vc*8] [tangents?: vc*4]
                // Only present streams occupy bytes, so offsets are computed cumulatively.
                uint vc         = (uint)mesh.vertexCount;
                bool hasNormal  = mesh.HasVertexAttribute(VertexAttribute.Normal);
                bool hasUV      = mesh.HasVertexAttribute(VertexAttribute.TexCoord0);
                bool hasTangent = mesh.HasVertexAttribute(VertexAttribute.Tangent);

                uint streamOffset = 0u;
                uint posOff       = streamOffset;
                streamOffset += vc * 12u;
                uint normOff                = hasNormal ? streamOffset : 0xFFFFFFFFu;
                if (hasNormal) streamOffset += vc * 4u;
                uint uvOff                  = hasUV ? streamOffset : 0xFFFFFFFFu;
                if (hasUV) streamOffset     += vc * 8u;
                uint tanOff                 = hasTangent ? streamOffset : 0xFFFFFFFFu;

                Material[] mats       = mr.sharedMaterials ?? Array.Empty<Material>();
                int        subMeshCnt = mesh.subMeshCount;
                // (firstGeom and instIdx are computed per-branch below)

#if UNITY_EDITOR
                if (mr.name == "Bistro_Research_Interior_Paris_Flower_Pot_01A_2442" ||
                    mr.name == "Bistro_Research_Interior_Paris_ToffeeJar_01_3262" ||
                    mr.name == "Bistro_Research_Interior_Paris_Wall_Light_Interior_01_3266")
                {
                    var groups_dbg = target.SubmeshGroups;
                    var sb         = new System.Text.StringBuilder();
                    sb.AppendLine($"[GPUScene DEBUG] Renderer: '{mr.name}'");
                    sb.AppendLine($"  Mesh: '{mesh.name}'  vertexCount={vc}  subMeshCount={subMeshCnt}");
                    sb.AppendLine($"  SoA offsets: pos={posOff}  norm={normOff}  uv={uvOff}  tan={tanOff}");
                    sb.AppendLine($"  Buffer slots: vb={slots.vb}  ib={slots.ib}");
                    sb.AppendLine($"  instList.Count={instList.Count}  geomList.Count={geomList.Count}  groups={groups_dbg?.Length ?? 0}");
                    for (int _s = 0; _s < subMeshCnt; _s++)
                    {
                        var _sub = mesh.GetSubMesh(_s);
                        sb.AppendLine($"  subMesh[{_s}]: indexStart={_sub.indexStart}  indexCount={_sub.indexCount}  baseVertex={_sub.baseVertex}  topology={_sub.topology}");
                        var _mat = _s < mats.Length ? mats[_s] : null;
                        sb.AppendLine($"  subMesh[{_s}]: mat='{(_mat != null ? _mat.name : "null")}'");
                    }

                    if (groups_dbg != null)
                        for (int _gi = 0; _gi < groups_dbg.Length; _gi++)
                        {
                            var _grp = groups_dbg[_gi];
                            sb.AppendLine($"  Group[{_gi}]: emissive={_grp.isEmissive} alphaClip={_grp.isAlphaClip} submeshIndices=[{string.Join(",", _grp.submeshIndices)}]");
                        }

                    Debug.Log(sb.ToString());
                }
#endif

                var   matOverride        = mr.GetComponent<RtxptRenderer>();
                int[] overrideMatIndices = matOverride != null ? new int[subMeshCnt] : null;

                Matrix4x4 m    = target.transform.localToWorldMatrix;
                var       row0 = new Vector4(m.m00, m.m01, m.m02, m.m03);
                var       row1 = new Vector4(m.m10, m.m11, m.m12, m.m13);
                var       row2 = new Vector4(m.m20, m.m21, m.m22, m.m23);

                // Local helper: append geometry + sub-instance data for one sub-mesh index.
                // Only ever called for sub-meshes that have a pre-baked RtxptMaterial assigned
                // (callers filter via SubmeshHasMaterial), so matOverride.Slots[s] is non-null here.
                void AddSubmeshData(int s)
                {
                    SubMeshDescriptor sub                                 = mesh.GetSubMesh(s);
                    int               matIdx                              = GetOrAddMaterial(s, matOverride, ptMatList, texPtrs);
                    if (overrideMatIndices != null) overrideMatIndices[s] = matIdx;

                    int globalGeomIdx = geomList.Count;

                    // SubInstanceData.GlobalGeometryIndex_PTMaterialDataIndex packs both fields as 16-bit.
                    if (globalGeomIdx > 0xFFFF)
                        Debug.LogError($"[NativeRtxptGPUScene] GlobalGeometryIndex overflow: geomIndex={globalGeomIdx} exceeds 16-bit limit (65535). Rendering will be corrupted. Renderer='{mr.name}' subMesh={s}.");
                    if (matIdx > 0xFFFF)
                        Debug.LogError($"[NativeRtxptGPUScene] PTMaterialDataIndex overflow: matIndex={matIdx} exceeds 16-bit limit (65535). Rendering will be corrupted. Renderer='{mr.name}' subMesh={s}.");

                    geomList.Add(new DonutGeometryData
                    {
                        numIndices         = (uint)sub.indexCount,
                        numVertices        = (uint)mesh.vertexCount,
                        indexBufferIndex   = slots.ib,
                        indexOffset        = (uint)sub.indexStart * 4u, // always uint32
                        vertexBufferIndex  = slots.vb,
                        positionOffset     = posOff,
                        prevPositionOffset = posOff, // no skinning / morph support yet
                        texCoord1Offset    = uvOff,
                        texCoord2Offset    = 0xFFFFFFFFu,
                        normalOffset       = normOff,
                        tangentOffset      = tanOff,
                        curveRadiusOffset  = 0xFFFFFFFFu,
                        materialIndex      = (uint)matIdx,
                    });

                    // --- SubInstanceData (RTXPT t1) ---
                    RtxptMaterial overrideSlot = matOverride.Slots[s];
                    bool  isAlphaTested  = overrideSlot.EnableAlphaTesting;
                    float aCutoff        = isAlphaTested ? overrideSlot.AlphaCutoff : 0f;
                    bool  excludeFromNEE = overrideSlot.ExcludeFromNEE;
                    uint alphaU8     = (uint)Mathf.RoundToInt(Mathf.Clamp01(aCutoff) * 255f);
                    // AlphaTextureIndex in lo16: use base texture slot if alpha-tested, else 0
                    uint alphaTexIdx = isAlphaTested && ptMatList.Count > matIdx
                        ? (uint)Mathf.Max(0, ptMatList[matIdx].BaseOrDiffuseTextureIndex)
                        : 0u;
                    uint siFlags = alphaTexIdx & 0xFFFFu;
                    if (isAlphaTested)  siFlags |= SubInstanceFlags.AlphaTested;
                    if (excludeFromNEE) siFlags |= SubInstanceFlags.ExcludeFromNEE;
                    siFlags |= (alphaU8 << SubInstanceFlags.AlphaOffsetOffset);

                    subInstList.Add(new SubInstanceData
                    {
                        FlagsAndAlphaInfo                       = siFlags,
                        GlobalGeometryIndex_PTMaterialDataIndex = ((uint)globalGeomIdx << 16) | ((uint)matIdx & 0xFFFFu),
                        EmissiveLightMappingOffset              = 0xFFFFFFFFu, // no light baker yet
                        // RTXPT_INVALID_LIGHT_INDEX — the analytic-light-proxy feature
                        // (PrepareEmissiveTriangleTasks) is not ported, so leave this at the sentinel.
                        // The shader only reads it for materials flagged EnableAsAnalyticLightProxy;
                        // INVALID disables the proxy lookup, whereas 0u would alias env-quad light 0.
                        AnalyticProxyLightIndex                 = 0xFFFFFFFFu,
                        IndexBufferIndex_VertexBufferIndex      = ((uint)slots.ib << 16) | ((uint)slots.vb & 0xFFFFu),
                        IndexOffset                             = (uint)sub.indexStart * 4u,
                        TexCoord1Offset                         = uvOff,
                        padding0                                = 0u,
                    });

                    // --- GeometryDebugData (RTXPT t4, all zero = no OMM) ---
                    geomDbgList.Add(default);
                }

                var groups = target.SubmeshGroups;
                if (groups != null && groups.Length > 0)
                {
                    // Each SubmeshGroup is a separate TLAS instance (AddInstanceGroup), so each needs
                    // its own DonutInstanceData with firstGeometryIndex pointing to its own geomList slice.
                    // GeometryIndex() in the shader is 0-based within each group instance.
                    int mrId              = mr.GetInstanceID();
                    int firstGroupInstIdx = _sceneInstances.Count;
                    int addedGroups       = 0;
                    for (int gi = 0; gi < groups.Length; gi++)
                    {
                        var grp               = groups[gi];
                        int firstGeomForGroup = geomList.Count;

                        // Skip sub-meshes with no assigned RtxptMaterial. Must match the BLAS filter
                        // in RegisterScene exactly so the shader's per-geometry indices stay aligned.
                        for (int si = 0; si < grp.submeshIndices.Length; si++)
                        {
                            int sIdx = grp.submeshIndices[si];
                            if (!SubmeshHasMaterial(matOverride, sIdx))
                            {
                                Debug.LogWarning($"[NativeRtxptGPUScene] '{mr.name}' sub-mesh {sIdx} has no RtxptMaterial assigned — skipping (not rendered).");
                                continue;
                            }
                            AddSubmeshData(sIdx);
                        }

                        int numGeom = geomList.Count - firstGeomForGroup;
                        if (numGeom == 0) continue; // whole group unassigned — matches RegisterScene

                        int  groupInstIdx = instList.Count;
                        uint groupHandle  = MakeGroupHandle(mrId, gi);
                        _accelStructure.SetInstanceID(groupHandle, (uint)groupInstIdx);

                        instList.Add(new DonutInstanceData
                        {
                            flags                      = 0u,
                            firstGeometryInstanceIndex = (uint)firstGeomForGroup,
                            firstGeometryIndex         = (uint)firstGeomForGroup,
                            numGeometries              = (uint)numGeom,
                            transformRow0              = row0,
                            transformRow1              = row1,
                            transformRow2              = row2,
                            prevTransformRow0          = row0,
                            prevTransformRow1          = row1,
                            prevTransformRow2          = row2,
                        });
                        _sceneInstances.Add(new SceneInstance
                        {
                            renderer    = mr,
                            groupHandle = groupHandle,
                        });
                        addedGroups++;
                    }
                    if (addedGroups > 0)
                        _rendererEntries[mrId] = new RendererEntry
                        {
                            transform        = mr.transform,
                            firstInstanceIdx = firstGroupInstIdx,
                            instanceCount    = addedGroups,
                        };
                }
                else
                {
                    // Non-grouped: one TLAS instance (AddInstance) covers all sub-meshes. The native
                    // AddInstance(mr) path can't disable individual sub-meshes, so RegisterScene only
                    // registers this renderer when every sub-mesh is assigned — mirror that here.
                    if (!AllSubmeshesAssigned(matOverride, subMeshCnt))
                        continue;

                    int firstGeom = geomList.Count;
                    int instIdx   = instList.Count;

                    for (int s = 0; s < subMeshCnt; s++)
                        AddSubmeshData(s);

                    _accelStructure.SetInstanceID(mr, (uint)instIdx);
                    instList.Add(new DonutInstanceData
                    {
                        flags                      = 0u,
                        firstGeometryInstanceIndex = (uint)firstGeom,
                        firstGeometryIndex         = (uint)firstGeom,
                        numGeometries              = (uint)subMeshCnt,
                        transformRow0              = row0,
                        transformRow1              = row1,
                        transformRow2              = row2,
                        prevTransformRow0          = row0,
                        prevTransformRow1          = row1,
                        prevTransformRow2          = row2,
                    });
                    int singleInstIdx = _sceneInstances.Count;
                    _sceneInstances.Add(new SceneInstance
                    {
                        renderer    = mr,
                        groupHandle = 0u,
                    });
                    _rendererEntries[mr.GetInstanceID()] = new RendererEntry
                    {
                        transform        = mr.transform,
                        firstInstanceIdx = singleInstIdx,
                        instanceCount    = 1,
                    };
                }

                if (overrideMatIndices != null)
                {
                    _overrideMaterialIndices[mr.GetInstanceID()] = overrideMatIndices;
                    _overrideCache.Add((matOverride, overrideMatIndices));
                }
            }

            // Append environment map to bindless texture array (if registered)
            if (_pendingEnvMap != null)
            {
                _environmentMapTextureIndex = texPtrs.Count;
                texPtrs.Add(_pendingEnvMap.GetNativeTexturePtr());
            }

            _sceneBuffers = new BindlessBuffer(Mathf.Max(bufPtrs.Count, 1));
            for (int i = 0; i < bufPtrs.Count; i++)
                _sceneBuffers.SetNativePtr(i, bufPtrs[i]);

            int texCount = Mathf.Max(texPtrs.Count, 1);
            _sceneTextures = new BindlessTexture(texCount);
            for (int i = 0; i < texPtrs.Count; i++)
                _sceneTextures.SetNativePtr(i, texPtrs[i]);

            if (instList.Count == 0)
            {
                instList.Add(default);
                geomList.Add(default);
                subInstList.Add(default);
                ptMatList.Add(default);
                geomDbgList.Add(default);
            }

            _instanceCpu    = instList.ToArray();
            _geometryCpu    = geomList.ToArray();
            _subInstanceCpu = subInstList.ToArray();
            _ptMaterialCpu  = ptMatList.ToArray();
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

            _ptMaterialGpuBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _ptMaterialCpu.Length, Marshal.SizeOf<PTMaterialData>()) { name = "PTMaterialDataStorage" };
            _ptMaterialGpuBuf.SetData(_ptMaterialCpu);
            _ptMaterialGpuBufPtr = _ptMaterialGpuBuf.GetNativeBufferPtr();

            _geomDebugGpuBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _geomDebugCpu.Length, Marshal.SizeOf<GeometryDebugData>()) { name = "BindlessGeometryDebug" };
            _geomDebugGpuBuf.SetData(_geomDebugCpu);
            _geomDebugGpuBufPtr = _geomDebugGpuBuf.GetNativeBufferPtr();

            _sceneGpuDirty = false;
        }

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
                    RefreshMaterialCpuFromOverride(matOverride.Slots[s], ref _ptMaterialCpu[idx]);
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

        /// <summary>
        /// Refreshes all scalar/color/flag fields of <paramref name="data"/> from <paramref name="slot"/>,
        /// preserving the existing texture index fields (which are set only during a full scene rebuild).
        /// </summary>
        private static void RefreshMaterialCpuFromOverride(RtxptMaterial slot, ref PTMaterialData data)
        {
            // Texture flags: preserve loaded state (indices set during full rebuild) AND'd with slot enables.
            uint flags = 0;
            if (slot.UseSpecularGlossModel) flags |= PTMaterialFlags.UseSpecularGlossModel;
            if (data.BaseOrDiffuseTextureIndex        != 0xFFFFFFFFu && slot.EnableBaseTexture)                                             flags |= PTMaterialFlags.UseBaseOrDiffuseTexture;
            if (data.MetalRoughOrSpecularTextureIndex != 0xFFFFFFFFu && slot.EnableOcclusionRoughnessMetallicTexture)                       flags |= PTMaterialFlags.UseMetalRoughOrSpecularTexture;
            if (data.EmissiveTextureIndex             != 0xFFFFFFFFu && slot.EnableEmissiveTexture)                                         flags |= PTMaterialFlags.UseEmissiveTexture;
            if (data.NormalTextureIndex               != 0xFFFFFFFFu && slot.EnableNormalTexture)                                           flags |= PTMaterialFlags.UseNormalTexture;
            if (data.TransmissionTextureIndex         != 0xFFFFFFFFu && slot.EnableTransmissionTexture && slot.EnableTransmission)           flags |= PTMaterialFlags.UseTransmissionTexture;
            if (slot.MetalnessInRedChannel)                                                             flags |= PTMaterialFlags.MetalnessInRedChannel;
            if (slot.ThinSurface || !slot.EnableTransmission)                                           flags |= PTMaterialFlags.ThinSurface;
            if (slot.PSDExclude)                                                                        flags |= PTMaterialFlags.PSDExclude;
            if (slot.EnableAsAnalyticLightProxy)                                                        flags |= PTMaterialFlags.EnableAsAnalyticLightProxy;
            if (slot.IgnoreMeshTangentSpace)                                                            flags |= PTMaterialFlags.IgnoreMeshTangentSpace;
            if (slot.PSDBlockMotionVectorsAtSurfaceType % 2 != 0)                                       flags |= PTMaterialFlags.PSDBlockMVsAtSurfaceTypeB0;
            if (slot.PSDBlockMotionVectorsAtSurfaceType / 2 != 0)                                       flags |= PTMaterialFlags.PSDBlockMVsAtSurfaceTypeB1;
            flags |= (uint)Mathf.Clamp(slot.NestedPriority, 0, 14) << PTMaterialFlags.NestedPriorityShift;
            flags |= (uint)Mathf.Clamp(slot.PSDDominantDeltaLobe + 1, 0, 7) << PTMaterialFlags.PSDDominantDeltaLobeP1Shift;

            data.Flags                     = flags;
            data.BaseOrDiffuseColor        = new Vector3(slot.BaseColorFactor.r, slot.BaseColorFactor.g, slot.BaseColorFactor.b);
            data.SpecularColor             = new Vector3(slot.SpecularColor.r, slot.SpecularColor.g, slot.SpecularColor.b);
            data.EmissiveColor             = new Vector3(slot.EmissiveColor.r, slot.EmissiveColor.g, slot.EmissiveColor.b) * slot.EmissiveIntensity;
            data.ShadowNoLFadeout          = Mathf.Clamp(slot.ShadowNoLFadeout, 0f, 0.25f);
            data.Opacity                   = slot.Opacity;
            data.Roughness                 = slot.Roughness;
            data.Metalness                 = slot.Metalness;
            data.NormalTextureScale        = slot.NormalTextureScale;
            data.AlphaCutoff               = slot.AlphaCutoff;
            data.TransmissionFactor        = slot.EnableTransmission ? slot.TransmissionFactor       : 0f;
            data.DiffuseTransmissionFactor = slot.EnableTransmission ? slot.DiffuseTransmissionFactor : 0f;
            data.IoR                       = slot.IoR;
            data.ThicknessFactor           = slot.ThicknessFactor;
            data.VolumeAttenuationColor    = new Vector3(slot.VolumeAttenuationColor.r, slot.VolumeAttenuationColor.g, slot.VolumeAttenuationColor.b);
            data.VolumeAttenuationDistance = slot.VolumeAttenuationDistance;
        }

        private void UpdateInstanceTransforms()
        {
            if (_instanceCpu == null || _instanceGpuBuf == null) return;

            foreach (var entry in _rendererEntries.Values)
            {
                if (entry.transform == null) continue;

                bool moved = entry.transform.hasChanged;
                if (moved) entry.transform.hasChanged = false;

                if (!moved && !entry.wasMoving) continue;

                int start = entry.firstInstanceIdx;
                int count = Mathf.Min(entry.instanceCount, _instanceCpu.Length - start);
                if (count <= 0) continue;

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

                        var si = _sceneInstances[i];
                        if (si.groupHandle != 0)
                            _accelStructure.SetInstanceTransform(si.groupHandle, m);
                        else
                            _accelStructure.SetInstanceTransform(si.renderer, m);
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

        private int BuildMaterialFromOverride(RtxptMaterial slot, List<PTMaterialData> ptMatList, List<IntPtr> texPtrs)
        {
            int idx = ptMatList.Count;

            int baseTexIdx    = AddTexture(slot.BaseOrDiffuseTexture,              texPtrs);
            int ormTexIdx     = AddTexture(slot.OcclusionRoughnessMetallicTexture, texPtrs);
            int normalTexIdx  = AddTexture(slot.NormalTexture,                     texPtrs);
            int emissiveTexIdx = AddTexture(slot.EmissiveTexture,                  texPtrs);
            int transmTexIdx  = AddTexture(slot.TransmissionTexture,               texPtrs);

            uint SafeIdx(int i) => i >= 0 ? (uint)i : 0xFFFFFFFFu;

            uint flags = 0;
            if (slot.UseSpecularGlossModel)                                                         flags |= PTMaterialFlags.UseSpecularGlossModel;
            if (baseTexIdx    >= 0 && slot.EnableBaseTexture)                                        flags |= PTMaterialFlags.UseBaseOrDiffuseTexture;
            if (ormTexIdx     >= 0 && slot.EnableOcclusionRoughnessMetallicTexture)                  flags |= PTMaterialFlags.UseMetalRoughOrSpecularTexture;
            if (emissiveTexIdx >= 0 && slot.EnableEmissiveTexture)                                   flags |= PTMaterialFlags.UseEmissiveTexture;
            if (normalTexIdx  >= 0 && slot.EnableNormalTexture)                                      flags |= PTMaterialFlags.UseNormalTexture;
            if (transmTexIdx  >= 0 && slot.EnableTransmissionTexture && slot.EnableTransmission)     flags |= PTMaterialFlags.UseTransmissionTexture;
            if (slot.MetalnessInRedChannel)                                                          flags |= PTMaterialFlags.MetalnessInRedChannel;
            if (slot.ThinSurface || !slot.EnableTransmission)                                        flags |= PTMaterialFlags.ThinSurface;
            if (slot.PSDExclude)                                                                     flags |= PTMaterialFlags.PSDExclude;
            if (slot.EnableAsAnalyticLightProxy)                                                     flags |= PTMaterialFlags.EnableAsAnalyticLightProxy;
            if (slot.IgnoreMeshTangentSpace)                                                         flags |= PTMaterialFlags.IgnoreMeshTangentSpace;
            if (slot.PSDBlockMotionVectorsAtSurfaceType % 2 != 0)                                    flags |= PTMaterialFlags.PSDBlockMVsAtSurfaceTypeB0;
            if (slot.PSDBlockMotionVectorsAtSurfaceType / 2 != 0)                                    flags |= PTMaterialFlags.PSDBlockMVsAtSurfaceTypeB1;
            flags |= (uint)Mathf.Clamp(slot.NestedPriority, 0, 14) << PTMaterialFlags.NestedPriorityShift;
            flags |= (uint)Mathf.Clamp(slot.PSDDominantDeltaLobe + 1, 0, 7) << PTMaterialFlags.PSDDominantDeltaLobeP1Shift;

            ptMatList.Add(new PTMaterialData
            {
                BaseOrDiffuseColor               = new Vector3(slot.BaseColorFactor.r, slot.BaseColorFactor.g, slot.BaseColorFactor.b),
                Flags                            = flags,
                SpecularColor                    = new Vector3(slot.SpecularColor.r, slot.SpecularColor.g, slot.SpecularColor.b),
                _padding0                        = 42,
                EmissiveColor                    = new Vector3(slot.EmissiveColor.r, slot.EmissiveColor.g, slot.EmissiveColor.b) * slot.EmissiveIntensity,
                ShadowNoLFadeout                 = Mathf.Clamp(slot.ShadowNoLFadeout, 0f, 0.25f),
                Opacity                          = slot.Opacity,
                Roughness                        = slot.Roughness,
                Metalness                        = slot.Metalness,
                NormalTextureScale               = slot.NormalTextureScale,
                _padding1                        = 42f,
                AlphaCutoff                      = slot.AlphaCutoff,
                TransmissionFactor               = slot.EnableTransmission ? slot.TransmissionFactor        : 0f,
                BaseOrDiffuseTextureIndex        = SafeIdx(baseTexIdx),
                MetalRoughOrSpecularTextureIndex = SafeIdx(ormTexIdx),
                EmissiveTextureIndex             = SafeIdx(emissiveTexIdx),
                NormalTextureIndex               = SafeIdx(normalTexIdx),
                OcclusionTextureIndex            = 0u, // C++ FillData never writes this field; PTMaterialData is zero-initialized so it stays 0 (occlusion is packed into ORM, UseOcclusionTexture is disabled). See MaterialsBaker.cpp:516/907
                TransmissionTextureIndex         = SafeIdx(transmTexIdx),
                IoR                              = slot.IoR,
                ThicknessFactor                  = slot.ThicknessFactor,
                DiffuseTransmissionFactor        = slot.EnableTransmission ? slot.DiffuseTransmissionFactor : 0f,
                VolumeAttenuationColor           = new Vector3(slot.VolumeAttenuationColor.r, slot.VolumeAttenuationColor.g, slot.VolumeAttenuationColor.b),
                VolumeAttenuationDistance        = slot.VolumeAttenuationDistance,
            });
            return idx;
        }

        // Returns the GPU material index for a sub-mesh's pre-baked RtxptMaterial. Only called for
        // sub-meshes the build passes have already confirmed are assigned (see SubmeshHasMaterial),
        // so matOverride.Slots[subMeshIndex] is non-null. Runtime baking of Unity Materials is no
        // longer supported — materials must be authored as RtxptMaterial assets in advance.
        private int GetOrAddMaterial(int subMeshIndex, RtxptRenderer matOverride,
            List<PTMaterialData> ptMatList, List<IntPtr> texPtrs)
        {
            // RtxptMaterial assets are shareable across renderers/sub-meshes, so dedup on the asset
            // reference: identical references reuse the same GPU material entry rather than emitting
            // one per (renderer, subMesh). Keeps MaterialCount = unique materials, matching the C++
            // baker (Sample.cpp:2095).
            RtxptMaterial asset = matOverride.Slots[subMeshIndex];
            int assetId = asset.GetInstanceID();
            if (_overrideSlots.TryGetValue(assetId, out int existing))
                return existing;

            int newIdx = BuildMaterialFromOverride(asset, ptMatList, texPtrs);
            _overrideSlots[assetId] = newIdx;
            return newIdx;
        }

        // True when renderer rr has a pre-baked RtxptMaterial assigned for the given sub-mesh.
        private static bool SubmeshHasMaterial(RtxptRenderer rr, int subMesh)
            => rr != null && subMesh < rr.Slots.Count && rr.Slots[subMesh] != null;

        // True when every sub-mesh [0, subMeshCount) has an assigned RtxptMaterial.
        private static bool AllSubmeshesAssigned(RtxptRenderer rr, int subMeshCount)
        {
            if (rr == null) return false;
            for (int s = 0; s < subMeshCount; s++)
                if (!SubmeshHasMaterial(rr, s)) return false;
            return true;
        }

        private int AddTexture(Texture tex, List<IntPtr> texPtrs)
        {
            if (tex == null) return -1;
            int texId = tex.GetInstanceID();
            if (_textureSlots.TryGetValue(texId, out int slot)) return slot;
            slot = texPtrs.Count;
            texPtrs.Add(tex.GetNativeTexturePtr());
            _textureSlots[texId] = slot;
            return slot;
        }

        private bool TargetSetChanged(IReadOnlyList<NativeRayTracingTarget> current)
        {
            if (current.Count != _registeredTargets.Count) return true;
            for (int i = 0; i < current.Count; i++)
                if (current[i] != _registeredTargets[i])
                    return true;
            return false;
        }

        /// <summary>
        /// Builds donut-compatible SoA vertex buffer and uint32 index buffer for the given mesh.
        /// VB layout: [Position: float3 × vc][Normal: RGB8_SNORM × vc][TexCoord: float2 × vc][Tangent: RGBA8_SNORM × vc]
        /// IB layout: uint32 per index, same slot layout as Unity submesh indexStart.
        /// Both returned as <c>GraphicsBuffer.Target.Raw</c> (ByteAddressBuffer).
        /// </summary>
        private (GraphicsBuffer vb, GraphicsBuffer ib) GetOrCreateDonutBuffers(Mesh src)
        {
            if (src == null) return (null, null);
            int key = src.GetInstanceID();
            if (_donutBufferCache.TryGetValue(key, out var cached)) return cached;
 
            int  vc         = src.vertexCount;
            bool hasNormal  = src.HasVertexAttribute(VertexAttribute.Normal);
            bool hasUV      = src.HasVertexAttribute(VertexAttribute.TexCoord0);
            bool hasTangent = src.HasVertexAttribute(VertexAttribute.Tangent);

            // ---- VB (SoA) ----
            int vbBytes             = vc * 12; // position always present
            if (hasNormal) vbBytes  += vc * 4; // RGB8_SNORM
            if (hasUV) vbBytes      += vc * 8; // float2
            if (hasTangent) vbBytes += vc * 4; // RGBA8_SNORM

            var vbData = new byte[vbBytes];

            // Position stream (float3, no compression)
            Vector3[] positions = src.vertices;
            int       writePos  = 0;
            for (int i = 0; i < vc; i++)
            {
                Buffer.BlockCopy(BitConverter.GetBytes(positions[i].x), 0, vbData, writePos, 4);
                Buffer.BlockCopy(BitConverter.GetBytes(positions[i].y), 0, vbData, writePos + 4, 4);
                Buffer.BlockCopy(BitConverter.GetBytes(positions[i].z), 0, vbData, writePos + 8, 4);
                writePos += 12;
            }

            // Normal stream (RGB8_SNORM, 4 bytes each)
            if (hasNormal)
            {
                Vector3[] normals = src.normals;
                for (int i = 0; i < vc; i++)
                {
                    uint packed = PackRGB8Snorm(normals[i]);
                    Buffer.BlockCopy(BitConverter.GetBytes(packed), 0, vbData, writePos, 4);
                    writePos += 4;
                }
            }

            // TexCoord stream (float2, 8 bytes each)
            if (hasUV)
            {
                Vector2[] uvs = src.uv;
                for (int i = 0; i < vc; i++)
                {
                    Buffer.BlockCopy(BitConverter.GetBytes(uvs[i].x), 0, vbData, writePos, 4);
                    Buffer.BlockCopy(BitConverter.GetBytes(uvs[i].y), 0, vbData, writePos + 4, 4);
                    writePos += 8;
                }
            }

            // Tangent stream (RGBA8_SNORM, 4 bytes each)
            if (hasTangent)
            {
                Vector4[] tangents = src.tangents;
                for (int i = 0; i < vc; i++)
                {
                    // todo 这里取反了
                    uint packed = PackRGBA8Snorm(new Vector4(tangents[i].x, tangents[i].y, tangents[i].z, -tangents[i].w));
                    Buffer.BlockCopy(BitConverter.GetBytes(packed), 0, vbData, writePos, 4);
                    writePos += 4;
                }
            }

            var vbUint = new uint[vbBytes / 4];
            Buffer.BlockCopy(vbData, 0, vbUint, 0, vbBytes);
            var vbGfx = new GraphicsBuffer(GraphicsBuffer.Target.Raw, vbBytes / 4, 4) { name = "VertexBuffer" };
            vbGfx.SetData(vbUint);

            // ---- IB (uint32, matching Unity submesh indexStart layout) ----
            int totalIndexSlots = 0;
            for (int s = 0; s < src.subMeshCount; s++)
            {
                var sub = src.GetSubMesh(s);
                totalIndexSlots = Mathf.Max(totalIndexSlots, sub.indexStart + sub.indexCount);
            }

            var ibData = new uint[Mathf.Max(totalIndexSlots, 3)];
            for (int s = 0; s < src.subMeshCount; s++)
            {
                var   sub    = src.GetSubMesh(s);
                int[] subIdx = src.GetIndices(s, applyBaseVertex: true);
                for (int k = 0; k < subIdx.Length; k++)
                    ibData[sub.indexStart + k] = (uint)subIdx[k];
            }

            int ibBytes = ibData.Length * 4;
            var ibGfx   = new GraphicsBuffer(GraphicsBuffer.Target.Raw, ibBytes / 4, 4) { name = "IndexBuffer" };
            ibGfx.SetData(ibData);

            _ownedGfxBuffers.Add(vbGfx);
            _ownedGfxBuffers.Add(ibGfx);
            var result = (vbGfx, ibGfx);
            _donutBufferCache[key] = result;
            return result;
        }

        private static uint PackRGB8Snorm(Vector3 v)
        {
            byte r = (byte)(Mathf.RoundToInt(Mathf.Clamp(v.x, -1f, 1f) * 127f) & 0xFF);
            byte g = (byte)(Mathf.RoundToInt(Mathf.Clamp(v.y, -1f, 1f) * 127f) & 0xFF);
            byte b = (byte)(Mathf.RoundToInt(Mathf.Clamp(v.z, -1f, 1f) * 127f) & 0xFF);
            return (uint)(r | (g << 8) | (b << 16));
        }

        private static uint PackRGBA8Snorm(Vector4 v)
        {
            byte r = (byte)(Mathf.RoundToInt(Mathf.Clamp(v.x, -1f, 1f) * 127f) & 0xFF);
            byte g = (byte)(Mathf.RoundToInt(Mathf.Clamp(v.y, -1f, 1f) * 127f) & 0xFF);
            byte b = (byte)(Mathf.RoundToInt(Mathf.Clamp(v.z, -1f, 1f) * 127f) & 0xFF);
            byte a = (byte)(Mathf.RoundToInt(Mathf.Clamp(v.w, -1f, 1f) * 127f) & 0xFF);
            return (uint)(r | (g << 8) | (b << 16) | (a << 24));
        }
    }
}
