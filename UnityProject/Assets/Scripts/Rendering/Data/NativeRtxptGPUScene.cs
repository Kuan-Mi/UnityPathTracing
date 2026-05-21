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
        private GraphicsBuffer _instanceGpuBuf; // t_InstanceData  (t2)
        private GraphicsBuffer _geometryGpuBuf; // t_GeometryData  (t3)

        // RTXPT-specific structured buffers
        private GraphicsBuffer _subInstanceGpuBuf; // t_SubInstanceData    (t1)
        private GraphicsBuffer _ptMaterialGpuBuf; // t_PTMaterialData     (t5)
        private GraphicsBuffer _geomDebugGpuBuf; // t_GeometryDebugData  (t4)

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

            public Transform transform;

            // Previous frame's world matrix rows (for prevTransform)
            public Vector4 prevRow0;
            public Vector4 prevRow1;
            public Vector4 prevRow2;
        }

        private readonly List<SceneInstance>                                     _sceneInstances    = new();
        private readonly Dictionary<int, (int vb, int ib)>                       _meshBufferSlots   = new();
        private readonly Dictionary<int, int>                                    _materialSlots     = new();
        private readonly Dictionary<int, int>                                    _textureSlots      = new();
        private readonly Dictionary<int, (GraphicsBuffer vb, GraphicsBuffer ib)> _donutBufferCache  = new();
        private readonly List<GraphicsBuffer>                                    _ownedGfxBuffers   = new();
        private readonly List<NativeRayTracingTarget>                            _registeredTargets = new();

        private bool _sceneGpuDirty = true;
        private bool _forceRebuild  = false;
        private bool _disposed;

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
            ds.SetStructuredBuffer("t_SubInstanceData", _subInstanceGpuBuf.GetNativeBufferPtr(), _subInstanceGpuBuf.count, _subInstanceGpuBuf.stride);
            ds.SetStructuredBuffer("t_InstanceData", _instanceGpuBuf.GetNativeBufferPtr(), _instanceGpuBuf.count, _instanceGpuBuf.stride);
            ds.SetStructuredBuffer("t_GeometryData", _geometryGpuBuf.GetNativeBufferPtr(), _geometryGpuBuf.count, _geometryGpuBuf.stride);
            ds.SetStructuredBuffer("t_GeometryDebugData", _geomDebugGpuBuf.GetNativeBufferPtr(), _geomDebugGpuBuf.count, _geomDebugGpuBuf.stride);
            ds.SetStructuredBuffer("t_PTMaterialData", _ptMaterialGpuBuf.GetNativeBufferPtr(), _ptMaterialGpuBuf.count, _ptMaterialGpuBuf.stride);
            ds.SetBindlessBuffer("t_BindlessBuffers", _sceneBuffers);
            ds.SetBindlessTexture("t_BindlessTextures", _sceneTextures);
        }

        public void BindToShader(NativeRayTraceDescriptorSet ds)
        {
            if (ds == null) return;
            ds.SetStructuredBuffer("t_SubInstanceData", _subInstanceGpuBuf.GetNativeBufferPtr(), _subInstanceGpuBuf.count, _subInstanceGpuBuf.stride);
            ds.SetStructuredBuffer("t_InstanceData", _instanceGpuBuf.GetNativeBufferPtr(), _instanceGpuBuf.count, _instanceGpuBuf.stride);
            ds.SetStructuredBuffer("t_GeometryData", _geometryGpuBuf.GetNativeBufferPtr(), _geometryGpuBuf.count, _geometryGpuBuf.stride);
            ds.SetStructuredBuffer("t_GeometryDebugData", _geomDebugGpuBuf.GetNativeBufferPtr(), _geomDebugGpuBuf.count, _geomDebugGpuBuf.stride);
            ds.SetStructuredBuffer("t_PTMaterialData", _ptMaterialGpuBuf.GetNativeBufferPtr(), _ptMaterialGpuBuf.count, _ptMaterialGpuBuf.stride);
            ds.SetBindlessBuffer("t_BindlessBuffers", _sceneBuffers);
            ds.SetBindlessTexture("t_BindlessTextures", _sceneTextures);
        }

        /// <summary>Builds / updates the TLAS. Call inside a CommandBuffer recording.</summary>
        public void BuildAccelerationStructure(CommandBuffer cmd)
        {
            _accelStructure.BuildOrUpdate(cmd);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            DisposeGpuBuffers();
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
                    if (mr != null) _accelStructure.RemoveInstance(mr);
                }
            }

            foreach (var t in targets)
            {
                if (t == null) continue;
                var mr = t.GetComponent<MeshRenderer>();
                if (mr == null) continue;
                _accelStructure.AddInstance(mr);
            }
        }

        private void DisposeGpuBuffers()
        {
            _instanceGpuBuf?.Release();
            _instanceGpuBuf = null;
            _geometryGpuBuf?.Release();
            _geometryGpuBuf = null;
            _subInstanceGpuBuf?.Release();
            _subInstanceGpuBuf = null;
            _ptMaterialGpuBuf?.Release();
            _ptMaterialGpuBuf = null;
            _geomDebugGpuBuf?.Release();
            _geomDebugGpuBuf = null;
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
            _meshBufferSlots.Clear();
            _materialSlots.Clear();
            _textureSlots.Clear();
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
                    bufPtrs.Add(donutVb.GetNativeBufferPtr());
                    bufPtrs.Add(donutIb.GetNativeBufferPtr());
                    _meshBufferSlots[meshKey] = slots;
                }

                // SoA offsets (matches c_SizeOfPosition=12, c_SizeOfNormal=4, c_SizeOfTexcoord=8)
                uint vc      = (uint)mesh.vertexCount;
                uint posOff  = 0u;
                uint normOff = mesh.HasVertexAttribute(VertexAttribute.Normal) ? vc * 12u : 0xFFFFFFFFu;
                uint uvOff   = mesh.HasVertexAttribute(VertexAttribute.TexCoord0) ? vc * (12u + 4u) : 0xFFFFFFFFu;
                uint tanOff  = mesh.HasVertexAttribute(VertexAttribute.Tangent) ? vc * (12u + 4u + 8u) : 0xFFFFFFFFu;

                Material[] mats       = mr.sharedMaterials ?? Array.Empty<Material>();
                int        subMeshCnt = mesh.subMeshCount;
                int        firstGeom  = geomList.Count;
                int        instIdx    = instList.Count;

                var matOverride = mr.GetComponent<NativeRtxptMaterialOverride>();

                for (int s = 0; s < subMeshCnt; s++)
                {
                    SubMeshDescriptor sub    = mesh.GetSubMesh(s);
                    Material          mat    = s < mats.Length ? mats[s] : (mats.Length > 0 ? mats[mats.Length - 1] : null);
                    int               matIdx = GetOrAddMaterial(mat, s, matOverride, ptMatList, texPtrs);

                    int globalGeomIdx = geomList.Count; // index we're about to push

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
                    bool  isAlphaTested = mat != null && TryGetFloat(mat, "_Cutoff", 0f) > 0f;
                    float aCutoff       = isAlphaTested ? TryGetFloat(mat, "_Cutoff", 0f) : 0f;
                    uint  alphaU8       = (uint)Mathf.RoundToInt(Mathf.Clamp01(aCutoff) * 255f);
                    // AlphaTextureIndex in lo16: use base texture slot if alpha-tested, else 0
                    uint alphaTexIdx = isAlphaTested && ptMatList.Count > matIdx
                        ? (uint)Mathf.Max(0, ptMatList[matIdx].BaseOrDiffuseTextureIndex)
                        : 0u;
                    uint siFlags               = alphaTexIdx & 0xFFFFu;
                    if (isAlphaTested) siFlags |= SubInstanceFlags.AlphaTested;
                    siFlags |= (alphaU8 << SubInstanceFlags.AlphaOffsetOffset);

                    subInstList.Add(new SubInstanceData
                    {
                        FlagsAndAlphaInfo                       = siFlags,
                        GlobalGeometryIndex_PTMaterialDataIndex = ((uint)globalGeomIdx << 16) | ((uint)matIdx & 0xFFFFu),
                        EmissiveLightMappingOffset              = 0xFFFFFFFFu, // no light baker yet
                        AnalyticProxyLightIndex                 = 0u,
                        IndexBufferIndex_VertexBufferIndex      = ((uint)slots.ib << 16) | ((uint)slots.vb & 0xFFFFu),
                        IndexOffset                             = (uint)sub.indexStart * 4u,
                        TexCoord1Offset                         = uvOff,
                        padding0                                = 0u,
                    });

                    // --- GeometryDebugData (RTXPT t4, all zero = no OMM) ---
                    geomDbgList.Add(default);
                }

                Matrix4x4 m    = target.transform.localToWorldMatrix;
                var       row0 = new Vector4(m.m00, m.m01, m.m02, m.m03);
                var       row1 = new Vector4(m.m10, m.m11, m.m12, m.m13);
                var       row2 = new Vector4(m.m20, m.m21, m.m22, m.m23);

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
                _sceneInstances.Add(new SceneInstance
                {
                    renderer  = mr,
                    transform = mr.transform,
                    prevRow0  = row0,
                    prevRow1  = row1,
                    prevRow2  = row2,
                });
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

            _instanceGpuBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _instanceCpu.Length, Marshal.SizeOf<DonutInstanceData>());
            _instanceGpuBuf.SetData(_instanceCpu);

            _geometryGpuBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _geometryCpu.Length, Marshal.SizeOf<DonutGeometryData>());
            _geometryGpuBuf.SetData(_geometryCpu);

            _subInstanceGpuBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _subInstanceCpu.Length, Marshal.SizeOf<SubInstanceData>());
            _subInstanceGpuBuf.SetData(_subInstanceCpu);

            _ptMaterialGpuBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _ptMaterialCpu.Length, Marshal.SizeOf<PTMaterialData>());
            _ptMaterialGpuBuf.SetData(_ptMaterialCpu);

            _geomDebugGpuBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _geomDebugCpu.Length, Marshal.SizeOf<GeometryDebugData>());
            _geomDebugGpuBuf.SetData(_geomDebugCpu);

            _sceneGpuDirty = false;
        }

        private void UpdateInstanceTransforms()
        {
            if (_instanceCpu == null || _instanceGpuBuf == null) return;

            int instanceCount = Mathf.Min(_sceneInstances.Count, _instanceCpu.Length);
            int dirtyStart    = -1;

            for (int i = 0; i < instanceCount; i++)
            {
                var si = _sceneInstances[i];
                if (si.transform == null || !si.transform.hasChanged)
                {
                    if (dirtyStart >= 0)
                    {
                        _instanceGpuBuf.SetData(_instanceCpu, dirtyStart, dirtyStart, i - dirtyStart);
                        dirtyStart = -1;
                    }

                    continue;
                }

                Matrix4x4 m    = si.transform.localToWorldMatrix;
                var       row0 = new Vector4(m.m00, m.m01, m.m02, m.m03);
                var       row1 = new Vector4(m.m10, m.m11, m.m12, m.m13);
                var       row2 = new Vector4(m.m20, m.m21, m.m22, m.m23);

                // shift current → prev
                _instanceCpu[i].prevTransformRow0 = _instanceCpu[i].transformRow0;
                _instanceCpu[i].prevTransformRow1 = _instanceCpu[i].transformRow1;
                _instanceCpu[i].prevTransformRow2 = _instanceCpu[i].transformRow2;
                // write new current
                _instanceCpu[i].transformRow0 = row0;
                _instanceCpu[i].transformRow1 = row1;
                _instanceCpu[i].transformRow2 = row2;

                _accelStructure.SetInstanceTransform(si.renderer, m);
                if (dirtyStart < 0) dirtyStart = i;

                // update cached prev for next frame
                var updated = si;
                updated.prevRow0   = row0;
                updated.prevRow1   = row1;
                updated.prevRow2   = row2;
                _sceneInstances[i] = updated;
            }

            if (dirtyStart >= 0)
                _instanceGpuBuf.SetData(_instanceCpu, dirtyStart, dirtyStart, instanceCount - dirtyStart);
        }

        private int BuildMaterialFromOverride(RtxptMaterialSlot slot, List<PTMaterialData> ptMatList, List<IntPtr> texPtrs)
        {
            int idx = ptMatList.Count;

            int baseTexIdx       = AddTexture(slot.BaseOrDiffuseTexture, texPtrs);
            int normalTexIdx     = AddTexture(slot.NormalTexture,        texPtrs);
            int metalRoughTexIdx = AddTexture(slot.MetalRoughTexture,    texPtrs);
            int emissiveTexIdx   = AddTexture(slot.EmissiveTexture,      texPtrs);
            int occlusionTexIdx  = AddTexture(slot.OcclusionTexture,     texPtrs);
            int transmTexIdx     = AddTexture(slot.TransmissionTexture,  texPtrs);

            uint SafeIdx(int i) => i >= 0 ? (uint)i : 0xFFFFFFFFu;

            uint flags = 0;
            if (baseTexIdx >= 0)       flags |= PTMaterialFlags.UseBaseOrDiffuseTexture;
            if (normalTexIdx >= 0)     flags |= PTMaterialFlags.UseNormalTexture;
            if (metalRoughTexIdx >= 0) flags |= PTMaterialFlags.UseMetalRoughOrSpecularTexture;
            if (emissiveTexIdx >= 0)   flags |= PTMaterialFlags.UseEmissiveTexture;
            if (occlusionTexIdx >= 0)  flags |= (uint)DonutMaterialFlags.UseOcclusionTexture;
            if (transmTexIdx >= 0)     flags |= PTMaterialFlags.UseTransmissionTexture;
            if (slot.ThinSurface)              flags |= PTMaterialFlags.ThinSurface;
            if (slot.MetalnessInRedChannel)    flags |= PTMaterialFlags.MetalnessInRedChannel;
            if (slot.PSDExclude)               flags |= PTMaterialFlags.PSDExclude;
            if (slot.IgnoreMeshTangentSpace)   flags |= PTMaterialFlags.IgnoreMeshTangentSpace;
            if (slot.PSDBlockMotionVectorsAtSurfaceType % 2 != 0) flags |= PTMaterialFlags.PSDBlockMVsAtSurfaceTypeB0;
            if (slot.PSDBlockMotionVectorsAtSurfaceType / 2 != 0) flags |= PTMaterialFlags.PSDBlockMVsAtSurfaceTypeB1;
            flags |= (uint)Mathf.Clamp(slot.NestedPriority, 0, 15)        << PTMaterialFlags.NestedPriorityShift;
            flags |= (uint)Mathf.Clamp(slot.PSDDominantDeltaLobe + 1, 0, 7) << PTMaterialFlags.PSDDominantDeltaLobeP1Shift;

            ptMatList.Add(new PTMaterialData
            {
                BaseOrDiffuseColor               = new Vector3(slot.BaseOrDiffuseColor.r, slot.BaseOrDiffuseColor.g, slot.BaseOrDiffuseColor.b),
                Flags                            = flags,
                SpecularColor                    = new Vector3(slot.SpecularColor.r, slot.SpecularColor.g, slot.SpecularColor.b),
                _padding0                        = 0,
                EmissiveColor                    = new Vector3(slot.EmissiveColor.r, slot.EmissiveColor.g, slot.EmissiveColor.b),
                ShadowNoLFadeout                 = Mathf.Clamp(slot.ShadowNoLFadeout, 0f, 0.25f),
                Opacity                          = slot.Opacity,
                Roughness                        = slot.Roughness,
                Metalness                        = slot.Metalness,
                NormalTextureScale               = slot.NormalTextureScale,
                _padding1                        = 0f,
                AlphaCutoff                      = slot.AlphaCutoff,
                TransmissionFactor               = slot.TransmissionFactor,
                BaseOrDiffuseTextureIndex        = SafeIdx(baseTexIdx),
                MetalRoughOrSpecularTextureIndex = SafeIdx(metalRoughTexIdx),
                EmissiveTextureIndex             = SafeIdx(emissiveTexIdx),
                NormalTextureIndex               = SafeIdx(normalTexIdx),
                OcclusionTextureIndex            = SafeIdx(occlusionTexIdx),
                TransmissionTextureIndex         = SafeIdx(transmTexIdx),
                IoR                              = slot.IoR,
                ThicknessFactor                  = slot.ThicknessFactor,
                DiffuseTransmissionFactor        = slot.DiffuseTransmissionFactor,
                VolumeAttenuationColor           = new Vector3(slot.VolumeAttenuationColor.r, slot.VolumeAttenuationColor.g, slot.VolumeAttenuationColor.b),
                VolumeAttenuationDistance        = slot.VolumeAttenuationDistance,
            });
            return idx;
        }

        private int GetOrAddMaterial(Material mat, int subMeshIndex, NativeRtxptMaterialOverride matOverride,
                                        List<PTMaterialData> ptMatList, List<IntPtr> texPtrs)
        {
            // ----------------------------------------------------------------
            // Fast path: renderer has a manual override for this sub-mesh
            // ----------------------------------------------------------------
            if (matOverride != null && subMeshIndex < matOverride.Slots.Count)
            {
                // Overrides are per-(renderer, subMesh), so skip the shared material cache
                // and always produce a unique GPU material entry.
                return BuildMaterialFromOverride(matOverride.Slots[subMeshIndex], ptMatList, texPtrs);
            }

            // ----------------------------------------------------------------
            // Normal path: derive from Unity Material
            // ----------------------------------------------------------------
            int matId = mat != null ? mat.GetInstanceID() : -1;
            if (_materialSlots.TryGetValue(matId, out int existing))
                return existing;

            int idx = ptMatList.Count;
            _materialSlots[matId] = idx;

            int   baseTexIdx, normalTexIdx, metalRoughTexIdx, emissiveTexIdx, occlusionTexIdx;
            Color baseColor;
            Color emissive;
            float roughness, metalness, alphaCutoff, normalScale, occStr;
            int   domain;

            bool isGltf = mat != null && mat.shader.name == "Shader Graphs/glTF-pbrMetallicRoughness";
            if (isGltf)
            {
                baseTexIdx       = AddTexture(TryGetTex(mat, "baseColorTexture"), texPtrs);
                normalTexIdx     = AddTexture(TryGetTex(mat, "normalTexture"), texPtrs);
                metalRoughTexIdx = AddTexture(TryGetTex(mat, "metallicRoughnessTexture"), texPtrs);
                emissiveTexIdx   = AddTexture(TryGetTex(mat, "emissiveTexture"), texPtrs);
                occlusionTexIdx  = AddTexture(TryGetTex(mat, "occlusionTexture"), texPtrs);

                baseColor   = TryGetColor(mat, "baseColorFactor", Color.white);
                emissive    = TryGetColor(mat, "emissiveFactor", Color.black);
                roughness   = TryGetFloat(mat, "roughnessFactor", 0.5f);
                metalness   = TryGetFloat(mat, "metallicFactor", 0f);
                alphaCutoff = mat.IsKeywordEnabled("_ALPHATEST_ON") ? TryGetFloat(mat, "alphaCutoff", 0.5f) : 0f;
                normalScale = 1f;
                occStr      = TryGetFloat(mat, "occlusionStrength", 1f);

                domain = mat.IsKeywordEnabled("_SURFACE_TYPE_TRANSPARENT") ? 1 : 0;
            }
            else
            {
                // URP/Lit, RayTracing/Lit, and unknown-shader fallback
                baseTexIdx       = AddTexture(TryGetTex(mat, "_BaseMap"), texPtrs);
                normalTexIdx     = AddTexture(TryGetTex(mat, "_BumpMap"), texPtrs);
                metalRoughTexIdx = AddTexture(TryGetTex(mat, "_MetallicGlossMap"), texPtrs);
                emissiveTexIdx   = AddTexture(TryGetTex(mat, "_EmissionMap"), texPtrs);
                occlusionTexIdx  = AddTexture(TryGetTex(mat, "_OcclusionMap"), texPtrs);

                baseColor   = TryGetColor(mat, "_BaseColor", Color.white);
                emissive    = TryGetColor(mat, "_EmissionColor", Color.black);
                roughness   = 1f - TryGetFloat(mat, "_Smoothness", 0.5f);
                metalness   = TryGetFloat(mat, "_Metallic", 0f);
                alphaCutoff = TryGetFloat(mat, "_Cutoff", 0f);
                normalScale = TryGetFloat(mat, "_BumpScale", 1f);
                occStr      = TryGetFloat(mat, "_OcclusionStrength", 1f);

                domain = 0;
                if (mat != null && mat.HasProperty("_Surface"))
                    domain = (int)mat.GetFloat("_Surface");
                else if (alphaCutoff > 0f)
                    domain = 1; // AlphaTested
            }

            // Approximate F0 specular colour (donut convention)
            Vector3 dielectricF0   = Vector3.one * 0.04f;
            Vector3 metalBaseColor = new Vector3(baseColor.r, baseColor.g, baseColor.b);
            Vector3 specularColor  = Vector3.Lerp(dielectricF0, metalBaseColor, metalness);

            int matFlags                        = 0;
            if (baseTexIdx >= 0) matFlags       |= DonutMaterialFlags.UseBaseOrDiffuseTexture;
            if (normalTexIdx >= 0) matFlags     |= DonutMaterialFlags.UseNormalTexture;
            if (metalRoughTexIdx >= 0) matFlags |= DonutMaterialFlags.UseMetalRoughOrSpecularTexture;
            if (emissiveTexIdx >= 0) matFlags   |= DonutMaterialFlags.UseEmissiveTexture;
            if (occlusionTexIdx >= 0) matFlags  |= DonutMaterialFlags.UseOcclusionTexture;

            if (metalRoughTexIdx >= 0)
                metalness = 1;

            // --- PTMaterialData (same flag bit values as DonutMaterialFlags for texture flags) ---
            // PTMaterialData texture indices use uint with 0xFFFFFFFF = none
            uint SafeTexIdx(int i) => i >= 0 ? (uint)i : 0xFFFFFFFFu;

            // Additional PT-specific flags (beyond the shared donut texture flags)
            uint ptExtraFlags = 0;
            // All Unity materials use TransmissionFactor=0 (no transmission), so they are thin surfaces.
            // Matches C++ MaterialsBaker: if (ThinSurface || !EnableTransmission) data.Flags |= ThinSurface
            ptExtraFlags |= PTMaterialFlags.ThinSurface;
            // URP _MetallicGlossMap stores metalness in the R channel
            if (metalRoughTexIdx >= 0)
                ptExtraFlags |= PTMaterialFlags.MetalnessInRedChannel;

            ptMatList.Add(new PTMaterialData
            {
                BaseOrDiffuseColor               = new Vector3(baseColor.r, baseColor.g, baseColor.b),
                Flags                            = (uint)matFlags | ptExtraFlags, // flag bits 1..8 are identical between donut and RTXPT
                SpecularColor                    = specularColor,
                _padding0                        = 0,
                EmissiveColor                    = new Vector3(emissive.r, emissive.g, emissive.b),
                ShadowNoLFadeout                 = 0f, // C++ clamps to [0, 0.25f]; 0 = full shadow fadeout always shown
                Opacity                          = baseColor.a,
                Roughness                        = roughness,
                Metalness                        = metalness,
                NormalTextureScale               = normalScale,
                _padding1                        = 0f,
                AlphaCutoff                      = alphaCutoff,
                TransmissionFactor               = 0f,
                BaseOrDiffuseTextureIndex        = SafeTexIdx(baseTexIdx),
                MetalRoughOrSpecularTextureIndex = SafeTexIdx(metalRoughTexIdx),
                EmissiveTextureIndex             = SafeTexIdx(emissiveTexIdx),
                NormalTextureIndex               = SafeTexIdx(normalTexIdx),
                OcclusionTextureIndex            = SafeTexIdx(occlusionTexIdx),
                TransmissionTextureIndex         = 0xFFFFFFFFu,
                IoR                              = 1.5f,
                ThicknessFactor                  = 0f,
                DiffuseTransmissionFactor        = 0f,
                VolumeAttenuationColor           = Vector3.one,
                VolumeAttenuationDistance        = float.MaxValue,
            });

            return idx;
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

        private static Texture TryGetTex(Material mat, string prop)
        {
            if (mat == null || !mat.HasProperty(prop)) return null;
            return mat.GetTexture(prop);
        }

        private static Color TryGetColor(Material mat, string prop, Color fallback)
        {
            if (mat == null || !mat.HasProperty(prop)) return fallback;
            return mat.GetColor(prop);
        }

        private static float TryGetFloat(Material mat, string prop, float fallback)
        {
            if (mat == null || !mat.HasProperty(prop)) return fallback;
            return mat.GetFloat(prop);
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
            var vbGfx = new GraphicsBuffer(GraphicsBuffer.Target.Raw, vbBytes / 4, 4);
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
            var ibGfx   = new GraphicsBuffer(GraphicsBuffer.Target.Raw, ibBytes / 4, 4);
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