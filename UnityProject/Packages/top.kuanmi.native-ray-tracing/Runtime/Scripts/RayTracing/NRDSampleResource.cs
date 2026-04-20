using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

namespace NativeRender
{
    /// <summary>
    /// Scene-resident resources mirroring NRDSample.cpp's resource taxonomy:
    /// dual TLAS (world + emissive), per-instance / per-geometry / per-material
    /// structured buffers, per-triangle primitive buffer, SHARC ring buffers,
    /// Morph stubs (kept as 1-element allocations for a fully static scene),
    /// and bindless VB/IB + texture arrays.
    ///
    /// Usage mirrors <see cref="GPUScene"/>: call <see cref="UpdateForFrame"/>
    /// from the feature, <see cref="BuildAccelerationStructures"/> inside the
    /// command buffer, then <see cref="BindToShader"/> on any ray-tracing shader.
    /// </summary>
    public sealed class NRDSampleResource : IDisposable
    {
        // Matches "#define SHARC_CAPACITY ( 1 << 23 )" in Shared.hlsl.
        public const int SharcCapacity = 1 << 23;

        // ----- Acceleration structures -----
        private RayTracingAccelerationStructure _worldAS;
        private RayTracingAccelerationStructure _emissiveAS;

        // ----- Scene structured buffers (matches GPUScene layout) -----
        private GraphicsBuffer _instanceBuf;
        private GraphicsBuffer _geometryBuf;
        private GraphicsBuffer _materialBuf;
        private GraphicsBuffer _primitiveBuf;

        // ----- SHARC ring buffers -----
        private GraphicsBuffer _sharcHashEntries;     // uint64 per entry
        private GraphicsBuffer _sharcAccumulated;     // uint4 per entry
        private GraphicsBuffer _sharcResolved;        // uint4 per entry

        // ----- Morph resource stubs (static scene => size 1) -----
        private GraphicsBuffer _morphMeshIndices;
        private GraphicsBuffer _morphMeshVertices;
        private GraphicsBuffer _morphPositions;
        private GraphicsBuffer _morphAttributes;
        private GraphicsBuffer _morphPrimitivePositions;
        private GraphicsBuffer _morphMeshScratch;

        // ----- Bindless arrays (VB/IB + textures) -----
        private BindlessBuffer  _sceneBuffers;
        private BindlessTexture _sceneTextures;

        // ----- CPU mirrors -----
        private InstanceDataGPU[]      _instanceCpu;
        private GeometryDataGPU[]      _geometryCpu;
        private MaterialConstantsGPU[] _materialCpu;
        private PrimitiveDataGPU[]     _primitiveCpu;

        // ----- Per-instance tracking -----
        private struct SceneInstance
        {
            public MeshRenderer renderer;
            public Transform    transform;
            public bool         isEmissive;
        }
        private readonly List<SceneInstance>               _sceneInstances  = new();
        private readonly Dictionary<int, (int vb, int ib)> _meshBufferSlots = new();
        private readonly Dictionary<int, int>              _materialSlots   = new();
        private readonly Dictionary<int, int>              _textureSlots    = new();
        private readonly List<NativeRayTracingTarget>      _registeredTargets = new();

        // Single-stream mesh normalisation cache.
        private readonly Dictionary<int, Mesh> _normalizedMeshCache = new();
        private readonly List<Mesh>            _ownedMeshes         = new();

        // Dirty flags.
        private bool _sceneGpuDirty = true;
        private bool _forceRebuild  = false;
        private readonly HashSet<Material> _dirtyMaterials = new();

        private bool _disposed;

        public RayTracingAccelerationStructure WorldAS    => _worldAS;
        public RayTracingAccelerationStructure EmissiveAS => _emissiveAS;

        public NRDSampleResource()
        {
            _worldAS    = new RayTracingAccelerationStructure();
            _emissiveAS = new RayTracingAccelerationStructure();
            AllocateStaticResources();
        }

        public void MarkMaterialDirty(Material mat)
        {
            if (mat != null) _dirtyMaterials.Add(mat);
        }

        public void MarkRebuildDirty()
        {
            _forceRebuild = true;
        }

        /// <summary>Dirty detection + incremental scene registration + GPU data rebuild + transform update.</summary>
        public void UpdateForFrame()
        {
            var targets = NativeRayTracingTarget.All;

            if (_dirtyMaterials.Count > 0)
            {
                _dirtyMaterials.Clear();
                _sceneGpuDirty = true;
            }
            else if (_forceRebuild || TargetSetChanged(targets))
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

        /// <summary>Build / update both TLASes (call inside a CommandBuffer).</summary>
        public void BuildAccelerationStructures(CommandBuffer cmd)
        {
            _worldAS.BuildOrUpdate(cmd);
            _emissiveAS.BuildOrUpdate(cmd);
        }

        /// <summary>Bind all scene GPU resources to a ray tracing shader.</summary>
        public void BindToShader(RayTraceShader shader)
        {
            if (shader == null || !shader.IsValid) return;

            shader.SetAccelerationStructure("gWorldTlas",    _worldAS);
            shader.SetAccelerationStructure("gEmissiveTlas", _emissiveAS);

            shader.SetStructuredBuffer("t_InstanceData",      _instanceBuf);
            shader.SetStructuredBuffer("t_GeometryData",      _geometryBuf);
            shader.SetStructuredBuffer("t_MaterialConstants", _materialBuf);
            shader.SetStructuredBuffer("t_PrimitiveData",     _primitiveBuf);

            shader.SetStructuredBuffer("t_SharcHashEntries", _sharcHashEntries);
            shader.SetStructuredBuffer("t_SharcAccumulated", _sharcAccumulated);
            shader.SetStructuredBuffer("t_SharcResolved",    _sharcResolved);

            shader.SetStructuredBuffer("t_MorphMeshIndices",        _morphMeshIndices);
            shader.SetStructuredBuffer("t_MorphMeshVertices",       _morphMeshVertices);
            shader.SetStructuredBuffer("t_MorphPositions",          _morphPositions);
            shader.SetStructuredBuffer("t_MorphAttributes",         _morphAttributes);
            shader.SetStructuredBuffer("t_MorphPrimitivePositions", _morphPrimitivePositions);
            shader.SetStructuredBuffer("t_MorphMeshScratch",        _morphMeshScratch);

            if (_sceneBuffers  != null) shader.SetBindlessBuffer ("t_BindlessBuffers",  _sceneBuffers);
            if (_sceneTextures != null) shader.SetBindlessTexture("t_BindlessTextures", _sceneTextures);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            DisposeSceneGpuBuffers();
            DisposeStaticResources();

            _worldAS?.Dispose();    _worldAS    = null;
            _emissiveAS?.Dispose(); _emissiveAS = null;
        }

        // =====================================================================
        // Static (size-invariant) resource allocation
        // =====================================================================

        private void AllocateStaticResources()
        {
            // SHARC buffers — NRDSample uses these sizes in Buffer::Sharc*.
            _sharcHashEntries = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                SharcCapacity, sizeof(ulong));
            _sharcAccumulated = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                SharcCapacity, sizeof(uint) * 4);
            _sharcResolved = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                SharcCapacity, sizeof(uint) * 4);

            // Morph stubs — scene is fully static so we keep a single-element
            // allocation per buffer to satisfy shader bindings without wasting memory.
            _morphMeshIndices        = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, sizeof(uint));
            _morphMeshVertices       = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, sizeof(uint) * 4);
            _morphPositions          = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, sizeof(float) * 4);
            _morphAttributes         = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, sizeof(uint) * 4);
            _morphPrimitivePositions = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, sizeof(float) * 4);
            _morphMeshScratch        = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, sizeof(uint) * 4);
        }

        private void DisposeStaticResources()
        {
            _sharcHashEntries?.Release(); _sharcHashEntries = null;
            _sharcAccumulated?.Release(); _sharcAccumulated = null;
            _sharcResolved?.Release();    _sharcResolved    = null;

            _morphMeshIndices?.Release();        _morphMeshIndices        = null;
            _morphMeshVertices?.Release();       _morphMeshVertices       = null;
            _morphPositions?.Release();          _morphPositions          = null;
            _morphAttributes?.Release();         _morphAttributes         = null;
            _morphPrimitivePositions?.Release(); _morphPrimitivePositions = null;
            _morphMeshScratch?.Release();        _morphMeshScratch        = null;
        }

        // =====================================================================
        // Dynamic scene GPU data
        // =====================================================================

        private void DisposeSceneGpuBuffers()
        {
            _instanceBuf?.Release();  _instanceBuf  = null;
            _geometryBuf?.Release();  _geometryBuf  = null;
            _materialBuf?.Release();  _materialBuf  = null;
            _primitiveBuf?.Release(); _primitiveBuf = null;

            _sceneBuffers?.Dispose();  _sceneBuffers  = null;
            _sceneTextures?.Dispose(); _sceneTextures = null;

            _instanceCpu  = null;
            _geometryCpu  = null;
            _materialCpu  = null;
            _primitiveCpu = null;

            _sceneInstances.Clear();
            _meshBufferSlots.Clear();
            _materialSlots.Clear();
            _textureSlots.Clear();

            for (int i = 0; i < _ownedMeshes.Count; i++)
            {
                var m = _ownedMeshes[i];
                if (m == null) continue;
                if (Application.isPlaying) UnityEngine.Object.Destroy(m);
                else UnityEngine.Object.DestroyImmediate(m);
            }
            _ownedMeshes.Clear();
            _normalizedMeshCache.Clear();
        }

        private bool TargetSetChanged(IReadOnlyList<NativeRayTracingTarget> current)
        {
            if (current.Count != _registeredTargets.Count) return true;
            for (int i = 0; i < current.Count; i++)
                if (current[i] != _registeredTargets[i])
                    return true;
            return false;
        }

        private void RegisterScene(IReadOnlyList<NativeRayTracingTarget> targets)
        {
            // If any previously registered target was destroyed we cannot call
            // RemoveInstance (no live MeshRenderer), so tear down and rebuild both AS.
            bool hasDestroyed = false;
            foreach (var target in _registeredTargets)
            {
                if (target == null) { hasDestroyed = true; break; }
            }

            if (hasDestroyed)
            {
                _worldAS?.Dispose();    _worldAS    = new RayTracingAccelerationStructure();
                _emissiveAS?.Dispose(); _emissiveAS = new RayTracingAccelerationStructure();

                foreach (var target in targets)
                {
                    if (target == null) continue;
                    var mr = target.GetComponent<MeshRenderer>();
                    if (mr == null)
                    {
                        Debug.LogWarning($"[NRDSampleResource] Target '{target.name}' has no MeshRenderer — skipping");
                        continue;
                    }

                    Matrix4x4 xform = target.transform.localToWorldMatrix;
                    if (_worldAS.AddInstance(mr, target.ommCaches))
                        _worldAS.SetInstanceTransform(mr, xform);

                    if (IsTargetEmissive(target) && _emissiveAS.AddInstance(mr, target.ommCaches))
                        _emissiveAS.SetInstanceTransform(mr, xform);
                }
                return;
            }

            var incoming = new HashSet<NativeRayTracingTarget>(targets);

            foreach (var target in _registeredTargets)
            {
                if (incoming.Contains(target)) continue;
                var mr = target.GetComponent<MeshRenderer>();
                if (mr == null) continue;
                _worldAS.RemoveInstance(mr);
                _emissiveAS.RemoveInstance(mr);
            }

            var registered = new HashSet<NativeRayTracingTarget>(_registeredTargets);
            foreach (var target in targets)
            {
                if (registered.Contains(target)) continue;
                var mr = target.GetComponent<MeshRenderer>();
                if (mr == null)
                {
                    Debug.LogWarning($"[NRDSampleResource] Target '{target.name}' has no MeshRenderer — skipping");
                    continue;
                }

                Matrix4x4 xform = target.transform.localToWorldMatrix;
                if (_worldAS.AddInstance(mr, target.ommCaches))
                    _worldAS.SetInstanceTransform(mr, xform);

                if (IsTargetEmissive(target) && _emissiveAS.AddInstance(mr, target.ommCaches))
                    _emissiveAS.SetInstanceTransform(mr, xform);
            }
        }

        private static bool IsTargetEmissive(NativeRayTracingTarget target)
        {
            var mr = target != null ? target.GetComponent<MeshRenderer>() : null;
            if (mr == null) return false;
            var mats = mr.sharedMaterials;
            if (mats == null) return false;
            for (int i = 0; i < mats.Length; i++)
            {
                var mat = mats[i];
                if (mat == null) continue;
                if (mat.HasProperty("_EmissionColor"))
                {
                    Color e = mat.GetColor("_EmissionColor").linear;
                    if (e.r > 0f || e.g > 0f || e.b > 0f) return true;
                }
                if (mat.HasProperty("_EmissionMap") && mat.GetTexture("_EmissionMap") != null)
                    return true;
            }
            return false;
        }

        private void RebuildSceneGpuData(IReadOnlyList<NativeRayTracingTarget> targets)
        {
            DisposeSceneGpuBuffers();

            var instList = new List<InstanceDataGPU>();
            var geomList = new List<GeometryDataGPU>();
            var matList  = new List<MaterialConstantsGPU>();
            var primList = new List<PrimitiveDataGPU>();
            var bufPtrs  = new List<IntPtr>();
            var texPtrs  = new List<IntPtr>();

            int emissiveSlot = 0;   // index within _emissiveAS
            foreach (var target in targets)
            {
                var mr = target.GetComponent<MeshRenderer>();
                if (mr == null) continue;
                var mf = mr.GetComponent<MeshFilter>();
                if (mf == null || mf.sharedMesh == null) continue;

                Mesh mesh = GetOrCreateSingleStreamMesh(mf.sharedMesh);
                if (mesh == null) continue;
                int meshKey = mesh.GetInstanceID();
                mesh.UploadMeshData(false);

                if (!_meshBufferSlots.TryGetValue(meshKey, out var slots))
                {
                    IntPtr vbPtr = mesh.GetNativeVertexBufferPtr(0);
                    IntPtr ibPtr = mesh.GetNativeIndexBufferPtr();
                    if (vbPtr == IntPtr.Zero || ibPtr == IntPtr.Zero)
                    {
                        Debug.LogWarning($"[NRDSampleResource] '{mesh.name}': failed to get GPU buffer ptrs — skipping");
                        continue;
                    }

                    slots = (bufPtrs.Count, bufPtrs.Count + 1);
                    bufPtrs.Add(vbPtr);
                    bufPtrs.Add(ibPtr);
                    _meshBufferSlots[meshKey] = slots;
                }

                uint vertexStride = (uint)mesh.GetVertexBufferStride(0);
                uint indexStride  = mesh.indexFormat == IndexFormat.UInt16 ? 2u : 4u;
                uint posOff       = mesh.HasVertexAttribute(VertexAttribute.Position)  ? (uint)mesh.GetVertexAttributeOffset(VertexAttribute.Position)  : 0u;
                uint normOff      = mesh.HasVertexAttribute(VertexAttribute.Normal)    ? (uint)mesh.GetVertexAttributeOffset(VertexAttribute.Normal)    : 0xFFFFFFFFu;
                uint uvOff        = mesh.HasVertexAttribute(VertexAttribute.TexCoord0) ? (uint)mesh.GetVertexAttributeOffset(VertexAttribute.TexCoord0) : 0xFFFFFFFFu;
                uint tanOff       = mesh.HasVertexAttribute(VertexAttribute.Tangent)   ? (uint)mesh.GetVertexAttributeOffset(VertexAttribute.Tangent)   : 0xFFFFFFFFu;

                Material[] mats       = mr.sharedMaterials ?? Array.Empty<Material>();
                int        subMeshCnt = mesh.subMeshCount;
                int        firstGeom  = geomList.Count;
                int        instanceIdx = instList.Count;

                for (int s = 0; s < subMeshCnt; s++)
                {
                    SubMeshDescriptor sub    = mesh.GetSubMesh(s);
                    Material          mat    = s < mats.Length ? mats[s] : (mats.Length > 0 ? mats[^1] : null);
                    int               matIdx = GetOrAddMaterialGpu(mat, matList, texPtrs);

                    geomList.Add(new GeometryDataGPU
                    {
                        numIndices        = (uint)sub.indexCount,
                        numVertices       = (uint)mesh.vertexCount,
                        indexBufferIndex  = slots.ib,
                        indexOffset       = (uint)sub.indexStart * indexStride,
                        vertexBufferIndex = slots.vb,
                        positionOffset    = posOff,
                        normalOffset      = normOff,
                        texCoord1Offset   = uvOff,
                        tangentOffset     = tanOff,
                        vertexStride      = vertexStride,
                        indexStride       = indexStride,
                        materialIndex     = (uint)matIdx,
                    });
                }

                // Per-triangle PrimitiveData (walk the managed index buffer for each submesh).
                AppendPrimitiveData(mesh, (uint)instanceIdx, primList);

                Matrix4x4 m = target.transform.localToWorldMatrix;
                _worldAS.SetInstanceID(mr, (uint)instanceIdx);
                instList.Add(new InstanceDataGPU
                {
                    firstGeometryIndex = (uint)firstGeom,
                    numGeometries      = (uint)subMeshCnt,
                    pad0               = 0, pad1 = 0,
                    transformRow0      = new Vector4(m.m00, m.m01, m.m02, m.m03),
                    transformRow1      = new Vector4(m.m10, m.m11, m.m12, m.m13),
                    transformRow2      = new Vector4(m.m20, m.m21, m.m22, m.m23),
                });

                bool isEmissive = IsTargetEmissive(target);
                if (isEmissive)
                {
                    _emissiveAS.SetInstanceID(mr, (uint)emissiveSlot);
                    emissiveSlot++;
                }
                _sceneInstances.Add(new SceneInstance { renderer = mr, transform = mr.transform, isEmissive = isEmissive });
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
                matList.Add(default);
            }
            if (primList.Count == 0) primList.Add(default);

            _instanceCpu  = instList.ToArray();
            _geometryCpu  = geomList.ToArray();
            _materialCpu  = matList.ToArray();
            _primitiveCpu = primList.ToArray();

            _instanceBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                _instanceCpu.Length, Marshal.SizeOf<InstanceDataGPU>());
            _instanceBuf.SetData(_instanceCpu);

            _geometryBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                _geometryCpu.Length, Marshal.SizeOf<GeometryDataGPU>());
            _geometryBuf.SetData(_geometryCpu);

            _materialBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                _materialCpu.Length, Marshal.SizeOf<MaterialConstantsGPU>());
            _materialBuf.SetData(_materialCpu);

            _primitiveBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                _primitiveCpu.Length, Marshal.SizeOf<PrimitiveDataGPU>());
            _primitiveBuf.SetData(_primitiveCpu);

            _sceneGpuDirty = false;
        }

        private static void AppendPrimitiveData(Mesh mesh, uint instanceId, List<PrimitiveDataGPU> dst)
        {
            Vector3[] verts = mesh.vertices;
            Vector2[] uvs   = mesh.uv;
            if (verts == null || verts.Length == 0) return;

            int subCnt = mesh.subMeshCount;
            for (int s = 0; s < subCnt; s++)
            {
                int[] tris = mesh.GetTriangles(s);
                for (int i = 0; i + 2 < tris.Length; i += 3)
                {
                    int i0 = tris[i + 0], i1 = tris[i + 1], i2 = tris[i + 2];
                    var p = new PrimitiveDataGPU
                    {
                        uv0 = (uvs != null && i0 < uvs.Length) ? uvs[i0] : Vector2.zero,
                        uv1 = (uvs != null && i1 < uvs.Length) ? uvs[i1] : Vector2.zero,
                        uv2 = (uvs != null && i2 < uvs.Length) ? uvs[i2] : Vector2.zero,
                        pos0 = new Vector4(verts[i0].x, verts[i0].y, verts[i0].z, 0f),
                        pos1 = new Vector4(verts[i1].x, verts[i1].y, verts[i1].z, 0f),
                        pos2 = new Vector4(verts[i2].x, verts[i2].y, verts[i2].z, 0f),
                        instanceId = instanceId,
                    };
                    dst.Add(p);
                }
            }
        }

        private int GetOrAddMaterialGpu(Material mat,
            List<MaterialConstantsGPU> matList,
            List<IntPtr> texPtrs)
        {
            int matId = mat != null ? mat.GetInstanceID() : -1;
            if (_materialSlots.TryGetValue(matId, out int existing))
                return existing;

            int idx = matList.Count;
            _materialSlots[matId] = idx;

            Texture baseTex       = TryGetTex(mat, "_BaseMap");
            Texture normalTex     = TryGetTex(mat, "_BumpMap");
            Texture metalRoughTex = TryGetTex(mat, "_MetallicGlossMap");
            Texture emissiveTex   = TryGetTex(mat, "_EmissionMap");
            Texture occlusionTex  = TryGetTex(mat, "_OcclusionMap");

            int baseTexIdx       = AddTexture(baseTex, texPtrs);
            int normalTexIdx     = AddTexture(normalTex, texPtrs);
            int metalRoughTexIdx = AddTexture(metalRoughTex, texPtrs);
            int emissiveTexIdx   = AddTexture(emissiveTex, texPtrs);
            int occlusionTexIdx  = AddTexture(occlusionTex, texPtrs);

            Color baseColor   = TryGetColor(mat, "_BaseColor",     Color.white);
            Color emissive    = TryGetColor(mat, "_EmissionColor", Color.black);
            float roughness   = 1f - TryGetFloat(mat, "_Smoothness", 0.5f);
            float metalness   = TryGetFloat(mat, "_Metallic",           0f);
            float alphaCutoff = TryGetFloat(mat, "_Cutoff",             0f);
            float normalScale = TryGetFloat(mat, "_BumpScale",          1f);
            float occStr      = TryGetFloat(mat, "_OcclusionStrength",  1f);

            int domain = 0;
            if (mat != null && mat.HasProperty("_Surface"))
                domain = (int)mat.GetFloat("_Surface");
            else if (alphaCutoff > 0f)
                domain = 1;

            int flags = 0;
            if (baseTexIdx       >= 0) flags |= MaterialFlags.UseBaseOrDiffuseTexture;
            if (normalTexIdx     >= 0) flags |= MaterialFlags.UseNormalTexture;
            if (metalRoughTexIdx >= 0) flags |= MaterialFlags.UseMetalRoughOrSpecularTexture;
            if (emissiveTexIdx   >= 0) flags |= MaterialFlags.UseEmissiveTexture;

            matList.Add(new MaterialConstantsGPU
            {
                baseOrDiffuseColor               = new Vector3(baseColor.r, baseColor.g, baseColor.b),
                flags                            = flags,
                emissiveColor                    = new Vector3(emissive.r, emissive.g, emissive.b),
                domain                           = domain,
                opacity                          = baseColor.a,
                roughness                        = roughness,
                metalness                        = metalness,
                normalTextureScale               = normalScale,
                occlusionStrength                = occStr,
                alphaCutoff                      = alphaCutoff,
                transmissionFactor               = 0f,
                baseOrDiffuseTextureIndex        = baseTexIdx,
                metalRoughOrSpecularTextureIndex = metalRoughTexIdx,
                emissiveTextureIndex             = emissiveTexIdx,
                normalTextureIndex               = normalTexIdx,
                occlusionTextureIndex            = occlusionTexIdx,
            });

            return idx;
        }

        private int AddTexture(Texture tex, List<IntPtr> texPtrs)
        {
            if (tex == null) return -1;
            int texId = tex.GetInstanceID();
            if (_textureSlots.TryGetValue(texId, out int slot))
                return slot;
            slot = texPtrs.Count;
            texPtrs.Add(tex.GetNativeTexturePtr());
            _textureSlots[texId] = slot;
            return slot;
        }

        private void UpdateInstanceTransforms()
        {
            if (_instanceCpu == null || _instanceBuf == null) return;

            int instanceCount = Mathf.Min(_sceneInstances.Count, _instanceCpu.Length);

            int dirtyStart = -1;
            for (int i = 0; i < instanceCount; i++)
            {
                Transform t = _sceneInstances[i].transform;
                if (t == null || !t.hasChanged)
                {
                    if (dirtyStart >= 0)
                    {
                        _instanceBuf.SetData(_instanceCpu, dirtyStart, dirtyStart, i - dirtyStart);
                        dirtyStart = -1;
                    }
                    continue;
                }

                Matrix4x4 m = t.localToWorldMatrix;
                _instanceCpu[i].transformRow0 = new Vector4(m.m00, m.m01, m.m02, m.m03);
                _instanceCpu[i].transformRow1 = new Vector4(m.m10, m.m11, m.m12, m.m13);
                _instanceCpu[i].transformRow2 = new Vector4(m.m20, m.m21, m.m22, m.m23);

                var inst = _sceneInstances[i];
                _worldAS.SetInstanceTransform(inst.renderer, m);
                if (inst.isEmissive)
                    _emissiveAS.SetInstanceTransform(inst.renderer, m);

                if (dirtyStart < 0) dirtyStart = i;
            }

            if (dirtyStart >= 0)
                _instanceBuf.SetData(_instanceCpu, dirtyStart, dirtyStart, instanceCount - dirtyStart);
        }

        // Clones multi-stream source meshes into a single-stream layout (required for
        // bindless sampling that assumes one VB per mesh with a single stride).
        private Mesh GetOrCreateSingleStreamMesh(Mesh src)
        {
            if (src == null) return null;

            int id = src.GetInstanceID();
            if (_normalizedMeshCache.TryGetValue(id, out var cached) && cached != null)
                return cached;

            var attrs = src.GetVertexAttributes();
            bool multiStream = false;
            for (int i = 0; i < attrs.Length; i++)
            {
                if (attrs[i].stream != 0) { multiStream = true; break; }
            }

            if (!multiStream)
            {
                _normalizedMeshCache[id] = src;
                return src;
            }

            var positions = src.vertices;
            var normals   = src.normals;
            var tangents  = src.tangents;
            var colors    = src.colors;
            var uv0 = new List<Vector4>(); src.GetUVs(0, uv0);
            var uv1 = new List<Vector4>(); src.GetUVs(1, uv1);
            var uv2 = new List<Vector4>(); src.GetUVs(2, uv2);
            var uv3 = new List<Vector4>(); src.GetUVs(3, uv3);

            var clone = UnityEngine.Object.Instantiate(src);
            clone.name = src.name + " (NRDSampleResource SingleStream)";

            var newDescs = new VertexAttributeDescriptor[attrs.Length];
            for (int i = 0; i < attrs.Length; i++)
            {
                var a = attrs[i];
                newDescs[i] = new VertexAttributeDescriptor(a.attribute, a.format, a.dimension, 0);
            }
            clone.SetVertexBufferParams(clone.vertexCount, newDescs);

            if (positions != null && positions.Length > 0) clone.vertices = positions;
            if (normals   != null && normals.Length   > 0) clone.normals  = normals;
            if (tangents  != null && tangents.Length  > 0) clone.tangents = tangents;
            if (colors    != null && colors.Length    > 0) clone.colors   = colors;
            if (uv0.Count > 0) clone.SetUVs(0, uv0);
            if (uv1.Count > 0) clone.SetUVs(1, uv1);
            if (uv2.Count > 0) clone.SetUVs(2, uv2);
            if (uv3.Count > 0) clone.SetUVs(3, uv3);

            clone.UploadMeshData(false);

            _normalizedMeshCache[id] = clone;
            _ownedMeshes.Add(clone);
            return clone;
        }

        private static Texture TryGetTex(Material mat, string prop) =>
            mat != null && mat.HasProperty(prop) ? mat.GetTexture(prop) : null;

        private static Color TryGetColor(Material mat, string prop, Color def) =>
            mat != null && mat.HasProperty(prop) ? mat.GetColor(prop).linear : def;

        private static float TryGetFloat(Material mat, string prop, float def) =>
            mat != null && mat.HasProperty(prop) ? mat.GetFloat(prop) : def;
    }
}
