using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

namespace NativeRender
{
    /// <summary>
    /// Owns all GPU scene resources shared across multiple render passes:
    /// acceleration structure, structured buffers (instance/geometry/material),
    /// and bindless VB/IB/texture arrays.
    /// Lifetime is managed by NativeRayTracingFeature.
    /// </summary>
    public sealed class GPUScene : IDisposable
    {
        // Acceleration structure — owns BLAS/TLAS for the registered scene.
        private RayTracingAccelerationStructure _accelStructure;

        // Tracks which targets are currently registered with the native plugin.
        private readonly List<NativeRayTracingTarget> _registeredTargets = new();

        // space0 structured buffers: t_InstanceData, t_GeometryData, t_MaterialConstants
        private GraphicsBuffer _instanceGpuBuf;
        private GraphicsBuffer _geometryGpuBuf;
        private GraphicsBuffer _materialGpuBuf;

        // space1 bindless VB/IB buffers, space2 bindless material textures
        private BindlessBuffer  _sceneBuffers;
        private BindlessTexture _sceneTextures;

        // CPU-side mirrors (updated per-frame for transforms, then SetData'd)
        private InstanceDataGPU[]      _instanceCpu;
        private GeometryDataGPU[]      _geometryCpu;
        private MaterialConstantsGPU[] _materialCpu;

        // Instance tracking (one entry per registered MeshRenderer, in TLAS order)
        private struct SceneInstance
        {
            public MeshRenderer renderer;
            public Transform    transform;
        }

        private readonly List<SceneInstance>               _sceneInstances  = new();
        private readonly Dictionary<int, (int vb, int ib)> _meshBufferSlots = new();
        private readonly Dictionary<int, int>              _materialSlots   = new();
        private readonly Dictionary<int, int>              _textureSlots    = new();

        // Dirty flags
        private bool _sceneGpuDirty = true;
        private bool _forceRebuild  = false;

        private readonly HashSet<Material> _dirtyMaterials = new();

        private bool _disposed;

        public RayTracingAccelerationStructure AccelerationStructure => _accelStructure;

        public GPUScene()
        {
            _accelStructure = new RayTracingAccelerationStructure();
        }

        /// <summary>Queues a material for in-place property update next frame.</summary>
        public void MarkMaterialDirty(Material mat)
        {
            if (mat != null) _dirtyMaterials.Add(mat);
        }

        /// <summary>Forces a full scene rebuild next frame (e.g. material assignment changed).</summary>
        public void MarkRebuildDirty()
        {
            _forceRebuild = true;
        }

        /// <summary>
        /// Called once per frame from the render pass. Handles dirty detection,
        /// incremental scene registration, GPU data rebuild, and transform updates.
        /// </summary>
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

        /// <summary>Builds / updates the acceleration structure. Call inside a CommandBuffer recording.</summary>
        public void BuildAccelerationStructure(CommandBuffer cmd)
        {
            _accelStructure.BuildOrUpdate(cmd);
        }

        /// <summary>Binds all scene GPU buffers to the given shader.</summary>
        public void BindToShader(RayTraceShader shader)
        {
            if (shader == null || !shader.IsValid) return;
            shader.SetStructuredBuffer("t_InstanceData", _instanceGpuBuf);
            shader.SetStructuredBuffer("t_GeometryData", _geometryGpuBuf);
            shader.SetStructuredBuffer("t_MaterialConstants", _materialGpuBuf);
            if (_sceneBuffers != null) shader.SetBindlessBuffer("t_BindlessBuffers", _sceneBuffers);
            if (_sceneTextures != null) shader.SetBindlessTexture("t_BindlessTextures", _sceneTextures);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            DisposeSceneGpuBuffers();

            _accelStructure?.Dispose();
            _accelStructure = null;
        }

        // -----------------------------------------------------------------------
        // Private helpers
        // -----------------------------------------------------------------------

        private void DisposeSceneGpuBuffers()
        {
            _instanceGpuBuf?.Release();
            _instanceGpuBuf = null;
            _geometryGpuBuf?.Release();
            _geometryGpuBuf = null;
            _materialGpuBuf?.Release();
            _materialGpuBuf = null;
            _sceneBuffers?.Dispose();
            _sceneBuffers = null;
            _sceneTextures?.Dispose();
            _sceneTextures = null;
            _instanceCpu   = null;
            _geometryCpu   = null;
            _materialCpu   = null;
            _sceneInstances.Clear();
            _meshBufferSlots.Clear();
            _materialSlots.Clear();
            _textureSlots.Clear();
        }

        private void RebuildSceneGpuData(IReadOnlyList<NativeRayTracingTarget> targets)
        {
            DisposeSceneGpuBuffers();

            var instList = new List<InstanceDataGPU>();
            var geomList = new List<GeometryDataGPU>();
            var matList  = new List<MaterialConstantsGPU>();
            var bufPtrs  = new List<IntPtr>();
            var texPtrs  = new List<IntPtr>();

            foreach (var target in targets)
            {
                var mr = target.GetComponent<MeshRenderer>();
                if (mr == null) continue;
                var mf = mr.GetComponent<MeshFilter>();
                if (mf == null || mf.sharedMesh == null) continue;

                Mesh mesh    = mf.sharedMesh;
                int  meshKey = mesh.GetInstanceID();
                mesh.UploadMeshData(false);

                if (!_meshBufferSlots.TryGetValue(meshKey, out var slots))
                {
                    IntPtr vbPtr = mesh.GetNativeVertexBufferPtr(0);
                    IntPtr ibPtr = mesh.GetNativeIndexBufferPtr();
                    if (vbPtr == IntPtr.Zero || ibPtr == IntPtr.Zero)
                    {
                        Debug.LogWarning($"[GPUScene] '{mesh.name}': failed to get GPU buffer ptrs — skipping");
                        continue;
                    }

                    slots = (bufPtrs.Count, bufPtrs.Count + 1);
                    bufPtrs.Add(vbPtr);
                    bufPtrs.Add(ibPtr);
                    _meshBufferSlots[meshKey] = slots;
                }

                uint vertexStride = (uint)mesh.GetVertexBufferStride(0);
                uint indexStride  = mesh.indexFormat == IndexFormat.UInt16 ? 2u : 4u;
                uint posOff       = mesh.HasVertexAttribute(VertexAttribute.Position) ? (uint)mesh.GetVertexAttributeOffset(VertexAttribute.Position) : 0u;
                uint normOff      = mesh.HasVertexAttribute(VertexAttribute.Normal) ? (uint)mesh.GetVertexAttributeOffset(VertexAttribute.Normal) : 0xFFFFFFFFu;
                uint uvOff        = mesh.HasVertexAttribute(VertexAttribute.TexCoord0) ? (uint)mesh.GetVertexAttributeOffset(VertexAttribute.TexCoord0) : 0xFFFFFFFFu;
                uint tanOff       = mesh.HasVertexAttribute(VertexAttribute.Tangent) ? (uint)mesh.GetVertexAttributeOffset(VertexAttribute.Tangent) : 0xFFFFFFFFu;

                Material[] mats       = mr.sharedMaterials ?? Array.Empty<Material>();
                int        subMeshCnt = mesh.subMeshCount;
                int        firstGeom  = geomList.Count;

                for (int s = 0; s < subMeshCnt; s++)
                {
                    SubMeshDescriptor sub    = mesh.GetSubMesh(s);
                    Material          mat    = s < mats.Length ? mats[s] : (mats.Length > 0 ? mats[mats.Length - 1] : null);
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

                Matrix4x4 m = target.transform.localToWorldMatrix;
                // Set the TLAS InstanceID to match this instance's index in t_InstanceData,
                // so that InstanceID() in HLSL shaders can directly index t_InstanceData.
                _accelStructure.SetInstanceID(mr, (uint)instList.Count);
                instList.Add(new InstanceDataGPU
                {
                    firstGeometryIndex = (uint)firstGeom,
                    numGeometries      = (uint)subMeshCnt,
                    pad0               = 0, pad1 = 0,
                    transformRow0      = new Vector4(m.m00, m.m01, m.m02, m.m03),
                    transformRow1      = new Vector4(m.m10, m.m11, m.m12, m.m13),
                    transformRow2      = new Vector4(m.m20, m.m21, m.m22, m.m23),
                });
                _sceneInstances.Add(new SceneInstance { renderer = mr, transform = mr.transform });
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
                _sceneGpuDirty = false;
                return;
            }

            _instanceCpu = instList.ToArray();
            _geometryCpu = geomList.ToArray();
            _materialCpu = matList.ToArray();

            _instanceGpuBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                _instanceCpu.Length, Marshal.SizeOf<InstanceDataGPU>());
            _instanceGpuBuf.SetData(_instanceCpu);

            _geometryGpuBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                _geometryCpu.Length, Marshal.SizeOf<GeometryDataGPU>());
            _geometryGpuBuf.SetData(_geometryCpu);

            _materialGpuBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                _materialCpu.Length, Marshal.SizeOf<MaterialConstantsGPU>());
            _materialGpuBuf.SetData(_materialCpu);

            _sceneGpuDirty = false;
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

            Color baseColor   = TryGetColor(mat, "_BaseColor", Color.white);
            Color emissive    = TryGetColor(mat, "_EmissionColor", Color.black);
            float roughness   = 1f - TryGetFloat(mat, "_Smoothness", 0.5f);
            float metalness   = TryGetFloat(mat, "_Metallic", 0f);
            float alphaCutoff = TryGetFloat(mat, "_Cutoff", 0f);
            float normalScale = TryGetFloat(mat, "_BumpScale", 1f);
            float occStr      = TryGetFloat(mat, "_OcclusionStrength", 1f);

            int domain = 0;
            if (mat != null && mat.HasProperty("_Surface"))
                domain = (int)mat.GetFloat("_Surface");
            else if (alphaCutoff > 0f)
                domain = 1;

            int flags                        = 0;
            if (baseTexIdx >= 0) flags       |= MaterialFlags.UseBaseOrDiffuseTexture;
            if (normalTexIdx >= 0) flags     |= MaterialFlags.UseNormalTexture;
            if (metalRoughTexIdx >= 0) flags |= MaterialFlags.UseMetalRoughOrSpecularTexture;
            if (emissiveTexIdx >= 0) flags   |= MaterialFlags.UseEmissiveTexture;

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
            if (_instanceCpu == null || _instanceGpuBuf == null) return;

            // Guard against _instanceCpu / _sceneInstances length mismatch.
            int instanceCount = Mathf.Min(_sceneInstances.Count, _instanceCpu.Length);

            // _instanceCpu / GPU buffer: only write & upload when the transform actually changed.
            // Flush dirty regions in contiguous segments to minimise GPU upload bandwidth.
            int dirtyStart = -1;
            for (int i = 0; i < instanceCount; i++)
            {
                Transform t = _sceneInstances[i].transform;
                if (t == null || !t.hasChanged)
                {
                    if (dirtyStart >= 0)
                    {
                        _instanceGpuBuf.SetData(_instanceCpu, dirtyStart, dirtyStart, i - dirtyStart);
                        dirtyStart = -1;
                    }
                
                    continue;
                }

                Matrix4x4 m = t.localToWorldMatrix;
                _instanceCpu[i].transformRow0 = new Vector4(m.m00, m.m01, m.m02, m.m03);
                _instanceCpu[i].transformRow1 = new Vector4(m.m10, m.m11, m.m12, m.m13);
                _instanceCpu[i].transformRow2 = new Vector4(m.m20, m.m21, m.m22, m.m23);

                _accelStructure.SetInstanceTransform(_sceneInstances[i].renderer, m);
                if (dirtyStart < 0) dirtyStart = i;
            }

            if (dirtyStart >= 0)
                _instanceGpuBuf.SetData(_instanceCpu, dirtyStart, dirtyStart, instanceCount - dirtyStart);
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
            var incoming = new HashSet<NativeRayTracingTarget>(targets);

            foreach (var target in _registeredTargets)
            {
                if (incoming.Contains(target)) continue;
                var mr = target.GetComponent<MeshRenderer>();
                if (mr != null)
                    _accelStructure.RemoveInstance(mr);
            }

            var registered = new HashSet<NativeRayTracingTarget>(_registeredTargets);
            foreach (var target in targets)
            {
                if (registered.Contains(target)) continue;
                var meshRenderer = target.GetComponent<MeshRenderer>();
                if (meshRenderer == null)
                {
                    Debug.LogWarning($"[GPUScene] Target '{target.name}' has no MeshRenderer — skipping");
                    continue;
                }

                bool ok = _accelStructure.AddInstance(meshRenderer, target.ommCaches);
                if (ok)
                    _accelStructure.SetInstanceTransform(meshRenderer, target.transform.localToWorldMatrix);
            }
        }

        private static Texture TryGetTex(Material mat, string prop) =>
            mat != null && mat.HasProperty(prop) ? mat.GetTexture(prop) : null;

        private static Color TryGetColor(Material mat, string prop, Color def) =>
            mat != null && mat.HasProperty(prop) ? mat.GetColor(prop).linear : def;

        private static float TryGetFloat(Material mat, string prop, float def) =>
            mat != null && mat.HasProperty(prop) ? mat.GetFloat(prop) : def;
    }
}