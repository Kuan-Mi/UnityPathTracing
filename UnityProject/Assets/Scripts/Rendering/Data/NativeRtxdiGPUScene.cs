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
    // Donut-compatible GPU structs (exact mirror of bindless.h / material_cb.h)
    // =========================================================================

    /// <summary>
    /// Mirrors donut <c>InstanceData</c> from <c>bindless.h</c>.
    /// Size: 7 × 16 = 112 bytes  (<c>c_SizeOfInstanceData</c>).
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct DonutInstanceData
    {
        public uint flags;                       // +0
        public uint firstGeometryInstanceIndex;  // +4   = firstGeometryIndex in our flat list
        public uint firstGeometryIndex;          // +8
        public uint numGeometries;               // +12
        // float3x4 transform — row 0..2 of object-to-world
        public Vector4 transformRow0;            // +16
        public Vector4 transformRow1;            // +32
        public Vector4 transformRow2;            // +48
        // float3x4 prevTransform — same layout, previous frame
        public Vector4 prevTransformRow0;        // +64
        public Vector4 prevTransformRow1;        // +80
        public Vector4 prevTransformRow2;        // +96
    }                                            // Total: 112 bytes

    /// <summary>
    /// Mirrors donut <c>GeometryData</c> from <c>bindless.h</c>.
    /// Size: 4 × 16 = 64 bytes  (<c>c_SizeOfGeometryData</c>).
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct DonutGeometryData
    {
        public uint numIndices;         // +0
        public uint numVertices;        // +4
        public int  indexBufferIndex;   // +8
        public uint indexOffset;        // +12  byte offset

        public int  vertexBufferIndex;  // +16
        public uint positionOffset;     // +20  byte offset (float3)
        public uint prevPositionOffset; // +24  byte offset; = positionOffset for static meshes
        public uint texCoord1Offset;    // +28  byte offset (float2), or ~0u

        public uint texCoord2Offset;    // +32  byte offset, ~0u (not used)
        public uint normalOffset;       // +36  byte offset (oct-encoded or float3), or ~0u
        public uint tangentOffset;      // +40  byte offset, or ~0u
        public uint curveRadiusOffset;  // +44  ~0u (curves not supported)

        public uint materialIndex;      // +48
        public uint pad0;               // +52
        public uint pad1;               // +56
        public uint pad2;               // +60
    }                                   // Total: 64 bytes

    /// <summary>
    /// Mirrors donut <c>MaterialConstants</c> from <c>material_cb.h</c>.
    /// Size: 13 × 16 = 208 bytes  (<c>c_SizeOfMaterialConstants</c>).
    /// SSS / hair fields are zeroed — unused in the GBuffer pass.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct DonutMaterialConstants
    {
        // row 0
        public Vector3 baseOrDiffuseColor;               // +0
        public int     flags;                             // +12

        // row 1
        public Vector3 specularColor;                     // +16  (derived F0)
        public int     materialID;                        // +28  = material slot index

        // row 2
        public Vector3 emissiveColor;                     // +32
        public int     domain;                            // +44

        // row 3
        public float opacity;                             // +48
        public float roughness;                           // +52
        public float metalness;                           // +56
        public float normalTextureScale;                  // +60

        // row 4
        public float occlusionStrength;                   // +64
        public float alphaCutoff;                         // +68
        public float transmissionFactor;                  // +72
        public int   baseOrDiffuseTextureIndex;           // +76

        // row 5
        public int metalRoughOrSpecularTextureIndex;      // +80
        public int emissiveTextureIndex;                  // +84
        public int normalTextureIndex;                    // +88
        public int occlusionTextureIndex;                 // +92

        // row 6
        public int   transmissionTextureIndex;            // +96  = -1
        public int   opacityTextureIndex;                 // +100 = -1
        public float normalTextureTransformScaleX;        // +104 = 1
        public float normalTextureTransformScaleY;        // +108 = 1

        // rows 7-12  (SSS / hair — zeroed)
        public uint   _padding7_x;                        // +112
        public uint   _padding7_y;                        // +116
        public uint   _padding7_z;                        // +120
        public float  sssScale;                           // +124 = 0

        public float  sssTransmissionColorR;              // +128
        public float  sssTransmissionColorG;              // +132
        public float  sssTransmissionColorB;              // +136
        public float  sssAnisotropy;                      // +140

        public float  sssScatteringColorR;                // +144
        public float  sssScatteringColorG;                // +148
        public float  sssScatteringColorB;                // +152
        public float  hairMelanin;                        // +156

        public float  hairBaseColorR;                     // +160
        public float  hairBaseColorG;                     // +164
        public float  hairBaseColorB;                     // +168
        public float  hairMelaninRedness;                 // +172

        public float  hairLongitudinalRoughness;          // +176
        public float  hairAzimuthalRoughness;             // +180
        public float  hairIor;                            // +184
        public float  hairCuticleAngle;                   // +188

        public float  hairDiffuseReflectionTintR;         // +192
        public float  hairDiffuseReflectionTintG;         // +196
        public float  hairDiffuseReflectionTintB;         // +200
        public float  hairDiffuseReflectionWeight;        // +204
    }                                                     // Total: 208 bytes

    // =========================================================================
    // MaterialFlags matching donut material_cb.h
    // =========================================================================
    public static class DonutMaterialFlags
    {
        public const int UseSpecularGlossModel          = 0x00000001;
        public const int DoubleSided                    = 0x00000002;
        public const int UseMetalRoughOrSpecularTexture = 0x00000004;
        public const int UseBaseOrDiffuseTexture        = 0x00000008;
        public const int UseEmissiveTexture             = 0x00000010;
        public const int UseNormalTexture               = 0x00000020;
        public const int UseOcclusionTexture            = 0x00000040;
        public const int UseTransmissionTexture         = 0x00000080;
    }

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
    public sealed class NativeRtxdiGPUScene : IDisposable
    {
        // Acceleration structure
        private RayTracingAccelerationStructure _accelStructure;

        // Structured buffers
        private GraphicsBuffer _instanceGpuBuf;
        private GraphicsBuffer _geometryGpuBuf;
        private GraphicsBuffer _materialGpuBuf;

        // Bindless
        private BindlessBuffer  _sceneBuffers;
        private BindlessTexture _sceneTextures;

        // CPU-side mirrors
        private DonutInstanceData[]      _instanceCpu;
        private DonutGeometryData[]      _geometryCpu;
        private DonutMaterialConstants[] _materialCpu;

        // Per-instance tracking for transforms
        private struct SceneInstance
        {
            public MeshRenderer renderer;
            public Transform    transform;
            // Previous frame's world matrix rows (for prevTransform)
            public Vector4 prevRow0;
            public Vector4 prevRow1;
            public Vector4 prevRow2;
        }

        private readonly List<SceneInstance>                _sceneInstances  = new();
        private readonly Dictionary<int, (int vb, int ib)> _meshBufferSlots = new();
        private readonly Dictionary<int, int>              _materialSlots   = new();
        private readonly Dictionary<int, int>              _textureSlots    = new();
        private readonly Dictionary<int, Mesh>             _normalizedMeshCache = new();
        private readonly List<Mesh>                        _ownedMeshes     = new();
        private readonly List<NativeRayTracingTarget>      _registeredTargets = new();

        private bool _sceneGpuDirty = true;
        private bool _forceRebuild  = false;
        private bool _disposed;

        public RayTracingAccelerationStructure AccelerationStructure => _accelStructure;

        public NativeRtxdiGPUScene()
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
        /// Binds scene buffers to a native compute descriptor set.
        /// Binds: t_InstanceData, t_GeometryData, t_MaterialConstants,
        ///        t_BindlessBuffers (space1), t_BindlessTextures (space2).
        /// </summary>
        public void BindToShader(NativeComputeDescriptorSet ds)
        {
            if (ds == null) return;
            if (_instanceGpuBuf != null)
                ds.SetStructuredBuffer("t_InstanceData",      _instanceGpuBuf.GetNativeBufferPtr(), _instanceGpuBuf.count, _instanceGpuBuf.stride);
            if (_geometryGpuBuf != null)
                ds.SetStructuredBuffer("t_GeometryData",      _geometryGpuBuf.GetNativeBufferPtr(), _geometryGpuBuf.count, _geometryGpuBuf.stride);
            if (_materialGpuBuf != null)
                ds.SetStructuredBuffer("t_MaterialConstants", _materialGpuBuf.GetNativeBufferPtr(), _materialGpuBuf.count, _materialGpuBuf.stride);
            if (_sceneBuffers  != null) ds.SetBindlessBuffer("t_BindlessBuffers",   _sceneBuffers);
            if (_sceneTextures != null) ds.SetBindlessTexture("t_BindlessTextures", _sceneTextures);
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
            _instanceGpuBuf?.Release(); _instanceGpuBuf = null;
            _geometryGpuBuf?.Release(); _geometryGpuBuf = null;
            _materialGpuBuf?.Release(); _materialGpuBuf = null;
            _sceneBuffers?.Dispose();   _sceneBuffers   = null;
            _sceneTextures?.Dispose();  _sceneTextures  = null;
            _instanceCpu = null;
            _geometryCpu = null;
            _materialCpu = null;
            _sceneInstances.Clear();
            _meshBufferSlots.Clear();
            _materialSlots.Clear();
            _textureSlots.Clear();

            foreach (var m in _ownedMeshes)
            {
                if (m == null) continue;
                if (Application.isPlaying) UnityEngine.Object.Destroy(m);
                else UnityEngine.Object.DestroyImmediate(m);
            }
            _ownedMeshes.Clear();
            _normalizedMeshCache.Clear();
        }

        private void RebuildSceneGpuData(IReadOnlyList<NativeRayTracingTarget> targets)
        {
            DisposeGpuBuffers();

            var instList = new List<DonutInstanceData>();
            var geomList = new List<DonutGeometryData>();
            var matList  = new List<DonutMaterialConstants>();
            var bufPtrs  = new List<IntPtr>();
            var texPtrs  = new List<IntPtr>();

            foreach (var target in targets)
            {
                if (target == null) continue;
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
                        Debug.LogWarning($"[NativeRtxdiGPUScene] '{mesh.name}': failed to get GPU buffer ptrs — skipping");
                        continue;
                    }
                    slots = (bufPtrs.Count, bufPtrs.Count + 1);
                    bufPtrs.Add(vbPtr);
                    bufPtrs.Add(ibPtr);
                    _meshBufferSlots[meshKey] = slots;
                }

                uint indexStride = mesh.indexFormat == IndexFormat.UInt16 ? 2u : 4u;
                uint posOff  = mesh.HasVertexAttribute(VertexAttribute.Position) ? (uint)mesh.GetVertexAttributeOffset(VertexAttribute.Position)  : 0u;
                uint normOff = mesh.HasVertexAttribute(VertexAttribute.Normal)   ? (uint)mesh.GetVertexAttributeOffset(VertexAttribute.Normal)     : 0xFFFFFFFFu;
                uint uvOff   = mesh.HasVertexAttribute(VertexAttribute.TexCoord0) ? (uint)mesh.GetVertexAttributeOffset(VertexAttribute.TexCoord0) : 0xFFFFFFFFu;
                uint tanOff  = mesh.HasVertexAttribute(VertexAttribute.Tangent)  ? (uint)mesh.GetVertexAttributeOffset(VertexAttribute.Tangent)    : 0xFFFFFFFFu;

                Material[] mats       = mr.sharedMaterials ?? Array.Empty<Material>();
                int        subMeshCnt = mesh.subMeshCount;
                int        firstGeom  = geomList.Count;
                int        instIdx    = instList.Count;

                for (int s = 0; s < subMeshCnt; s++)
                {
                    SubMeshDescriptor sub    = mesh.GetSubMesh(s);
                    Material          mat    = s < mats.Length ? mats[s] : (mats.Length > 0 ? mats[mats.Length - 1] : null);
                    int               matIdx = GetOrAddMaterial(mat, instIdx, matList, texPtrs);

                    geomList.Add(new DonutGeometryData
                    {
                        numIndices        = (uint)sub.indexCount,
                        numVertices       = (uint)mesh.vertexCount,
                        indexBufferIndex  = slots.ib,
                        indexOffset       = (uint)sub.indexStart * indexStride,
                        vertexBufferIndex = slots.vb,
                        positionOffset    = posOff,
                        prevPositionOffset = posOff,  // no skinning / morph support yet
                        texCoord1Offset   = uvOff,
                        texCoord2Offset   = 0xFFFFFFFFu,
                        normalOffset      = normOff,
                        tangentOffset     = tanOff,
                        curveRadiusOffset = 0xFFFFFFFFu,
                        materialIndex     = (uint)matIdx,
                    });
                }

                Matrix4x4 m = target.transform.localToWorldMatrix;
                var row0 = new Vector4(m.m00, m.m01, m.m02, m.m03);
                var row1 = new Vector4(m.m10, m.m11, m.m12, m.m13);
                var row2 = new Vector4(m.m20, m.m21, m.m22, m.m23);

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
                    renderer = mr,
                    transform = mr.transform,
                    prevRow0 = row0,
                    prevRow1 = row1,
                    prevRow2 = row2,
                });
            }

            _sceneBuffers = new BindlessBuffer(Mathf.Max(bufPtrs.Count, 1));
            for (int i = 0; i < bufPtrs.Count; i++)
                _sceneBuffers.SetNativePtr(i, bufPtrs[i]);

            int texCount = Mathf.Max(texPtrs.Count, 1);
            _sceneTextures = new BindlessTexture(texCount);
            for (int i = 0; i < texPtrs.Count; i++)
                _sceneTextures.SetNativePtr(i, texPtrs[i]);

            if (instList.Count == 0) { instList.Add(default); geomList.Add(default); matList.Add(default); }

            _instanceCpu = instList.ToArray();
            _geometryCpu = geomList.ToArray();
            _materialCpu = matList.ToArray();

            _instanceGpuBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _instanceCpu.Length, Marshal.SizeOf<DonutInstanceData>());
            _instanceGpuBuf.SetData(_instanceCpu);

            _geometryGpuBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _geometryCpu.Length, Marshal.SizeOf<DonutGeometryData>());
            _geometryGpuBuf.SetData(_geometryCpu);

            _materialGpuBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _materialCpu.Length, Marshal.SizeOf<DonutMaterialConstants>());
            _materialGpuBuf.SetData(_materialCpu);

            _sceneGpuDirty = false;
        }

        private void UpdateInstanceTransforms()
        {
            if (_instanceCpu == null || _instanceGpuBuf == null) return;

            int instanceCount = Mathf.Min(_sceneInstances.Count, _instanceCpu.Length);
            int dirtyStart = -1;

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

                Matrix4x4 m = si.transform.localToWorldMatrix;
                var row0 = new Vector4(m.m00, m.m01, m.m02, m.m03);
                var row1 = new Vector4(m.m10, m.m11, m.m12, m.m13);
                var row2 = new Vector4(m.m20, m.m21, m.m22, m.m23);

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
                updated.prevRow0 = row0; updated.prevRow1 = row1; updated.prevRow2 = row2;
                _sceneInstances[i] = updated;
            }

            if (dirtyStart >= 0)
                _instanceGpuBuf.SetData(_instanceCpu, dirtyStart, dirtyStart, instanceCount - dirtyStart);
        }

        private int GetOrAddMaterial(Material mat, int instanceSlotIndex,
            List<DonutMaterialConstants> matList, List<IntPtr> texPtrs)
        {
            int matId = mat != null ? mat.GetInstanceID() : -1;
            if (_materialSlots.TryGetValue(matId, out int existing))
                return existing;

            int idx = matList.Count;
            _materialSlots[matId] = idx;

            int baseTexIdx       = AddTexture(TryGetTex(mat, "_BaseMap"),          texPtrs);
            int normalTexIdx     = AddTexture(TryGetTex(mat, "_BumpMap"),          texPtrs);
            int metalRoughTexIdx = AddTexture(TryGetTex(mat, "_MetallicGlossMap"), texPtrs);
            int emissiveTexIdx   = AddTexture(TryGetTex(mat, "_EmissionMap"),      texPtrs);
            int occlusionTexIdx  = AddTexture(TryGetTex(mat, "_OcclusionMap"),     texPtrs);

            Color  baseColor   = TryGetColor(mat, "_BaseColor", Color.white);
            Color  emissive    = TryGetColor(mat, "_EmissionColor", Color.black);
            float  roughness   = 1f - TryGetFloat(mat, "_Smoothness", 0.5f);
            float  metalness   = TryGetFloat(mat, "_Metallic", 0f);
            float  alphaCutoff = TryGetFloat(mat, "_Cutoff", 0f);
            float  normalScale = TryGetFloat(mat, "_BumpScale", 1f);
            float  occStr      = TryGetFloat(mat, "_OcclusionStrength", 1f);

            // Approximate F0 specular colour (donut convention)
            Vector3 dielectricF0  = Vector3.one * 0.04f;
            Vector3 metalBaseColor = new Vector3(baseColor.r, baseColor.g, baseColor.b);
            Vector3 specularColor  = Vector3.Lerp(dielectricF0, metalBaseColor, metalness);

            int domain = 0;
            if (mat != null && mat.HasProperty("_Surface"))
                domain = (int)mat.GetFloat("_Surface");
            else if (alphaCutoff > 0f)
                domain = 1; // AlphaTested

            int matFlags = 0;
            if (baseTexIdx       >= 0) matFlags |= DonutMaterialFlags.UseBaseOrDiffuseTexture;
            if (normalTexIdx     >= 0) matFlags |= DonutMaterialFlags.UseNormalTexture;
            if (metalRoughTexIdx >= 0) matFlags |= DonutMaterialFlags.UseMetalRoughOrSpecularTexture;
            if (emissiveTexIdx   >= 0) matFlags |= DonutMaterialFlags.UseEmissiveTexture;
            if (occlusionTexIdx  >= 0) matFlags |= DonutMaterialFlags.UseOcclusionTexture;

            matList.Add(new DonutMaterialConstants
            {
                baseOrDiffuseColor               = new Vector3(baseColor.r, baseColor.g, baseColor.b),
                flags                            = matFlags,
                specularColor                    = specularColor,
                materialID                       = idx,
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
                transmissionTextureIndex         = -1,
                opacityTextureIndex              = -1,
                normalTextureTransformScaleX     = 1f,
                normalTextureTransformScaleY     = 1f,
                // SSS / hair — all zero (default struct init)
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
        /// Returns a single-stream copy of the mesh if needed; original otherwise.
        /// Mirrors the same logic in <c>NativeRender.GPUScene</c>.
        /// </summary>
        private Mesh GetOrCreateSingleStreamMesh(Mesh source)
        {
            if (source == null) return null;
            int key = source.GetInstanceID();
            if (_normalizedMeshCache.TryGetValue(key, out var cached)) return cached;

            // If only one vertex stream, use as-is.
            if (source.vertexBufferCount <= 1)
            {
                _normalizedMeshCache[key] = source;
                return source;
            }

            // Combine into a single stream.
            var desc = new Mesh();
            desc.name = source.name + "_SingleStream";
            desc.indexFormat = source.indexFormat;

            var positions = source.vertices;
            var normals   = source.normals;
            var tangents  = source.tangents;
            var uvs       = source.uv;

            desc.vertices = positions;
            if (normals.Length  > 0) desc.normals  = normals;
            if (tangents.Length > 0) desc.tangents = tangents;
            if (uvs.Length      > 0) desc.uv       = uvs;

            desc.subMeshCount = source.subMeshCount;
            for (int s = 0; s < source.subMeshCount; s++)
                desc.SetIndices(source.GetIndices(s), source.GetSubMesh(s).topology, s);

            desc.UploadMeshData(false);

            _normalizedMeshCache[key] = desc;
            _ownedMeshes.Add(desc);
            return desc;
        }
    }
}
