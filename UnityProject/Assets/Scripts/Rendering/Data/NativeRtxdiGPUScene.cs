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
        public uint flags; // +0
        public uint firstGeometryInstanceIndex; // +4   = firstGeometryIndex in our flat list
        public uint firstGeometryIndex; // +8

        public uint numGeometries; // +12

        // float3x4 transform — row 0..2 of object-to-world
        public Vector4 transformRow0; // +16
        public Vector4 transformRow1; // +32

        public Vector4 transformRow2; // +48

        // float3x4 prevTransform — same layout, previous frame
        public Vector4 prevTransformRow0; // +64
        public Vector4 prevTransformRow1; // +80
        public Vector4 prevTransformRow2; // +96
    } // Total: 112 bytes

    /// <summary>
    /// Mirrors donut <c>GeometryData</c> from <c>bindless.h</c>.
    /// Size: 4 × 16 = 64 bytes  (<c>c_SizeOfGeometryData</c>).
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct DonutGeometryData
    {
        public uint numIndices; // +0
        public uint numVertices; // +4
        public int  indexBufferIndex; // +8
        public uint indexOffset; // +12  byte offset

        public int  vertexBufferIndex; // +16
        public uint positionOffset; // +20  byte offset (float3)
        public uint prevPositionOffset; // +24  byte offset; = positionOffset for static meshes
        public uint texCoord1Offset; // +28  byte offset (float2), or ~0u

        public uint texCoord2Offset; // +32  byte offset, ~0u (not used)
        public uint normalOffset; // +36  byte offset (oct-encoded or float3), or ~0u
        public uint tangentOffset; // +40  byte offset, or ~0u
        public uint curveRadiusOffset; // +44  ~0u (curves not supported)

        public uint materialIndex; // +48
        public uint pad0; // +52
        public uint pad1; // +56
        public uint pad2; // +60
    } // Total: 64 bytes

    /// <summary>
    /// Mirrors donut <c>MaterialConstants</c> from <c>material_cb.h</c>.
    /// Size: 13 × 16 = 208 bytes  (<c>c_SizeOfMaterialConstants</c>).
    /// SSS / hair fields are zeroed — unused in the GBuffer pass.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct DonutMaterialConstants
    {
        // row 0
        public Vector3 baseOrDiffuseColor; // +0
        public int     flags; // +12

        // row 1
        public Vector3 specularColor; // +16  (derived F0)
        public int     materialID; // +28  = material slot index

        // row 2
        public Vector3 emissiveColor; // +32
        public int     domain; // +44

        // row 3
        public float opacity; // +48
        public float roughness; // +52
        public float metalness; // +56
        public float normalTextureScale; // +60

        // row 4
        public float occlusionStrength; // +64
        public float alphaCutoff; // +68
        public float transmissionFactor; // +72
        public int   baseOrDiffuseTextureIndex; // +76

        // row 5
        public int metalRoughOrSpecularTextureIndex; // +80
        public int emissiveTextureIndex; // +84
        public int normalTextureIndex; // +88
        public int occlusionTextureIndex; // +92

        // row 6
        public int   transmissionTextureIndex; // +96  = -1
        public int   opacityTextureIndex; // +100 = -1
        public float normalTextureTransformScaleX; // +104 = 1
        public float normalTextureTransformScaleY; // +108 = 1

        // rows 7-12  (SSS / hair — zeroed)
        public uint  _padding7_x; // +112
        public uint  _padding7_y; // +116
        public uint  _padding7_z; // +120
        public float sssScale; // +124 = 0

        public float sssTransmissionColorR; // +128
        public float sssTransmissionColorG; // +132
        public float sssTransmissionColorB; // +136
        public float sssAnisotropy; // +140

        public float sssScatteringColorR; // +144
        public float sssScatteringColorG; // +148
        public float sssScatteringColorB; // +152
        public float hairMelanin; // +156

        public float hairBaseColorR; // +160
        public float hairBaseColorG; // +164
        public float hairBaseColorB; // +168
        public float hairMelaninRedness; // +172

        public float hairLongitudinalRoughness; // +176
        public float hairAzimuthalRoughness; // +180
        public float hairIor; // +184
        public float hairCuticleAngle; // +188

        public float hairDiffuseReflectionTintR; // +192
        public float hairDiffuseReflectionTintG; // +196
        public float hairDiffuseReflectionTintB; // +200
        public float hairDiffuseReflectionWeight; // +204
    } // Total: 208 bytes

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
        public const int MetalnessInRedChannel          = 0x00000100;
        public const int UseOpacityTexture              = 0x00000200;
        public const int SubsurfaceScattering           = 0x00000400;
        public const int Hair                           = 0x00000800;
    }

    // =========================================================================
    // Emissive geometry enumeration (used by NativeRtxdiPrepareLightsPass)
    // =========================================================================

    /// <summary>
    /// One entry per emissive sub-mesh in the scene.
    /// Mirrors the per-geometry task record that PrepareLights.computeshader needs.
    /// </summary>
    public struct EmissiveGeometryEntry
    {
        /// <summary>Flat index into the t_InstanceData GPU buffer.</summary>
        public int InstanceIndex;

        /// <summary>Sub-geometry index within the instance (0 … numGeometries-1).</summary>
        public int GeometrySubIndex;

        /// <summary>Number of triangles (numIndices / 3) for this sub-mesh.</summary>
        public uint TriangleCount;

        /// <summary>instance.firstGeometryInstanceIndex — used to fill GeometryInstanceToLight.</summary>
        public uint FirstGeometryInstanceIndex;
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
        private Texture  _pendingEnvMap;
        private int      _environmentMapTextureIndex = -1;

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
            _pendingEnvMap  = envMap;
            _sceneGpuDirty  = true;
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
            if (_instanceCpu == null || _geometryCpu == null || _materialCpu == null)
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
                    if (matIdx < 0 || matIdx >= _materialCpu.Length) continue;

                    var mat = _materialCpu[matIdx];
                    if (mat.emissiveColor.x <= 0f && mat.emissiveColor.y <= 0f && mat.emissiveColor.z <= 0f)
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
                ds.SetStructuredBuffer("t_InstanceData", _instanceGpuBuf.GetNativeBufferPtr(), _instanceGpuBuf.count, _instanceGpuBuf.stride);
            if (_geometryGpuBuf != null)
                ds.SetStructuredBuffer("t_GeometryData", _geometryGpuBuf.GetNativeBufferPtr(), _geometryGpuBuf.count, _geometryGpuBuf.stride);
            if (_materialGpuBuf != null)
                ds.SetStructuredBuffer("t_MaterialConstants", _materialGpuBuf.GetNativeBufferPtr(), _materialGpuBuf.count, _materialGpuBuf.stride);
            if (_sceneBuffers != null) ds.SetBindlessBuffer("t_BindlessBuffers", _sceneBuffers);
            if (_sceneTextures != null) ds.SetBindlessTexture("t_BindlessTextures", _sceneTextures);
        }
        
        public void BindToShader(NativeRayTraceDescriptorSet ds)
        {
            if (ds == null) return;
            if (_instanceGpuBuf != null)
                ds.SetStructuredBuffer("t_InstanceData", _instanceGpuBuf.GetNativeBufferPtr(), _instanceGpuBuf.count, _instanceGpuBuf.stride);
            if (_geometryGpuBuf != null)
                ds.SetStructuredBuffer("t_GeometryData", _geometryGpuBuf.GetNativeBufferPtr(), _geometryGpuBuf.count, _geometryGpuBuf.stride);
            if (_materialGpuBuf != null)
                ds.SetStructuredBuffer("t_MaterialConstants", _materialGpuBuf.GetNativeBufferPtr(), _materialGpuBuf.count, _materialGpuBuf.stride);
            if (_sceneBuffers != null) ds.SetBindlessBuffer("t_BindlessBuffers", _sceneBuffers);
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
            _environmentMapTextureIndex = -1;

            foreach (var buf in _ownedGfxBuffers)
                buf?.Release();
            _ownedGfxBuffers.Clear();
            _donutBufferCache.Clear();
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

                for (int s = 0; s < subMeshCnt; s++)
                {
                    SubMeshDescriptor sub    = mesh.GetSubMesh(s);
                    Material          mat    = s < mats.Length ? mats[s] : (mats.Length > 0 ? mats[mats.Length - 1] : null);
                    int               matIdx = GetOrAddMaterial(mat, instIdx, matList, texPtrs);

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
                matList.Add(default);
            }

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

        private int GetOrAddMaterial(Material mat, int instanceSlotIndex,
            List<DonutMaterialConstants> matList, List<IntPtr> texPtrs)
        {
            int matId = mat != null ? mat.GetInstanceID() : -1;
            if (_materialSlots.TryGetValue(matId, out int existing))
                return existing;

            int idx = matList.Count;
            _materialSlots[matId] = idx;

            int baseTexIdx       = AddTexture(TryGetTex(mat, "_BaseMap"), texPtrs);
            int normalTexIdx     = AddTexture(TryGetTex(mat, "_BumpMap"), texPtrs);
            int metalRoughTexIdx = AddTexture(TryGetTex(mat, "_MetallicGlossMap"), texPtrs);
            int emissiveTexIdx   = AddTexture(TryGetTex(mat, "_EmissionMap"), texPtrs);
            int occlusionTexIdx  = AddTexture(TryGetTex(mat, "_OcclusionMap"), texPtrs);

            Color baseColor   = TryGetColor(mat, "_BaseColor", Color.white);
            Color emissive    = TryGetColor(mat, "_EmissionColor", Color.black);
            float roughness   = 1f - TryGetFloat(mat, "_Smoothness", 0.5f);
            float metalness   = TryGetFloat(mat, "_Metallic", 0f);
            float alphaCutoff = TryGetFloat(mat, "_Cutoff", 0f);
            float normalScale = TryGetFloat(mat, "_BumpScale", 1f);
            float occStr      = TryGetFloat(mat, "_OcclusionStrength", 1f);

            // Approximate F0 specular colour (donut convention)
            Vector3 dielectricF0   = Vector3.one * 0.04f;
            Vector3 metalBaseColor = new Vector3(baseColor.r, baseColor.g, baseColor.b);
            Vector3 specularColor  = Vector3.Lerp(dielectricF0, metalBaseColor, metalness);

            int domain = 0;
            if (mat != null && mat.HasProperty("_Surface"))
                domain = (int)mat.GetFloat("_Surface");
            else if (alphaCutoff > 0f)
                domain = 1; // AlphaTested

            int matFlags                        = 0;
            if (baseTexIdx >= 0) matFlags       |= DonutMaterialFlags.UseBaseOrDiffuseTexture;
            if (normalTexIdx >= 0) matFlags     |= DonutMaterialFlags.UseNormalTexture;
            if (metalRoughTexIdx >= 0) matFlags |= DonutMaterialFlags.UseMetalRoughOrSpecularTexture;
            if (emissiveTexIdx >= 0) matFlags   |= DonutMaterialFlags.UseEmissiveTexture;
            if (occlusionTexIdx >= 0) matFlags  |= DonutMaterialFlags.UseOcclusionTexture;

            if (metalRoughTexIdx >= 0)
                metalness = 1;

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