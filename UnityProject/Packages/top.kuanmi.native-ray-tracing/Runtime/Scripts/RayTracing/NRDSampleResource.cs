using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

namespace NativeRender
{
    /// <summary>
    /// Scene-resident resources mirroring NRDSample.cpp's resource taxonomy and
    /// binding names: two TLAS (world/light), InstanceData + PrimitiveData
    /// structured buffers, SHARC UAV ring buffers, a MorphPrimitivePositions stub
    /// (static scene), and a bindless material texture array.
    ///
    /// Data layouts match the <c>InstanceData</c> / <c>PrimitiveData</c> /
    /// <c>MorphPrimitivePositions</c> structs in NRDSample.cpp exactly.
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
        public const uint FLAG_MORPH           = 0x10;

        // ----- Acceleration structures -----
        private RayTracingAccelerationStructure _worldAS;   // gWorldTlas
        private RayTracingAccelerationStructure _lightAS;   // gLightTlas

        // ----- Structured buffers (NRDSample names) -----
        private GraphicsBuffer _instanceDataBuf;                 // gIn_InstanceData
        private GraphicsBuffer _primitiveDataBuf;                // gIn_PrimitiveData
        private GraphicsBuffer _morphPrimitivePositionsPrevBuf;  // gIn_MorphPrimitivePositionsPrev (stub)

        // ----- SHARC UAV ring buffers -----
        private GraphicsBuffer _sharcHashEntries;   // gInOut_SharcHashEntriesBuffer
        private GraphicsBuffer _sharcAccumulated;   // gInOut_SharcAccumulated
        private GraphicsBuffer _sharcResolved;      // gInOut_SharcResolved

        // ----- Material texture array (gIn_Textures) -----
        // Interleaved TEXTURES_PER_MATERIAL per material, per NRDSample layout:
        // slot [matIdx*4 + 0] = baseColor
        //      [matIdx*4 + 1] = normal
        //      [matIdx*4 + 2] = roughness/metallic
        //      [matIdx*4 + 3] = emissive
        private BindlessTexture _textures;

        // ----- CPU mirrors -----
        private InstanceDataNRD[]  _instanceCpu;
        private PrimitiveDataNRD[] _primitiveCpu;

        // ----- Per-instance tracking -----
        private struct SceneInstance
        {
            public MeshRenderer renderer;
            public Transform    transform;
            public bool         isEmissive;
        }
        private readonly List<SceneInstance>           _sceneInstances    = new();
        private readonly Dictionary<Material, int>     _materialSlots     = new();
        private readonly Dictionary<int, int>          _textureSlots      = new();
        private readonly List<NativeRayTracingTarget>  _registeredTargets = new();
        private readonly Dictionary<int, Mesh>         _normalizedMeshCache = new();
        private readonly List<Mesh>                    _ownedMeshes         = new();

        private bool _sceneGpuDirty = true;
        private bool _forceRebuild;
        private readonly HashSet<Material> _dirtyMaterials = new();

        private bool _disposed;

        public RayTracingAccelerationStructure WorldAS => _worldAS;
        public RayTracingAccelerationStructure LightAS => _lightAS;

        public NRDSampleResource()
        {
            _worldAS = new RayTracingAccelerationStructure();
            _lightAS = new RayTracingAccelerationStructure();
            AllocateStaticResources();
        }

        public void MarkMaterialDirty(Material mat)
        {
            if (mat != null) _dirtyMaterials.Add(mat);
        }

        public void MarkRebuildDirty() => _forceRebuild = true;

        /// <summary>Dirty detection + scene registration + GPU rebuild + transform update.</summary>
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
            _lightAS.BuildOrUpdate(cmd);
        }

        /// <summary>Bind all scene GPU resources to a ray tracing shader using NRDSample names.</summary>
        public void BindToShader(RayTraceShader shader)
        {
            if (shader == null || !shader.IsValid) return;

            shader.SetAccelerationStructure("gWorldTlas", _worldAS);
            shader.SetAccelerationStructure("gLightTlas", _lightAS);

            shader.SetStructuredBuffer("gIn_InstanceData",               _instanceDataBuf);
            shader.SetStructuredBuffer("gIn_PrimitiveData",              _primitiveDataBuf);
            shader.SetStructuredBuffer("gIn_MorphPrimitivePositionsPrev", _morphPrimitivePositionsPrevBuf);

            // SHARC UAV bindings.
            shader.SetRWBuffer("gInOut_SharcHashEntriesBuffer", _sharcHashEntries);
            shader.SetRWBuffer("gInOut_SharcAccumulated",       _sharcAccumulated);
            shader.SetRWBuffer("gInOut_SharcResolved",          _sharcResolved);

            if (_textures != null) shader.SetBindlessTexture("gIn_Textures", _textures);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            DisposeSceneGpuBuffers();
            DisposeStaticResources();

            _worldAS?.Dispose(); _worldAS = null;
            _lightAS?.Dispose(); _lightAS = null;
        }

        // =====================================================================
        // Static resources
        // =====================================================================

        private void AllocateStaticResources()
        {
            const GraphicsBuffer.Target rwTarget = GraphicsBuffer.Target.Structured;

            _sharcHashEntries = new GraphicsBuffer(rwTarget, SharcCapacity, sizeof(ulong));
            _sharcAccumulated = new GraphicsBuffer(rwTarget, SharcCapacity, sizeof(uint) * 4);
            _sharcResolved    = new GraphicsBuffer(rwTarget, SharcCapacity, sizeof(uint) * 4);

            // Static scene: single-element stub buffer so the binding is never null.
            _morphPrimitivePositionsPrevBuf = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured, 1, Marshal.SizeOf<MorphPrimitivePositionsNRD>());
            var stub = new MorphPrimitivePositionsNRD[1];
            _morphPrimitivePositionsPrevBuf.SetData(stub);
        }

        private void DisposeStaticResources()
        {
            _sharcHashEntries?.Release();               _sharcHashEntries               = null;
            _sharcAccumulated?.Release();               _sharcAccumulated               = null;
            _sharcResolved?.Release();                  _sharcResolved                  = null;
            _morphPrimitivePositionsPrevBuf?.Release(); _morphPrimitivePositionsPrevBuf = null;
        }

        // =====================================================================
        // Dynamic scene GPU data
        // =====================================================================

        private void DisposeSceneGpuBuffers()
        {
            _instanceDataBuf?.Release();  _instanceDataBuf  = null;
            _primitiveDataBuf?.Release(); _primitiveDataBuf = null;
            _textures?.Dispose();         _textures         = null;

            _instanceCpu  = null;
            _primitiveCpu = null;

            _sceneInstances.Clear();
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
            bool hasDestroyed = false;
            foreach (var target in _registeredTargets)
                if (target == null) { hasDestroyed = true; break; }

            if (hasDestroyed)
            {
                _worldAS?.Dispose(); _worldAS = new RayTracingAccelerationStructure();
                _lightAS?.Dispose(); _lightAS = new RayTracingAccelerationStructure();

                foreach (var target in targets)
                    AddToAS(target);
                return;
            }

            var incoming = new HashSet<NativeRayTracingTarget>(targets);
            foreach (var target in _registeredTargets)
            {
                if (incoming.Contains(target)) continue;
                var mr = target.GetComponent<MeshRenderer>();
                if (mr == null) continue;
                _worldAS.RemoveInstance(mr);
                _lightAS.RemoveInstance(mr);
            }

            var registered = new HashSet<NativeRayTracingTarget>(_registeredTargets);
            foreach (var target in targets)
            {
                if (registered.Contains(target)) continue;
                AddToAS(target);
            }
        }

        private void AddToAS(NativeRayTracingTarget target)
        {
            if (target == null) return;
            var mr = target.GetComponent<MeshRenderer>();
            if (mr == null)
            {
                Debug.LogWarning($"[NRDSampleResource] Target '{target.name}' has no MeshRenderer — skipping");
                return;
            }

            Matrix4x4 xform = target.transform.localToWorldMatrix;
            if (_worldAS.AddInstance(mr, target.ommCaches))
                _worldAS.SetInstanceTransform(mr, xform);

            if (IsTargetEmissive(target) && _lightAS.AddInstance(mr, target.ommCaches))
                _lightAS.SetInstanceTransform(mr, xform);
        }

        private static bool IsTargetEmissive(NativeRayTracingTarget target)
        {
            var mr = target != null ? target.GetComponent<MeshRenderer>() : null;
            if (mr == null) return false;
            var mats = mr.sharedMaterials;
            if (mats == null) return false;
            for (int i = 0; i < mats.Length; i++)
            {
                var m = mats[i];
                if (m == null) continue;
                if (m.HasProperty("_EmissionColor"))
                {
                    Color e = m.GetColor("_EmissionColor").linear;
                    if (e.r > 0f || e.g > 0f || e.b > 0f) return true;
                }
                if (m.HasProperty("_EmissionMap") && m.GetTexture("_EmissionMap") != null)
                    return true;
            }
            return false;
        }

        private void RebuildSceneGpuData(IReadOnlyList<NativeRayTracingTarget> targets)
        {
            DisposeSceneGpuBuffers();

            var instList = new List<InstanceDataNRD>();
            var primList = new List<PrimitiveDataNRD>();
            var texPtrs  = new List<IntPtr>();

            int lightInstanceSlot = 0;
            uint primitiveRunning = 0;

            foreach (var target in targets)
            {
                var mr = target.GetComponent<MeshRenderer>();
                if (mr == null) continue;
                var mf = mr.GetComponent<MeshFilter>();
                if (mf == null || mf.sharedMesh == null) continue;

                Mesh mesh = GetOrCreateSingleStreamMesh(mf.sharedMesh);
                if (mesh == null) continue;

                // Use the first submesh's material as the "representative" material
                // (NRDSample stores one Material per Instance). Multi-submesh renderers
                // all share the same instance slot; additional material variation can be
                // added later by splitting into multiple NativeRayTracingTargets.
                Material[] mats = mr.sharedMaterials ?? Array.Empty<Material>();
                Material   mat  = mats.Length > 0 ? mats[0] : null;
                int        matIdx = GetOrAddMaterial(mat, texPtrs);

                uint flags = FLAG_STATIC | FLAG_NON_TRANSPARENT;
                bool isEmissive = IsMaterialEmissive(mat);
                if (isEmissive) flags |= FLAG_EMISSIVE;

                uint baseTextureIndex = (uint)(matIdx * TexturesPerMaterial);

                // Append PrimitiveData for every triangle of the mesh.
                uint primitiveOffset = primitiveRunning;
                AppendPrimitiveData(mesh, primList);
                uint triCount = (uint)(primList.Count - (int)primitiveOffset);
                primitiveRunning += triCount;

                // Build InstanceData.
                Matrix4x4 local = target.transform.localToWorldMatrix;
                Vector3   s     = new Vector3(
                    new Vector3(local.m00, local.m10, local.m20).magnitude,
                    new Vector3(local.m01, local.m11, local.m21).magnitude,
                    new Vector3(local.m02, local.m12, local.m22).magnitude);
                float     scaleMax    = Mathf.Max(s.x, Mathf.Max(s.y, s.z));
                bool      leftHanded  = Vector3.Dot(Vector3.Cross(
                                          new Vector3(local.m00, local.m10, local.m20),
                                          new Vector3(local.m01, local.m11, local.m21)),
                                          new Vector3(local.m02, local.m12, local.m22)) < 0f;

                var inst = new InstanceDataNRD
                {
                    mOverloadedMatrix0   = TransposedCol(local, 0),
                    mOverloadedMatrix1   = TransposedCol(local, 1),
                    mOverloadedMatrix2   = TransposedCol(local, 2),
                    textureOffsetAndFlags = baseTextureIndex | (flags << FlagFirstBit),
                    primitiveOffset       = primitiveOffset,
                    scale                 = (leftHanded ? -1f : 1f) * scaleMax,
                    morphPrimitiveOffset  = 0,
                };
                EncodeMaterial(mat, ref inst);

                uint instanceIdx = (uint)instList.Count;
                _worldAS.SetInstanceID(mr, instanceIdx);
                instList.Add(inst);

                if (isEmissive)
                {
                    _lightAS.SetInstanceID(mr, (uint)lightInstanceSlot);
                    lightInstanceSlot++;
                }

                _sceneInstances.Add(new SceneInstance
                {
                    renderer   = mr,
                    transform  = mr.transform,
                    isEmissive = isEmissive,
                });
            }

            int texCount = Mathf.Max(texPtrs.Count, 1);
            _textures = new BindlessTexture(texCount);
            for (int i = 0; i < texPtrs.Count; i++)
                _textures.SetNativePtr(i, texPtrs[i]);

            if (instList.Count == 0) instList.Add(default);
            if (primList.Count == 0) primList.Add(default);

            _instanceCpu  = instList.ToArray();
            _primitiveCpu = primList.ToArray();

            _instanceDataBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                _instanceCpu.Length, Marshal.SizeOf<InstanceDataNRD>());
            _instanceDataBuf.SetData(_instanceCpu);

            _primitiveDataBuf = new GraphicsBuffer(GraphicsBuffer.Target.Structured,
                _primitiveCpu.Length, Marshal.SizeOf<PrimitiveDataNRD>());
            _primitiveDataBuf.SetData(_primitiveCpu);

            _sceneGpuDirty = false;
        }

        private static Vector4 TransposedCol(Matrix4x4 m, int col)
        {
            // Transpose3x4: column `col` of the transposed matrix is row `col` of original,
            // i.e. (m[col,0], m[col,1], m[col,2], m[col,3]).
            switch (col)
            {
                case 0: return new Vector4(m.m00, m.m01, m.m02, m.m03);
                case 1: return new Vector4(m.m10, m.m11, m.m12, m.m13);
                case 2: return new Vector4(m.m20, m.m21, m.m22, m.m23);
            }
            return Vector4.zero;
        }

        private static void EncodeMaterial(Material mat, ref InstanceDataNRD inst)
        {
            Color baseColor = TryGetColor(mat, "_BaseColor",     Color.white);
            Color emission  = TryGetColor(mat, "_EmissionColor", Color.black);
            float metal     = TryGetFloat(mat, "_Metallic",   0f);
            float smooth    = TryGetFloat(mat, "_Smoothness", 0.5f);
            float roughness = 1f - smooth;
            float normScale = TryGetFloat(mat, "_BumpScale",  1f);

            inst.baseColorR      = Mathf.FloatToHalf(baseColor.r);
            inst.baseColorG      = Mathf.FloatToHalf(baseColor.g);
            inst.baseColorB      = Mathf.FloatToHalf(baseColor.b);
            inst.metalnessScaleH = Mathf.FloatToHalf(metal);

            inst.emissionR       = Mathf.FloatToHalf(emission.r);
            inst.emissionG       = Mathf.FloatToHalf(emission.g);
            inst.emissionB       = Mathf.FloatToHalf(emission.b);
            inst.roughnessScaleH = Mathf.FloatToHalf(roughness);

            inst.normalUvScaleX  = Mathf.FloatToHalf(normScale);
            inst.normalUvScaleY  = Mathf.FloatToHalf(normScale);
        }

        private int GetOrAddMaterial(Material mat, List<IntPtr> texPtrs)
        {
            // Use a sentinel key for null materials so they all share slot 0.
            if (mat != null && _materialSlots.TryGetValue(mat, out int existing))
                return existing;

            int idx = _materialSlots.Count;
            if (mat != null) _materialSlots[mat] = idx;

            // Emit 4 texture slots per material — null textures become IntPtr.Zero.
            Texture baseTex     = TryGetTex(mat, "_BaseMap");
            Texture normalTex   = TryGetTex(mat, "_BumpMap");
            Texture metalTex    = TryGetTex(mat, "_MetallicGlossMap");
            Texture emissiveTex = TryGetTex(mat, "_EmissionMap");

            AppendTexture(baseTex,     texPtrs);
            AppendTexture(normalTex,   texPtrs);
            AppendTexture(metalTex,    texPtrs);
            AppendTexture(emissiveTex, texPtrs);

            return idx;
        }

        private static void AppendTexture(Texture tex, List<IntPtr> texPtrs)
        {
            texPtrs.Add(tex != null ? tex.GetNativeTexturePtr() : IntPtr.Zero);
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

        private void UpdateInstanceTransforms()
        {
            if (_instanceCpu == null || _instanceDataBuf == null) return;

            int instanceCount = Mathf.Min(_sceneInstances.Count, _instanceCpu.Length);
            int dirtyStart = -1;

            for (int i = 0; i < instanceCount; i++)
            {
                Transform t = _sceneInstances[i].transform;
                if (t == null || !t.hasChanged)
                {
                    if (dirtyStart >= 0)
                    {
                        _instanceDataBuf.SetData(_instanceCpu, dirtyStart, dirtyStart, i - dirtyStart);
                        dirtyStart = -1;
                    }
                    continue;
                }

                Matrix4x4 m = t.localToWorldMatrix;
                _instanceCpu[i].mOverloadedMatrix0 = TransposedCol(m, 0);
                _instanceCpu[i].mOverloadedMatrix1 = TransposedCol(m, 1);
                _instanceCpu[i].mOverloadedMatrix2 = TransposedCol(m, 2);

                var inst = _sceneInstances[i];
                _worldAS.SetInstanceTransform(inst.renderer, m);
                if (inst.isEmissive)
                    _lightAS.SetInstanceTransform(inst.renderer, m);

                if (dirtyStart < 0) dirtyStart = i;
            }

            if (dirtyStart >= 0)
                _instanceDataBuf.SetData(_instanceCpu, dirtyStart, dirtyStart, instanceCount - dirtyStart);
        }

        // =====================================================================
        // PrimitiveData construction
        // =====================================================================

        private static void AppendPrimitiveData(Mesh mesh, List<PrimitiveDataNRD> dst)
        {
            Vector3[] verts   = mesh.vertices;
            Vector3[] normals = mesh.normals;
            Vector4[] tangs   = mesh.tangents;
            Vector2[] uvs     = mesh.uv;
            if (verts == null || verts.Length == 0) return;

            int subCnt = mesh.subMeshCount;
            for (int s = 0; s < subCnt; s++)
            {
                int[] tris = mesh.GetTriangles(s);
                for (int i = 0; i + 2 < tris.Length; i += 3)
                {
                    int i0 = tris[i + 0], i1 = tris[i + 1], i2 = tris[i + 2];

                    Vector3 p0 = verts[i0], p1 = verts[i1], p2 = verts[i2];
                    Vector2 uv0 = (uvs != null && i0 < uvs.Length) ? uvs[i0] : Vector2.zero;
                    Vector2 uv1 = (uvs != null && i1 < uvs.Length) ? uvs[i1] : Vector2.zero;
                    Vector2 uv2 = (uvs != null && i2 < uvs.Length) ? uvs[i2] : Vector2.zero;

                    Vector3 e1 = p1 - p0, e2 = p2 - p0;
                    float worldArea = 0.5f * Vector3.Cross(e1, e2).magnitude;

                    Vector2 du1 = uv1 - uv0, du2 = uv2 - uv0;
                    float uvArea = 0.5f * Mathf.Abs(du1.x * du2.y - du1.y * du2.x);

                    Vector3 n0v = (normals != null && i0 < normals.Length) ? normals[i0].normalized : Vector3.up;
                    Vector3 n1v = (normals != null && i1 < normals.Length) ? normals[i1].normalized : Vector3.up;
                    Vector3 n2v = (normals != null && i2 < normals.Length) ? normals[i2].normalized : Vector3.up;

                    Vector4 t0v = (tangs != null && i0 < tangs.Length) ? tangs[i0] : new Vector4(1, 0, 0, 1);
                    Vector4 t1v = (tangs != null && i1 < tangs.Length) ? tangs[i1] : new Vector4(1, 0, 0, 1);
                    Vector4 t2v = (tangs != null && i2 < tangs.Length) ? tangs[i2] : new Vector4(1, 0, 0, 1);

                    Vector2 n0e = EncodeUnitVectorSigned(n0v);
                    Vector2 n1e = EncodeUnitVectorSigned(n1v);
                    Vector2 n2e = EncodeUnitVectorSigned(n2v);

                    Vector3 t0d = new Vector3(t0v.x, t0v.y, t0v.z);
                    Vector3 t1d = new Vector3(t1v.x, t1v.y, t1v.z);
                    Vector3 t2d = new Vector3(t2v.x, t2v.y, t2v.z);
                    Vector2 t0e = EncodeUnitVectorSigned(t0d.sqrMagnitude > 0f ? t0d.normalized : Vector3.right);
                    Vector2 t1e = EncodeUnitVectorSigned(t1d.sqrMagnitude > 0f ? t1d.normalized : Vector3.right);
                    Vector2 t2e = EncodeUnitVectorSigned(t2d.sqrMagnitude > 0f ? t2d.normalized : Vector3.right);

                    var d = new PrimitiveDataNRD
                    {
                        uv0x = Mathf.FloatToHalf(uv0.x), uv0y = Mathf.FloatToHalf(uv0.y),
                        uv1x = Mathf.FloatToHalf(uv1.x), uv1y = Mathf.FloatToHalf(uv1.y),
                        uv2x = Mathf.FloatToHalf(uv2.x), uv2y = Mathf.FloatToHalf(uv2.y),
                        worldArea = worldArea,

                        n0x = Mathf.FloatToHalf(n0e.x), n0y = Mathf.FloatToHalf(n0e.y),
                        n1x = Mathf.FloatToHalf(n1e.x), n1y = Mathf.FloatToHalf(n1e.y),
                        n2x = Mathf.FloatToHalf(n2e.x), n2y = Mathf.FloatToHalf(n2e.y),
                        uvArea = uvArea,

                        t0x = Mathf.FloatToHalf(t0e.x), t0y = Mathf.FloatToHalf(t0e.y),
                        t1x = Mathf.FloatToHalf(t1e.x), t1y = Mathf.FloatToHalf(t1e.y),
                        t2x = Mathf.FloatToHalf(t2e.x), t2y = Mathf.FloatToHalf(t2e.y),
                        bitangentSign = t0v.w,
                    };
                    dst.Add(d);
                }
            }
        }

        /// <summary>Signed octahedral unit-vector encoding, matching MathLib Packing::EncodeUnitVector(v, true).</summary>
        private static Vector2 EncodeUnitVectorSigned(Vector3 v)
        {
            float absSum = Mathf.Abs(v.x) + Mathf.Abs(v.y) + Mathf.Abs(v.z);
            if (absSum < 1e-8f) return Vector2.zero;
            v /= absSum;
            if (v.z >= 0f)
                return new Vector2(v.x, v.y);
            float sx = v.x >= 0f ?  1f : -1f;
            float sy = v.y >= 0f ?  1f : -1f;
            return new Vector2((1f - Mathf.Abs(v.y)) * sx, (1f - Mathf.Abs(v.x)) * sy);
        }

        // =====================================================================
        // Single-stream mesh normalisation
        // =====================================================================

        private Mesh GetOrCreateSingleStreamMesh(Mesh src)
        {
            if (src == null) return null;

            int id = src.GetInstanceID();
            if (_normalizedMeshCache.TryGetValue(id, out var cached) && cached != null)
                return cached;

            var attrs = src.GetVertexAttributes();
            bool multiStream = false;
            for (int i = 0; i < attrs.Length; i++)
                if (attrs[i].stream != 0) { multiStream = true; break; }

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