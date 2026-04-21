using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace NativeRender
{
    /// <summary>
    /// Scene-resident resources mirroring NRDSample.cpp's resource taxonomy and
    /// merged-BLAS acceleration-structure layout:
    ///
    ///     BLAS_MergedOpaque      – non-transparent, non-emissive geometry (pre-transformed)
    ///     BLAS_MergedTransparent – transparent geometry                   (pre-transformed)
    ///     BLAS_MergedEmissive    – emissive geometry                      (pre-transformed)
    ///
    ///     gWorldTlas (one instance each for MergedOpaque + MergedTransparent)
    ///     gLightTlas (one instance for MergedEmissive)
    ///
    /// For each merged BLAS we build, on the CPU, one tightly-packed float3
    /// vertex buffer whose positions are already in world space and one uint32
    /// index buffer. The BLAS is then constructed from those buffers with an
    /// identity TLAS transform, exactly like <c>Sample::CreateBottomLevelAccelerationStructures</c>.
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
        private InstanceDataNRD[]  _instanceCpu;
        private PrimitiveDataNRD[] _primitiveCpu;

        // ----- Tracking for dirty-detection -----
        private readonly List<NativeRayTracingTarget> _registeredTargets = new();
        private readonly Dictionary<Material, int>    _materialSlots     = new();

        private bool _sceneDirty = true;
        private bool _disposed;

        public RayTracingAccelerationStructure WorldAS => _worldAS;
        public RayTracingAccelerationStructure LightAS => _lightAS;

        public NRDSampleResource()
        {
            _worldAS = new RayTracingAccelerationStructure();
            _lightAS = new RayTracingAccelerationStructure();
            AllocateStaticResources();
        }

        public void MarkRebuildDirty() => _sceneDirty = true;

        /// <summary>Dirty detection + full scene rebuild when needed.</summary>
        public void UpdateForFrame()
        {
            var targets = NativeRayTracingTarget.All;

            if (!_sceneDirty && !TargetSetChanged(targets) && !AnyTransformChanged(targets))
                return;

            RebuildScene(targets);

            _registeredTargets.Clear();
            _registeredTargets.AddRange(targets);
            foreach (var t in targets)
                if (t != null)
                    t.transform.hasChanged = false;

            _sceneDirty = false;
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

            shader.SetStructuredBuffer("gIn_InstanceData", _instanceDataBuf);
            shader.SetStructuredBuffer("gIn_PrimitiveData", _primitiveDataBuf);
            shader.SetStructuredBuffer("gIn_MorphPrimitivePositionsPrev", _morphPrimitivePositionsPrevBuf);

            shader.SetRWBuffer("gInOut_SharcHashEntriesBuffer", _sharcHashEntries);
            shader.SetRWBuffer("gInOut_SharcAccumulated", _sharcAccumulated);
            shader.SetRWBuffer("gInOut_SharcResolved", _sharcResolved);

            shader.SetBindlessTexture("gIn_Textures", _textures);
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

        private void DisposeSceneGpuBuffers()
        {
            _instanceDataBuf?.Release();
            _instanceDataBuf = null;
            _primitiveDataBuf?.Release();
            _primitiveDataBuf = null;
            _textures?.Dispose();
            _textures = null;

            _blasOpaque?.Dispose();
            _blasOpaque = null;
            _blasTransparent?.Dispose();
            _blasTransparent = null;
            _blasEmissive?.Dispose();
            _blasEmissive = null;

            _instanceCpu  = null;
            _primitiveCpu = null;
            _materialSlots.Clear();
        }

        private bool TargetSetChanged(IReadOnlyList<NativeRayTracingTarget> current)
        {
            if (current.Count != _registeredTargets.Count) return true;
            for (int i = 0; i < current.Count; i++)
                if (current[i] != _registeredTargets[i])
                    return true;
            return false;
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
        private void RebuildScene(IReadOnlyList<NativeRayTracingTarget> targets)
        {
            DisposeSceneGpuBuffers();

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
                else if (isEmissive) emissive.Add(t);
                else opaque.Add(t);
            }

            // Running primitive / instance offsets (NRDSample orders opaque → transparent → emissive).
            uint primitiveCursor = 0;
            uint instanceCursor  = 0;

            var instList = new List<InstanceDataNRD>();
            var primList = new List<PrimitiveDataNRD>();
            var texPtrs  = new List<IntPtr>();

            uint opaqueFirstInstance = instanceCursor;
            _blasOpaque = BuildMergedBlas(opaque, ref instanceCursor, ref primitiveCursor,
                instList, primList, texPtrs,
                FLAG_STATIC | FLAG_NON_TRANSPARENT);

            uint transparentFirstInstance = instanceCursor;
            _blasTransparent = BuildMergedBlas(transparent, ref instanceCursor, ref primitiveCursor,
                instList, primList, texPtrs,
                FLAG_STATIC | FLAG_TRANSPARENT);

            uint emissiveFirstInstance = instanceCursor;
            _blasEmissive = BuildMergedBlas(emissive, ref instanceCursor, ref primitiveCursor,
                instList, primList, texPtrs,
                FLAG_STATIC | FLAG_NON_TRANSPARENT | FLAG_EMISSIVE);

            // Texture array.
            int texCount = Mathf.Max(texPtrs.Count, 1);
            _textures = new BindlessTexture(texCount);
            for (int i = 0; i < texPtrs.Count; i++)
                _textures.SetNativePtr(i, texPtrs[i]);

            // Instance / primitive GPU buffers.
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
            List<PrimitiveDataNRD> primList,
            List<IntPtr> texPtrs,
            uint baseFlags)
        {
            if (group.Count == 0) return null;

            // First pass – sum sizes.
            int totalVerts = 0, totalIndices = 0;
            foreach (var t in group)
            {
                var mesh = t.GetComponent<MeshFilter>().sharedMesh;
                totalVerts += mesh.vertexCount;
                for (int s = 0; s < mesh.subMeshCount; s++)
                    totalIndices += (int)mesh.GetIndexCount(s);
            }

            if (totalVerts == 0 || totalIndices == 0) return null;

            var positions    = new Vector3[totalVerts];
            var indices      = new uint[totalIndices];
            var submeshDescs = new List<NativeRenderPlugin.SubmeshDesc>();
            int vWrite       = 0, iWrite = 0;

            foreach (var target in group)
            {
                var       mr    = target.GetComponent<MeshRenderer>();
                var       mesh  = mr.GetComponent<MeshFilter>().sharedMesh;
                Matrix4x4 xform = target.transform.localToWorldMatrix;

                Vector3[] src   = mesh.vertices;
                Vector3[] srcN  = mesh.normals;
                Vector4[] srcT  = mesh.tangents;
                Vector2[] srcUV = mesh.uv;

                int vertBase = vWrite;

                // Pre-transform positions into world space.
                for (int k = 0; k < src.Length; k++)
                    positions[vWrite++] = xform.MultiplyPoint3x4(src[k]);

                // Normal-space transform (handles non-uniform scale).
                Matrix4x4 normalMatrix = xform.inverse.transpose;
                bool      leftHanded   = xform.determinant < 0f;
                Vector3 s = new Vector3(
                    new Vector3(xform.m00, xform.m10, xform.m20).magnitude,
                    new Vector3(xform.m01, xform.m11, xform.m21).magnitude,
                    new Vector3(xform.m02, xform.m12, xform.m22).magnitude);
                float scaleMax = Mathf.Max(s.x, Mathf.Max(s.y, s.z));

                Material[] sharedMaterials = mr.sharedMaterials;

                int subCnt = mesh.subMeshCount;
                for (int sub = 0; sub < subCnt; sub++)
                {
                    // Each submesh gets its own InstanceDataNRD with its own primitiveOffset and material.
                    uint primitiveOffsetForSubMesh = primitiveCursor;

                    Material subMat    = (sub < sharedMaterials.Length) ? sharedMaterials[sub] : GetRepresentativeMaterial(mr);
                    int      subMatIdx = GetOrAddMaterial(subMat, texPtrs);

                    int[] tris = mesh.GetTriangles(sub);

                    // Record this submesh's IB offset/length inside the merged IB
                    // (NRDSample stores one geometry per submesh in the merged BLAS).
                    int submeshIndexStart = iWrite;
                    submeshDescs.Add(new NativeRenderPlugin.SubmeshDesc
                    {
                        indexCount      = (uint)tris.Length,
                        indexByteOffset = (uint)(submeshIndexStart * sizeof(uint)),
                    });

                    for (int i = 0; i + 2 < tris.Length; i += 3)
                    {
                        int i0 = tris[i + 0], i1 = tris[i + 1], i2 = tris[i + 2];

                        // Append to merged IB (remap to global vertex space).
                        indices[iWrite++] = (uint)(vertBase + i0);
                        indices[iWrite++] = (uint)(vertBase + i1);
                        indices[iWrite++] = (uint)(vertBase + i2);

                        // PrimitiveData uses world-space normals / tangents / area.
                        Vector3 p0 = positions[vertBase + i0];
                        Vector3 p1 = positions[vertBase + i1];
                        Vector3 p2 = positions[vertBase + i2];

                        Vector2 uv0 = (srcUV != null && i0 < srcUV.Length) ? srcUV[i0] : Vector2.zero;
                        Vector2 uv1 = (srcUV != null && i1 < srcUV.Length) ? srcUV[i1] : Vector2.zero;
                        Vector2 uv2 = (srcUV != null && i2 < srcUV.Length) ? srcUV[i2] : Vector2.zero;

                        float   worldArea = 0.5f * Vector3.Cross(p1 - p0, p2 - p0).magnitude;
                        Vector2 du1       = uv1 - uv0, du2 = uv2 - uv0;
                        float   uvArea    = 0.5f * Mathf.Abs(du1.x * du2.y - du1.y * du2.x);

                        Vector3 n0 = TransformNormal(normalMatrix, srcN, i0);
                        Vector3 n1 = TransformNormal(normalMatrix, srcN, i1);
                        Vector3 n2 = TransformNormal(normalMatrix, srcN, i2);

                        Vector4 t0 = TransformTangent(xform, srcT, i0);
                        Vector4 t1 = TransformTangent(xform, srcT, i1);
                        Vector4 t2 = TransformTangent(xform, srcT, i2);

                        primList.Add(new PrimitiveDataNRD
                        {
                            uv0       = new half2(uv0),
                            uv1       = new half2(uv1),
                            uv2       = new half2(uv2),
                            worldArea = worldArea,

                            n0     = EncodeUnitVectorSigned(n0),
                            n1     = EncodeUnitVectorSigned(n1),
                            n2     = EncodeUnitVectorSigned(n2),
                            uvArea = uvArea,

                            t0            = EncodeUnitVectorSigned(new Vector3(t0.x, t0.y, t0.z)),
                            t1            = EncodeUnitVectorSigned(new Vector3(t1.x, t1.y, t1.z)),
                            t2            = EncodeUnitVectorSigned(new Vector3(t2.x, t2.y, t2.z)),
                            bitangentSign = t0.w,
                        });
                        primitiveCursor++;
                    }

                    // Emit one InstanceDataNRD per submesh.
                    uint baseTextureIndex = (uint)(subMatIdx * TexturesPerMaterial);
                    var inst = new InstanceDataNRD
                    {
                        // Vertices are already world-space → mOverloadedMatrix encodes identity.
                        mOverloadedMatrix0 = new Vector4(1f, 0f, 0f, 0f),
                        mOverloadedMatrix1 = new Vector4(0f, 1f, 0f, 0f),
                        mOverloadedMatrix2 = new Vector4(0f, 0f, 1f, 0f),

                        textureOffsetAndFlags = baseTextureIndex | (baseFlags << FlagFirstBit),
                        primitiveOffset       = primitiveOffsetForSubMesh,
                        scale                 = (leftHanded ? -1f : 1f) * scaleMax,
                        morphPrimitiveOffset  = 0,
                    };
                    EncodeMaterial(subMat, ref inst);
                    instList.Add(inst);
                    instanceCursor++;
                }
            }

            // Upload merged VB/IB.
            var blas = new MergedBlas
            {
                vertexCount  = (uint)totalVerts,
                submeshDescs = submeshDescs.ToArray(),
                vb           = new GraphicsBuffer(GraphicsBuffer.Target.Structured, totalVerts, sizeof(float) * 3),
                ib           = new GraphicsBuffer(GraphicsBuffer.Target.Structured, totalIndices, sizeof(uint)),
            };
            blas.vb.SetData(positions);
            blas.ib.SetData(indices);
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

        private static Material GetRepresentativeMaterial(MeshRenderer mr)
        {
            var mats = mr.sharedMaterials;
            return (mats != null && mats.Length > 0) ? mats[0] : null;
        }

        private int GetOrAddMaterial(Material mat, List<IntPtr> texPtrs)
        {
            if (mat != null && _materialSlots.TryGetValue(mat, out int existing))
                return existing;

            int idx                              = _materialSlots.Count;
            if (mat != null) _materialSlots[mat] = idx;

            AppendTexture(TryGetTex(mat, "_BaseMap"), PlaceholderKind.White, texPtrs);
            AppendTexture(TryGetTex(mat, "_BumpMap"), PlaceholderKind.FlatNormal, texPtrs);
            AppendTexture(TryGetTex(mat, "_MetallicGlossMap"), PlaceholderKind.Black, texPtrs);
            AppendTexture(TryGetTex(mat, "_EmissionMap"), PlaceholderKind.Black, texPtrs);
            return idx;
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
            var t = tex != null ? tex : (Texture)GetPlaceholder(fallback);
            texPtrs.Add(t.GetNativeTexturePtr());
        }

        private static void EncodeMaterial(Material mat, ref InstanceDataNRD inst)
        {
            Color baseColor = TryGetColor(mat, "_BaseColor", Color.white);
            Color emission  = TryGetColor(mat, "_EmissionColor", Color.black);
            float metal     = TryGetFloat(mat, "_Metallic", 0f);
            float smooth    = TryGetFloat(mat, "_Smoothness", 0.5f);
            float roughness = 1f - smooth;
            float normScale = TryGetFloat(mat, "_BumpScale", 1f);

            inst.baseColorR      = Mathf.FloatToHalf(baseColor.r);
            inst.baseColorG      = Mathf.FloatToHalf(baseColor.g);
            inst.baseColorB      = Mathf.FloatToHalf(baseColor.b);
            inst.metalnessScaleH = Mathf.FloatToHalf(metal);

            inst.emissionR       = Mathf.FloatToHalf(emission.r);
            inst.emissionG       = Mathf.FloatToHalf(emission.g);
            inst.emissionB       = Mathf.FloatToHalf(emission.b);
            inst.roughnessScaleH = Mathf.FloatToHalf(roughness);

            inst.normalUvScaleX = Mathf.FloatToHalf(normScale);
            inst.normalUvScaleY = Mathf.FloatToHalf(normScale);
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