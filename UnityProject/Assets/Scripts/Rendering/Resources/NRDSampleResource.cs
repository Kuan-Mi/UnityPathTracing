using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Rendering.Resources;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using Debug = UnityEngine.Debug;

namespace NativeRender
{
    /// <summary>
    /// Scene-resident resources mirroring NRDSample.cpp's resource taxonomy.
    ///
    /// In play mode, Unity-static objects are batched into merged pre-transformed
    /// BLASes (FLAG_STATIC set; mOverloaded = identity). Non-static (dynamic)
    /// objects get one BLAS each; mOverloaded stores the previous-frame motion
    /// matrix (prevT * inv(currT)) so the shader can compute correct Xprev.
    ///
    /// In editor non-play mode, all objects use separate BLASes with FLAG_STATIC
    /// (no motion vectors needed; mOverloaded = current rotation matrix).
    ///
    /// </summary>
    public sealed class NRDSampleResource : IDisposable
    {
        public const int SharcCapacity = 1 << 22;

        // Matches "constexpr uint32_t TEXTURES_PER_MATERIAL = 4;" in NRDSample.cpp.
        public const int TexturesPerMaterial = 4;

        // Matches "#define FLAG_FIRST_BIT 24" in Shared.hlsl.
        public const int FlagFirstBit = 24;

        // Scene flag constants (Shared.hlsl).
        public const uint FLAG_NON_TRANSPARENT = 0x01;

        public const uint FLAG_TRANSPARENT = 0x02;

        // public const uint FLAG_EMISSIVE        = 0x04;
        public const uint FLAG_STATIC = 0x08;
        public const uint FLAG_MORPH  = 0x80;

        private const uint kHandleOpaque      = 0xFFFF0001u;
        private const uint kHandleTransparent = 0xFFFF0002u;
        private const uint kHandleEmissive    = 0xFFFF0003u;

        // ----- Mode helpers -----
        // Static objects are merged into one BLAS in play mode only.
        private static bool ShouldMerge() => Application.isPlaying;

        // ----- Acceleration structures -----
        private RayTracingAccelerationStructure _worldAS; // gWorldTlas
        private RayTracingAccelerationStructure _lightAS; // gLightTlas

        private MergedBlas _blasOpaque;
        private MergedBlas _blasTransparent;
        private MergedBlas _blasEmissive;

        // ----- Scene structured buffers -----
        private NativeStructuredBuffer _instanceDataBuf; // gIn_InstanceData
        private GraphicsBuffer         _primitiveDataBuf; // gIn_PrimitiveData
        private GraphicsBuffer         _morphPrimitivePositionsPrevBuf; // gIn_MorphPrimitivePositionsPrev (stub)

        // ----- SHARC UAV ring buffers -----
        private GraphicsBuffer _sharcHashEntries; // gInOut_SharcHashEntriesBuffer
        private GraphicsBuffer _sharcAccumulated; // gInOut_SharcAccumulated
        private GraphicsBuffer _sharcResolved; // gInOut_SharcResolved

        // ----- Material texture array (gIn_Textures) -----
        private BindlessTexture _textures;

        // ----- CPU mirrors -----
        private InstanceDataNRD[]             _instanceCpu;
        private NativeArray<PrimitiveDataNRD> _primitiveCpu;

        // ----- Separate-BLAS per-target tracking -----

        /// <summary>
        /// Represents one group of submeshes that share the same (isTransparent, isEmissive) pair
        /// and are therefore registered as a single TLAS entry with a unique customHandle.
        ///
        /// 一个GO下的一个或多个 submesh 可能会被分成多个 SubmeshGroup 注册到 TLAS 中，条件是它们的 isTransparent/isEmissive/isAlphaClip 标志不同。
        /// </summary>
        private sealed class SubmeshGroup
        {
            /// <summary>Whether all submeshes in this group are transparent.</summary>
            public bool isTransparent;

            /// <summary>Whether all submeshes in this group are emissive.</summary>
            public bool isEmissive;

            public bool isAlphaClip;

            /// <summary>Indices into the original mesh.subMeshCount.</summary>
            public int[] submeshIndices;

            /// <summary>Materials for each submesh (parallel to submeshIndices).</summary>
            public Material[] materials;

            /// <summary>
            /// Unique handle used for all TLAS Set*/Remove calls.
            /// Encoded as: high-4-bits = groupIndex, low-28-bits = mrInstanceId.
            /// </summary>
            public uint customHandle;

            /// <summary>First contiguous slot in _instanceCpu/_instanceDataBuf for this group.</summary>
            public uint firstInstanceIdx;

            /// <summary>Starting element index in _primitiveCpu for each submesh (parallel to submeshIndices).</summary>
            public uint[] primitiveOffsets;

            /// <summary>Triangle count for each submesh (parallel to submeshIndices).</summary>
            public int[] primitiveCounts;

            /// <summary>Which TLAS(es) this group was registered in.</summary>
            public List<RayTracingAccelerationStructure> tlasList;
        }

        /// <summary>Tracks per-renderer state when running in separate (non-merged) BLAS mode.</summary>
        private sealed class PerTargetBlas
        {
            /// <summary>All submesh groups for this renderer, each potentially in different TLAS(es).</summary>
            public List<SubmeshGroup> groups;

            /// <summary>Cached transform to detect changes.</summary>
            public Matrix4x4 lastTransform;

            /// <summary>
            /// True when this entry uses FLAG_STATIC semantics:
            /// in edit mode all objects, in play mode nothing (dynamic only reach separate BLAS).
            /// Controls mOverloaded update strategy in UpdateTransformsOnly.
            /// </summary>
            public bool isStatic;

            /// <summary>
            /// True if the object was moving last frame. Used to detect the first stationary frame
            /// so we can upload identity mOverloaded exactly once when movement stops.
            /// </summary>
            public bool wasMoving = true;
        }

        // Keyed by MeshRenderer.GetInstanceID()
        private readonly Dictionary<int, PerTargetBlas> _perTargetBlas = new();

        // Identifies a single submesh within a NativeRayTracingTarget.
        private readonly struct SubmeshRef
        {
            public readonly NativeRayTracingTarget Target;
            public readonly int                    SubIndex;

            public SubmeshRef(NativeRayTracingTarget t, int s)
            {
                Target   = t;
                SubIndex = s;
            }
        }

        private sealed class SceneBuildPlan
        {
            public readonly List<SubmeshRef>             StaticOpaque      = new();
            public readonly List<SubmeshRef>             StaticTransparent = new();
            public readonly List<SubmeshRef>             StaticEmissive    = new();
            public readonly List<NativeRayTracingTarget> Dynamic           = new();
        }

        // ----- SkinnedMeshRenderer tracking -----
        // Keyed by SkinnedMeshRenderer.GetInstanceID()
        private sealed class SkinnedEntry
        {
            public SkinnedMeshRenderer                   smr;
            public List<RayTracingAccelerationStructure> tlasList;
            public Matrix4x4                             lastRootTransform;

            // GPU buffer slot tracking (populated by AddSkinnedInstance)
            public uint   firstInstanceDataIndex;
            public int    submeshCount;
            public uint[] morphPrimitiveOffsets; // per-submesh offset into _morphPrimitivePositionsPrevBuf
            public uint[] primitiveOffsets; // per-submesh offset into _primitiveDataBuf
            public int[]  primitiveCounts; // per-submesh triangle count
            public int    indexStride; // 2 (UInt16) or 4 (UInt32)
        }

        private readonly Dictionary<int, SkinnedEntry> _skinnedInstances = new();

        // ----- Separate-BLAS incremental slot management -----
        // One allocator per buffer; both track element-level free/used ranges.
        private readonly PrimitiveSlotAllocator _instanceAlloc = new PrimitiveSlotAllocator { Name = "InstanceAlloc" };
        private readonly PrimitiveSlotAllocator _primAlloc     = new PrimitiveSlotAllocator { Name = "PrimitiveAlloc" };

        // Next free slot in _morphPrimitivePositionsPrevBuf; reset on scene rebuild.
        private uint _morphPrimCursor;

        // Fragmentation thresholds: trigger MarkRebuildDirty when both conditions are met.
        private const float kFragThreshold    = 0.5f;
        private const uint  kFragMinFreeCount = 10_000;

        // ----- Tracking for dirty-detection -----
        private readonly Dictionary<Material, int> _materialSlots = new();

        // Reference counts per material slot (how many submeshes reference it).
        private readonly Dictionary<Material, int> _materialRefCounts = new();

        // Freed material slot indices available for reuse (each slot = TexturesPerMaterial descriptors).
        private readonly Queue<int> _freeMatSlots = new();

        private bool _sceneDirty = true;
        private bool _disposed;

        public RayTracingAccelerationStructure WorldAS => _worldAS;
        public RayTracingAccelerationStructure LightAS => _lightAS;

        public NativeStructuredBuffer InstanceDataBuf => _instanceDataBuf;
        public GraphicsBuffer PrimitiveDataBuf => _primitiveDataBuf;
        public GraphicsBuffer MorphPrimitivePositionsPrevBuf => _morphPrimitivePositionsPrevBuf;

        public IntPtr InstanceDataBufPtr { get; private set; }
        public IntPtr PrimitiveDataBufPtr { get; private set; }
        public IntPtr MorphPrimitivePositionsPrevBufPtr { get; private set; }

        public BindlessTexture Textures => _textures;

        public GraphicsBuffer HashEntriesBuffer => _sharcHashEntries;
        public GraphicsBuffer AccumulationBuffer => _sharcAccumulated;
        public GraphicsBuffer ResolvedBuffer => _sharcResolved;

        // Cached native pointers (valid for the lifetime of the buffers, set once in AllocateStaticResources).
        public IntPtr HashEntriesBufferPtr { get; private set; }
        public IntPtr AccumulationBufferPtr { get; private set; }
        public IntPtr ResolvedBufferPtr { get; private set; }

        public NRDSampleResource()
        {
            _worldAS = new RayTracingAccelerationStructure();
            _lightAS = new RayTracingAccelerationStructure();
            AllocateStaticResources();
        }

        public void MarkRebuildDirty() => _sceneDirty = true;

        /// <summary>打印 _instanceCpu 和 _textures 的全部信息到 Console。</summary>
        public void PrintDebugInfo()
        {
            var sb = new System.Text.StringBuilder();

            // ---------- _instanceCpu ----------
            if (_instanceCpu == null)
            {
                sb.AppendLine("[NRDSampleResource] _instanceCpu: null");
            }
            else
            {
                sb.AppendLine($"[NRDSampleResource] _instanceCpu: {_instanceCpu.Length} entries");
                for (int i = 0; i < _instanceCpu.Length; i++)
                {
                    var inst = _instanceCpu[i];
                    // sb.AppendLine($"  [{i}] mOverloaded0={inst.mOverloadedMatrix0}  mOverloaded1={inst.mOverloadedMatrix1}  mOverloaded2={inst.mOverloadedMatrix2}");
                    // sb.AppendLine($"  [{i}] mOverloaded0={inst.mOverloadedMatrix0}  mOverloaded1={inst.mOverloadedMatrix1}  mOverloaded2={inst.mOverloadedMatrix2}");
                    sb.AppendLine($"  [{i}] textureOffsetAndFlags=0x{inst.textureOffsetAndFlags:X8}  primitiveOffset={inst.primitiveOffset}  scale={inst.scale}  morphPrimitiveOffset={inst.morphPrimitiveOffset}");
                }
            }

            // ---------- _textures ----------
            if (_textures == null)
            {
                sb.AppendLine("[NRDSampleResource] _textures: null");
            }
            else
            {
                sb.AppendLine($"[NRDSampleResource] _textures: capacity={_textures.Capacity}  handle=0x{_textures.Handle:X}  isValid={_textures.IsValid}");
                for (int i = 0; i < _textures.Capacity; i++)
                {
                    var tex = _textures[i];
                    sb.AppendLine($"  [{i}] {(tex != null ? $"{tex.name} ({tex.GetType().Name}) dim={tex.dimension} {tex.width}x{tex.height}" : "<null>")}");
                }
            }

            Debug.Log(sb.ToString());
        }

        /// <summary>Dirty detection + full scene rebuild when needed.</summary>
        public void UpdateForFrame()
        {
            var targets = NativeRayTracingTarget.All;

            if (_sceneDirty)
            {
                RebuildScene(targets);
                foreach (var t in targets)
                    if (t != null)
                        t.transform.hasChanged = false;
                _sceneDirty = false;

                NativeRayTracingTarget.RemoveQueue.Clear();
                NativeRayTracingTarget.AddQueue.Clear();
                NativeRayTracingSkinnedTarget.RemoveQueue.Clear();
                NativeRayTracingSkinnedTarget.AddQueue.Clear();
                return;
            }

            DrainChangeQueue();

            UpdateTransformsOnly(targets);

            UpdateSkinnedInstances();
        }

        // =====================================================================
        // SkinnedMeshRenderer public API
        // =====================================================================

        /// <summary>
        /// Registers a <see cref="SkinnedMeshRenderer"/> into <see cref="WorldAS"/> (and optionally
        /// <see cref="LightAS"/> when the representative material is emissive). The BLAS is rebuilt
        /// every frame via <see cref="UpdateSkinnedInstances"/>.
        /// </summary>
        public void AddSkinnedInstance(SkinnedMeshRenderer smr, NativeRayTracingSkinnedTarget stTarget)
        {
            if (smr == null) return;
            int id = smr.GetInstanceID();
            if (_skinnedInstances.ContainsKey(id))
            {
                Debug.LogWarning($"[NRDSampleResource] SkinnedMeshRenderer '{smr.name}' already registered.");
                return;
            }

            Mesh mesh = smr.sharedMesh;
            if (mesh == null) return;

            if (!_worldAS.AddInstance(smr))
            {
                Debug.LogError($"[NRDSampleResource] AddSkinnedInstance: AddInstance failed for '{smr.name}'");
                return;
            }

            // GPU-skinned vertices are in root-bone space (or smr.transform space if no rootBone).
            // Use that transform so the TLAS places geometry correctly in world space.
            Matrix4x4 xform = GetSkinnedRootTransform(smr);
            _worldAS.SetInstanceTransform(smr, xform);

            var  smi           = stTarget?.SubmeshMaterialInfos;
            bool isEmissive    = smi != null && smi.Length > 0 && smi[0].isEmissive;
            bool isTransparent = smi != null && smi.Length > 0 && smi[0].isTransparent;
            uint baseFlags     = isTransparent ? FLAG_TRANSPARENT : FLAG_NON_TRANSPARENT;
            _worldAS.SetInstanceMask(smr, GetMaskForFlags(baseFlags));

            var tlasList = new List<RayTracingAccelerationStructure> { _worldAS };

            if (isEmissive)
            {
                if (_lightAS.AddInstance(smr))
                {
                    _lightAS.SetInstanceTransform(smr, xform);
                    _lightAS.SetInstanceMask(smr, GetMaskForFlags(baseFlags));
                    tlasList.Add(_lightAS);
                }
            }

            // ------------------------------------------------------------------
            // Allocate GPU buffer slots and initialise PrimitiveData / InstanceData.
            // ------------------------------------------------------------------
            int subCnt = mesh.subMeshCount;

            // Count total triangles so we can pre-grow the primitive buffer.
            int totalTris = 0;
            for (int s = 0; s < subCnt; s++)
                totalTris += (int)mesh.GetIndexCount(s) / 3;

            // Allocate a contiguous block of instance slots.
            uint instBase = _instanceAlloc.Allocate(subCnt);
            if (instBase == PrimitiveSlotAllocator.InvalidOffset)
            {
                int newInstCap = Mathf.Max((int)_instanceAlloc.Capacity * 2,
                    (int)_instanceAlloc.Capacity + subCnt);
                EnsureInstanceCapacity(newInstCap);
                instBase = _instanceAlloc.Allocate(subCnt);
            }

            _worldAS.SetInstanceID(smr, instBase);
            if (isEmissive)
                _lightAS.SetInstanceID(smr, instBase);

            // Ensure primitive buffer has room for all triangles.
            if (_primAlloc.TotalFreeCount < (uint)totalTris)
            {
                int newPrimCap = Mathf.Max((int)_primAlloc.Capacity * 2,
                    (int)_primAlloc.Capacity + totalTris);
                EnsurePrimitiveCapacity(newPrimCap);
            }

            // Ensure morph primitive buffer has room.
            EnsureMorphPrimCapacity(_morphPrimCursor + (uint)totalTris);

            // Uniform scale from the root transform (for worldArea scaling in shader).
            Vector3 sc = new Vector3(
                new Vector3(xform.m00, xform.m10, xform.m20).magnitude,
                new Vector3(xform.m01, xform.m11, xform.m21).magnitude,
                new Vector3(xform.m02, xform.m12, xform.m22).magnitude);
            float scaleMax   = Mathf.Max(sc.x, Mathf.Max(sc.y, sc.z));
            bool  leftHanded = xform.determinant < 0f;

            // Read base-mesh data (rest pose) for UV / worldArea initialisation.
            var meshDataArr = Mesh.AcquireReadOnlyMeshData(mesh);
            var meshData    = meshDataArr[0];

            int vertCount = mesh.vertexCount;
            var nativePos = new NativeArray<Vector3>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
            var nativeN   = new NativeArray<Vector3>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
            var nativeT   = new NativeArray<Vector4>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
            var nativeUV  = new NativeArray<Vector2>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

            meshData.GetVertices(nativePos);
            bool hasN  = meshData.HasVertexAttribute(VertexAttribute.Normal);
            bool hasT  = meshData.HasVertexAttribute(VertexAttribute.Tangent);
            bool hasUV = meshData.HasVertexAttribute(VertexAttribute.TexCoord0);
            if (hasN) meshData.GetNormals(nativeN);
            if (hasT) meshData.GetTangents(nativeT);
            if (hasUV) meshData.GetUVs(0, nativeUV);

            var posF3 = nativePos.Reinterpret<float3>(sizeof(float) * 3);
            var norF3 = nativeN.Reinterpret<float3>(sizeof(float) * 3);
            var tanF4 = nativeT.Reinterpret<float4>(sizeof(float) * 4);
            var uvF2  = nativeUV.Reinterpret<float2>(sizeof(float) * 2);

            var jobHandles = new List<JobHandle>(subCnt);
            var tempTris   = new List<NativeArray<int>>(subCnt);

            Material[] sharedMats      = smr.sharedMaterials;
            var        subPrimOffsets  = new uint[subCnt];
            var        subPrimCounts   = new int[subCnt];
            var        subMorphOffsets = new uint[subCnt];

            for (int sub = 0; sub < subCnt; sub++)
            {
                if (sub >= sharedMats.Length)
                {
                    Debug.LogError($"[NRDSampleResource] Submesh {sub} of skinned '{smr.name}' has no material assigned; skipping submesh");
                    continue;
                }

                int subMatIdx = GetOrAddMaterial(stTarget.SubmeshMaterialInfos[sub], null);

                int indexCount = (int)mesh.GetIndexCount(sub);
                int triCount   = indexCount / 3;

                uint primOffset  = _primAlloc.Allocate(triCount);
                uint morphOffset = _morphPrimCursor;
                _morphPrimCursor += (uint)triCount;

                subPrimOffsets[sub]  = primOffset;
                subPrimCounts[sub]   = triCount;
                subMorphOffsets[sub] = morphOffset;

                // Schedule Burst job to initialise UV / worldArea from the rest-pose mesh.
                // Normals + tangents will be overwritten every frame by the compute shader.
                // For skinned meshes, we skip writing normals/tangents here because:
                // - Rest-pose normals are in local space
                // - But mObjectToWorld is root-bone's localToWorldMatrix
                // - UpdateSkinnedPrimitives.compute will write root-bone space normals
                var nativeTris = new NativeArray<int>(indexCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
                meshData.GetIndices(nativeTris, sub);
                tempTris.Add(nativeTris);

                jobHandles.Add(new BuildPrimitivesJob
                {
                    Indices   = nativeTris,
                    Positions = posF3,
                    Normals   = norF3,
                    Tangents  = tanF4,
                    UVs       = uvF2,
                    HasN      = false, // Skip normals for skinned meshes (wrong space)
                    HasT      = false, // Skip tangents for skinned meshes (wrong space)
                    HasUV     = hasUV,
                    Output    = _primitiveCpu.GetSubArray((int)primOffset, triCount),
                }.Schedule(triCount, 64));

                // Build InstanceDataNRD for this sub-mesh.
                uint instSlot         = instBase + (uint)sub;
                uint baseTextureIndex = (uint)(subMatIdx * TexturesPerMaterial);
                // FLAG_MORPH: shader uses gIn_MorphPrimitivePositionsPrev for Xprev;
                //             mOverloaded holds prev root-bone-to-world (written each frame).
                // On the first frame mOverloaded = identity → Xprev = X (no motion).
                _instanceCpu[instSlot] = new InstanceDataNRD
                {
                    mOverloadedMatrix0    = new Vector4(1f, 0f, 0f, 0f),
                    mOverloadedMatrix1    = new Vector4(0f, 1f, 0f, 0f),
                    mOverloadedMatrix2    = new Vector4(0f, 0f, 1f, 0f),
                    textureOffsetAndFlags = baseTextureIndex | ((baseFlags | FLAG_MORPH) << FlagFirstBit),
                    primitiveOffset       = primOffset,
                    scale                 = (leftHanded ? -1f : 1f) * scaleMax,
                    morphPrimitiveOffset  = morphOffset,
                };
                EncodeMaterial(stTarget.SubmeshMaterialInfos[sub], ref _instanceCpu[instSlot]);
            }

            // Wait for all Burst jobs.
            var allHandles                                           = new NativeArray<JobHandle>(jobHandles.Count, Allocator.Temp);
            for (int i = 0; i < jobHandles.Count; i++) allHandles[i] = jobHandles[i];
            JobHandle.CombineDependencies(allHandles).Complete();
            allHandles.Dispose();

            foreach (var arr in tempTris) arr.Dispose();
            nativePos.Dispose();
            nativeN.Dispose();
            nativeT.Dispose();
            nativeUV.Dispose();
            meshDataArr.Dispose();

            // Partial GPU uploads.
            _instanceDataBuf.UploadRange(_instanceCpu, (int)instBase, subCnt);
            for (int sub = 0; sub < subCnt; sub++)
                _primitiveDataBuf.SetData(_primitiveCpu,
                    (int)subPrimOffsets[sub], (int)subPrimOffsets[sub], subPrimCounts[sub]);
            InstanceDataBufPtr  = _instanceDataBuf.NativePtr;
            PrimitiveDataBufPtr = _primitiveDataBuf.GetNativeBufferPtr();

            // Note: vertexBufferTarget and skinnedMotionVectors are now set in
            // NativeRayTracingSkinnedTarget.OnEnable() to ensure they're set before the first render.

            _skinnedInstances[id] = new SkinnedEntry
            {
                smr                    = smr,
                tlasList               = tlasList,
                lastRootTransform      = xform,
                firstInstanceDataIndex = instBase,
                submeshCount           = subCnt,
                morphPrimitiveOffsets  = subMorphOffsets,
                primitiveOffsets       = subPrimOffsets,
                primitiveCounts        = subPrimCounts,
                indexStride            = mesh.indexFormat == IndexFormat.UInt16 ? 2 : 4,
            };
        }

        /// <summary>Removes a previously registered <see cref="SkinnedMeshRenderer"/> from all TLASes.</summary>
        public void RemoveSkinnedInstance(SkinnedMeshRenderer smr)
        {
            if (smr == null) return;
            int id = smr.GetInstanceID();
            if (!_skinnedInstances.TryGetValue(id, out var entry)) return;

            foreach (var tlas in entry.tlasList)
                tlas.RemoveInstance(smr);

            _skinnedInstances.Remove(id);
        }

        // =====================================================================
        // Private: per-frame skinned update
        // =====================================================================

        /// <summary>
        /// Returns the transform that maps GPU-skinned vertices (root-bone space) to world space.
        /// Unity's skinning uses <c>bone.localToWorldMatrix * bindpose</c>, so outputs are in
        /// root-bone local space. If no root bone is assigned, falls back to the renderer's own transform.
        /// </summary>
        private static Matrix4x4 GetSkinnedRootTransform(SkinnedMeshRenderer smr)
        {
            Transform root = smr.rootBone != null ? smr.rootBone : smr.transform;
            return root.localToWorldMatrix;
        }

        private void UpdateSkinnedInstances()
        {
            bool anyInstChanged = false;
            // Debug.Log("UpdateSkinnedInstances " + _skinnedInstances.Count);

            foreach (var kv in _skinnedInstances)
            {
                var entry = kv.Value;
                var smr   = entry.smr;
                if (smr == null) continue;

                // // Refresh the GPU vertex buffer to the current frame's skinning result.
                // foreach (var tlas in entry.tlasList)
                //     tlas.UpdateSkinnedInstance(smr);

                // Store the PREVIOUS frame's root-bone-to-world transform into mOverloaded.
                // The FLAG_MORPH path in HLSL uses mOverloaded to transform Xprev from
                // local (root-bone) space to world space: Xprev = AffineTransform(mOverloaded, prevLocalPos).
                if (_instanceCpu != null && entry.submeshCount > 0)
                {
                    Matrix4x4 prevXform = entry.lastRootTransform;
                    for (int sub = 0; sub < entry.submeshCount; sub++)
                    {
                        int idx = (int)entry.firstInstanceDataIndex + sub;
                        if (idx >= _instanceCpu.Length) break;
                        _instanceCpu[idx].mOverloadedMatrix0 = new Vector4(prevXform.m00, prevXform.m01, prevXform.m02, prevXform.m03);
                        _instanceCpu[idx].mOverloadedMatrix1 = new Vector4(prevXform.m10, prevXform.m11, prevXform.m12, prevXform.m13);
                        _instanceCpu[idx].mOverloadedMatrix2 = new Vector4(prevXform.m20, prevXform.m21, prevXform.m22, prevXform.m23);
                    }

                    anyInstChanged = true;
                }

                // Update TLAS instance transform and save current transform for next frame.
                Matrix4x4 rootXform = GetSkinnedRootTransform(smr);
                if (rootXform != entry.lastRootTransform)
                {
                    foreach (var tlas in entry.tlasList)
                        tlas.SetInstanceTransform(smr, rootXform);
                }

                // CRITICAL: Always update lastRootTransform so it's used as prevXform next frame
                entry.lastRootTransform = rootXform;
            }

            if (anyInstChanged && _instanceCpu != null && _instanceDataBuf != null)
            {
                foreach (var kv in _skinnedInstances)
                {
                    var entry = kv.Value;
                    if (entry.submeshCount > 0)
                        _instanceDataBuf.UploadRange(_instanceCpu,
                            (int)entry.firstInstanceDataIndex,
                            entry.submeshCount);
                }

                InstanceDataBufPtr = _instanceDataBuf.NativePtr;
            }
        }

        /// <summary>
        /// Records compute-shader dispatches that, for each registered <see cref="SkinnedMeshRenderer"/>:
        /// <list type="bullet">
        ///   <item>Writes current-frame normals + tangents into <see cref="_primitiveDataBuf"/>.</item>
        ///   <item>Writes previous-frame vertex positions into <see cref="_morphPrimitivePositionsPrevBuf"/>.</item>
        /// </list>
        /// Call this every frame inside a <see cref="CommandBuffer"/> <b>before</b> the ray-tracing dispatches,
        /// passing the <c>UpdateSkinnedPrimitives.compute</c> shader asset.
        /// </summary>
        public void RecordSkinnedMorphUpdate(CommandBuffer cmd, ComputeShader cs)
        {
            if (cs == null || _skinnedInstances.Count == 0) return;
            if (_primitiveDataBuf == null || _morphPrimitivePositionsPrevBuf == null) return;

            int kernel = cs.FindKernel("UpdateSkinnedPrimitives");

            // Buffers that are the same for all SMRs - use CommandBuffer API
            cmd.SetComputeBufferParam(cs, kernel, "gInOut_PrimitiveData", _primitiveDataBuf);
            cmd.SetComputeBufferParam(cs, kernel, "gOut_MorphPrimitivePositions", _morphPrimitivePositionsPrevBuf);

            foreach (var kv in _skinnedInstances)
            {
                var entry = kv.Value;
                var smr   = entry.smr;
                if (smr == null || entry.submeshCount == 0) continue;

                Mesh mesh = smr.sharedMesh;
                if (mesh == null) continue;

                GraphicsBuffer vbCurr = smr.GetVertexBuffer();
                if (vbCurr == null) continue;

                // GetPreviousVertexBuffer() returns null on the very first frame or when
                // skinnedMotionVectors is off; fall back to the current buffer (Xprev = X).
                GraphicsBuffer vbPrev = smr.GetPreviousVertexBuffer() ?? vbCurr;

                // Expose the index buffer as a ByteAddressBuffer for the compute shader.
                mesh.indexBufferTarget |= GraphicsBuffer.Target.Raw;
                GraphicsBuffer ib = mesh.GetIndexBuffer();
                if (ib == null) continue;

                int vertexStride = mesh.GetVertexBufferStride(0);
                int posOff       = mesh.GetVertexAttributeOffset(VertexAttribute.Position);
                int normOff      = mesh.GetVertexAttributeOffset(VertexAttribute.Normal);
                int tanOff       = mesh.GetVertexAttributeOffset(VertexAttribute.Tangent);

                if (normOff < 0 || tanOff < 0)
                {
                    // Mesh has no normals or tangents — skip morph update for this SMR.
                    ib.Release();
                    continue;
                }

                cmd.SetComputeBufferParam(cs, kernel, "gIn_VB_Curr", vbCurr);
                cmd.SetComputeBufferParam(cs, kernel, "gIn_VB_Prev", vbPrev);
                cmd.SetComputeBufferParam(cs, kernel, "gIn_IB", ib);

                cmd.SetComputeIntParam(cs, "gVertexStride", vertexStride);
                cmd.SetComputeIntParam(cs, "gPositionOffset", posOff);
                cmd.SetComputeIntParam(cs, "gNormalOffset", normOff);
                cmd.SetComputeIntParam(cs, "gTangentOffset", tanOff);
                cmd.SetComputeIntParam(cs, "gIndexStride", entry.indexStride);


                for (int sub = 0; sub < entry.submeshCount; sub++)
                {
                    var sm              = mesh.GetSubMesh(sub);
                    int indexByteOffset = sm.indexStart * entry.indexStride;
                    int triCount        = entry.primitiveCounts[sub];
                    if (triCount <= 0) continue;

                    cmd.SetComputeIntParam(cs, "gIndexOffset", indexByteOffset);
                    cmd.SetComputeIntParam(cs, "gBaseVertex", sm.baseVertex);
                    cmd.SetComputeIntParam(cs, "gNumPrimitives", triCount);
                    cmd.SetComputeIntParam(cs, "gPrimitiveOffset", (int)entry.primitiveOffsets[sub]);
                    cmd.SetComputeIntParam(cs, "gMorphPrimitiveOffset", (int)entry.morphPrimitiveOffsets[sub]);

                    cmd.DispatchCompute(cs, kernel, (triCount + 63) / 64, 1, 1);
                }

                ib.Release();
            }
        }

        /// <summary>
        /// Drains all pending <see cref="TargetAddEvent"/> / <see cref="TargetRemoveEvent"/> events.
        /// <para>
        /// In play mode, static objects trigger a full scene rebuild (they belong to merged BLASes).
        /// Dynamic objects are handled incrementally without FLAG_STATIC.
        /// </para>
        /// <para>In editor non-play mode all objects are treated as dynamic separate BLASes.</para>
        /// </summary>
        private void DrainChangeQueue()
        {
            // Process removals first so freed slots can be reused by additions.
            while (NativeRayTracingTarget.RemoveQueue.Count > 0)
            {
                var ev = NativeRayTracingTarget.RemoveQueue.Dequeue();

                if (!_perTargetBlas.ContainsKey(ev.RendererInstanceId))
                {
                    // Not tracked in separate BLAS → was part of a merged (static) BLAS; need full rebuild.
                    _sceneDirty = true;
                }
                else
                {
                    RemoveTargetIncremental(ev.RendererInstanceId);
                }
            }

            while (NativeRayTracingTarget.AddQueue.Count > 0)
            {
                var ev = NativeRayTracingTarget.AddQueue.Dequeue();

                if (ev.Target == null || ev.Renderer == null) continue;

                // Static objects in play mode live in merged BLASes → full rebuild required.
                if (ShouldMerge() && (ev.Target.IsStatic))
                {
                    _sceneDirty = true;
                    continue;
                }

                // Dynamic objects (or any object in edit mode): add incrementally without FLAG_STATIC.
                AddTargetIncremental(ev.Target);
            }

            // ---- Skinned add/remove events ----
            while (NativeRayTracingSkinnedTarget.RemoveQueue.Count > 0)
            {
                var ev = NativeRayTracingSkinnedTarget.RemoveQueue.Dequeue();
                if (_skinnedInstances.TryGetValue(ev.RendererInstanceId, out var entry))
                {
                    foreach (var tlas in entry.tlasList)
                        tlas.RemoveInstance(entry.smr);
                    _skinnedInstances.Remove(ev.RendererInstanceId);
                }
            }

            while (NativeRayTracingSkinnedTarget.AddQueue.Count > 0)
            {
                var ev = NativeRayTracingSkinnedTarget.AddQueue.Dequeue();
                if (ev.Target == null || ev.Renderer == null) continue;
                AddSkinnedInstance(ev.Renderer, ev.Target);
            }

            // Auto-schedule full rebuild when fragmentation becomes excessive.
            if (_primAlloc.FragmentationRatio > kFragThreshold &&
                _primAlloc.TotalFreeCount > kFragMinFreeCount)
            {
                Debug.Log($"[NRDSampleResource] Primitive buffer fragmentation {_primAlloc.FragmentationRatio:P0} — scheduling full rebuild.");
                _sceneDirty = true;
            }
        }

        /// <summary>Build / update both TLASes (call inside a CommandBuffer).</summary>
        public void FlushPendingCopies(CommandBuffer cmd)
        {
            _instanceDataBuf?.FlushPendingCopies(cmd);
        }

        public void BuildAccelerationStructures(CommandBuffer cmd)
        {
            // Flush any staged instance-data writes into the GPU-resident buffer before
            // the acceleration-structure build (and subsequent ray-tracing dispatches) read it.

            _worldAS.BuildOrUpdate(cmd);
            _lightAS.BuildOrUpdate(cmd);
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

            HashEntriesBufferPtr  = _sharcHashEntries.GetNativeBufferPtr();
            AccumulationBufferPtr = _sharcAccumulated.GetNativeBufferPtr();
            ResolvedBufferPtr     = _sharcResolved.GetNativeBufferPtr();

            // Allocate a 1-element stub so the shader binding is never null.
            // EnsureMorphPrimCapacity() will grow this when skinned instances are registered.
            _morphPrimitivePositionsPrevBuf = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.Raw, 1,
                Marshal.SizeOf<MorphPrimitivePositionsNRD>());
            MorphPrimitivePositionsPrevBufPtr = _morphPrimitivePositionsPrevBuf.GetNativeBufferPtr();
        }

        private void DisposeStaticResources()
        {
            _sharcHashEntries?.Release();
            _sharcHashEntries = null;
            _sharcAccumulated?.Release();
            _sharcAccumulated = null;
            _sharcResolved?.Release();
            _sharcResolved = null;
        }

        // =====================================================================
        // Dynamic scene GPU data + merged BLAS construction
        // =====================================================================

        private void DisposeSceneGpuBuffers(bool preserveTextures = false)
        {
            _instanceDataBuf?.Dispose();
            _instanceDataBuf = null;
            _primitiveDataBuf?.Release();
            _primitiveDataBuf = null;

            // Morph primitive positions buffer is sized to match registered skinned instances;
            // release here so it is rebuilt fresh after the next scene build.
            _morphPrimitivePositionsPrevBuf?.Release();
            _morphPrimitivePositionsPrevBuf = null;
            _morphPrimCursor                = 0;

            if (!preserveTextures)
            {
                _textures?.Dispose();
                _textures = null;
            }

            _blasOpaque?.Dispose();
            _blasOpaque = null;
            _blasTransparent?.Dispose();
            _blasTransparent = null;
            _blasEmissive?.Dispose();
            _blasEmissive = null;

            _perTargetBlas.Clear();

            // Remove all skinned instances from both TLASes and clear tracking.
            foreach (var kv in _skinnedInstances)
            {
                foreach (var tlas in kv.Value.tlasList)
                    tlas.RemoveInstance(kv.Value.smr);
            }

            _skinnedInstances.Clear();

            _instanceCpu = null;
            if (_primitiveCpu.IsCreated)
            {
                _primitiveCpu.Dispose();
                _primitiveCpu = default;
            }

            _instanceAlloc.Reset(0);
            _primAlloc.Reset(0);

            if (!preserveTextures)
            {
                _materialSlots.Clear();
                _materialRefCounts.Clear();
                _freeMatSlots.Clear();
            }
        }

        /// <summary>
        /// Classifies targets into static (merged BLAS in play mode) and dynamic (separate BLAS),
        /// builds GPU resources, and registers all BLASes with the appropriate TLAS.
        ///
        /// Play mode:
        ///   Unity-static objects → merged pre-transformed BLASes with FLAG_STATIC (mOverloaded = identity).
        ///   Dynamic objects → separate BLAS per renderer, no FLAG_STATIC (mOverloaded = identity first frame).
        ///
        /// Editor non-play mode:
        ///   All objects → separate BLAS with FLAG_STATIC (mOverloaded = current transform for normals).
        /// </summary>
        private void RebuildScene(IReadOnlyList<NativeRayTracingTarget> targets, bool preserveTextures = false)
        {
            DisposeSceneGpuBuffers(preserveTextures);

            _morphPrimitivePositionsPrevBuf   = new GraphicsBuffer(GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.Raw, 1, Marshal.SizeOf<MorphPrimitivePositionsNRD>());
            MorphPrimitivePositionsPrevBufPtr = _morphPrimitivePositionsPrevBuf.GetNativeBufferPtr();

            _worldAS?.Clear();
            _lightAS?.Clear();

            bool mergeStatics = ShouldMerge();

            var plan = BuildScenePlan(targets, mergeStatics);

            var totalPrims = CountSceneTriangles(plan);

            _primitiveCpu = new NativeArray<PrimitiveDataNRD>(Mathf.Max(totalPrims, 1), Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

            uint primitiveCursor = 0;
            uint instanceCursor  = 0;

            var instList = new List<InstanceDataNRD>();
            var texPtrs  = new List<Texture>();

            if (mergeStatics)
            {
                BuildMergedBlases(
                    plan,
                    ref instanceCursor,
                    ref primitiveCursor,
                    instList,
                    texPtrs,
                    out uint staticOpaqueFirstInstance,
                    out uint staticTransparentFirstInstance,
                    out uint staticEmissiveFirstInstance);
                RegisterMergedBlases(staticOpaqueFirstInstance, staticTransparentFirstInstance, staticEmissiveFirstInstance);
            }

            ProcessSeparateGroup(plan.Dynamic, ref instanceCursor, ref primitiveCursor, instList, texPtrs);

            UploadTextureArray(texPtrs, preserveTextures);

            UploadSceneGeometryBuffers(instList);

            InitializeIncrementalAllocators(totalPrims);
            RegisterSkinnedTargets();
        }

        private void InitializeIncrementalAllocators(int totalPrims)
        {
            _instanceAlloc.ResetFullyAllocated(_instanceCpu.Length);
            _primAlloc.ResetFullyAllocated(Mathf.Max(totalPrims, 1));
        }

        private void RegisterSkinnedTargets()
        {
            foreach (var st in NativeRayTracingSkinnedTarget.All)
            {
                if (st != null)
                    AddSkinnedInstance(st.GetComponent<SkinnedMeshRenderer>(), st);
            }
        }

        private static SceneBuildPlan BuildScenePlan(IReadOnlyList<NativeRayTracingTarget> targets, bool mergeStatics)
        {
            var plan = new SceneBuildPlan();

            foreach (var t in targets)
            {
                if (t == null) continue;
                var mr = t.GetComponent<MeshRenderer>();
                if (mr == null) continue;
                var mf = mr.GetComponent<MeshFilter>();
                if (mf == null || mf.sharedMesh == null) continue;

                bool goesToMerged = mergeStatics && t.IsStatic;
                if (!goesToMerged)
                {
                    plan.Dynamic.Add(t);
                    continue;
                }

                Material[] mats   = mr.sharedMaterials;
                Mesh       mesh   = mf.sharedMesh;
                int        subCnt = mesh.subMeshCount;
                for (int s = 0; s < subCnt; s++)
                {
                    if (s >= mats.Length)
                    {
                        Debug.LogError($"[NRDSampleResource] Submesh {s} of '{mr.name}' has no material assigned; skipping submesh");
                        continue;
                    }

                    bool isTrans    = t.SubmeshMaterialInfos[s].isTransparent;
                    bool isEmissive = t.SubmeshMaterialInfos[s].isEmissive;

                    var sr = new SubmeshRef(t, s);
                    if (isTrans)
                        plan.StaticTransparent.Add(sr);
                    else
                        plan.StaticOpaque.Add(sr);

                    if (isEmissive) plan.StaticEmissive.Add(sr);
                }
            }

            return plan;
        }

        private static int CountSceneTriangles(SceneBuildPlan plan)
        {
            return CountGroupTriangles(plan.StaticOpaque)
                   + CountGroupTriangles(plan.StaticTransparent)
                   + CountGroupTriangles(plan.StaticEmissive)
                   + CountGroupTriangles(plan.Dynamic);
        }

        private static void LogSceneTriangleCounts(SceneBuildPlan plan, int totalPrims)
        {
            Debug.Log(
                $"[RebuildScene] Total triangles: {totalPrims}  (Opaque: {CountGroupTriangles(plan.StaticOpaque)}, Transparent: {CountGroupTriangles(plan.StaticTransparent)}, Emissive: {CountGroupTriangles(plan.StaticEmissive)}, Dynamic: {CountGroupTriangles(plan.Dynamic)})");
        }

        private void BuildMergedBlases(
            SceneBuildPlan plan,
            ref uint instanceCursor,
            ref uint primitiveCursor,
            List<InstanceDataNRD> instList,
            List<Texture> texPtrs,
            out uint staticOpaqueFirstInstance,
            out uint staticTransparentFirstInstance,
            out uint staticEmissiveFirstInstance)
        {
            staticOpaqueFirstInstance      = instanceCursor;
            staticTransparentFirstInstance = 0;
            staticEmissiveFirstInstance    = 0;

            _blasOpaque = BuildMergedBlas(plan.StaticOpaque, ref instanceCursor, ref primitiveCursor, instList, _primitiveCpu, texPtrs, FLAG_STATIC | FLAG_NON_TRANSPARENT);
            Debug.Log($"Opaque Num {instanceCursor - staticOpaqueFirstInstance} instances, {primitiveCursor} primitives");

            staticTransparentFirstInstance = instanceCursor;
            _blasTransparent               = BuildMergedBlas(plan.StaticTransparent, ref instanceCursor, ref primitiveCursor, instList, _primitiveCpu, texPtrs, FLAG_STATIC | FLAG_TRANSPARENT);
            Debug.Log($"Transparent Num {instanceCursor - staticTransparentFirstInstance} instances, {primitiveCursor - staticOpaqueFirstInstance} primitives");

            staticEmissiveFirstInstance = instanceCursor;
            _blasEmissive               = BuildMergedBlas(plan.StaticEmissive, ref instanceCursor, ref primitiveCursor, instList, _primitiveCpu, texPtrs, FLAG_STATIC | FLAG_NON_TRANSPARENT);
            Debug.Log($"Emissive Num {instanceCursor - staticEmissiveFirstInstance} instances, {primitiveCursor - staticTransparentFirstInstance} primitives");
        }

        private void RegisterMergedBlases(
            uint staticOpaqueFirstInstance,
            uint staticTransparentFirstInstance,
            uint staticEmissiveFirstInstance)
        {
            if (_blasOpaque != null)
                _worldAS.RegisterMergedBlas(_blasOpaque, kHandleOpaque, staticOpaqueFirstInstance, (byte)FLAG_NON_TRANSPARENT);
            if (_blasTransparent != null)
                _worldAS.RegisterMergedBlas(_blasTransparent, kHandleTransparent, staticTransparentFirstInstance, (byte)FLAG_TRANSPARENT);
            if (_blasEmissive != null)
                _lightAS.RegisterMergedBlas(_blasEmissive, kHandleEmissive, staticEmissiveFirstInstance, (byte)FLAG_NON_TRANSPARENT);
        }

        private void UploadTextureArray(List<Texture> texPtrs, bool preserveTextures)
        {
            if (preserveTextures) return;

            int texCount = Mathf.Max(texPtrs.Count, 1);
            _textures = new BindlessTexture(texCount);
            for (int i = 0; i < texPtrs.Count; i++)
                _textures[i] = texPtrs[i];
        }

        private void UploadSceneGeometryBuffers(List<InstanceDataNRD> instList)
        {
            if (instList.Count == 0) instList.Add(default);
            _instanceCpu = instList.ToArray();

            _instanceDataBuf = new NativeStructuredBuffer(_instanceCpu.Length, Marshal.SizeOf<InstanceDataNRD>());
            _instanceDataBuf.UploadRange(_instanceCpu, 0, _instanceCpu.Length);
            InstanceDataBufPtr = _instanceDataBuf.NativePtr;

            Debug.Log($"Geometries {_instanceCpu.Length}");

            _primitiveDataBuf = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.Raw,
                _primitiveCpu.Length, Marshal.SizeOf<PrimitiveDataNRD>());
            _primitiveDataBuf.SetData(_primitiveCpu);
            PrimitiveDataBufPtr = _primitiveDataBuf.GetNativeBufferPtr();
            Debug.Log($"Primitives {_primitiveCpu.Length}");
        }

        /// <summary>
        /// Non-merged path: one BLAS per MeshRenderer (using the mesh's native VB/IB).
        /// TLAS transform = localToWorldMatrix. InstanceID = firstSubmeshIndex in _instanceDataBuf.
        /// </summary>
        // Count triangles for a group without building full validPairs — used to pre-allocate _primitiveCpu.
        private static int CountGroupTriangles(List<NativeRayTracingTarget> group)
        {
            int count = 0;
            foreach (var t in group)
            {
                if (t == null) continue;
                var mf = t.GetComponent<MeshFilter>();
                if (mf == null || mf.sharedMesh == null) continue;
                Mesh mesh = mf.sharedMesh;
                for (int s = 0; s < mesh.subMeshCount; s++)
                    count += (int)mesh.GetIndexCount(s) / 3;
            }

            return count;
        }

        // Submesh-granularity overload used by merged BLAS paths.
        private static int CountGroupTriangles(List<SubmeshRef> group)
        {
            int count = 0;
            foreach (var sr in group)
            {
                if (sr.Target == null) continue;
                var mf = sr.Target.GetComponent<MeshFilter>();
                if (mf == null || mf.sharedMesh == null) continue;
                count += (int)mf.sharedMesh.GetIndexCount(sr.SubIndex) / 3;
            }

            return count;
        }

        private void ProcessSeparateGroup(
            List<NativeRayTracingTarget> group,
            ref uint instanceCursor,
            ref uint primitiveCursor,
            List<InstanceDataNRD> instList,
            List<Texture> texPtrs)
        {
            var swTotal = Stopwatch.StartNew();

            // First pass: collect valid pairs and build mesh list.
            var validPairs    = new List<(NativeRayTracingTarget target, MeshRenderer mr, Mesh mesh, int meshIndex)>(group.Count);
            var meshList      = new List<Mesh>(group.Count);
            int totalTriCount = 0;

            foreach (var t in group)
            {
                if (t == null) continue;
                var mr = t.GetComponent<MeshRenderer>();
                if (mr == null) continue;
                var mf = mr.GetComponent<MeshFilter>();
                if (mf == null || mf.sharedMesh == null) continue;

                Mesh mesh = mf.sharedMesh;
                for (int s = 0; s < mesh.subMeshCount; s++)
                    totalTriCount += (int)mesh.GetIndexCount(s) / 3;

                validPairs.Add((t, mr, mesh, meshList.Count));
                meshList.Add(mesh);
            }

            if (validPairs.Count == 0)
            {
                swTotal.Stop();
                Debug.Log($"[ProcessSeparateGroup] Total: {swTotal.Elapsed.TotalMilliseconds:F2} ms  (0 renderers)");
                return;
            }

            // Single AcquireReadOnlyMeshData call for all meshes.
            var meshDataArr = Mesh.AcquireReadOnlyMeshData(meshList);

            // Per-job tracking — kept alive until all jobs Complete().
            var jobHandles = new List<JobHandle>(validPairs.Count * 4);
            var tempArrays = new List<IDisposable>(validPairs.Count * 5);

            foreach (var (target, mr, mesh, meshIndex) in validPairs)
            {
                Matrix4x4 xform       = target.transform.localToWorldMatrix;
                int       mrId        = mr.GetInstanceID();
                uint      indexStride = mesh.indexFormat == IndexFormat.UInt16 ? 2u : 4u;

                List<SubmeshGroup> groups = BuildSubmeshGroupsFromDescs(target, mrId);
                if (groups.Count == 0) continue;

                bool leftHanded = xform.determinant < 0f;
                Vector3 sc = new Vector3(
                    new Vector3(xform.m00, xform.m10, xform.m20).magnitude,
                    new Vector3(xform.m01, xform.m11, xform.m21).magnitude,
                    new Vector3(xform.m02, xform.m12, xform.m22).magnitude);
                float scaleMax = Mathf.Max(sc.x, Mathf.Max(sc.y, sc.z));

                int vertCount = mesh.vertexCount;
                var meshData  = meshDataArr[meshIndex];

                // Vertex attribute arrays — TempJob so they stay valid across job scheduling.
                var nativePos = new NativeArray<Vector3>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
                var nativeN   = new NativeArray<Vector3>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
                var nativeT   = new NativeArray<Vector4>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
                var nativeUV  = new NativeArray<Vector2>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

                meshData.GetVertices(nativePos);
                bool hasN  = meshData.HasVertexAttribute(VertexAttribute.Normal);
                bool hasT  = meshData.HasVertexAttribute(VertexAttribute.Tangent);
                bool hasUV = meshData.HasVertexAttribute(VertexAttribute.TexCoord0);
                if (hasN) meshData.GetNormals(nativeN);
                if (hasT) meshData.GetTangents(nativeT);
                if (hasUV) meshData.GetUVs(0, nativeUV);

                var posF3 = nativePos.Reinterpret<float3>(sizeof(float) * 3);
                var norF3 = nativeN.Reinterpret<float3>(sizeof(float) * 3);
                var tanF4 = nativeT.Reinterpret<float4>(sizeof(float) * 4);
                var uvF2  = nativeUV.Reinterpret<float2>(sizeof(float) * 2);

                tempArrays.Add(nativePos);
                tempArrays.Add(nativeN);
                tempArrays.Add(nativeT);
                tempArrays.Add(nativeUV);

                bool anyGroupRegistered = false;

                foreach (var grp in groups)
                {
                    var  groupDescs = BuildGroupSubmeshDescs(mesh, grp, indexStride);
                    uint groupFlags = grp.isTransparent ? FLAG_TRANSPARENT : FLAG_NON_TRANSPARENT;
                    byte groupMask  = GetMaskForFlags(groupFlags);

                    var ommPinnedHandles = new List<GCHandle>();
                    var groupOmmDescs    = BuildGroupOmmDescs(target, grp.submeshIndices, ommPinnedHandles);
                    try
                    {
                        if (!RegisterSeparateBlasGroup(mr, mesh, grp, groupDescs, groupOmmDescs, xform, instanceCursor, groupMask, "ProcessSeparateGroup"))
                            continue;
                    }
                    finally
                    {
                        foreach (var h in ommPinnedHandles) h.Free();
                    }

                    // Build per-submesh primitive + instance data for this group.
                    for (int gi = 0; gi < grp.submeshIndices.Length; gi++)
                    {
                        int sub = grp.submeshIndices[gi];

                        uint subFlags  = grp.isTransparent ? FLAG_TRANSPARENT : FLAG_NON_TRANSPARENT;
                        int  subMatIdx = GetOrAddMaterial(target.SubmeshMaterialInfos[sub], texPtrs);

                        int indexCount = (int)mesh.GetIndexCount(sub);
                        int triCount   = indexCount / 3;

                        grp.primitiveOffsets[gi] = primitiveCursor;
                        grp.primitiveCounts[gi]  = triCount;

                        var nativeTris = new NativeArray<int>(indexCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
                        meshData.GetIndices(nativeTris, sub);
                        tempArrays.Add(nativeTris);

                        jobHandles.Add(new BuildPrimitivesJob
                        {
                            Indices   = nativeTris,
                            Positions = posF3,
                            Normals   = norF3,
                            Tangents  = tanF4,
                            UVs       = uvF2,
                            HasN      = hasN,
                            HasT      = hasT,
                            HasUV     = hasUV,
                            Output    = _primitiveCpu.GetSubArray((int)primitiveCursor, triCount),
                        }.Schedule(triCount, 64));

                        primitiveCursor += (uint)triCount;

                        uint      baseTextureIndex = (uint)(subMatIdx * TexturesPerMaterial);
                        Matrix4x4 mOverloaded      = xform;
                        var inst = new InstanceDataNRD
                        {
                            mOverloadedMatrix0    = new Vector4(mOverloaded.m00, mOverloaded.m01, mOverloaded.m02, mOverloaded.m03),
                            mOverloadedMatrix1    = new Vector4(mOverloaded.m10, mOverloaded.m11, mOverloaded.m12, mOverloaded.m13),
                            mOverloadedMatrix2    = new Vector4(mOverloaded.m20, mOverloaded.m21, mOverloaded.m22, mOverloaded.m23),
                            textureOffsetAndFlags = baseTextureIndex | (subFlags << FlagFirstBit),
                            primitiveOffset       = grp.primitiveOffsets[gi],
                            scale                 = (leftHanded ? -1f : 1f) * scaleMax,
                            morphPrimitiveOffset  = 0,
                        };
                        EncodeMaterial(target.SubmeshMaterialInfos[sub], ref inst);
                        instList.Add(inst);
                        instanceCursor++;
                    }

                    anyGroupRegistered = true;
                }

                if (!anyGroupRegistered) continue;

                // Record per-target state for transform-only updates.
                _perTargetBlas[mrId] = new PerTargetBlas
                {
                    groups        = groups,
                    lastTransform = xform,
                    isStatic      = false,
                };
            }

            // Wait for all Burst jobs to finish.
            var allHandles = new NativeArray<JobHandle>(jobHandles.Count, Allocator.Temp);
            for (int i = 0; i < jobHandles.Count; i++)
                allHandles[i] = jobHandles[i];
            JobHandle.CombineDependencies(allHandles).Complete();
            allHandles.Dispose();

            // Dispose all temporary vertex / index arrays.
            foreach (var arr in tempArrays)
                arr.Dispose();

            meshDataArr.Dispose();

            swTotal.Stop();
            Debug.Log($"[ProcessSeparateGroup] Total: {swTotal.Elapsed.TotalMilliseconds:F2} ms  ({validPairs.Count} renderers, {totalTriCount} tris)");
        }

        /// <summary>
        /// Per-frame transform update for separate-BLAS entries.
        ///
        /// Static entries (editor non-play mode, FLAG_STATIC set):
        ///   Only patched when the transform actually changed.
        ///   mOverloaded = current localToWorld (HLSL uses it as rotation matrix for normals).
        ///
        /// Dynamic entries (play mode, FLAG_STATIC clear):
        ///   Written every frame regardless of movement so Xprev is always valid.
        ///   mOverloaded = prevT * inv(currT)  (motion matrix; identity when stationary).
        ///   HLSL: Xprev = AffineTransform(mOverloaded, X) = correct previous world position.
        /// </summary>
        private void UpdateTransformsOnly(IReadOnlyList<NativeRayTracingTarget> targets)
        {
            if (_instanceCpu == null) return;

            foreach (var target in targets)
            {
                if (target == null) continue;
                var mr = target.meshRenderer;
                if (mr == null) continue;

                if (!_perTargetBlas.TryGetValue(target.instanceId, out var info)) continue;

                bool moved = target.transform.hasChanged;
                if (moved)
                {
                    target.transform.hasChanged = false; // reset for next frame
                }

                if (info.isStatic)
                {
                    // Edit-mode static: skip unless the transform actually changed.
                    if (!moved) continue;

                    Matrix4x4 xform = target.transform.localToWorldMatrix;
                    foreach (var grp in info.groups)
                    {
                        foreach (var tlas in grp.tlasList)
                            tlas.SetInstanceTransform(grp.customHandle, xform);

                        for (int gi = 0; gi < grp.submeshIndices.Length; gi++)
                        {
                            int idx = (int)grp.firstInstanceIdx + gi;
                            if (idx >= _instanceCpu.Length) break;
                            _instanceCpu[idx].mOverloadedMatrix0 = new Vector4(xform.m00, xform.m01, xform.m02, xform.m03);
                            _instanceCpu[idx].mOverloadedMatrix1 = new Vector4(xform.m10, xform.m11, xform.m12, xform.m13);
                            _instanceCpu[idx].mOverloadedMatrix2 = new Vector4(xform.m20, xform.m21, xform.m22, xform.m23);
                        }
                    }

                    info.lastTransform = xform;
                }
                else
                {
                    // Dynamic: compute motion matrix = prevT * inv(currT).
                    if (moved || info.wasMoving)
                    {
                        Matrix4x4 xform        = target.transform.localToWorldMatrix;
                        Matrix4x4 motionMatrix = info.lastTransform * xform.inverse;

                        Vector3 s = new Vector3(
                            new Vector3(xform.m00, xform.m10, xform.m20).magnitude,
                            new Vector3(xform.m01, xform.m11, xform.m21).magnitude,
                            new Vector3(xform.m02, xform.m12, xform.m22).magnitude);
                        float scaleMax   = Mathf.Max(s.x, Mathf.Max(s.y, s.z));
                        bool  leftHanded = xform.determinant < 0f;

                        foreach (var grp in info.groups)
                        {
                            foreach (var tlas in grp.tlasList)
                                tlas.SetInstanceTransform(grp.customHandle, xform);

                            for (int gi = 0; gi < grp.submeshIndices.Length; gi++)
                            {
                                int idx = (int)grp.firstInstanceIdx + gi;
                                if (idx >= _instanceCpu.Length) break;
                                _instanceCpu[idx].mOverloadedMatrix0 = new Vector4(motionMatrix.m00, motionMatrix.m01, motionMatrix.m02, motionMatrix.m03);
                                _instanceCpu[idx].mOverloadedMatrix1 = new Vector4(motionMatrix.m10, motionMatrix.m11, motionMatrix.m12, motionMatrix.m13);
                                _instanceCpu[idx].mOverloadedMatrix2 = new Vector4(motionMatrix.m20, motionMatrix.m21, motionMatrix.m22, motionMatrix.m23);
                                _instanceCpu[idx].scale              = (leftHanded ? -1f : 1f) * scaleMax;
                            }

                            // Upload changed instance slots for this group.
                            _instanceDataBuf.UploadRange(_instanceCpu, (int)grp.firstInstanceIdx, grp.submeshIndices.Length);
                        }

                        info.lastTransform = xform;
                        info.wasMoving     = moved;
                    }
                }
            }
        }

        // =====================================================================
        // Incremental add / remove (separate-BLAS mode only)
        // =====================================================================

        /// <summary>
        /// Diffs <paramref name="current"/> against <see cref="_registeredTargets"/>, removes
        /// <summary>
        /// Removes a single target's TLAS instance and returns its GPU slots to the free pools.
        /// </summary>
        private void RemoveTargetIncremental(int rendererInstanceId)
        {
            if (!_perTargetBlas.TryGetValue(rendererInstanceId, out var info)) return;

            foreach (var grp in info.groups)
            {
                foreach (var tlas in grp.tlasList)
                    tlas.RemoveInstance(grp.customHandle);

                // Return contiguous instance slots.
                int grpSubCount = grp.submeshIndices.Length;
                _instanceAlloc.Free(grp.firstInstanceIdx, grpSubCount);
                for (int gi = 0; gi < grpSubCount; gi++)
                {
                    int slotIdx = (int)grp.firstInstanceIdx + gi;
                    if (slotIdx < _instanceCpu.Length)
                        _instanceCpu[slotIdx] = default;
                }

                // Return per-submesh primitive slots.
                for (int gi = 0; gi < grp.primitiveOffsets.Length; gi++)
                    _primAlloc.Free(grp.primitiveOffsets[gi], grp.primitiveCounts[gi]);

                // Release material reference counts.
                foreach (var mat in grp.materials)
                    ReleaseMaterial(mat);
            }

            _perTargetBlas.Remove(rendererInstanceId);

            // Upload zeroed instance slots. We do per-group uploads since slots may not be contiguous.
            foreach (var grp in info.groups)
                _instanceDataBuf?.UploadRange(_instanceCpu, (int)grp.firstInstanceIdx, grp.submeshIndices.Length);

            InstanceDataBufPtr = _instanceDataBuf?.NativePtr ?? IntPtr.Zero;
        }

        /// <summary>
        /// Adds a single target to <paramref name="tlas"/> without touching any other renderer.
        /// Allocates contiguous instance and primitive slots, grows backing buffers if needed,
        /// schedules a Burst job for primitive data, then partial-uploads changed ranges to the GPU.
        /// </summary>
        private void AddTargetIncremental(NativeRayTracingTarget target)
        {
            if (target == null) return;
            var mr = target.GetComponent<MeshRenderer>();
            if (mr == null) return;
            var mf = mr.GetComponent<MeshFilter>();
            if (mf == null || mf.sharedMesh == null) return;

            Mesh mesh        = mf.sharedMesh;
            int  subCnt      = mesh.subMeshCount;
            int  mrId        = mr.GetInstanceID();
            uint indexStride = mesh.indexFormat == IndexFormat.UInt16 ? 2u : 4u;

            List<SubmeshGroup> groups = BuildSubmeshGroupsFromDescs(target, mrId);
            if (groups.Count == 0) return;

            Matrix4x4 xform = target.transform.localToWorldMatrix;

            bool leftHanded = xform.determinant < 0f;
            Vector3 sc = new Vector3(
                new Vector3(xform.m00, xform.m10, xform.m20).magnitude,
                new Vector3(xform.m01, xform.m11, xform.m21).magnitude,
                new Vector3(xform.m02, xform.m12, xform.m22).magnitude);
            float scaleMax = Mathf.Max(sc.x, Mathf.Max(sc.y, sc.z));

            // ------------------------------------------------------------------
            // Compute total submesh count across all groups to pre-allocate instance slots.
            // ------------------------------------------------------------------
            int totalSubCount = subCnt; // same as mesh.subMeshCount

            uint instBase = _instanceAlloc.Allocate(totalSubCount);
            if (instBase == PrimitiveSlotAllocator.InvalidOffset)
            {
                int newInstCap = Mathf.Max((int)_instanceAlloc.Capacity * 2,
                    (int)_instanceAlloc.Capacity + totalSubCount);
                EnsureInstanceCapacity(newInstCap);
                instBase = _instanceAlloc.Allocate(totalSubCount);
            }

            // ------------------------------------------------------------------
            // Build per-submesh primitive + instance data.
            // ------------------------------------------------------------------
            var meshDataArr = Mesh.AcquireReadOnlyMeshData(mesh);
            var meshData    = meshDataArr[0];

            int vertCount = mesh.vertexCount;
            var nativePos = new NativeArray<Vector3>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
            var nativeN   = new NativeArray<Vector3>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
            var nativeT   = new NativeArray<Vector4>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
            var nativeUV  = new NativeArray<Vector2>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

            meshData.GetVertices(nativePos);
            bool hasN  = meshData.HasVertexAttribute(VertexAttribute.Normal);
            bool hasT  = meshData.HasVertexAttribute(VertexAttribute.Tangent);
            bool hasUV = meshData.HasVertexAttribute(VertexAttribute.TexCoord0);
            if (hasN) meshData.GetNormals(nativeN);
            if (hasT) meshData.GetTangents(nativeT);
            if (hasUV) meshData.GetUVs(0, nativeUV);

            var posF3 = nativePos.Reinterpret<float3>(sizeof(float) * 3);
            var norF3 = nativeN.Reinterpret<float3>(sizeof(float) * 3);
            var tanF4 = nativeT.Reinterpret<float4>(sizeof(float) * 4);
            var uvF2  = nativeUV.Reinterpret<float2>(sizeof(float) * 2);

            var jobHandles = new List<JobHandle>(subCnt);
            var tempTris   = new List<NativeArray<int>>(subCnt);

            // Pre-calculate total tri count and grow _primitiveCpu once.
            {
                int totalTriCount = 0;
                for (int s = 0; s < subCnt; s++)
                    totalTriCount += (int)(mesh.GetIndexCount(s) / 3);
                if (_primAlloc.TotalFreeCount < (uint)totalTriCount)
                {
                    int newPrimCap = Mathf.Max((int)_primAlloc.Capacity * 2,
                        (int)_primAlloc.Capacity + totalTriCount);
                    EnsurePrimitiveCapacity(newPrimCap);
                }
            }

            // Track the next instance slot — groups are processed in order, submeshes in order.
            uint instCursor = instBase;

            foreach (var grp in groups)
            {
                var groupDescs = BuildGroupSubmeshDescs(mesh, grp, indexStride);

                uint groupFlags = grp.isTransparent ? FLAG_TRANSPARENT : FLAG_NON_TRANSPARENT;
                byte groupMask  = GetMaskForFlags(groupFlags);

                var ommPinnedHandles = new List<GCHandle>();
                var groupOmmDescs    = BuildGroupOmmDescs(target, grp.submeshIndices, ommPinnedHandles);
                try
                {
                    if (!RegisterSeparateBlasGroup(mr, mesh, grp, groupDescs, groupOmmDescs, xform, instCursor, groupMask, "AddTargetIncremental"))
                        continue;
                }
                finally
                {
                    foreach (var h in ommPinnedHandles) h.Free();
                }

                for (int gi = 0; gi < grp.submeshIndices.Length; gi++)
                {
                    int sub = grp.submeshIndices[gi];

                    uint subFlags  = grp.isTransparent ? FLAG_TRANSPARENT : FLAG_NON_TRANSPARENT;
                    int  subMatIdx = GetOrAddMaterial(target.SubmeshMaterialInfos[sub], null);

                    int indexCount = (int)mesh.GetIndexCount(sub);
                    int triCount   = indexCount / 3;

                    uint primOffset = _primAlloc.Allocate(triCount);
                    grp.primitiveOffsets[gi] = primOffset;
                    grp.primitiveCounts[gi]  = triCount;

                    var nativeTris = new NativeArray<int>(indexCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
                    meshData.GetIndices(nativeTris, sub);
                    tempTris.Add(nativeTris);

                    jobHandles.Add(new BuildPrimitivesJob
                    {
                        Indices   = nativeTris,
                        Positions = posF3,
                        Normals   = norF3,
                        Tangents  = tanF4,
                        UVs       = uvF2,
                        HasN      = hasN,
                        HasT      = hasT,
                        HasUV     = hasUV,
                        Output    = _primitiveCpu.GetSubArray((int)primOffset, triCount),
                    }.Schedule(triCount, 64));

                    uint instSlot         = instCursor;
                    uint baseTextureIndex = (uint)(subMatIdx * TexturesPerMaterial);
                    _instanceCpu[instSlot] = new InstanceDataNRD
                    {
                        mOverloadedMatrix0    = new Vector4(1f, 0f, 0f, 0f),
                        mOverloadedMatrix1    = new Vector4(0f, 1f, 0f, 0f),
                        mOverloadedMatrix2    = new Vector4(0f, 0f, 1f, 0f),
                        textureOffsetAndFlags = baseTextureIndex | (subFlags << FlagFirstBit),
                        primitiveOffset       = primOffset,
                        scale                 = (leftHanded ? -1f : 1f) * scaleMax,
                        morphPrimitiveOffset  = 0,
                    };
                    EncodeMaterial(target.SubmeshMaterialInfos[sub], ref _instanceCpu[instSlot]);
                    instCursor++;
                }
            }

            // Wait for all Burst jobs.
            var allHandles                                           = new NativeArray<JobHandle>(jobHandles.Count, Allocator.Temp);
            for (int i = 0; i < jobHandles.Count; i++) allHandles[i] = jobHandles[i];
            JobHandle.CombineDependencies(allHandles).Complete();
            allHandles.Dispose();

            foreach (var arr in tempTris) arr.Dispose();
            nativePos.Dispose();
            nativeN.Dispose();
            nativeT.Dispose();
            nativeUV.Dispose();
            meshDataArr.Dispose();

            // Partial GPU uploads — only the ranges we touched.
            _instanceDataBuf.UploadRange(_instanceCpu, (int)instBase, totalSubCount);
            foreach (var grp in groups)
                for (int gi = 0; gi < grp.submeshIndices.Length; gi++)
                    _primitiveDataBuf.SetData(_primitiveCpu,
                        (int)grp.primitiveOffsets[gi], (int)grp.primitiveOffsets[gi], grp.primitiveCounts[gi]);

            InstanceDataBufPtr  = _instanceDataBuf.NativePtr;
            PrimitiveDataBufPtr = _primitiveDataBuf.GetNativeBufferPtr();

            _perTargetBlas[mrId] = new PerTargetBlas
            {
                groups        = groups,
                lastTransform = xform,
                isStatic      = false,
            };
        }

        /// <summary>
        /// Grows <see cref="_morphPrimitivePositionsPrevBuf"/> to hold at least
        /// <paramref name="requiredCount"/> elements, preserving existing content.
        /// </summary>
        private void EnsureMorphPrimCapacity(uint requiredCount)
        {
            int needed = (int)requiredCount;
            if (_morphPrimitivePositionsPrevBuf != null && _morphPrimitivePositionsPrevBuf.count >= needed)
                return;

            int cap = _morphPrimitivePositionsPrevBuf != null
                ? Mathf.Max(needed, _morphPrimitivePositionsPrevBuf.count * 2)
                : Mathf.Max(needed, 64);

            _morphPrimitivePositionsPrevBuf?.Release();
            _morphPrimitivePositionsPrevBuf = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.Raw, cap,
                Marshal.SizeOf<MorphPrimitivePositionsNRD>());
            MorphPrimitivePositionsPrevBufPtr = _morphPrimitivePositionsPrevBuf.GetNativeBufferPtr();
        }

        /// <summary>
        /// Grows <see cref="_instanceCpu"/> and <see cref="_instanceDataBuf"/> to at least
        /// <paramref name="newCapacity"/> elements, and updates <see cref="_instanceAlloc"/>.
        /// </summary>
        private void EnsureInstanceCapacity(int newCapacity)
        {
            if (_instanceCpu != null && _instanceCpu.Length >= newCapacity) return;

            int cap = Mathf.Max(newCapacity, _instanceCpu != null ? _instanceCpu.Length * 2 : newCapacity);

            var newArr = new InstanceDataNRD[cap];
            if (_instanceCpu != null) Array.Copy(_instanceCpu, newArr, _instanceCpu.Length);
            _instanceCpu = newArr;

            _instanceAlloc.GrowTo(cap);

            if (_instanceDataBuf == null)
            {
                _instanceDataBuf = new NativeStructuredBuffer(cap, Marshal.SizeOf<InstanceDataNRD>());
            }
            else
            {
                _instanceDataBuf.Grow(cap);
            }

            _instanceDataBuf.UploadRange(_instanceCpu, 0, _instanceCpu.Length);
            InstanceDataBufPtr = _instanceDataBuf.NativePtr;
        }

        /// <summary>
        /// Grows <see cref="_primitiveCpu"/> and <see cref="_primitiveDataBuf"/> to at least
        /// <paramref name="newCapacity"/> elements, and updates <see cref="_primAlloc"/>.
        /// </summary>
        private void EnsurePrimitiveCapacity(int newCapacity)
        {
            if (_primitiveCpu.IsCreated && _primitiveCpu.Length >= newCapacity) return;

            int cap = Mathf.Max(newCapacity, _primitiveCpu.IsCreated ? _primitiveCpu.Length * 2 : newCapacity);

            var newArr = new NativeArray<PrimitiveDataNRD>(cap, Allocator.Persistent, NativeArrayOptions.ClearMemory);
            if (_primitiveCpu.IsCreated)
            {
                NativeArray<PrimitiveDataNRD>.Copy(_primitiveCpu, newArr, _primitiveCpu.Length);
                _primitiveCpu.Dispose();
            }

            _primitiveCpu = newArr;

            _primAlloc.GrowTo(cap);

            _primitiveDataBuf?.Release();
            _primitiveDataBuf = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.Raw,
                cap, Marshal.SizeOf<PrimitiveDataNRD>());
            // Only upload the actually-used range to avoid overwhelming the D3D12 upload
            // ring buffer when cap doubles to >~50 MB in a single SetData call.
            int usedPrims = (int)_primAlloc.UsedCount;
            if (usedPrims > 0)
                _primitiveDataBuf.SetData(_primitiveCpu, 0, 0, usedPrims);
            PrimitiveDataBufPtr = _primitiveDataBuf.GetNativeBufferPtr();
        }

        // =====================================================================
        // Merged BLAS construction
        // =====================================================================

        /// <summary>
        /// Builds one merged BLAS for the given list of submesh references. Returns null when the list is empty.
        /// Each <see cref="SubmeshRef"/> identifies a single submesh within a target. Targets that contribute
        /// multiple submeshes to this bucket share one vertex buffer entry (vertices are copied once per target).
        /// Writes one <see cref="InstanceDataNRD"/> per submesh and one <see cref="PrimitiveDataNRD"/> per
        /// triangle into <paramref name="instList"/> / <paramref name="primOutput"/>. Advances
        /// <paramref name="instanceCursor"/> and <paramref name="primitiveCursor"/> accordingly.
        /// </summary>
        private MergedBlas BuildMergedBlas(
            List<SubmeshRef> group,
            ref uint instanceCursor,
            ref uint primitiveCursor,
            List<InstanceDataNRD> instList,
            NativeArray<PrimitiveDataNRD> primOutput,
            List<Texture> texPtrs,
            uint baseFlags)
        {
            if (group.Count == 0) return null;

            // Group SubmeshRefs by target so each mesh's vertices are copied only once.
            var targetOrder = new List<(NativeRayTracingTarget target, MeshRenderer mr, Mesh mesh, List<int> subIndices)>();
            var targetIndex = new Dictionary<int, int>(); // instanceID → index in targetOrder

            foreach (var sr in group)
            {
                if (sr.Target == null) continue;
                var mr = sr.Target.GetComponent<MeshRenderer>();
                if (mr == null) continue;
                var mf = mr.GetComponent<MeshFilter>();
                if (mf == null || mf.sharedMesh == null) continue;
                int id = sr.Target.GetInstanceID();
                if (!targetIndex.TryGetValue(id, out int idx))
                {
                    idx             = targetOrder.Count;
                    targetIndex[id] = idx;
                    targetOrder.Add((sr.Target, mr, mf.sharedMesh, new List<int>()));
                }

                var entry = targetOrder[idx];
                entry.subIndices.Add(sr.SubIndex);
                targetOrder[idx] = entry;
            }

            if (targetOrder.Count == 0) return null;

            // First pass – sum sizes.
            var meshList   = new List<Mesh>(targetOrder.Count);
            int totalVerts = 0, totalIndices = 0;
            foreach (var (t, mr, mesh, subIndices) in targetOrder)
            {
                totalVerts += mesh.vertexCount;
                foreach (var s in subIndices)
                    totalIndices += (int)mesh.GetIndexCount(s);
                meshList.Add(mesh);
            }

            if (totalVerts == 0 || totalIndices == 0) return null;

            // Allocate merged VB/IB as NativeArrays so Burst jobs can write into them.
            var mergedPos = new NativeArray<float3>(totalVerts, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
            var mergedIdx = new NativeArray<uint>(totalIndices, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

            var meshDataArr  = Mesh.AcquireReadOnlyMeshData(meshList);
            var submeshDescs = new List<NativeRenderPlugin.SubmeshDesc>();
            var ommCacheList = new List<OMMCache>();
            var jobHandles   = new List<JobHandle>(targetOrder.Count * 4);
            var tempArrays   = new List<IDisposable>(targetOrder.Count * 5);

            int vertBase = 0, iBase = 0;

            for (int mi = 0; mi < targetOrder.Count; mi++)
            {
                var (target, mr, mesh, subIndices) = targetOrder[mi];
                var meshData = meshDataArr[mi];

                Matrix4x4 xform        = target.transform.localToWorldMatrix;
                Matrix4x4 normalMatrix = xform.inverse.transpose;

                int vertCount = mesh.vertexCount;

                // Local vertex attribute arrays (TempJob lifetime spans job scheduling).
                var localPos = new NativeArray<Vector3>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
                var localN   = new NativeArray<Vector3>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
                var localT   = new NativeArray<Vector4>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
                var localUV  = new NativeArray<Vector2>(vertCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

                meshData.GetVertices(localPos);
                bool hasN  = meshData.HasVertexAttribute(VertexAttribute.Normal);
                bool hasT  = meshData.HasVertexAttribute(VertexAttribute.Tangent);
                bool hasUV = meshData.HasVertexAttribute(VertexAttribute.TexCoord0);
                if (hasN) meshData.GetNormals(localN);
                if (hasT) meshData.GetTangents(localT);
                if (hasUV) meshData.GetUVs(0, localUV);

                var posF3 = localPos.Reinterpret<float3>(sizeof(float) * 3);
                var norF3 = localN.Reinterpret<float3>(sizeof(float) * 3);
                var tanF4 = localT.Reinterpret<float4>(sizeof(float) * 4);
                var uvF2  = localUV.Reinterpret<float2>(sizeof(float) * 2);

                tempArrays.Add(localPos);
                tempArrays.Add(localN);
                tempArrays.Add(localT);
                tempArrays.Add(localUV);

                // Transform positions into world space → merged VB slice.
                jobHandles.Add(new TransformVerticesJob
                {
                    LocalPositions = posF3,
                    LocalToWorld   = xform,
                    Output         = mergedPos.GetSubArray(vertBase, vertCount),
                }.Schedule(vertCount, 64));

                Material[] sharedMaterials = mr.sharedMaterials;

                // Only process the submeshes that belong to this BLAS bucket.
                foreach (int sub in subIndices)
                {
                    uint primitiveOffsetForSubMesh = primitiveCursor;

                    if (sub >= sharedMaterials.Length)
                    {
                        Debug.LogError($"[NRDSampleResource] Submesh {sub} of '{mr.name}' has no material assigned; skipping submesh");
                        continue;
                    }

                    Material subMat    = sharedMaterials[sub];
                    int      subMatIdx = GetOrAddMaterial(target.SubmeshMaterialInfos[sub], texPtrs);

                    int indexCount = (int)mesh.GetIndexCount(sub);
                    int triCount   = indexCount / 3;

                    var localTris = new NativeArray<int>(indexCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
                    meshData.GetIndices(localTris, sub);
                    tempArrays.Add(localTris);

                    // Record submesh descriptor (IB offset inside merged IB, before remapping).
                    submeshDescs.Add(new NativeRenderPlugin.SubmeshDesc
                    {
                        indexCount      = (uint)indexCount,
                        indexByteOffset = (uint)(iBase * sizeof(uint)),
                        flags           = target.SubmeshMaterialInfos[sub].isAlphaClip ? 0u : NativeRenderPlugin.SUBMESH_FLAG_GEOMETRY_OPAQUE,
                    });
                    ommCacheList.Add((target.ommCaches != null && sub < target.ommCaches.Length) ? target.ommCaches[sub] : null);

                    // Remap local indices to global merged-VB space.
                    jobHandles.Add(new RemapIndicesJob
                    {
                        LocalIndices = localTris,
                        VertBase     = vertBase,
                        Output       = mergedIdx.GetSubArray(iBase, indexCount),
                    }.Schedule(indexCount, 128));

                    // Compute world-space PrimitiveDataNRD for this submesh.
                    jobHandles.Add(new BuildMergedPrimitivesJob
                    {
                        Indices        = localTris,
                        LocalPositions = posF3,
                        LocalNormals   = norF3,
                        LocalTangents  = tanF4,
                        UVs            = uvF2,
                        HasN           = hasN,
                        HasT           = hasT,
                        HasUV          = hasUV,
                        LocalToWorld   = xform,
                        NormalMatrix   = normalMatrix,
                        Output         = primOutput.GetSubArray((int)primitiveOffsetForSubMesh, triCount),
                    }.Schedule(triCount, 64));

                    iBase           += indexCount;
                    primitiveCursor += (uint)triCount;

                    // InstanceDataNRD is built on the main thread.
                    uint baseTextureIndex = (uint)(subMatIdx * TexturesPerMaterial);
                    var inst = new InstanceDataNRD
                    {
                        // Vertices are already world-space → mOverloadedMatrix encodes identity.
                        mOverloadedMatrix0 = new Vector4(1f, 0f, 0f, 0f),
                        mOverloadedMatrix1 = new Vector4(0f, 1f, 0f, 0f),
                        mOverloadedMatrix2 = new Vector4(0f, 0f, 1f, 0f),

                        textureOffsetAndFlags = baseTextureIndex | (baseFlags << FlagFirstBit),
                        primitiveOffset       = primitiveOffsetForSubMesh,
                        scale                 = 1f,
                        morphPrimitiveOffset  = 0,
                    };
                    EncodeMaterial(target.SubmeshMaterialInfos[sub], ref inst);
                    instList.Add(inst);
                    instanceCursor++;
                }

                vertBase += vertCount;
            }

            // Wait for all jobs.
            var allHandles                                           = new NativeArray<JobHandle>(jobHandles.Count, Allocator.Temp);
            for (int i = 0; i < jobHandles.Count; i++) allHandles[i] = jobHandles[i];
            JobHandle.CombineDependencies(allHandles).Complete();
            allHandles.Dispose();

            foreach (var arr in tempArrays) arr.Dispose();
            meshDataArr.Dispose();

            // Upload merged VB/IB to GPU buffers.
            var blas = new MergedBlas
            {
                vertexCount  = (uint)totalVerts,
                submeshDescs = submeshDescs.ToArray(),
                ommCaches    = ommCacheList.ToArray(),
                vb           = new GraphicsBuffer(GraphicsBuffer.Target.Structured, totalVerts, sizeof(float) * 3),
                ib           = new GraphicsBuffer(GraphicsBuffer.Target.Structured, totalIndices, sizeof(uint)),
            };
            blas.vb.SetData(mergedPos);
            blas.ib.SetData(mergedIdx);

            mergedPos.Dispose();
            mergedIdx.Dispose();

            return blas;
        }

        /// <summary>Passes the merged BLAS's VB/IB pointers to the native AS as a single instance.</summary>

        // =====================================================================
        // Material / texture helpers
        // =====================================================================
        private static NativeRenderPlugin.SubmeshDesc[] BuildGroupSubmeshDescs(
            Mesh mesh,
            SubmeshGroup group,
            uint indexStride)
        {
            var descs = new NativeRenderPlugin.SubmeshDesc[group.submeshIndices.Length];
            for (int gi = 0; gi < group.submeshIndices.Length; gi++)
            {
                int               sub = group.submeshIndices[gi];
                SubMeshDescriptor sd  = mesh.GetSubMesh(sub);
                descs[gi] = new NativeRenderPlugin.SubmeshDesc
                {
                    indexCount      = (uint)sd.indexCount,
                    indexByteOffset = (uint)sd.indexStart * indexStride,
                    baseVertex      = (uint)sd.baseVertex,
                    flags           = group.isAlphaClip ? 0u : NativeRenderPlugin.SUBMESH_FLAG_GEOMETRY_OPAQUE,
                };
            }

            return descs;
        }

        private bool RegisterSeparateBlasGroup(
            MeshRenderer mr,
            Mesh mesh,
            SubmeshGroup group,
            NativeRenderPlugin.SubmeshDesc[] groupDescs,
            NativeRenderPlugin.SubmeshOMMDesc[] groupOmmDescs,
            Matrix4x4 transform,
            uint firstInstanceIndex,
            byte groupMask,
            string logContext)
        {
            if (!_worldAS.AddInstanceGroup(mesh, groupDescs, group.customHandle, groupOmmDescs: groupOmmDescs))
            {
                Debug.LogWarning($"[NRDSampleResource] {logContext}: AddInstanceGroup failed for '{mr.name}' handle={group.customHandle}");
                return false;
            }

            group.firstInstanceIdx = firstInstanceIndex;
            _worldAS.SetInstanceID(group.customHandle, firstInstanceIndex);
            _worldAS.SetInstanceTransform(group.customHandle, transform);
            _worldAS.SetInstanceMask(group.customHandle, groupMask);
            group.tlasList.Add(_worldAS);

            if (group.isEmissive)
            {
                if (_lightAS.AddInstanceGroup(mesh, groupDescs, group.customHandle, groupOmmDescs: groupOmmDescs))
                {
                    _lightAS.SetInstanceID(group.customHandle, firstInstanceIndex);
                    _lightAS.SetInstanceTransform(group.customHandle, transform);
                    _lightAS.SetInstanceMask(group.customHandle, groupMask);
                    group.tlasList.Add(_lightAS);
                }
                else
                {
                    Debug.LogWarning($"[NRDSampleResource] {logContext}: AddInstanceGroup on lightAS failed for '{mr.name}' handle={group.customHandle}");
                }
            }

            return true;
        }

        /// <summary>
        /// Builds a <see cref="NativeRenderPlugin.SubmeshOMMDesc"/> array for the given submesh index
        /// subset of <paramref name="target"/>.ommCaches, pinning the underlying byte arrays.
        /// Returns null if no valid OMM entry exists.  Caller must free <paramref name="pinnedHandles"/>
        /// after the native AddInstanceGroup call completes.
        /// </summary>
        private static NativeRenderPlugin.SubmeshOMMDesc[] BuildGroupOmmDescs(
            NativeRayTracingTarget target,
            int[] submeshIndices,
            List<GCHandle> pinnedHandles)
        {
            if (target.ommCaches == null) return null;
            bool hasAny = false;
            foreach (int s in submeshIndices)
                if (s < target.ommCaches.Length && target.ommCaches[s] != null && target.ommCaches[s].IsValid)
                {
                    hasAny = true;
                    break;
                }

            if (!hasAny) return null;

            var descs = new NativeRenderPlugin.SubmeshOMMDesc[submeshIndices.Length];
            for (int gi = 0; gi < submeshIndices.Length; gi++)
            {
                int      s     = submeshIndices[gi];
                OMMCache cache = (s < target.ommCaches.Length) ? target.ommCaches[s] : null;
                if (cache == null || !cache.IsValid) continue;
                pinnedHandles.Add(GCHandle.Alloc(cache.bakedArrayData, GCHandleType.Pinned));
                pinnedHandles.Add(GCHandle.Alloc(cache.bakedDescArray, GCHandleType.Pinned));
                pinnedHandles.Add(GCHandle.Alloc(cache.bakedIndexBuffer, GCHandleType.Pinned));
                pinnedHandles.Add(GCHandle.Alloc(cache.histogramFlat, GCHandleType.Pinned));
                descs[gi] = new NativeRenderPlugin.SubmeshOMMDesc
                {
                    arrayData      = pinnedHandles[pinnedHandles.Count - 4].AddrOfPinnedObject(),
                    arrayDataSize  = (uint)cache.bakedArrayData.Length,
                    descArray      = pinnedHandles[pinnedHandles.Count - 3].AddrOfPinnedObject(),
                    descArrayCount = cache.bakedDescArrayCount,
                    indexBuffer    = pinnedHandles[pinnedHandles.Count - 2].AddrOfPinnedObject(),
                    indexCount     = cache.bakedIndexCount,
                    indexStride    = cache.bakedIndexStride,
                    histogramFlat  = pinnedHandles[pinnedHandles.Count - 1].AddrOfPinnedObject(),
                    histogramCount = (uint)cache.HistogramEntryCount,
                };
            }

            return descs;
        }

        private static byte GetMaskForFlags(uint flags)
        {
            // Use FLAG bits 0–2 as the visibility mask (matches merged-BLAS usage).
            return (byte)(flags & 0xFF);
        }

        /// <summary>
        /// Returns the material slot index for <paramref name="mat"/>, registering it if new.
        /// <para>
        /// <b>Bulk path</b> (<paramref name="texPtrs"/> != null): appends 4 native texture pointers
        /// to <paramref name="texPtrs"/>; the caller creates <see cref="_textures"/> afterwards.
        /// </para>
        /// <para>
        /// <b>Incremental path</b> (<paramref name="texPtrs"/> == null): writes directly into
        /// <see cref="_textures"/>, growing it via <see cref="BindlessTexture.Resize"/> as needed.
        /// Reuses freed slots from <see cref="_freeMatSlots"/> before appending.
        /// </para>
        /// Always increments the material reference count.
        /// </summary>
        private int GetOrAddMaterial(SubmeshMaterialData matData, List<Texture> texPtrs)
        {
            Material mat = matData?.material;

            if (mat != null && _materialSlots.TryGetValue(mat, out int existingD))
            {
                _materialRefCounts[mat] = (_materialRefCounts.TryGetValue(mat, out int rcD) ? rcD : 0) + 1;
                return existingD;
            }

            int idxD;
            if (_freeMatSlots.Count > 0)
                idxD = _freeMatSlots.Dequeue();
            else if (texPtrs != null)
                idxD = _materialSlots.Count;
            else
                idxD = _textures != null ? _textures.Capacity / TexturesPerMaterial : _materialSlots.Count;

            if (mat != null)
            {
                _materialSlots[mat]     = idxD;
                _materialRefCounts[mat] = 1;
            }

            if (texPtrs != null && matData != null)
            {
                for (int i = 0; i < TexturesPerMaterial; i++)
                {
                    var tex = matData.textures[i];
                    if (tex == null)
                    {
                        switch (i)
                        {
                            case 0: // BaseColor
                                tex = Texture2D.whiteTexture;
                                break;
                            case 1: // metallicRoughness
                                tex = Texture2D.blackTexture;
                                break;
                            case 2: // normalTexture
                                tex = Texture2D.normalTexture;
                                break;
                            case 3: // emission
                                tex = Texture2D.blackTexture;
                                break;
                        }
                    }

                    texPtrs.Add(tex);
                }
            }
            else if (_textures != null && matData != null)
            {
                int base4D = idxD * TexturesPerMaterial;
                int needD  = base4D + TexturesPerMaterial;
                if (needD > _textures.Capacity)
                    _textures.Resize(needD);
                for (int i = 0; i < TexturesPerMaterial; i++)
                {
                    var tex = matData.textures[i];
                    if (tex == null)
                    {
                        switch (i)
                        {
                            case 0: // BaseColor
                                tex = Texture2D.whiteTexture;
                                break;
                            case 1: // metallicRoughness
                                tex = Texture2D.blackTexture;
                                break;
                            case 2: // normalTexture
                                tex = Texture2D.normalTexture;
                                break;
                            case 3: // emission
                                tex = Texture2D.blackTexture;
                                break;
                        }
                    }

                    _textures[base4D + i] = tex;
                }
            }

            return idxD;
        }

        /// <summary>
        /// Decrements the reference count for <paramref name="mat"/>.
        /// When the count reaches zero, the material slot is freed:
        /// its 4 descriptor entries are cleared to null SRVs and the slot index
        /// is enqueued in <see cref="_freeMatSlots"/> for future reuse.
        /// </summary>
        private void ReleaseMaterial(Material mat)
        {
            if (mat == null || !_materialSlots.TryGetValue(mat, out int slotIdx)) return;

            int newRc = _materialRefCounts.GetValueOrDefault(mat, 1) - 1;
            if (newRc > 0)
            {
                _materialRefCounts[mat] = newRc;
                return;
            }

            // Reference count hit zero — free the slot.
            _materialSlots.Remove(mat);
            _materialRefCounts.Remove(mat);
            _freeMatSlots.Enqueue(slotIdx);

            // Write null SRVs so stale GPU resources don't linger.
            if (_textures != null)
            {
                int base4 = slotIdx * TexturesPerMaterial;
                for (int i = 0; i < TexturesPerMaterial; i++)
                    _textures[base4 + i] = null;
            }
        }

        private static void EncodeMaterial(SubmeshMaterialData data, ref InstanceDataNRD inst)
        {
            inst.baseColorAndMetalnessScale.x = new half(data.baseColor.r);
            inst.baseColorAndMetalnessScale.y = new half(data.baseColor.g);
            inst.baseColorAndMetalnessScale.z = new half(data.baseColor.b);
            inst.baseColorAndMetalnessScale.w = new half(data.metallic);

            inst.emissionAndRoughnessScale.x = new half(data.emissionColor.r);
            inst.emissionAndRoughnessScale.y = new half(data.emissionColor.g);
            inst.emissionAndRoughnessScale.z = new half(data.emissionColor.b);
            inst.emissionAndRoughnessScale.w = new half(data.roughnessScale);

            inst.normalUvScale.x = new half(data.normalScale);
            inst.normalUvScale.y = new half(data.normalScale);
        }

        // =====================================================================
        // Submesh-group helpers
        // =====================================================================

        /// <summary>
        /// Encodes a unique TLAS instance handle for a specific submesh group of a MeshRenderer.
        /// High 4 bits = groupIndex (max 16 groups per renderer), low 28 bits = mrInstanceId.
        /// </summary>
        private static uint MakeGroupHandle(int mrInstanceId, int groupIndex)
            => (uint)(mrInstanceId & 0x0FFFFFFF) | ((uint)groupIndex << 28);

        /// <summary>
        /// Builds the internal <see cref="SubmeshGroup"/> list for a target from its pre-computed
        /// <see cref="SubmeshGroupDesc"/> array (populated by <see cref="NativeRayTracingTarget.RebuildMaterialData"/>).
        /// </summary>
        private static List<SubmeshGroup> BuildSubmeshGroupsFromDescs(NativeRayTracingTarget target, int mrId)
        {
            var descs  = target.SubmeshGroups;
            var result = new List<SubmeshGroup>(descs.Length);
            for (int gi = 0; gi < descs.Length; gi++)
            {
                var desc      = descs[gi];
                int subCount  = desc.submeshIndices.Length;
                var materials = new Material[subCount];
                for (int si = 0; si < subCount; si++)
                    materials[si] = desc.materialDatas[si].material;

                result.Add(new SubmeshGroup
                {
                    isTransparent    = desc.isTransparent,
                    isEmissive       = desc.isEmissive,
                    isAlphaClip      = desc.isAlphaClip,
                    submeshIndices   = (int[])desc.submeshIndices.Clone(),
                    materials        = materials,
                    tlasList         = new List<RayTracingAccelerationStructure>(),
                    customHandle     = MakeGroupHandle(mrId, gi),
                    primitiveOffsets = new uint[subCount],
                    primitiveCounts  = new int[subCount],
                });
            }

            return result;
        }
    }
}