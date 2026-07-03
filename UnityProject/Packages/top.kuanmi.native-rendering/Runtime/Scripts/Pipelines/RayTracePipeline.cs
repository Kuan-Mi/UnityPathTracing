using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Rendering;

namespace NativeRender
{
    /// <summary>
    /// A DXR pipeline instance created from a <see cref="RayTraceShader"/> asset.
    /// Owns the native D3D12 pipeline handle and all resource bindings.
    ///
    /// Multiple pipelines can be created from the same <see cref="RayTraceShader"/>,
    /// each with independent resource bindings (textures, buffers, acceleration structures, etc.).
    ///
    /// Lifetime: must be explicitly disposed via <see cref="Dispose"/>.
    /// </summary>
    public sealed class RayTracePipeline : IDisposable
    {
        private ulong               _handle;
        private RayTraceShader      _shader;
        private HitGroupShader[]    _hitGroupShaders; // null when not using multi-blob path
        private bool[]              _hitGroupAnyHit;
        private RootConstantsHint[] _rootConstantsHints;
        private string[]            _rootSRVHints;
        private SamplerHint[]       _samplerHints;
        private NativeBindingLayout _userLayout;   // hand-authored layout; null = auto-generate
        private NativeBindingLayout _layout;       // the effective (mandatory) layout

        /// <summary>True if the underlying D3D12 pipeline is valid and ready to dispatch.</summary>
        public bool IsValid => _handle != 0;

        /// <summary>Opaque native handle (pointer to RayTraceShader). Used by NativeRayTraceDescriptorSet.</summary>
        public ulong Handle => _handle;

        /// <summary>The effective binding layout (hand-authored or auto-generated).</summary>
        public NativeBindingLayout Layout => _layout;

        /// <summary>Number of resource binding slots (= layout item count).</summary>
        public uint SlotCount => _slotCount;

        /// <summary>Maps HLSL variable names to layout slot indices (for NativeRayTraceDescriptorSet).
        /// Built from the import-time reflection JSON against the layout.</summary>
        public IReadOnlyDictionary<string, uint> NameToSlot => _nameToSlot;

        /// <summary>Fired (on the main thread) whenever the pipeline is rebuilt after a hot-reload.</summary>
        public event Action<RayTracePipeline> OnRebuilt;

        private uint                     _slotCount;
        private Dictionary<string, uint> _nameToSlot = new Dictionary<string, uint>();

        // Persisted event data for hit-group-table rebuild dispatches.
        // Ring of 3 slots so consecutive rebuilds don't overwrite a slot the render
        // thread may still be reading from a previous (in-flight) frame.
        private const int                                                 kShtRingSize = 3;
        private       NativeArray<NativeRenderPlugin.ShtRebuildEventData> _shtEventData;
        private       int                                                 _shtRingIndex;

        // Pipeline-owned copy of the last-applied per-geometry variant indices
        // (main-thread master) plus a ring of pinned copies the render thread reads.
        // Owning the copies — instead of pointing the event at the caller's array —
        // survives the caller reallocating its array and lets a hot-reload rebuild
        // re-apply the hit-group table: a freshly built native pipeline has an empty
        // table and suppresses every DispatchRays until RebuildHitGroupTable runs.
        private NativeArray<uint>   _shtVariants;
        private NativeArray<uint>[] _shtVariantRing;
        private bool                _shtNeedsReapply;

        // -------------------------------------------------------------------
        // Construction
        // -------------------------------------------------------------------

        /// <summary>
        /// Creates a new DXR pipeline from the given shader asset.
        /// Root binding hints defined on the asset (via the importer) are applied automatically.
        /// Triggers HLSL compilation if the asset has not been compiled yet.
        /// Throws <see cref="InvalidOperationException"/> if pipeline creation fails.
        /// </summary>
        public RayTracePipeline(RayTraceShader shader)
            : this(shader,
                shader != null ? shader.RootConstantsHints : null,
                shader != null ? shader.RootSRVHints : null)
        {
        }

        public RayTracePipeline(RayTraceShader shader, RootConstantsHint[] rootConstantsHints)
            : this(shader, rootConstantsHints, null)
        {
        }

        public RayTracePipeline(RayTraceShader shader, RootConstantsHint[] rootConstantsHints, string[] rootSRVHints)
            : this(shader, rootConstantsHints, rootSRVHints, null)
        {
        }

        public RayTracePipeline(RayTraceShader shader, NativeBindingLayout sharedLayout)
            : this(shader,
                shader != null ? shader.RootConstantsHints : null,
                shader != null ? shader.RootSRVHints : null,
                sharedLayout)
        {
        }

        public RayTracePipeline(RayTraceShader shader, RootConstantsHint[] rootConstantsHints,
            string[] rootSRVHints, NativeBindingLayout sharedLayout)
        {
            if (shader == null)
                throw new ArgumentNullException(nameof(shader));

            _shader             = shader;
            _rootConstantsHints = rootConstantsHints;
            _rootSRVHints       = rootSRVHints;
            _samplerHints       = shader.ResolveSamplerHints();
            _userLayout         = (sharedLayout != null && !sharedLayout.IsEmpty) ? sharedLayout : null;
            BuildNativeHandle(shader);
            RayTraceShader.OnRecompiled += OnShaderRecompiled;
        }

        /// <summary>
        /// Creates a DXR pipeline from a primary shader (raygen + miss) and one or more
        /// <see cref="HitGroupShader"/> blobs (per-material hit-group permutations).
        /// All blobs are merged into one RTPSO.
        ///
        /// <paramref name="primaryShader"/> must contain at least one RayGeneration and one Miss entry.
        /// <paramref name="hitGroupShaders"/> must not be null or empty; use the single-shader
        /// constructor when no extra hit groups are needed.
        /// </summary>
        public RayTracePipeline(RayTraceShader primaryShader, HitGroupShader[] hitGroupShaders)
            : this(primaryShader, hitGroupShaders,
                primaryShader != null ? primaryShader.RootConstantsHints : null,
                primaryShader != null ? primaryShader.RootSRVHints : null)
        {
        }

        public RayTracePipeline(
            RayTraceShader primaryShader,
            HitGroupShader[] hitGroupShaders,
            RootConstantsHint[] rootConstantsHints,
            string[] rootSRVHints = null,
            bool[] hitGroupAnyHit = null)
            : this(primaryShader, hitGroupShaders, rootConstantsHints, rootSRVHints, hitGroupAnyHit, null)
        {
        }

        public RayTracePipeline(
            RayTraceShader primaryShader,
            HitGroupShader[] hitGroupShaders,
            NativeBindingLayout sharedLayout)
            : this(primaryShader, hitGroupShaders,
                primaryShader != null ? primaryShader.RootConstantsHints : null,
                primaryShader != null ? primaryShader.RootSRVHints : null,
                null,
                sharedLayout)
        {
        }

        public RayTracePipeline(
            RayTraceShader primaryShader,
            HitGroupShader[] hitGroupShaders,
            RootConstantsHint[] rootConstantsHints,
            string[] rootSRVHints,
            bool[] hitGroupAnyHit,
            NativeBindingLayout sharedLayout)
        {
            if (primaryShader == null)
                throw new ArgumentNullException(nameof(primaryShader));
            if (hitGroupShaders == null || hitGroupShaders.Length == 0)
                throw new ArgumentException("Use the single-shader constructor when there are no extra hit groups.",
                    nameof(hitGroupShaders));

            _shader             = primaryShader;
            _hitGroupShaders    = hitGroupShaders;
            _hitGroupAnyHit     = hitGroupAnyHit;
            _rootConstantsHints = rootConstantsHints;
            _rootSRVHints       = rootSRVHints;
            _samplerHints       = primaryShader.ResolveSamplerHints();
            _userLayout         = (sharedLayout != null && !sharedLayout.IsEmpty) ? sharedLayout : null;

            BuildNativeHandleMultiBlob(primaryShader, hitGroupShaders);
            RayTraceShader.OnRecompiled += OnShaderRecompiled;
            HitGroupShader.OnRecompiled += OnHitGroupShaderRecompiled;
        }

        private void BuildNativeHandle(RayTraceShader shader)
        {
            byte[] dxil = shader.GetOrCompileDxil();
            if (dxil == null || dxil.Length == 0)
                throw new InvalidOperationException(
                    $"[RayTracePipeline] Shader compilation failed for: {shader.GetHlslPath()}");

            // Contract first: layout + name map + validation, all C#-side.
            var reflected = ShaderReflectionInfo.Parse(shader.ReflectionJson);
            BuildBindingContract(reflected, shader.name);

            uint flags      = ProfileSupportsOpacityMicromaps(shader.TargetProfile) ? 1u : 0u;
            uint maxPayload = shader.MaxPayloadSizeInBytes;
            Debug.Log($"[RayTracePipeline] Creating pipeline for: {shader.name} (DXIL size: {dxil.Length} bytes, OMM support: {flags != 0}, MaxPayload: {maxPayload})");
            string rayGenName = string.IsNullOrEmpty(shader.RayGenName) ? null : shader.RayGenName;
            string hintsJson  = BuildCreationJson(_layout, null);
            _handle = NativeRenderPlugin.NR_CreateRayTraceShaderFromBytesEx(
                dxil, (uint)dxil.Length, shader.name, flags, maxPayload, rayGenName, hintsJson);
            if (_handle == 0)
                throw new InvalidOperationException(
                    $"[RayTracePipeline] NR_CreateRayTraceShaderFromBytesEx returned 0 for: {shader.name}");
        }

        private void BuildNativeHandleMultiBlob(RayTraceShader primaryShader, HitGroupShader[] hitGroupShaders)
        {
            int        totalBlobs = 1 + hitGroupShaders.Length;
            byte[][]   dxils      = new byte[totalBlobs][];
            GCHandle[] pins       = new GCHandle[totalBlobs];

            try
            {
                // Compile all blobs
                dxils[0] = primaryShader.GetOrCompileDxil();
                if (dxils[0] == null || dxils[0].Length == 0)
                    throw new InvalidOperationException(
                        $"[RayTracePipeline] Compilation failed for primary shader: {primaryShader.GetHlslPath()}");

                for (int i = 0; i < hitGroupShaders.Length; ++i)
                {
                    dxils[i + 1] = hitGroupShaders[i].GetOrCompileDxil();
                    if (dxils[i + 1] == null || dxils[i + 1].Length == 0)
                        throw new InvalidOperationException(
                            $"[RayTracePipeline] Compilation failed for hit-group shader[{i}]: {hitGroupShaders[i].GetHlslPath()}");
                }

                // Pin all byte arrays and build pointer / size arrays
                IntPtr[] ptrs  = new IntPtr[totalBlobs];
                uint[]   sizes = new uint[totalBlobs];
                for (int i = 0; i < totalBlobs; ++i)
                {
                    pins[i]  = GCHandle.Alloc(dxils[i], GCHandleType.Pinned);
                    ptrs[i]  = pins[i].AddrOfPinnedObject();
                    sizes[i] = (uint)dxils[i].Length;
                }

                // Contract first: merge the reflection of every blob (primary +
                // hit-group shaders), then resolve/validate against the layout.
                var reflected = new List<ReflectedBinding>();
                ShaderReflectionInfo.ParseInto(primaryShader.ReflectionJson, reflected);
                for (int i = 0; i < hitGroupShaders.Length; ++i)
                    ShaderReflectionInfo.ParseInto(hitGroupShaders[i].ReflectionJson, reflected);
                BuildBindingContract(reflected, primaryShader.name);

                uint   flags      = ProfileSupportsOpacityMicromaps(primaryShader.TargetProfile) ? 1u : 0u;
                uint   maxPayload = primaryShader.MaxPayloadSizeInBytes;
                string rayGenName = string.IsNullOrEmpty(primaryShader.RayGenName) ? null : primaryShader.RayGenName;

                Debug.Log($"[RayTracePipeline] Creating multi-blob pipeline for '{primaryShader.name}' ({totalBlobs} blobs)");
                string hintsJson = BuildCreationJson(_layout, _hitGroupAnyHit);
                _handle = NativeRenderPlugin.NR_CreateRayTracePipelineFromBlobsEx(
                    ptrs, sizes, (uint)totalBlobs,
                    primaryShader.name, flags, maxPayload, rayGenName, hintsJson);

                if (_handle == 0)
                    throw new InvalidOperationException(
                        $"[RayTracePipeline] NR_CreateRayTracePipelineFromBlobsEx returned 0 for: {primaryShader.name}");
            }
            finally
            {
                foreach (var pin in pins)
                    if (pin.IsAllocated)
                        pin.Free();
            }
        }

        /// <summary>
        /// Returns true if the given DXC target profile supports Opacity Micromaps (lib_6_9 and above).
        /// </summary>
        private static bool ProfileSupportsOpacityMicromaps(string profile)
        {
            // Expected format: "lib_X_Y" — OMM requires SM 6.9+
            if (string.IsNullOrEmpty(profile)) return false;
            // Strip leading "lib_" and parse major.minor
            const string prefix = "lib_";
            if (!profile.StartsWith(prefix, System.StringComparison.OrdinalIgnoreCase)) return false;
            string   version = profile.Substring(prefix.Length); // e.g. "6_9" or "6_6"
            string[] parts   = version.Split('_');
            if (parts.Length < 2) return false;
            if (!int.TryParse(parts[0], out int major) || !int.TryParse(parts[1], out int minor)) return false;
            return major > 6 || (major == 6 && minor >= 9);
        }

        /// <summary>
        /// Establishes the binding contract before the native pipeline exists:
        /// picks (or auto-generates) the layout, resolves every reflected HLSL
        /// binding into it (name → layout slot), and fails loudly — naming the
        /// HLSL variables — when a hand-authored layout doesn't cover the
        /// shaders. C#-side safety net replacing the plugin's runtime reflection.
        /// </summary>
        private void BuildBindingContract(List<ReflectedBinding> reflected, string shaderName)
        {
            _layout = _userLayout ?? NativeBindingLayout.FromReflection(
                reflected, _rootConstantsHints, _rootSRVHints, _samplerHints);

            _slotCount = (uint)_layout.Items.Count;
            _nameToSlot.Clear();

            List<string> missing = null;
            for (int i = 0; i < reflected.Count; i++)
            {
                var b = reflected[i];
                int slot = _layout.ResolveSlot(b);
                if (slot >= 0)
                {
                    _nameToSlot[b.Name] = (uint)slot;
                }
                else if (!(b.RegClass == BindingRegClass.Sampler && _layout.HasStaticSampler(b.Reg, b.Space)))
                {
                    (missing ??= new List<string>()).Add(
                        $"'{b.Name}' ({b.RegClass} reg {b.Reg}, space {b.Space})");
                }
            }

            if (missing != null)
                throw new InvalidOperationException(
                    $"[RayTracePipeline] '{shaderName}': binding layout has no item for " +
                    string.Join(", ", missing));
        }

        private static string BuildCreationJson(NativeBindingLayout layout, bool[] hitGroupAnyHit)
        {
            string layoutJson = layout.BuildCreationJson();
            if (hitGroupAnyHit == null || hitGroupAnyHit.Length == 0) return layoutJson;

            // Splice "hitGroupAnyHit" into the layout's top-level JSON object.
            var sb = new System.Text.StringBuilder(layoutJson, layoutJson.Length + 64);
            sb.Length -= 1; // drop trailing '}'
            sb.Append(",\"hitGroupAnyHit\":[");
            for (int i = 0; i < hitGroupAnyHit.Length; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append(hitGroupAnyHit[i] ? "true" : "false");
            }
            sb.Append("]}");
            return sb.ToString();
        }

        private void OnShaderRecompiled(RayTraceShader shader)
        {
            if (shader != _shader) return;
            RebuildPipeline();
        }

        private void OnHitGroupShaderRecompiled(HitGroupShader shader)
        {
            // Rebuild whenever any of our hit-group blobs changes.
            if (_hitGroupShaders == null) return;
            foreach (var hg in _hitGroupShaders)
                if (hg == shader)
                {
                    RebuildPipeline();
                    return;
                }
        }

        // Guards against re-entrance: BuildNativeHandle* → GetOrCompileDxil →
        // EnsureCompiled fires OnRecompiled when it has to (re)compile, which would
        // re-enter RebuildPipeline mid-rebuild and leak a native pipeline.
        private bool _rebuilding;

        private void RebuildPipeline()
        {
            if (_rebuilding) return;
            _rebuilding = true;
            try
            {
                RebuildPipelineCore();
            }
            finally
            {
                _rebuilding = false;
            }
        }

        private void RebuildPipelineCore()
        {
            if (_handle != 0)
            {
                // Drain the render thread + GPU before swapping handles: in-flight
                // dispatch / SHT-rebuild events still reference the old pipeline, and
                // OnRebuilt makes the descriptor sets free their pinned ring buffers.
                // GL.Flush() alone only queues commands — it does NOT wait for
                // execution (same fix as NativeComputePipeline.OnShaderRecompiled).
                using var flushCmd = new CommandBuffer { name = "NR_GpuFlush" };
                flushCmd.IssuePluginEvent(NativeRenderPlugin.NR_GetGpuFlushEventFunc(), 0);
                Graphics.ExecuteCommandBuffer(flushCmd);
                NativeRenderPlugin.NR_WaitForGpuFlush();
            }

            ulong oldHandle = _handle;
            _handle = 0;
            try
            {
                _samplerHints = _shader.ResolveSamplerHints();
                if (_hitGroupShaders != null && _hitGroupShaders.Length > 0)
                    BuildNativeHandleMultiBlob(_shader, _hitGroupShaders);
                else
                    BuildNativeHandle(_shader);
            }
            catch (Exception e)
            {
                // Keep the previous pipeline so rendering continues with the old shader.
                _handle = oldHandle;
                Debug.LogError($"[RayTracePipeline] Hot-reload rebuild failed for '{_shader.name}' — keeping previous pipeline: {e.Message}");
                return;
            }

            if (oldHandle != 0)
                NativeRenderPlugin.NR_DestroyRayTraceShader(oldHandle);

            // The fresh native pipeline starts with an empty hit-group table and
            // suppresses DispatchRays until it is rebuilt — re-apply the cached
            // variant indices on the next Dispatch.
            _shtNeedsReapply = _shtVariants.IsCreated;

            Debug.Log($"[RayTracePipeline] Rebuilt pipeline for: {_shader.name}");
            OnRebuilt?.Invoke(this);
        }

        // -------------------------------------------------------------------
        // IDisposable
        // -------------------------------------------------------------------

        public void Dispose()
        {
            RayTraceShader.OnRecompiled -= OnShaderRecompiled;
            HitGroupShader.OnRecompiled -= OnHitGroupShaderRecompiled;

            if (_shtEventData.IsCreated) _shtEventData.Dispose();
            if (_shtVariants.IsCreated) _shtVariants.Dispose();
            if (_shtVariantRing != null)
            {
                for (int i = 0; i < _shtVariantRing.Length; i++)
                    if (_shtVariantRing[i].IsCreated)
                        _shtVariantRing[i].Dispose();
                _shtVariantRing = null;
            }

            if (_handle != 0)
            {
                GL.Flush();
                NativeRenderPlugin.NR_DestroyRayTraceShader(_handle);
                _handle = 0;
            }
        }

        // -------------------------------------------------------------------
        // Dispatch
        // -------------------------------------------------------------------

        /// <summary>
        /// Enqueues a DispatchRays call into the CommandBuffer using a NativeRayTraceDescriptorSet.
        /// This is the preferred overload — supports per-dispatch bindings and XR stereo.
        /// </summary>
        public void Dispatch(CommandBuffer cmd, NativeRayTraceDescriptorSet ds, uint width, uint height)
        {
            if (!IsValid || ds == null) return;
            // After a hot-reload rebuild the fresh native pipeline has an empty
            // hit-group table (DispatchRays is suppressed until it is rebuilt) —
            // re-apply the cached variants ahead of the dispatch event; events
            // execute in command-buffer order on the render thread.
            if (_shtNeedsReapply && _shtVariants.IsCreated)
            {
                _shtNeedsReapply = false;
                IssueShtRebuild(cmd);
            }

            IntPtr ptr = ds.SnapshotAndBuildHeader(width, height);
            if (ptr == IntPtr.Zero) return;
            cmd.IssuePluginEventAndData(NativeRenderPlugin.NR_RTS_GetRenderEventFunc(), 1, ptr);
        }

        /// <summary>
        /// Issues a render event to rebuild this pipeline's hit-group shader table from the
        /// flat per-geometry <paramref name="variantIndices"/> array (one entry per TLAS
        /// geometry, selecting which hit-group export to use). The indices are copied into
        /// pipeline-owned storage, so the caller's array may be freed or reused immediately;
        /// the copy is also re-applied automatically after a shader hot-reload rebuild.
        ///
        /// No <see cref="RayTracingAccelerationStructure"/> dependency: only rebuild when the
        /// variant layout actually changes (e.g. on a scene topology change), not every frame.
        /// </summary>
        public void RebuildHitGroupTable(CommandBuffer cmd, NativeArray<uint> variantIndices)
        {
            if (!IsValid || !variantIndices.IsCreated || variantIndices.Length == 0) return;

            if (!_shtVariants.IsCreated || _shtVariants.Length != variantIndices.Length)
            {
                if (_shtVariants.IsCreated) _shtVariants.Dispose();
                _shtVariants = new NativeArray<uint>(variantIndices.Length, Allocator.Persistent);
            }

            _shtVariants.CopyFrom(variantIndices);
            _shtNeedsReapply = false;
            IssueShtRebuild(cmd);
        }

        /// <summary>
        /// Copies the master variant array into the next pinned ring slot and issues the
        /// SHT-rebuild render event pointing at it. The ring keeps the pointer valid while
        /// previous (in-flight) frames may still be read by the render thread.
        /// </summary>
        private unsafe void IssueShtRebuild(CommandBuffer cmd)
        {
            if (!_shtEventData.IsCreated)
                _shtEventData = new NativeArray<NativeRenderPlugin.ShtRebuildEventData>(kShtRingSize, Allocator.Persistent);
            _shtVariantRing ??= new NativeArray<uint>[kShtRingSize];

            // Advance ring index so we never overwrite the slot the render thread
            // may still be reading from a previous (in-flight) frame.
            _shtRingIndex = (_shtRingIndex + 1) % kShtRingSize;

            ref var ringSlot = ref _shtVariantRing[_shtRingIndex];
            if (!ringSlot.IsCreated || ringSlot.Length != _shtVariants.Length)
            {
                if (ringSlot.IsCreated) ringSlot.Dispose();
                ringSlot = new NativeArray<uint>(_shtVariants.Length, Allocator.Persistent);
            }

            ringSlot.CopyFrom(_shtVariants);

            _shtEventData[_shtRingIndex] = new NativeRenderPlugin.ShtRebuildEventData
            {
                shaderHandle      = _handle,
                variantIndicesPtr = (IntPtr)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(ringSlot),
                count             = (uint)ringSlot.Length,
                _pad              = 0,
            };

            var basePtr = (NativeRenderPlugin.ShtRebuildEventData*)NativeArrayUnsafeUtility.GetUnsafePtr(_shtEventData);
            cmd.IssuePluginEventAndData(
                NativeRenderPlugin.NR_RTS_GetRebuildHitGroupTableEventFunc(),
                2,
                (IntPtr)(basePtr + _shtRingIndex));
        }
    }
}
