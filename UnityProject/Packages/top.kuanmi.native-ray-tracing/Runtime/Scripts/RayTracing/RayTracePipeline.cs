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
        private ulong _handle;
        private RayTraceShader  _shader;
        private HitGroupShader[] _hitGroupShaders; // null when not using multi-blob path

        /// <summary>True if the underlying D3D12 pipeline is valid and ready to dispatch.</summary>
        public bool IsValid => _handle != 0;

        /// <summary>Opaque native handle (pointer to RayTraceShader). Used by NativeRayTraceDescriptorSet.</summary>
        public ulong Handle => _handle;

        /// <summary>Number of resource binding slots in this shader.</summary>
        public uint SlotCount => _slotCount;

        /// <summary>Maps HLSL variable names to slot indices (for NativeRayTraceDescriptorSet).</summary>
        public IReadOnlyDictionary<string, uint> NameToSlot => _nameToSlot;

        /// <summary>Fired (on the main thread) whenever the pipeline is rebuilt after a hot-reload.</summary>
        public event Action<RayTracePipeline> OnRebuilt;

        private uint                     _slotCount;
        private Dictionary<string, uint> _nameToSlot = new Dictionary<string, uint>();

        // -------------------------------------------------------------------
        // Construction
        // -------------------------------------------------------------------

        /// <summary>
        /// Creates a new DXR pipeline from the given shader asset.
        /// Triggers HLSL compilation if the asset has not been compiled yet.
        /// Throws <see cref="InvalidOperationException"/> if pipeline creation fails.
        /// </summary>
        public RayTracePipeline(RayTraceShader shader)
        {
            if (shader == null)
                throw new ArgumentNullException(nameof(shader));

            _shader = shader;
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
        {
            if (primaryShader == null)
                throw new ArgumentNullException(nameof(primaryShader));
            if (hitGroupShaders == null || hitGroupShaders.Length == 0)
                throw new ArgumentException("Use the single-shader constructor when there are no extra hit groups.",
                    nameof(hitGroupShaders));

            _shader          = primaryShader;
            _hitGroupShaders = hitGroupShaders;

            BuildNativeHandleMultiBlob(primaryShader, hitGroupShaders);
            RayTraceShader.OnRecompiled  += OnShaderRecompiled;
            HitGroupShader.OnRecompiled  += OnHitGroupShaderRecompiled;
        }

        private void BuildNativeHandle(RayTraceShader shader)
        {
            byte[] dxil = shader.GetOrCompileDxil();
            if (dxil == null || dxil.Length == 0)
                throw new InvalidOperationException(
                    $"[RayTracePipeline] Shader compilation failed for: {shader.GetHlslPath()}");

            uint flags = ProfileSupportsOpacityMicromaps(shader.TargetProfile) ? 1u : 0u;
            uint maxPayload = shader.MaxPayloadSizeInBytes;
            Debug.Log($"[RayTracePipeline] Creating pipeline for: {shader.name} (DXIL size: {dxil.Length} bytes, OMM support: {flags != 0}, MaxPayload: {maxPayload})");
            string rayGenName = string.IsNullOrEmpty(shader.RayGenName) ? null : shader.RayGenName;
            _handle = NativeRenderPlugin.NR_CreateRayTraceShaderFromBytes(dxil, (uint)dxil.Length, shader.name, flags, maxPayload, rayGenName);
            if (_handle == 0)
                throw new InvalidOperationException(
                    $"[RayTracePipeline] NR_CreateRayTraceShaderFromBytes returned 0 for: {shader.name}");

            RefreshSlotLayout();
        }

        private void BuildNativeHandleMultiBlob(RayTraceShader primaryShader, HitGroupShader[] hitGroupShaders)
        {
            int totalBlobs = 1 + hitGroupShaders.Length;
            byte[][]  dxils = new byte[totalBlobs][];
            GCHandle[] pins  = new GCHandle[totalBlobs];

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

                uint flags      = ProfileSupportsOpacityMicromaps(primaryShader.TargetProfile) ? 1u : 0u;
                uint maxPayload = primaryShader.MaxPayloadSizeInBytes;
                string rayGenName = string.IsNullOrEmpty(primaryShader.RayGenName) ? null : primaryShader.RayGenName;

                Debug.Log($"[RayTracePipeline] Creating multi-blob pipeline for '{primaryShader.name}' ({totalBlobs} blobs)");
                _handle = NativeRenderPlugin.NR_CreateRayTracePipelineFromBlobs(
                    ptrs, sizes, (uint)totalBlobs,
                    primaryShader.name, flags, maxPayload, rayGenName);

                if (_handle == 0)
                    throw new InvalidOperationException(
                        $"[RayTracePipeline] NR_CreateRayTracePipelineFromBlobs returned 0 for: {primaryShader.name}");
            }
            finally
            {
                foreach (var pin in pins)
                    if (pin.IsAllocated) pin.Free();
            }

            RefreshSlotLayout();
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
            string version = profile.Substring(prefix.Length); // e.g. "6_9" or "6_6"
            string[] parts = version.Split('_');
            if (parts.Length < 2) return false;
            if (!int.TryParse(parts[0], out int major) || !int.TryParse(parts[1], out int minor)) return false;
            return major > 6 || (major == 6 && minor >= 9);
        }

        private void RefreshSlotLayout()
        {
            _nameToSlot.Clear();
            if (_handle == 0) { _slotCount = 0; return; }
            _slotCount = NativeRenderPlugin.NR_RTS_GetBindingCount(_handle);
            for (uint i = 0; i < _slotCount; i++)
            {
                IntPtr namePtr = NativeRenderPlugin.NR_RTS_GetBindingName(_handle, i);
                if (namePtr != IntPtr.Zero)
                {
                    string n = Marshal.PtrToStringAnsi(namePtr);
                    if (!string.IsNullOrEmpty(n))
                        _nameToSlot[n] = i;
                }
            }
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
                if (hg == shader) { RebuildPipeline(); return; }
        }

        private void RebuildPipeline()
        {
            if (_handle != 0)
            {
                GL.Flush();
                NativeRenderPlugin.NR_DestroyRayTraceShader(_handle);
                _handle = 0;
            }

            try
            {
                if (_hitGroupShaders != null && _hitGroupShaders.Length > 0)
                    BuildNativeHandleMultiBlob(_shader, _hitGroupShaders);
                else
                    BuildNativeHandle(_shader);
                Debug.Log($"[RayTracePipeline] Rebuilt pipeline for: {_shader.name}");
                OnRebuilt?.Invoke(this);
            }
            catch (Exception e)
            {
                Debug.LogError(e.Message);
            }
        }

        // -------------------------------------------------------------------
        // IDisposable
        // -------------------------------------------------------------------

        public void Dispose()
        {
            RayTraceShader.OnRecompiled  -= OnShaderRecompiled;
            HitGroupShader.OnRecompiled  -= OnHitGroupShaderRecompiled;

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
            IntPtr ptr = ds.SnapshotAndBuildHeader(width, height);
            if (ptr == IntPtr.Zero) return;
            cmd.IssuePluginEventAndData(NativeRenderPlugin.NR_RTS_GetRenderEventFunc(), 1, ptr);
        }
    }
}
