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
        private RayTraceShader _shader;

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

        private void BuildNativeHandle(RayTraceShader shader)
        {
            byte[] dxil = shader.GetOrCompileDxil();
            if (dxil == null || dxil.Length == 0)
                throw new InvalidOperationException(
                    $"[RayTracePipeline] Shader compilation failed for: {shader.GetHlslPath()}");

            uint flags = ProfileSupportsOpacityMicromaps(shader.TargetProfile) ? 1u : 0u;
            uint maxPayload = shader.MaxPayloadSizeInBytes;
            Debug.Log($"[RayTracePipeline] Creating pipeline for: {shader.name} (DXIL size: {dxil.Length} bytes, OMM support: {flags != 0}, MaxPayload: {maxPayload})");
            _handle = NativeRenderPlugin.NR_CreateRayTraceShaderFromBytes(dxil, (uint)dxil.Length, shader.name, flags, maxPayload);
            if (_handle == 0)
                throw new InvalidOperationException(
                    $"[RayTracePipeline] NR_CreateRayTraceShaderFromBytes returned 0 for: {shader.name}");

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

            // Destroy the old native pipeline and rebuild from the freshly compiled DXIL.
            if (_handle != 0)
            {
                GL.Flush();
                NativeRenderPlugin.NR_DestroyRayTraceShader(_handle);
                _handle = 0;
            }

            try
            {
                BuildNativeHandle(shader);
                Debug.Log($"[RayTracePipeline] Rebuilt pipeline for: {shader.name}");
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
            RayTraceShader.OnRecompiled -= OnShaderRecompiled;

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
