using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace NativeRender
{
    /// <summary>
    /// Manages the D3D12 compute pipeline state (PSO + root signature + slot layout)
    /// created from a <see cref="NativeComputeShader"/> asset.
    ///
    /// Resource bindings live in <see cref="NativeComputeDescriptorSet"/>, which is
    /// created separately and passed to <see cref="Dispatch"/>.  This decouples
    /// pipeline state from per-pass resource bindings so that multiple passes can
    /// each hold their own descriptor set while sharing the same pipeline.
    ///
    /// Lifetime: must be explicitly disposed via <see cref="Dispose"/>.
    /// </summary>
    public sealed class NativeComputePipeline : IDisposable
    {
        private ulong _handle;
        private NativeComputeShader _shader;

        // Slot layout: name → slot index as reported by NR_CS_GetSlotIndex
        private Dictionary<string, uint> _nameToSlot;
        private uint _slotCount;

        /// <summary>True if the underlying D3D12 pipeline is valid and ready to dispatch.</summary>
        public bool IsValid => _handle != 0;

        // Internal access for NativeComputeDescriptorSet
        internal IReadOnlyDictionary<string, uint> NameToSlot => _nameToSlot;
        internal uint SlotCount => _slotCount;
        internal ulong Handle => _handle;

        /// <summary>
        /// Fired after a hot-reload successfully rebuilds the native pipeline.
        /// <see cref="NativeComputeDescriptorSet"/> subscribes to reallocate its ring buffers.
        /// </summary>
        internal event Action<NativeComputePipeline> OnRebuilt;

        // -------------------------------------------------------------------
        // Construction
        // -------------------------------------------------------------------

        /// <summary>
        /// Creates a new compute pipeline from the given shader asset.
        /// Triggers HLSL compilation if the asset has not been compiled yet.
        /// Throws <see cref="InvalidOperationException"/> if pipeline creation fails.
        /// </summary>
        public NativeComputePipeline(NativeComputeShader shader)
        {
            if (shader == null)
                throw new ArgumentNullException(nameof(shader));

            _shader = shader;
            BuildNativeHandle(shader);
            BuildSlotLayout(shader);
            NativeComputeShader.OnRecompiled += OnShaderRecompiled;
        }

        private void BuildNativeHandle(NativeComputeShader shader)
        {
            byte[] dxil = shader.GetOrCompileDxil();
            if (dxil == null || dxil.Length == 0)
                throw new InvalidOperationException(
                    $"[NativeComputePipeline] Shader compilation failed for: {shader.GetHlslPath()}");

            _handle = NativeRenderPlugin.NR_CreateComputeShader(dxil, (uint)dxil.Length, shader.name);
            if (_handle == 0)
                throw new InvalidOperationException(
                    $"[NativeComputePipeline] NR_CreateComputeShader returned 0 for: {shader.name}");
        }

        private void BuildSlotLayout(NativeComputeShader shader)
        {
            _slotCount  = NativeRenderPlugin.NR_CS_GetBindingCount(_handle);
            _nameToSlot = new Dictionary<string, uint>((int)_slotCount);

            // Parse binding names from the reflected JSON to build name→slot mapping.
            // JSON structure: { "bindings": [ { "name": "..." }, ... ] }
            string json = shader.ReflectionJson ?? "";
            if (_slotCount > 0 && json.Length > 0)
            {
                int arrayStart = -1;
                int bindingsIdx = json.IndexOf("\"bindings\"", StringComparison.Ordinal);
                if (bindingsIdx >= 0)
                    arrayStart = json.IndexOf('[', bindingsIdx);

                if (arrayStart >= 0)
                {
                    int pos = arrayStart + 1;
                    while (pos < json.Length)
                    {
                        int objStart = json.IndexOf('{', pos);
                        if (objStart < 0) break;
                        int objEnd = json.IndexOf('}', objStart);
                        if (objEnd < 0) break;

                        string obj  = json.Substring(objStart + 1, objEnd - objStart - 1);
                        string name = ExtractJsonString(obj, "name");
                        if (!string.IsNullOrEmpty(name))
                        {
                            uint idx = NativeRenderPlugin.NR_CS_GetSlotIndex(_handle, name);
                            if (idx != uint.MaxValue)
                                _nameToSlot[name] = idx;
                        }
                        pos = objEnd + 1;
                    }
                }
            }
        }

        private void OnShaderRecompiled(NativeComputeShader shader)
        {
            if (shader != _shader) return;

            if (_handle != 0)
            {
                GL.Flush();
                NativeRenderPlugin.NR_DestroyComputeShader(_handle);
                _handle = 0;
            }

            try
            {
                BuildNativeHandle(shader);
                BuildSlotLayout(shader);
                Debug.Log($"[NativeComputePipeline] Rebuilt pipeline for: {shader.name}");
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
            NativeComputeShader.OnRecompiled -= OnShaderRecompiled;

            if (_handle != 0)
            {
                GL.Flush();
                NativeRenderPlugin.NR_DestroyComputeShader(_handle);
                _handle = 0;
            }
        }

        // -------------------------------------------------------------------
        // Dispatch
        // -------------------------------------------------------------------

        /// <summary>
        /// Snapshots bindings from <paramref name="descriptorSet"/> into its ring buffer
        /// and enqueues a Dispatch call into the CommandBuffer.
        /// Safe to call multiple times per frame with different descriptor sets.
        /// </summary>
        public void Dispatch(CommandBuffer cmd, NativeComputeDescriptorSet descriptorSet,
                             uint threadGroupX, uint threadGroupY, uint threadGroupZ)
        {
            if (!IsValid || descriptorSet == null) return;

            IntPtr headerPtr = descriptorSet.SnapshotAndBuildHeader(
                threadGroupX, threadGroupY, threadGroupZ);
            if (headerPtr == IntPtr.Zero) return;

            cmd.IssuePluginEventAndData(
                NativeRenderPlugin.NR_CS_GetRenderEventFunc(),
                1,
                headerPtr);
        }

        // -------------------------------------------------------------------
        // Helpers
        // -------------------------------------------------------------------

        private static string ExtractJsonString(string obj, string key)
        {
            string search = "\"" + key + "\"";
            int ki = obj.IndexOf(search, StringComparison.Ordinal);
            if (ki < 0) return null;
            int colon = obj.IndexOf(':', ki + search.Length);
            if (colon < 0) return null;
            int q1 = obj.IndexOf('"', colon + 1);
            if (q1 < 0) return null;
            int q2 = obj.IndexOf('"', q1 + 1);
            if (q2 < 0) return null;
            return obj.Substring(q1 + 1, q2 - q1 - 1);
        }
    }
}
