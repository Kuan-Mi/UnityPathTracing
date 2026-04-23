using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Rendering;

namespace NativeRender
{
    /// <summary>
    /// A compute pipeline instance created from a <see cref="NativeComputeShader"/> asset.
    /// Owns a per-dispatch staging slot array that is snapshotted into a ring buffer before
    /// each <see cref="Dispatch"/> call, ensuring main-thread bindings and render-thread execution
    /// are fully decoupled. The same pipeline can be dispatched multiple times per frame.
    ///
    /// Multiple pipelines can be created from the same <see cref="NativeComputeShader"/>,
    /// each with independent resource bindings.
    ///
    /// Lifetime: must be explicitly disposed via <see cref="Dispose"/>.
    /// </summary>
    public sealed class NativeComputePipeline : IDisposable
    {
        // Number of ring-buffer entries. Supports this many in-flight Dispatch calls
        // from a single pipeline per frame before the ring wraps (GPU must be done by then).
        private const int RingSize = 8;

        // objectKind constants matching C++ CS_BindingObjectKind
        private const uint ObjKindNone            = 0;
        private const uint ObjKindAccelStruct      = 1;
        private const uint ObjKindBindlessTexture  = 2;
        private const uint ObjKindBindlessBuffer   = 3;

        private ulong _handle;
        private NativeComputeShader _shader;

        // Slot layout: name → slot index as reported by NR_CS_GetSlotIndex
        private Dictionary<string, uint> _nameToSlot;
        private uint _slotCount;

        // Staging array written by SetXxx on the main thread (not pinned, just managed)
        private NativeRenderPlugin.CS_BindingSlot[] _stagingSlots;

        // Ring buffer of pinned NativeArrays (Persistent) – one entry per in-flight dispatch
        private NativeArray<NativeRenderPlugin.CS_BindingSlot>[] _slotRing;
        private NativeArray<NativeRenderPlugin.CS_RenderEventData>[] _headerRing;
        private int _ringIdx;

        /// <summary>True if the underlying D3D12 pipeline is valid and ready to dispatch.</summary>
        public bool IsValid => _handle != 0;

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
            AllocateRingBuffers();
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

            _stagingSlots = new NativeRenderPlugin.CS_BindingSlot[_slotCount];
        }

        private void AllocateRingBuffers()
        {
            _slotRing   = new NativeArray<NativeRenderPlugin.CS_BindingSlot>[RingSize];
            _headerRing = new NativeArray<NativeRenderPlugin.CS_RenderEventData>[RingSize];
            for (int i = 0; i < RingSize; i++)
            {
                _slotRing[i]   = new NativeArray<NativeRenderPlugin.CS_BindingSlot>(
                    (int)(_slotCount > 0 ? _slotCount : 1), Allocator.Persistent);
                _headerRing[i] = new NativeArray<NativeRenderPlugin.CS_RenderEventData>(
                    1, Allocator.Persistent);
            }
            _ringIdx = 0;
        }

        private void FreeRingBuffers()
        {
            if (_slotRing == null) return;
            for (int i = 0; i < RingSize; i++)
            {
                if (_slotRing[i].IsCreated)   _slotRing[i].Dispose();
                if (_headerRing[i].IsCreated) _headerRing[i].Dispose();
            }
            _slotRing   = null;
            _headerRing = null;
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

            FreeRingBuffers();

            try
            {
                BuildNativeHandle(shader);
                BuildSlotLayout(shader);
                AllocateRingBuffers();
                Debug.Log($"[NativeComputePipeline] Rebuilt pipeline for: {shader.name}");
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

            FreeRingBuffers();
        }

        // -------------------------------------------------------------------
        // Resource binding  (main thread — writes _stagingSlots)
        // -------------------------------------------------------------------

        private bool TryGetSlot(string name, out uint idx)
        {
            if (name != null && _nameToSlot != null && _nameToSlot.TryGetValue(name, out idx))
                return true;
            idx = uint.MaxValue;
            return false;
        }

        /// <summary>Binds a ComputeBuffer as a read-only structured/byte-address buffer (SRV).</summary>
        public void SetBuffer(string name, ComputeBuffer buffer)
        {
            if (!IsValid || !TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = (ulong)buffer.GetNativeBufferPtr();
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        /// <summary>Binds a GraphicsBuffer as a read-only structured/byte-address buffer (SRV).</summary>
        public void SetBuffer(string name, GraphicsBuffer buffer)
        {
            if (!IsValid || !TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = (ulong)buffer.GetNativeBufferPtr();
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        /// <summary>Binds a ComputeBuffer as an RW (read-write) buffer (UAV).</summary>
        public void SetRWBuffer(string name, ComputeBuffer buffer)
        {
            if (!IsValid || !TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = (ulong)buffer.GetNativeBufferPtr();
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        /// <summary>Binds a GraphicsBuffer as an RW (read-write) buffer (UAV).</summary>
        public void SetRWBuffer(string name, GraphicsBuffer buffer)
        {
            if (!IsValid || !TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = (ulong)buffer.GetNativeBufferPtr();
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        /// <summary>Binds a GraphicsBuffer as an RWStructuredBuffer UAV with explicit element count and stride.</summary>
        public void SetRWStructuredBuffer(string name, GraphicsBuffer buffer)
        {
            if (!IsValid || !TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = buffer != null ? (ulong)buffer.GetNativeBufferPtr() : 0;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = buffer != null ? (uint)buffer.count  : 0;
            _stagingSlots[i].stride      = buffer != null ? (uint)buffer.stride : 0;
        }

        /// <summary>Binds a Texture2D or RenderTexture as a read-only texture (SRV).</summary>
        public void SetTexture(string name, Texture texture)
        {
            if (!IsValid || !TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = texture != null ? (ulong)texture.GetNativeTexturePtr() : 0;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        /// <summary>Binds a RenderTexture as a read-write texture (UAV).</summary>
        public void SetRWTexture(string name, RenderTexture texture)
        {
            if (!IsValid || !TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = texture != null ? (ulong)texture.GetNativeTexturePtr() : 0;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        /// <summary>
        /// Binds a ComputeBuffer as a constant buffer (CBV).
        /// The buffer must have been created with ComputeBufferType.Constant.
        /// </summary>
        public void SetConstantBuffer(string name, ComputeBuffer buffer)
        {
            if (!IsValid || !TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = buffer != null ? (ulong)buffer.GetNativeBufferPtr() : 0;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        /// <summary>Binds a GraphicsBuffer as a constant buffer (CBV).</summary>
        public void SetConstantBuffer(string name, GraphicsBuffer buffer)
        {
            if (!IsValid || !TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = buffer != null ? (ulong)buffer.GetNativeBufferPtr() : 0;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        /// <summary>
        /// Binds a GraphicsBuffer as a StructuredBuffer SRV.
        /// Passing null clears the binding.
        /// </summary>
        public void SetStructuredBuffer(string name, GraphicsBuffer buffer)
        {
            if (!IsValid || !TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = buffer != null ? (ulong)buffer.GetNativeBufferPtr() : 0;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = buffer != null ? (uint)buffer.count  : 0;
            _stagingSlots[i].stride      = buffer != null ? (uint)buffer.stride : 0;
        }

        /// <summary>Binds a ComputeBuffer as a StructuredBuffer SRV with explicit element count and stride.</summary>
        public void SetStructuredBuffer(string name, ComputeBuffer buffer, int elementCount, int elementStride)
        {
            if (!IsValid || !TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = buffer != null ? (ulong)buffer.GetNativeBufferPtr() : 0;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = buffer != null ? (uint)elementCount  : 0;
            _stagingSlots[i].stride      = buffer != null ? (uint)elementStride : 0;
        }

        /// <summary>Binds the TLAS of an acceleration structure by HLSL variable name.</summary>
        public void SetAccelerationStructure(string name, RayTracingAccelerationStructure accelStructure)
        {
            if (!IsValid || !TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = 0;
            _stagingSlots[i].objectPtr   = accelStructure != null ? accelStructure.Handle : 0;
            _stagingSlots[i].objectKind  = ObjKindAccelStruct;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        /// <summary>
        /// Binds a BindlessTexture to an unbounded Texture2D[] variable.
        /// Call again after BindlessTexture.Resize() to rebind the new descriptor range.
        /// </summary>
        public void SetBindlessTexture(string name, BindlessTexture bt)
        {
            if (!IsValid || !TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = 0;
            _stagingSlots[i].objectPtr   = bt != null ? bt.Handle : 0;
            _stagingSlots[i].objectKind  = ObjKindBindlessTexture;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        /// <summary>
        /// Binds a BindlessBuffer to an unbounded ByteAddressBuffer[] variable.
        /// Call again after BindlessBuffer.Resize() to rebind the new descriptor range.
        /// </summary>
        public void SetBindlessBuffer(string name, BindlessBuffer bb)
        {
            if (!IsValid || !TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = 0;
            _stagingSlots[i].objectPtr   = bb != null ? bb.Handle : 0;
            _stagingSlots[i].objectKind  = ObjKindBindlessBuffer;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        // -------------------------------------------------------------------
        // Dispatch
        // -------------------------------------------------------------------

        /// <summary>
        /// Snapshots the current staging bindings into the next ring-buffer slot,
        /// then enqueues a Dispatch call into the CommandBuffer.
        /// Safe to call multiple times per frame with different bindings.
        /// </summary>
        public void Dispatch(CommandBuffer cmd, uint threadGroupX, uint threadGroupY, uint threadGroupZ)
        {
            if (!IsValid) return;

            int ring = _ringIdx % RingSize;
            _ringIdx++;

            // Copy staging → pinned ring slot
            var slotArray = _slotRing[ring];
            for (int k = 0; k < (int)_slotCount; k++)
                slotArray[k] = _stagingSlots[k];

            // Fill header
            unsafe
            {
                var header = new NativeRenderPlugin.CS_RenderEventData
                {
                    shaderHandle    = _handle,
                    threadGroupX    = threadGroupX,
                    threadGroupY    = threadGroupY,
                    threadGroupZ    = threadGroupZ,
                    bindingCount    = _slotCount,
                    bindingSlotsPtr = (ulong)slotArray.GetUnsafePtr(),
                };
                _headerRing[ring][0] = header;

                cmd.IssuePluginEventAndData(
                    NativeRenderPlugin.NR_CS_GetRenderEventFunc(),
                    1,
                    (IntPtr)_headerRing[ring].GetUnsafePtr());
            }
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
