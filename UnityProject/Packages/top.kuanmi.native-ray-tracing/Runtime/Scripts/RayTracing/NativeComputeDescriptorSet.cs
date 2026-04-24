using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Rendering;

namespace NativeRender
{
    /// <summary>
    /// Holds resource bindings (descriptor set) for a <see cref="NativeComputePipeline"/>.
    ///
    /// Each pass creates one (or more) <see cref="NativeComputeDescriptorSet"/> instances
    /// tied to a <see cref="NativeComputePipeline"/> and calls <c>Set*</c> methods each frame
    /// before passing the descriptor set to
    /// <see cref="NativeComputePipeline.Dispatch(CommandBuffer, NativeComputeDescriptorSet, uint, uint, uint)"/>.
    ///
    /// Owns a ring buffer of pinned <see cref="NativeArray{T}"/> allocations so that
    /// main-thread bindings and render-thread execution are fully decoupled.
    /// The same descriptor set can be dispatched multiple times per frame.
    ///
    /// Lifetime: must be explicitly disposed via <see cref="Dispose"/>.
    /// </summary>
    public sealed class NativeComputeDescriptorSet : IDisposable
    {
        // Number of ring-buffer entries — supports this many in-flight Dispatch calls
        // from a single descriptor set per frame before the ring wraps.
        private const int RingSize = 8;

        // objectKind constants matching C++ CS_BindingObjectKind
        private const uint ObjKindNone           = 0;
        private const uint ObjKindAccelStruct     = 1;
        private const uint ObjKindBindlessTexture = 2;
        private const uint ObjKindBindlessBuffer  = 3;

        private readonly NativeComputePipeline _pipeline;

        // Slot layout mirrored from the pipeline (refreshed on hot-reload)
        private Dictionary<string, uint> _nameToSlot;
        private uint _slotCount;

        // Staging array written by Set* on the main thread (not pinned)
        private NativeRenderPlugin.CS_BindingSlot[] _stagingSlots;

        // Ring buffer of pinned NativeArrays (Persistent)
        private NativeArray<NativeRenderPlugin.CS_BindingSlot>[] _slotRing;
        private NativeArray<NativeRenderPlugin.CS_RenderEventData>[] _headerRing;
        private int _ringIdx;

        // -------------------------------------------------------------------
        // Construction
        // -------------------------------------------------------------------

        /// <summary>
        /// Creates a descriptor set for the given pipeline.
        /// The descriptor set subscribes to the pipeline's <c>OnRebuilt</c> event so
        /// that it automatically reallocates its ring buffer after a hot-reload.
        /// </summary>
        public NativeComputeDescriptorSet(NativeComputePipeline pipeline)
        {
            if (pipeline == null)
                throw new ArgumentNullException(nameof(pipeline));

            _pipeline = pipeline;
            CopySlotLayout(pipeline);
            AllocateRingBuffers();
            pipeline.OnRebuilt += OnPipelineRebuilt;
        }

        private void CopySlotLayout(NativeComputePipeline pipeline)
        {
            _slotCount  = pipeline.SlotCount;
            _nameToSlot = new Dictionary<string, uint>(_slotCount > 0 ? (int)_slotCount : 0);
            foreach (var kv in pipeline.NameToSlot)
                _nameToSlot[kv.Key] = kv.Value;
            _stagingSlots = new NativeRenderPlugin.CS_BindingSlot[_slotCount > 0 ? _slotCount : 1];
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

        private void OnPipelineRebuilt(NativeComputePipeline pipeline)
        {
            FreeRingBuffers();
            CopySlotLayout(pipeline);
            AllocateRingBuffers();
        }

        // -------------------------------------------------------------------
        // IDisposable
        // -------------------------------------------------------------------

        public void Dispose()
        {
            _pipeline.OnRebuilt -= OnPipelineRebuilt;
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
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = (ulong)buffer.GetNativeBufferPtr();
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        /// <summary>Binds a GraphicsBuffer as a read-only structured/byte-address buffer (SRV).</summary>
        public void SetBuffer(string name, GraphicsBuffer buffer)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = (ulong)buffer.GetNativeBufferPtr();
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        /// <summary>Binds a ComputeBuffer as an RW (read-write) buffer (UAV).</summary>
        public void SetRWBuffer(string name, ComputeBuffer buffer)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = (ulong)buffer.GetNativeBufferPtr();
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        /// <summary>Binds a GraphicsBuffer as an RW (read-write) buffer (UAV).</summary>
        public void SetRWBuffer(string name, GraphicsBuffer buffer)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = (ulong)buffer.GetNativeBufferPtr();
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        /// <summary>Binds a GraphicsBuffer as an RWStructuredBuffer UAV with explicit element count and stride.</summary>
        public void SetRWStructuredBuffer(string name, GraphicsBuffer buffer)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = buffer != null ? (ulong)buffer.GetNativeBufferPtr() : 0;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = buffer != null ? (uint)buffer.count  : 0;
            _stagingSlots[i].stride      = buffer != null ? (uint)buffer.stride : 0;
        }

        /// <summary>Binds a Texture2D or RenderTexture as a read-only texture (SRV).</summary>
        public void SetTexture(string name, Texture texture)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = texture != null ? (ulong)texture.GetNativeTexturePtr() : 0;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        /// <summary>Binds a RenderTexture as a read-write texture (UAV).</summary>
        public void SetRWTexture(string name, RenderTexture texture)
        {
            if (!TryGetSlot(name, out uint i)) return;
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
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = buffer != null ? (ulong)buffer.GetNativeBufferPtr() : 0;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        /// <summary>Binds a GraphicsBuffer as a constant buffer (CBV).</summary>
        public void SetConstantBuffer(string name, GraphicsBuffer buffer)
        {
            if (!TryGetSlot(name, out uint i)) return;
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
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = buffer != null ? (ulong)buffer.GetNativeBufferPtr() : 0;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = buffer != null ? (uint)buffer.count  : 0;
            _stagingSlots[i].stride      = buffer != null ? (uint)buffer.stride : 0;
        }

        /// <summary>Binds a ComputeBuffer as a StructuredBuffer SRV with explicit element count and stride.</summary>
        public void SetStructuredBuffer(string name, ComputeBuffer buffer, int elementCount, int elementStride)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = buffer != null ? (ulong)buffer.GetNativeBufferPtr() : 0;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = buffer != null ? (uint)elementCount  : 0;
            _stagingSlots[i].stride      = buffer != null ? (uint)elementStride : 0;
        }

        /// <summary>Binds the TLAS of an acceleration structure by HLSL variable name.</summary>
        public void SetAccelerationStructure(string name, RayTracingAccelerationStructure accelStructure)
        {
            if (!TryGetSlot(name, out uint i)) return;
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
            if (!TryGetSlot(name, out uint i)) return;
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
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = 0;
            _stagingSlots[i].objectPtr   = bb != null ? bb.Handle : 0;
            _stagingSlots[i].objectKind  = ObjKindBindlessBuffer;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        // -------------------------------------------------------------------
        // Internal dispatch helper (called by NativeComputePipeline.Dispatch)
        // -------------------------------------------------------------------

        /// <summary>
        /// Snapshots the current staging bindings into the next ring-buffer slot,
        /// fills the event-data header and returns a pointer to it for
        /// <c>IssuePluginEventAndData</c>.  Returns <see cref="IntPtr.Zero"/> on failure.
        /// </summary>
        internal unsafe IntPtr SnapshotAndBuildHeader(
            ulong shaderHandle, uint threadGroupX, uint threadGroupY, uint threadGroupZ)
        {
            if (_slotRing == null) return IntPtr.Zero;

            int ring = _ringIdx % RingSize;
            _ringIdx++;

            // Copy staging → pinned ring slot
            var slotArray = _slotRing[ring];
            for (int k = 0; k < (int)_slotCount; k++)
                slotArray[k] = _stagingSlots[k];

            var header = new NativeRenderPlugin.CS_RenderEventData
            {
                shaderHandle    = shaderHandle,
                threadGroupX    = threadGroupX,
                threadGroupY    = threadGroupY,
                threadGroupZ    = threadGroupZ,
                bindingCount    = _slotCount,
                bindingSlotsPtr = (ulong)slotArray.GetUnsafePtr(),
            };
            _headerRing[ring][0] = header;

            return (IntPtr)_headerRing[ring].GetUnsafePtr();
        }
    }
}
