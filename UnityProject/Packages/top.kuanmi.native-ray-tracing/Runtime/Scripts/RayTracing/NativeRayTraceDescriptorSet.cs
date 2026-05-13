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
    /// Holds per-dispatch resource bindings for a <see cref="RayTracePipeline"/>.
    ///
    /// Mirrors <see cref="NativeComputeDescriptorSet"/> but targets DXR
    /// (<c>DispatchRays</c>) instead of <c>Dispatch</c>.
    ///
    /// Create one (or more) per pass; call <c>Set*</c> each frame to update
    /// bindings, then hand the descriptor set to
    /// <see cref="RayTracePipeline.Dispatch(CommandBuffer,NativeRayTraceDescriptorSet,uint,uint)"/>.
    ///
    /// Owns a ring buffer of pinned <see cref="NativeArray{T}"/> allocations so
    /// main-thread writes and render-thread reads are safely decoupled.
    ///
    /// Lifetime: must be explicitly disposed via <see cref="Dispose"/>.
    /// </summary>
    public sealed class NativeRayTraceDescriptorSet : IDisposable
    {
        // Ring-buffer depth — supports this many in-flight Dispatch calls per frame.
        private const int RingSize = 8;

        // DX12 hard limit for root-constants: 64 DWORDs = 256 bytes per slot.
        private const int MaxRootConstantBytes = 256;

        // objectKind constants — must match C++ CS_BindingObjectKind
        private const uint ObjKindNone               = 0;
        private const uint ObjKindAccelStruct        = 1;
        private const uint ObjKindBindlessTexture    = 2;
        private const uint ObjKindBindlessBuffer     = 3;
        private const uint ObjKindRootConstants      = 4;
        private const uint ObjKindNativeBuffer       = 5;
        private const uint ObjKindBindlessUAVTexture = 6;

        private readonly RayTracePipeline _pipeline;

        // Native C++ RayTraceDescriptorSet handle.
        private ulong _descriptorSetHandle;

        // Slot layout mirrored from the pipeline (refreshed on hot-reload).
        private Dictionary<string, uint> _nameToSlot;
        private uint                     _slotCount;

        // Staging arrays written by Set* on the main thread (not pinned).
        private NativeRenderPlugin.CS_BindingSlot[] _stagingSlots;
        private byte[]                              _stagingConstants;

        // Pinned ring buffers — read by the render thread.
        private NativeArray<NativeRenderPlugin.CS_BindingSlot>[]    _slotRing;
        private NativeArray<NativeRenderPlugin.RTS_RenderEventData>[] _headerRing;
        private NativeArray<byte>[]                                  _constantsRing;
        private int                                                  _ringIdx;

        // -------------------------------------------------------------------
        // Construction
        // -------------------------------------------------------------------

        /// <summary>
        /// Creates a descriptor set tied to <paramref name="pipeline"/>.
        /// Subscribes to <see cref="RayTracePipeline.OnRebuilt"/> for automatic
        /// re-creation after shader hot-reload.
        /// </summary>
        public NativeRayTraceDescriptorSet(RayTracePipeline pipeline)
        {
            if (pipeline == null)
                throw new ArgumentNullException(nameof(pipeline));

            _pipeline = pipeline;
            CopySlotLayout(pipeline);
            AllocateRingBuffers();
            _descriptorSetHandle = NativeRenderPlugin.NR_RTS_CreateDescriptorSet(pipeline.Handle);
            pipeline.OnRebuilt  += OnPipelineRebuilt;
        }

        private void CopySlotLayout(RayTracePipeline pipeline)
        {
            _slotCount  = pipeline.SlotCount;
            _nameToSlot = new Dictionary<string, uint>(_slotCount > 0 ? (int)_slotCount : 0);
            foreach (var kv in pipeline.NameToSlot)
                _nameToSlot[kv.Key] = kv.Value;
            int eff           = _slotCount > 0 ? (int)_slotCount : 1;
            _stagingSlots     = new NativeRenderPlugin.CS_BindingSlot[eff];
            _stagingConstants = new byte[eff * MaxRootConstantBytes];
        }

        private void AllocateRingBuffers()
        {
            int eff        = _slotCount > 0 ? (int)_slotCount : 1;
            _slotRing      = new NativeArray<NativeRenderPlugin.CS_BindingSlot>[RingSize];
            _headerRing    = new NativeArray<NativeRenderPlugin.RTS_RenderEventData>[RingSize];
            _constantsRing = new NativeArray<byte>[RingSize];
            for (int i = 0; i < RingSize; i++)
            {
                _slotRing[i]      = new NativeArray<NativeRenderPlugin.CS_BindingSlot>(eff, Allocator.Persistent);
                _headerRing[i]    = new NativeArray<NativeRenderPlugin.RTS_RenderEventData>(1, Allocator.Persistent);
                _constantsRing[i] = new NativeArray<byte>(eff * MaxRootConstantBytes, Allocator.Persistent);
            }
            _ringIdx = 0;
        }

        private void FreeRingBuffers()
        {
            if (_slotRing == null) return;
            for (int i = 0; i < RingSize; i++)
            {
                if (_slotRing[i].IsCreated)      _slotRing[i].Dispose();
                if (_headerRing[i].IsCreated)    _headerRing[i].Dispose();
                if (_constantsRing[i].IsCreated) _constantsRing[i].Dispose();
            }
            _slotRing      = null;
            _headerRing    = null;
            _constantsRing = null;
        }

        private void OnPipelineRebuilt(RayTracePipeline pipeline)
        {
            if (_descriptorSetHandle != 0)
            {
                NativeRenderPlugin.NR_RTS_DestroyDescriptorSet(_descriptorSetHandle);
                _descriptorSetHandle = 0;
            }
            FreeRingBuffers();
            CopySlotLayout(pipeline);
            AllocateRingBuffers();
            _descriptorSetHandle = NativeRenderPlugin.NR_RTS_CreateDescriptorSet(pipeline.Handle);
        }

        // -------------------------------------------------------------------
        // IDisposable
        // -------------------------------------------------------------------

        public void Dispose()
        {
            _pipeline.OnRebuilt -= OnPipelineRebuilt;
            if (_descriptorSetHandle != 0)
            {
                NativeRenderPlugin.NR_RTS_DestroyDescriptorSet(_descriptorSetHandle);
                _descriptorSetHandle = 0;
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

        /// <summary>Binds a raw/structured buffer (SRV) by HLSL variable name.</summary>
        public void SetBuffer(string name, IntPtr bufferPtr)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = (ulong)bufferPtr;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
            _stagingSlots[i].format      = 0;
        }

        /// <summary>Binds an RW buffer (UAV) by HLSL variable name.</summary>
        public void SetRWBuffer(string name, IntPtr bufferPtr)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = (ulong)bufferPtr;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
            _stagingSlots[i].format      = 0;
        }

        /// <summary>Binds an RW typed buffer (UAV) with an explicit DXGI_FORMAT.</summary>
        public void SetRWTypedBuffer(string name, IntPtr bufferPtr, int count, uint dxgiFormat)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = (ulong)bufferPtr;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = (uint)count;
            _stagingSlots[i].stride      = 0;
            _stagingSlots[i].format      = dxgiFormat;
        }

        /// <summary>Binds a structured buffer (UAV) with explicit element count and stride.</summary>
        public void SetRWStructuredBuffer(string name, IntPtr bufferPtr, int count, int stride)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = (ulong)bufferPtr;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = (uint)count;
            _stagingSlots[i].stride      = (uint)stride;
        }

        /// <summary>Binds a Texture2D or RenderTexture as a read-only texture (SRV).</summary>
        public void SetTexture(string name, IntPtr texturePtr)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = (ulong)texturePtr;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        /// <summary>Binds a RenderTexture as a read-write texture (UAV).</summary>
        public void SetRWTexture(string name, IntPtr texturePtr)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = (ulong)texturePtr;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        /// <summary>Binds a GraphicsBuffer as a constant buffer (CBV).</summary>
        public void SetConstantBuffer(string name, IntPtr bufferPtr)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = (ulong)bufferPtr;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        /// <summary>Binds a <see cref="NativeBuffer"/> as a constant buffer (CBV).</summary>
        public void SetNativeBuffer(string name, NativeBuffer nb)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = 0;
            _stagingSlots[i].objectPtr   = nb != null ? nb.Handle : 0;
            _stagingSlots[i].objectKind  = ObjKindNativeBuffer;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        /// <summary>Binds a typed buffer SRV (e.g. Buffer&lt;float2&gt;) with explicit DXGI_FORMAT.</summary>
        public void SetTypedBuffer(string name, IntPtr bufferPtr, int count, uint dxgiFormat)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = (ulong)bufferPtr;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = (uint)count;
            _stagingSlots[i].stride      = 0;
            _stagingSlots[i].format      = dxgiFormat;
        }

        /// <summary>Binds a StructuredBuffer SRV with explicit element count and stride.</summary>
        public void SetStructuredBuffer(string name, IntPtr bufferPtr, int elementCount, int elementStride)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = (ulong)bufferPtr;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = (uint)elementCount;
            _stagingSlots[i].stride      = (uint)elementStride;
        }

        /// <summary>Binds a RaytracingAccelerationStructure (TLAS) by HLSL variable name.</summary>
        public void SetAccelerationStructure(string name, RayTracingAccelerationStructure accelStructure)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = 0;
            _stagingSlots[i].objectPtr   = accelStructure != null ? accelStructure.Handle : 0;
            _stagingSlots[i].objectKind  = ObjKindAccelStruct;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        /// <summary>Binds a BindlessTexture to an unbounded Texture2D[] variable.</summary>
        public void SetBindlessTexture(string name, BindlessTexture bt)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = 0;
            _stagingSlots[i].objectPtr   = bt != null ? bt.Handle : 0;
            _stagingSlots[i].objectKind  = ObjKindBindlessTexture;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        /// <summary>Binds a BindlessBuffer to an unbounded ByteAddressBuffer[] variable.</summary>
        public void SetBindlessBuffer(string name, BindlessBuffer bb)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = 0;
            _stagingSlots[i].objectPtr   = bb != null ? bb.Handle : 0;
            _stagingSlots[i].objectKind  = ObjKindBindlessBuffer;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        /// <summary>
        /// Binds a BindlessUAVTexture to an unbounded RWTexture2D[] variable.
        /// <paramref name="baseResource"/>: native ptr of the base texture for state tracking.
        /// </summary>
        public void SetBindlessRWTexture(string name, BindlessUAVTexture uav, IntPtr baseResource = default)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = (ulong)baseResource;
            _stagingSlots[i].objectPtr   = uav != null ? uav.Handle : 0;
            _stagingSlots[i].objectKind  = ObjKindBindlessUAVTexture;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        /// <summary>
        /// Pushes inline 32-bit constants directly into the root signature.
        /// The CBV must have been declared as a root-constants hint when the pipeline was created.
        /// Data is copied immediately; the pointer need not remain valid after this call.
        /// </summary>
        public unsafe void SetRootConstants<T>(
            string name, T* dataPtr, uint count32 = 0, uint destOffset32 = 0)
            where T : unmanaged
        {
            if (!TryGetSlot(name, out uint i)) return;
            uint byteCount      = count32 == 0 ? (uint)sizeof(T) : count32 * 4u;
            uint destByteOffset = destOffset32 * 4u;
            fixed (byte* dst = _stagingConstants)
                Buffer.MemoryCopy(dataPtr,
                    dst + i * MaxRootConstantBytes + destByteOffset,
                    MaxRootConstantBytes - destByteOffset,
                    byteCount);
            _stagingSlots[i].resourcePtr = 0;
            _stagingSlots[i].objectPtr   = 0; // resolved to pinned ptr at Snapshot time
            _stagingSlots[i].count       = count32;
            _stagingSlots[i].stride      = destOffset32;
            _stagingSlots[i].objectKind  = ObjKindRootConstants;
        }

        /// <summary>
        /// Pushes inline 32-bit constants from a <see cref="NativeArray{T}"/>.
        /// Data is copied immediately.
        /// </summary>
        public unsafe void SetRootConstants<T>(
            string name, NativeArray<T> data, uint count32 = 0, uint destOffset32 = 0)
            where T : unmanaged
        {
            if (!TryGetSlot(name, out uint i)) return;
            void* src           = NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(data);
            uint byteCount      = count32 == 0 ? (uint)(data.Length * sizeof(T)) : count32 * 4u;
            uint destByteOffset = destOffset32 * 4u;
            fixed (byte* dst = _stagingConstants)
                Buffer.MemoryCopy(src,
                    dst + i * MaxRootConstantBytes + destByteOffset,
                    MaxRootConstantBytes - destByteOffset,
                    byteCount);
            _stagingSlots[i].resourcePtr = 0;
            _stagingSlots[i].objectPtr   = 0;
            _stagingSlots[i].count       = count32;
            _stagingSlots[i].stride      = destOffset32;
            _stagingSlots[i].objectKind  = ObjKindRootConstants;
        }

        // -------------------------------------------------------------------
        // Internal dispatch helper (called by RayTracePipeline.Dispatch)
        // -------------------------------------------------------------------

        /// <summary>
        /// Snapshots the current staging bindings into the next ring-buffer slot,
        /// fills the event-data header, and returns a pointer to it for
        /// <c>IssuePluginEventAndData</c>.  Returns <see cref="IntPtr.Zero"/> on failure.
        /// </summary>
        internal unsafe IntPtr SnapshotAndBuildHeader(uint width, uint height)
        {
            if (_slotRing == null || _descriptorSetHandle == 0) return IntPtr.Zero;

            int ring = _ringIdx % RingSize;
            _ringIdx++;

            // Copy staging → pinned ring slot
            var slotArray = _slotRing[ring];
            for (int k = 0; k < (int)_slotCount; k++)
                slotArray[k] = _stagingSlots[k];

            // Copy root-constants staging data → pinned constants ring and fix up objectPtrs.
            var constArray = _constantsRing[ring];
            fixed (byte* src = _stagingConstants)
                Buffer.MemoryCopy(src, constArray.GetUnsafePtr(),
                    constArray.Length, _stagingConstants.Length);

            byte* constBase = (byte*)constArray.GetUnsafePtr();
            for (int k = 0; k < (int)_slotCount; k++)
            {
                if (slotArray[k].objectKind == ObjKindRootConstants)
                {
                    var s = slotArray[k];
                    s.objectPtr = (ulong)(constBase + k * MaxRootConstantBytes);
                    slotArray[k] = s;
                }
            }

            var header = new NativeRenderPlugin.RTS_RenderEventData
            {
                descriptorSetHandle = _descriptorSetHandle,
                bindingSlotsPtr     = (ulong)slotArray.GetUnsafePtr(),
                bindingCount        = _slotCount,
                width               = width,
                height              = height,
                _pad                = 0,
            };
            _headerRing[ring][0] = header;

            return (IntPtr)_headerRing[ring].GetUnsafePtr();
        }
    }
}
