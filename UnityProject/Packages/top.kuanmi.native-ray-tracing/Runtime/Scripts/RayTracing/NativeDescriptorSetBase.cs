using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Rendering;

namespace NativeRender
{
    /// <summary>
    /// Shared base for <see cref="NativeComputeDescriptorSet"/> and
    /// <see cref="NativeRayTraceDescriptorSet"/>.
    ///
    /// Holds all pipeline-agnostic state: slot layout, staging arrays, the
    /// slot/constants ring buffers, and every <c>Set*</c> binding method.
    /// Subclasses own the typed header ring and supply pipeline-specific
    /// create/destroy/snapshot logic.
    /// </summary>
    public abstract class NativeDescriptorSetBase : IDisposable
    {
        // Ring-buffer depth — supports this many in-flight Dispatch calls per frame.
        protected const int RingSize = 8;

        // DX12 hard limit for root-constants: 64 DWORDs = 256 bytes per slot.
        protected const int MaxRootConstantBytes = 256;

        // objectKind constants — must match C++ CS_BindingObjectKind
        protected const uint ObjKindNone               = 0;
        protected const uint ObjKindAccelStruct        = 1;
        protected const uint ObjKindBindlessTexture    = 2;
        protected const uint ObjKindBindlessBuffer     = 3;
        protected const uint ObjKindRootConstants      = 4;
        protected const uint ObjKindNativeBuffer       = 5;
        protected const uint ObjKindBindlessUAVTexture = 6;
        protected const uint ObjKindNativeGpuBuffer    = 7;

        // Native C++ descriptor-set handle.
        protected ulong _descriptorSetHandle;

        // Slot layout mirrored from the pipeline (refreshed on hot-reload).
        protected Dictionary<string, uint> _nameToSlot;
        protected uint                     _slotCount;

        // Staging arrays written by Set* on the main thread (not pinned).
        protected NativeRenderPlugin.CS_BindingSlot[] _stagingSlots;
        protected byte[]                              _stagingConstants;

        // Pinned ring buffers for slots and root-constants — read by the render thread.
        protected NativeArray<NativeRenderPlugin.CS_BindingSlot>[] _slotRing;
        protected NativeArray<byte>[]                              _constantsRing;
        protected int                                              _ringIdx;

        // -------------------------------------------------------------------
        // Shared ring-buffer helpers
        // -------------------------------------------------------------------

        protected void CopySlotLayoutCommon(uint slotCount, IReadOnlyDictionary<string, uint> nameToSlot)
        {
            _slotCount  = slotCount;
            _nameToSlot = new Dictionary<string, uint>(slotCount > 0 ? (int)slotCount : 0);
            foreach (var kv in nameToSlot)
                _nameToSlot[kv.Key] = kv.Value;
            int eff           = _slotCount > 0 ? (int)_slotCount : 1;
            _stagingSlots     = new NativeRenderPlugin.CS_BindingSlot[eff];
            _stagingConstants = new byte[eff * MaxRootConstantBytes];
        }

        protected void AllocateRingBuffersCommon()
        {
            int eff        = _slotCount > 0 ? (int)_slotCount : 1;
            _slotRing      = new NativeArray<NativeRenderPlugin.CS_BindingSlot>[RingSize];
            _constantsRing = new NativeArray<byte>[RingSize];
            for (int i = 0; i < RingSize; i++)
            {
                _slotRing[i]      = new NativeArray<NativeRenderPlugin.CS_BindingSlot>(eff, Allocator.Persistent);
                _constantsRing[i] = new NativeArray<byte>(eff * MaxRootConstantBytes, Allocator.Persistent);
            }
            _ringIdx = 0;
        }

        protected void FreeRingBuffersCommon()
        {
            if (_slotRing == null) return;
            for (int i = 0; i < RingSize; i++)
            {
                if (_slotRing[i].IsCreated)      _slotRing[i].Dispose();
                if (_constantsRing[i].IsCreated) _constantsRing[i].Dispose();
            }
            _slotRing      = null;
            _constantsRing = null;
        }

        /// <summary>
        /// Copies the current staging slots and root-constants into the next pinned
        /// ring-buffer slot, fixes up root-constant object pointers, and returns the
        /// ring index.  Returns -1 if the ring has not been allocated yet.
        /// </summary>
        protected unsafe int CopyStagingToRing()
        {
            if (_slotRing == null) return -1;

            int ring = _ringIdx % RingSize;
            _ringIdx++;

            // Copy staging → pinned ring slot
            var slotArray = _slotRing[ring];
            for (int k = 0; k < (int)_slotCount; k++)
                slotArray[k] = _stagingSlots[k];

            // Copy root-constants staging data → pinned constants ring.
            var constArray = _constantsRing[ring];
            fixed (byte* src = _stagingConstants)
                Buffer.MemoryCopy(src, constArray.GetUnsafePtr(),
                    constArray.Length, _stagingConstants.Length);

            // Fix up objectPtrs for root-constants slots to point into pinned memory.
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

            return ring;
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
        
        
        public void SetRWTypedBuffer(string name, NativeBuffer nativeBuffer, int count, uint dxgiFormat)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = 0;
            _stagingSlots[i].objectPtr   = nativeBuffer.Handle;
            _stagingSlots[i].objectKind  = ObjKindNativeBuffer;
            _stagingSlots[i].count       = (uint)count;
            _stagingSlots[i].stride      = 0;
            _stagingSlots[i].format      = dxgiFormat;
        }

        /// <summary>Binds a <see cref="NativeGpuBuffer"/> as a typed RW buffer UAV (e.g. RWBuffer&lt;float2&gt;).</summary>
        public void SetRWTypedBuffer(string name, NativeGpuBuffer gpuBuffer, int count, uint dxgiFormat)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = 0;
            _stagingSlots[i].objectPtr   = gpuBuffer.Handle;
            _stagingSlots[i].objectKind  = ObjKindNativeGpuBuffer;
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
        
        /// <summary>Binds a GraphicsBuffer as a constant buffer (CBV).</summary>
        public void SetConstantBuffer(string name, NativeBuffer nativeBuffer)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = 0;
            _stagingSlots[i].objectPtr   = nativeBuffer.Handle;
            _stagingSlots[i].objectKind  = ObjKindNativeBuffer;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }

        /// <summary>Binds a <see cref="NativeBuffer"/> as a constant buffer (CBV).</summary>
        public void SetNativeBuffer(string name, NativeBuffer nb)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = 0;
            _stagingSlots[i].objectPtr   = nb?.Handle ?? 0;
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
        
        public void SetTypedBuffer(string name, NativeBuffer nativeBuffer, int count, uint dxgiFormat)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = 0;
            _stagingSlots[i].objectPtr   = nativeBuffer.Handle;
            _stagingSlots[i].objectKind  = ObjKindNativeBuffer;
            _stagingSlots[i].count       = (uint)count;
            _stagingSlots[i].stride      = 0;
            _stagingSlots[i].format      = dxgiFormat;
        }

        /// <summary>Binds a <see cref="NativeGpuBuffer"/> as a typed buffer SRV (e.g. Buffer&lt;float2&gt;).</summary>
        public void SetTypedBuffer(string name, NativeGpuBuffer gpuBuffer, int count, uint dxgiFormat)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = 0;
            _stagingSlots[i].objectPtr   = gpuBuffer.Handle;
            _stagingSlots[i].objectKind  = ObjKindNativeGpuBuffer;
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

        public abstract void Dispose();
    }
}
