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
    ///
    /// Binding names can be resolved to a process-global integer id via
    /// <see cref="PropertyToID"/> (mirrors <c>UnityEngine.Shader.PropertyToID</c>).
    /// Cache the id once and pass it to the <c>int</c>-id <c>Set*</c> overloads to
    /// skip the per-call string hash.  Ids survive pipeline hot-reloads — each set
    /// rebuilds its id→slot map from the refreshed layout.
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
        protected const uint ObjKindNativeBuffer       = 5; // unified plugin buffer (VolatileConstantBuffer / DeviceBuffer / UploadBuffer)
        protected const uint ObjKindBindlessUAVTexture = 6;

        // -------------------------------------------------------------------
        // Global binding-name interning (à la UnityEngine.Shader.PropertyToID)
        // -------------------------------------------------------------------

        private static readonly object                  s_idLock   = new object();
        private static readonly Dictionary<string, int> s_nameToId = new Dictionary<string, int>(256);

        /// <summary>
        /// Interns a binding (HLSL variable) name to a process-global integer id, exactly like
        /// <c>UnityEngine.Shader.PropertyToID</c>.  The id is stable for the lifetime of the process,
        /// so the idiom is to cache it once in a <c>static readonly int</c> and pass it to the int-id
        /// <c>Set*</c> overloads, avoiding the per-call dictionary string-hash.
        /// </summary>
        public static int PropertyToID(string name)
        {
            if (name == null) return -1;
            lock (s_idLock)
            {
                if (!s_nameToId.TryGetValue(name, out int id))
                {
                    id = s_nameToId.Count;
                    s_nameToId[name] = id;
                }
                return id;
            }
        }

        private static int GlobalIdCount
        {
            get { lock (s_idLock) { return s_nameToId.Count; } }
        }

        // Native C++ descriptor-set handle.
        protected ulong _descriptorSetHandle;

        // Slot layout mirrored from the pipeline (refreshed on hot-reload).
        protected uint _slotCount;

        // Maps a global PropertyToID id → this set's slot index (-1 = not a binding of this set).
        // Rebuilt whenever the slot layout is (re)copied, so cached ids survive hot-reloads.
        private int[] _idToSlot;

        // Staging arrays written by Set* on the main thread (not pinned).
        protected NativeRenderPlugin.CS_BindingSlot[] _stagingSlots;
        protected byte[]                              _stagingConstants;

        // Which slots have ever been bound as root-constants, and whether any have.
        // Lets CopyStagingToRing skip the constants copy + objectPtr fix-up entirely
        // for shaders that use no root-constants (the common case) and otherwise touch
        // only the root-constant slots instead of scanning every slot.
        protected bool[] _slotIsRootConst;
        protected bool   _anyRootConst;

        // Pinned ring buffers for slots and root-constants — read by the render thread.
        protected NativeArray<NativeRenderPlugin.CS_BindingSlot>[] _slotRing;
        protected NativeArray<byte>[]                              _constantsRing;
        protected int                                              _ringIdx;

        // -------------------------------------------------------------------
        // Shared ring-buffer helpers
        // -------------------------------------------------------------------

        protected void CopySlotLayoutCommon(uint slotCount, IReadOnlyDictionary<string, uint> nameToSlot)
        {
            _slotCount = slotCount;

            // Intern every binding name, then map its global id → local slot index.  Sizing the
            // array to the current global id count keeps slot resolution a single bounds-checked
            // index — no string (or even dictionary) lookup on the binding hot path.
            foreach (var kv in nameToSlot)
                PropertyToID(kv.Key);

            _idToSlot = new int[GlobalIdCount];
            for (int k = 0; k < _idToSlot.Length; k++)
                _idToSlot[k] = -1;
            foreach (var kv in nameToSlot)
                _idToSlot[PropertyToID(kv.Key)] = (int)kv.Value;

            int eff = _slotCount > 0 ? (int)_slotCount : 1;
            _stagingSlots     = new NativeRenderPlugin.CS_BindingSlot[eff];
            _stagingConstants = new byte[eff * MaxRootConstantBytes];
            _slotIsRootConst  = new bool[eff];
            _anyRootConst     = false;
        }

        protected void AllocateRingBuffersCommon()
        {
            int eff = _slotCount > 0 ? (int)_slotCount : 1;
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
                if (_slotRing[i].IsCreated) _slotRing[i].Dispose();
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

            // Blit staging → pinned ring slot in one shot (CS_BindingSlot is blittable).
            var slotArray = _slotRing[ring];
            slotArray.CopyFrom(_stagingSlots);

            // Root-constants are the only bindings whose payload lives in the constants
            // ring; skip the whole copy + fix-up when the shader uses none.
            if (_anyRootConst)
            {
                // Copy root-constants staging data → pinned constants ring.
                var constArray = _constantsRing[ring];
                fixed (byte* src = _stagingConstants)
                    Buffer.MemoryCopy(src, constArray.GetUnsafePtr(),
                        constArray.Length, _stagingConstants.Length);

                // Fix up objectPtrs for root-constants slots to point into pinned memory.
                byte* constBase = (byte*)constArray.GetUnsafePtr();
                for (int k = 0; k < (int)_slotCount; k++)
                {
                    if (_slotIsRootConst[k] && slotArray[k].objectKind == ObjKindRootConstants)
                    {
                        var s = slotArray[k];
                        s.objectPtr  = (ulong)(constBase + k * MaxRootConstantBytes);
                        slotArray[k] = s;
                    }
                }
            }

            return ring;
        }

        // -------------------------------------------------------------------
        // Resource binding  (main thread — writes _stagingSlots)
        //
        // Each binder has two overloads: an int-id overload (the implementation,
        // resolving the slot via a single bounds-checked array index) and a thin
        // string overload that interns the name through PropertyToID.  Both no-op
        // when the name/id is not a binding of this descriptor set.
        // -------------------------------------------------------------------

        /// <summary>Resolves a global PropertyToID id to this set's slot index, or -1 if unknown.</summary>
        private int SlotFromId(int nameId)
        {
            var map = _idToSlot;
            return (map != null && (uint)nameId < (uint)map.Length) ? map[nameId] : -1;
        }

        /// <summary>
        /// Resolves a binding name to this set's slot index, or -1 if the name is not a binding
        /// of this set.  Equivalent to <c>SlotFromId(PropertyToID(name))</c>; exposed so callers
        /// can validate a name once and decide whether to bind.
        /// </summary>
        public int GetSlotIndex(string name) => SlotFromId(PropertyToID(name));

        /// <summary>Binds a raw/structured buffer (SRV).</summary>
        public void SetBuffer(int nameId, IntPtr bufferPtr)
        {
            int i = SlotFromId(nameId);
            if (i < 0) return;
            _stagingSlots[i].objectPtr   = (ulong)bufferPtr;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
            _stagingSlots[i].format      = 0;
        }
        /// <summary>Binds a raw/structured buffer (SRV) by HLSL variable name.</summary>
        public void SetBuffer(string name, IntPtr bufferPtr) => SetBuffer(PropertyToID(name), bufferPtr);

        /// <summary>Binds an RW buffer (UAV).</summary>
        public void SetRWBuffer(int nameId, IntPtr bufferPtr)
        {
            int i = SlotFromId(nameId);
            if (i < 0) return;
            _stagingSlots[i].objectPtr   = (ulong)bufferPtr;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
            _stagingSlots[i].format      = 0;
        }
        /// <summary>Binds an RW buffer (UAV) by HLSL variable name.</summary>
        public void SetRWBuffer(string name, IntPtr bufferPtr) => SetRWBuffer(PropertyToID(name), bufferPtr);

        /// <summary>Binds an <see cref="UploadBuffer"/> as a raw RW buffer UAV.</summary>
        public void SetRWBuffer(int nameId, UploadBuffer buffer)
        {
            int i = SlotFromId(nameId);
            if (i < 0) return;
            _stagingSlots[i].objectPtr   = buffer != null ? buffer.Handle : 0;
            _stagingSlots[i].objectKind  = ObjKindNativeBuffer;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
            _stagingSlots[i].format      = 0;
        }
        /// <summary>Binds an <see cref="UploadBuffer"/> as a raw RW buffer UAV.</summary>
        public void SetRWBuffer(string name, UploadBuffer buffer) => SetRWBuffer(PropertyToID(name), buffer);

        /// <summary>Binds an RW typed buffer (UAV) with an explicit DXGI_FORMAT.</summary>
        public void SetRWTypedBuffer(int nameId, IntPtr bufferPtr, int count, uint dxgiFormat)
        {
            int i = SlotFromId(nameId);
            if (i < 0) return;
            _stagingSlots[i].objectPtr   = (ulong)bufferPtr;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = (uint)count;
            _stagingSlots[i].stride      = 0;
            _stagingSlots[i].format      = dxgiFormat;
        }
        /// <summary>Binds an RW typed buffer (UAV) with an explicit DXGI_FORMAT.</summary>
        public void SetRWTypedBuffer(string name, IntPtr bufferPtr, int count, uint dxgiFormat)
            => SetRWTypedBuffer(PropertyToID(name), bufferPtr, count, dxgiFormat);

        /// <summary>Binds a <see cref="DeviceBuffer"/> as a typed RW buffer UAV (e.g. RWBuffer&lt;float2&gt;).</summary>
        public void SetRWTypedBuffer(int nameId, DeviceBuffer deviceBuffer, int count, uint dxgiFormat)
        {
            int i = SlotFromId(nameId);
            if (i < 0) return;
            _stagingSlots[i].objectPtr   = deviceBuffer.Handle;
            _stagingSlots[i].objectKind  = ObjKindNativeBuffer;
            _stagingSlots[i].count       = (uint)count;
            _stagingSlots[i].stride      = 0;
            _stagingSlots[i].format      = dxgiFormat;
        }
        /// <summary>Binds a <see cref="DeviceBuffer"/> as a typed RW buffer UAV (e.g. RWBuffer&lt;float2&gt;).</summary>
        public void SetRWTypedBuffer(string name, DeviceBuffer deviceBuffer, int count, uint dxgiFormat)
            => SetRWTypedBuffer(PropertyToID(name), deviceBuffer, count, dxgiFormat);

        /// <summary>Binds a structured buffer (UAV) with explicit element count and stride.</summary>
        public void SetRWStructuredBuffer(int nameId, IntPtr bufferPtr, int count, int stride)
        {
            int i = SlotFromId(nameId);
            if (i < 0) return;
            _stagingSlots[i].objectPtr   = (ulong)bufferPtr;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = (uint)count;
            _stagingSlots[i].stride      = (uint)stride;
        }
        /// <summary>Binds a structured buffer (UAV) with explicit element count and stride.</summary>
        public void SetRWStructuredBuffer(string name, IntPtr bufferPtr, int count, int stride)
            => SetRWStructuredBuffer(PropertyToID(name), bufferPtr, count, stride);

        public void SetRWStructuredBuffer(int nameId, GraphicsBuffer buffe)
            => SetRWStructuredBuffer(nameId, buffe.GetNativeBufferPtr(), buffe.count, buffe.stride);
        public void SetRWStructuredBuffer(string name, GraphicsBuffer buffe)
            => SetRWStructuredBuffer(PropertyToID(name), buffe.GetNativeBufferPtr(), buffe.count, buffe.stride);

        /// <summary>Binds an <see cref="UploadBuffer"/> as a RWStructuredBuffer UAV (resolved by handle).
        /// Requires the buffer to have been created with UAV support.</summary>
        public void SetRWStructuredBuffer(int nameId, UploadBuffer buffer, int count, int stride)
        {
            int i = SlotFromId(nameId);
            if (i < 0) return;
            _stagingSlots[i].objectPtr   = buffer != null ? buffer.Handle : 0;
            _stagingSlots[i].objectKind  = ObjKindNativeBuffer;
            _stagingSlots[i].count       = (uint)count;
            _stagingSlots[i].stride      = (uint)stride;
        }
        /// <summary>Binds an <see cref="UploadBuffer"/> as a RWStructuredBuffer UAV (resolved by handle).</summary>
        public void SetRWStructuredBuffer(string name, UploadBuffer buffer, int count, int stride)
            => SetRWStructuredBuffer(PropertyToID(name), buffer, count, stride);

        /// <summary>Binds a Texture2D or RenderTexture as a read-only texture (SRV).</summary>
        public void SetTexture(int nameId, IntPtr texturePtr)
        {
            int i = SlotFromId(nameId);
            if (i < 0) return;
            _stagingSlots[i].objectPtr   = (ulong)texturePtr;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }
        /// <summary>Binds a Texture2D or RenderTexture as a read-only texture (SRV).</summary>
        public void SetTexture(string name, IntPtr texturePtr) => SetTexture(PropertyToID(name), texturePtr);

        /// <summary>Binds a RenderTexture as a read-write texture (UAV).</summary>
        public void SetRWTexture(int nameId, IntPtr texturePtr)
        {
            int i = SlotFromId(nameId);
            if (i < 0) return;
            _stagingSlots[i].objectPtr   = (ulong)texturePtr;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }
        /// <summary>Binds a RenderTexture as a read-write texture (UAV).</summary>
        public void SetRWTexture(string name, IntPtr texturePtr) => SetRWTexture(PropertyToID(name), texturePtr);

        /// <summary>Binds a GraphicsBuffer as a constant buffer (CBV).</summary>
        public void SetConstantBuffer(int nameId, IntPtr bufferPtr)
        {
            int i = SlotFromId(nameId);
            if (i < 0) return;
            _stagingSlots[i].objectPtr   = (ulong)bufferPtr;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }
        /// <summary>Binds a GraphicsBuffer as a constant buffer (CBV).</summary>
        public void SetConstantBuffer(string name, IntPtr bufferPtr) => SetConstantBuffer(PropertyToID(name), bufferPtr);

        /// <summary>Binds a <see cref="VolatileConstantBuffer"/> as a constant buffer (CBV).</summary>
        public void SetConstantBuffer(int nameId, VolatileConstantBuffer volatileConstantBuffer)
        {
            int i = SlotFromId(nameId);
            if (i < 0) return;
            _stagingSlots[i].objectPtr   = volatileConstantBuffer.Handle;
            _stagingSlots[i].objectKind  = ObjKindNativeBuffer;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }
        /// <summary>Binds a <see cref="VolatileConstantBuffer"/> as a constant buffer (CBV).</summary>
        public void SetConstantBuffer(string name, VolatileConstantBuffer volatileConstantBuffer)
            => SetConstantBuffer(PropertyToID(name), volatileConstantBuffer);

        /// <summary>Binds a typed buffer SRV (e.g. Buffer&lt;float2&gt;) with explicit DXGI_FORMAT.</summary>
        public void SetTypedBuffer(int nameId, IntPtr bufferPtr, int count, uint dxgiFormat)
        {
            int i = SlotFromId(nameId);
            if (i < 0) return;
            _stagingSlots[i].objectPtr   = (ulong)bufferPtr;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = (uint)count;
            _stagingSlots[i].stride      = 0;
            _stagingSlots[i].format      = dxgiFormat;
        }
        /// <summary>Binds a typed buffer SRV (e.g. Buffer&lt;float2&gt;) with explicit DXGI_FORMAT.</summary>
        public void SetTypedBuffer(string name, IntPtr bufferPtr, int count, uint dxgiFormat)
            => SetTypedBuffer(PropertyToID(name), bufferPtr, count, dxgiFormat);

        /// <summary>Binds a <see cref="DeviceBuffer"/> as a typed buffer SRV (e.g. Buffer&lt;float2&gt;).</summary>
        public void SetTypedBuffer(int nameId, DeviceBuffer deviceBuffer, int count, uint dxgiFormat)
        {
            int i = SlotFromId(nameId);
            if (i < 0) return;
            _stagingSlots[i].objectPtr   = deviceBuffer.Handle;
            _stagingSlots[i].objectKind  = ObjKindNativeBuffer;
            _stagingSlots[i].count       = (uint)count;
            _stagingSlots[i].stride      = 0;
            _stagingSlots[i].format      = dxgiFormat;
        }
        /// <summary>Binds a <see cref="DeviceBuffer"/> as a typed buffer SRV (e.g. Buffer&lt;float2&gt;).</summary>
        public void SetTypedBuffer(string name, DeviceBuffer deviceBuffer, int count, uint dxgiFormat)
            => SetTypedBuffer(PropertyToID(name), deviceBuffer, count, dxgiFormat);

        /// <summary>Binds a StructuredBuffer SRV with explicit element count and stride.</summary>
        public void SetStructuredBuffer(int nameId, IntPtr bufferPtr, int elementCount, int elementStride)
        {
            int i = SlotFromId(nameId);
            if (i < 0) return;
            _stagingSlots[i].objectPtr   = (ulong)bufferPtr;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = (uint)elementCount;
            _stagingSlots[i].stride      = (uint)elementStride;
        }
        /// <summary>Binds a StructuredBuffer SRV with explicit element count and stride.</summary>
        public void SetStructuredBuffer(string name, IntPtr bufferPtr, int elementCount, int elementStride)
            => SetStructuredBuffer(PropertyToID(name), bufferPtr, elementCount, elementStride);

        public void SetStructuredBuffer(int nameId, GraphicsBuffer buffer)
            => SetStructuredBuffer(nameId, buffer.GetNativeBufferPtr(), buffer.count, buffer.stride);
        public void SetStructuredBuffer(string name, GraphicsBuffer buffer)
            => SetStructuredBuffer(PropertyToID(name), buffer.GetNativeBufferPtr(), buffer.count, buffer.stride);

        /// <summary>Binds an <see cref="UploadBuffer"/> as a StructuredBuffer SRV. Resolved by handle
        /// (the resource pointer is resolved natively at dispatch), so no cached NativePtr is needed.</summary>
        public void SetStructuredBuffer(int nameId, UploadBuffer buffer, int elementCount, int elementStride)
        {
            int i = SlotFromId(nameId);
            if (i < 0) return;
            _stagingSlots[i].objectPtr   = buffer != null ? buffer.Handle : 0;
            _stagingSlots[i].objectKind  = ObjKindNativeBuffer;
            _stagingSlots[i].count       = (uint)elementCount;
            _stagingSlots[i].stride      = (uint)elementStride;
        }
        /// <summary>Binds an <see cref="UploadBuffer"/> as a StructuredBuffer SRV (resolved by handle).</summary>
        public void SetStructuredBuffer(string name, UploadBuffer buffer, int elementCount, int elementStride)
            => SetStructuredBuffer(PropertyToID(name), buffer, elementCount, elementStride);

        public void SetStructuredBuffer(int nameId, UploadBuffer buffer)
            => SetStructuredBuffer(nameId, buffer, buffer != null ? buffer.count : 0, buffer != null ? buffer.stride : 0);
        public void SetStructuredBuffer(string name, UploadBuffer buffer)
            => SetStructuredBuffer(PropertyToID(name), buffer, buffer != null ? buffer.count : 0, buffer != null ? buffer.stride : 0);

        /// <summary>Binds a RaytracingAccelerationStructure (TLAS).</summary>
        public void SetAccelerationStructure(int nameId, RayTracingAccelerationStructure accelStructure)
        {
            int i = SlotFromId(nameId);
            if (i < 0) return;
            _stagingSlots[i].objectPtr   = accelStructure != null ? accelStructure.Handle : 0;
            _stagingSlots[i].objectKind  = ObjKindAccelStruct;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }
        /// <summary>Binds a RaytracingAccelerationStructure (TLAS) by HLSL variable name.</summary>
        public void SetAccelerationStructure(string name, RayTracingAccelerationStructure accelStructure)
            => SetAccelerationStructure(PropertyToID(name), accelStructure);

        /// <summary>Binds a BindlessTexture to an unbounded Texture2D[] variable.</summary>
        public void SetBindlessTexture(int nameId, BindlessTexture bt)
        {
            int i = SlotFromId(nameId);
            if (i < 0) return;
            _stagingSlots[i].objectPtr   = bt != null ? bt.Handle : 0;
            _stagingSlots[i].objectKind  = ObjKindBindlessTexture;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }
        /// <summary>Binds a BindlessTexture to an unbounded Texture2D[] variable.</summary>
        public void SetBindlessTexture(string name, BindlessTexture bt) => SetBindlessTexture(PropertyToID(name), bt);

        /// <summary>Binds a BindlessBuffer to an unbounded ByteAddressBuffer[] variable.</summary>
        public void SetBindlessBuffer(int nameId, BindlessBuffer bb)
        {
            int i = SlotFromId(nameId);
            if (i < 0) return;
            _stagingSlots[i].objectPtr   = bb != null ? bb.Handle : 0;
            _stagingSlots[i].objectKind  = ObjKindBindlessBuffer;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }
        /// <summary>Binds a BindlessBuffer to an unbounded ByteAddressBuffer[] variable.</summary>
        public void SetBindlessBuffer(string name, BindlessBuffer bb) => SetBindlessBuffer(PropertyToID(name), bb);

        /// <summary>
        /// Binds a BindlessUAVTexture to an unbounded RWTexture2D[] variable.
        /// </summary>
        /// <param name="baseResource">Unused — retained for source compatibility. Resource-state
        /// tracking now walks every element of the bindless array natively, so no base pointer
        /// is needed.</param>
        public void SetBindlessRWTexture(int nameId, BindlessUAVTexture uav, IntPtr baseResource = default)
        {
            int i = SlotFromId(nameId);
            if (i < 0) return;
            _stagingSlots[i].objectPtr   = uav != null ? uav.Handle : 0;
            _stagingSlots[i].objectKind  = ObjKindBindlessUAVTexture;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }
        /// <summary>Binds a BindlessUAVTexture to an unbounded RWTexture2D[] variable.</summary>
        public void SetBindlessRWTexture(string name, BindlessUAVTexture uav, IntPtr baseResource = default)
            => SetBindlessRWTexture(PropertyToID(name), uav, baseResource);

        /// <summary>
        /// Pushes inline 32-bit constants directly into the root signature.
        /// The CBV must have been declared as a root-constants hint when the pipeline was created.
        /// Data is copied immediately; the pointer need not remain valid after this call.
        /// </summary>
        public unsafe void SetRootConstants<T>(
            int nameId, T* dataPtr, uint count32 = 0, uint destOffset32 = 0)
            where T : unmanaged
        {
            int i = SlotFromId(nameId);
            if (i < 0) return;
            uint byteCount      = count32 == 0 ? (uint)sizeof(T) : count32 * 4u;
            uint destByteOffset = destOffset32 * 4u;
            fixed (byte* dst = _stagingConstants)
                Buffer.MemoryCopy(dataPtr,
                    dst + i * MaxRootConstantBytes + destByteOffset,
                    MaxRootConstantBytes - destByteOffset,
                    byteCount);
            _stagingSlots[i].objectPtr   = 0; // resolved to pinned ptr at Snapshot time
            _stagingSlots[i].count       = count32;
            _stagingSlots[i].stride      = destOffset32;
            _stagingSlots[i].objectKind  = ObjKindRootConstants;
            _slotIsRootConst[i]          = true;
            _anyRootConst                = true;
        }
        /// <summary>Pushes inline 32-bit constants directly into the root signature.</summary>
        public unsafe void SetRootConstants<T>(
            string name, T* dataPtr, uint count32 = 0, uint destOffset32 = 0)
            where T : unmanaged
            => SetRootConstants(PropertyToID(name), dataPtr, count32, destOffset32);

        /// <summary>
        /// Pushes inline 32-bit constants from a <see cref="NativeArray{T}"/>.
        /// Data is copied immediately.
        /// </summary>
        public unsafe void SetRootConstants<T>(
            int nameId, NativeArray<T> data, uint count32 = 0, uint destOffset32 = 0)
            where T : unmanaged
        {
            int i = SlotFromId(nameId);
            if (i < 0) return;
            void* src            = NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(data);
            uint  byteCount      = count32 == 0 ? (uint)(data.Length * sizeof(T)) : count32 * 4u;
            uint  destByteOffset = destOffset32 * 4u;
            fixed (byte* dst = _stagingConstants)
                Buffer.MemoryCopy(src,
                    dst + i * MaxRootConstantBytes + destByteOffset,
                    MaxRootConstantBytes - destByteOffset,
                    byteCount);
            _stagingSlots[i].objectPtr   = 0;
            _stagingSlots[i].count       = count32;
            _stagingSlots[i].stride      = destOffset32;
            _stagingSlots[i].objectKind  = ObjKindRootConstants;
            _slotIsRootConst[i]          = true;
            _anyRootConst                = true;
        }
        /// <summary>Pushes inline 32-bit constants from a <see cref="NativeArray{T}"/>.</summary>
        public unsafe void SetRootConstants<T>(
            string name, NativeArray<T> data, uint count32 = 0, uint destOffset32 = 0)
            where T : unmanaged
            => SetRootConstants(PropertyToID(name), data, count32, destOffset32);

        public abstract void Dispose();
    }
}
