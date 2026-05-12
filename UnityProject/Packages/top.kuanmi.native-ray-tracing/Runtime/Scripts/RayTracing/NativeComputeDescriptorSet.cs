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

        // DX12 hard limit: 64 DWORDs = 256 bytes per root-constants slot.
        private const int MaxRootConstantBytes = 256;

        // objectKind constants matching C++ CS_BindingObjectKind
        private const uint ObjKindNone            = 0;
        private const uint ObjKindAccelStruct     = 1;
        private const uint ObjKindBindlessTexture    = 2;
        private const uint ObjKindBindlessBuffer      = 3;
        private const uint ObjKindRootConstants       = 4;
        private const uint ObjKindBindlessUAVTexture  = 6;
        private const uint ObjKindNativeBuffer    = 5;

        private readonly NativeComputePipeline _pipeline;

        // Native C++ ComputeDescriptorSet handle — owns the GPU heap slice for this set.
        private ulong _descriptorSetHandle;

        // Slot layout mirrored from the pipeline (refreshed on hot-reload)
        private Dictionary<string, uint> _nameToSlot;
        private uint                     _slotCount;

        // Staging array written by Set* on the main thread (not pinned)
        private NativeRenderPlugin.CS_BindingSlot[] _stagingSlots;

        // Staging byte buffer for root-constants data (main thread, not pinned).
        // Layout: slot[i] occupies [i * MaxRootConstantBytes .. (i+1) * MaxRootConstantBytes).
        private byte[] _stagingConstants;

        // Ring buffer of pinned NativeArrays (Persistent)
        private NativeArray<NativeRenderPlugin.CS_BindingSlot>[]     _slotRing;
        private NativeArray<NativeRenderPlugin.CS_RenderEventData>[] _headerRing;
        // Pinned ring for root-constants data — the render thread reads from here.
        private NativeArray<byte>[]                                  _constantsRing;
        private int                                                  _ringIdx;

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
            _descriptorSetHandle =  NativeRenderPlugin.NR_CS_CreateDescriptorSet(pipeline.Handle);
            pipeline.OnRebuilt   += OnPipelineRebuilt;
        }

        private void CopySlotLayout(NativeComputePipeline pipeline)
        {
            _slotCount  = pipeline.SlotCount;
            _nameToSlot = new Dictionary<string, uint>(_slotCount > 0 ? (int)_slotCount : 0);
            foreach (var kv in pipeline.NameToSlot)
                _nameToSlot[kv.Key] = kv.Value;
            int effectiveSlots   = _slotCount > 0 ? (int)_slotCount : 1;
            _stagingSlots        = new NativeRenderPlugin.CS_BindingSlot[effectiveSlots];
            _stagingConstants    = new byte[effectiveSlots * MaxRootConstantBytes];
        }

        private void AllocateRingBuffers()
        {
            int effectiveSlots = _slotCount > 0 ? (int)_slotCount : 1;
            _slotRing      = new NativeArray<NativeRenderPlugin.CS_BindingSlot>[RingSize];
            _headerRing    = new NativeArray<NativeRenderPlugin.CS_RenderEventData>[RingSize];
            _constantsRing = new NativeArray<byte>[RingSize];
            for (int i = 0; i < RingSize; i++)
            {
                _slotRing[i]      = new NativeArray<NativeRenderPlugin.CS_BindingSlot>(
                    effectiveSlots, Allocator.Persistent);
                _headerRing[i]    = new NativeArray<NativeRenderPlugin.CS_RenderEventData>(
                    1, Allocator.Persistent);
                _constantsRing[i] = new NativeArray<byte>(
                    effectiveSlots * MaxRootConstantBytes, Allocator.Persistent);
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

        private void OnPipelineRebuilt(NativeComputePipeline pipeline)
        {
            // Destroy the old C++ descriptor set (its heap slice is now stale)
            if (_descriptorSetHandle != 0)
            {
                NativeRenderPlugin.NR_CS_DestroyDescriptorSet(_descriptorSetHandle);
                _descriptorSetHandle = 0;
            }

            FreeRingBuffers();
            CopySlotLayout(pipeline);
            AllocateRingBuffers();
            _descriptorSetHandle = NativeRenderPlugin.NR_CS_CreateDescriptorSet(pipeline.Handle);
        }

        // -------------------------------------------------------------------
        // IDisposable
        // -------------------------------------------------------------------

        public void Dispose()
        {
            _pipeline.OnRebuilt -= OnPipelineRebuilt;
            if (_descriptorSetHandle != 0)
            {
                NativeRenderPlugin.NR_CS_DestroyDescriptorSet(_descriptorSetHandle);
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

        /// <summary>Binds a ComputeBuffer as a read-only structured/byte-address buffer (SRV).</summary>
        public void SetBuffer(string name, IntPtr bufferPtr)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = (ulong)bufferPtr;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
            _stagingSlots[i].format      = 0;
        }

        /// <summary>Binds a ComputeBuffer as an RW (read-write) buffer (UAV).</summary>
        public void SetRWBuffer(string name, IntPtr bufferPtr)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = (ulong)bufferPtr;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
            _stagingSlots[i].format      = 0;
        }

        /// <summary>Binds a ComputeBuffer as a typed RWBuffer&lt;T&gt; UAV (e.g. RWBuffer&lt;uint2&gt;).
        /// <paramref name="dxgiFormat"/> must be the matching DXGI_FORMAT value (e.g. 0x18 = DXGI_FORMAT_R32G32_UINT).</summary>
        public void SetRWTypedBuffer(string name, IntPtr bufferPtr, int count, uint dxgiFormat)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = (ulong)bufferPtr;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = (uint)count;
            _stagingSlots[i].stride      = 0;
            _stagingSlots[i].format      = dxgiFormat;
        }

        /// <summary>Binds a GraphicsBuffer as an RWStructuredBuffer UAV with explicit element count and stride.</summary>
        public void SetRWStructuredBuffer(string name, IntPtr bufferPtr, int count, int stride)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = (ulong) bufferPtr;
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

        /// <summary>
        /// Binds a <see cref="NativeBuffer"/> as a constant buffer (CBV).
        /// The native plugin resolves the current frame's D3D12 resource at Dispatch time,
        /// so this only needs to be called once (or when the NativeBuffer instance changes).
        /// </summary>
        public void SetNativeBuffer(string name, NativeBuffer nb)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = 0;
            _stagingSlots[i].objectPtr   = nb != null ? nb.Handle : 0;
            _stagingSlots[i].objectKind  = ObjKindNativeBuffer;
            _stagingSlots[i].count       = 0;
            _stagingSlots[i].stride      = 0;
        }



        /// <summary>Binds a ComputeBuffer as a typed Buffer&lt;T&gt; SRV (e.g. Buffer&lt;float2&gt;).
        /// <paramref name="dxgiFormat"/> must be the matching DXGI_FORMAT value.</summary>
        public void SetTypedBuffer(string name, IntPtr bufferPtr, int count, uint dxgiFormat)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = (ulong)bufferPtr;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = (uint)count;
            _stagingSlots[i].stride      = 0;
            _stagingSlots[i].format      = dxgiFormat;
        }

        /// <summary>Binds a ComputeBuffer as a StructuredBuffer SRV with explicit element count and stride.</summary>
        public void SetStructuredBuffer(string name, IntPtr bufferPtr, int elementCount, int elementStride)
        {
            if (!TryGetSlot(name, out uint i)) return;
            _stagingSlots[i].resourcePtr = (ulong)bufferPtr ;
            _stagingSlots[i].objectKind  = ObjKindNone;
            _stagingSlots[i].count       = (uint)elementCount ;
            _stagingSlots[i].stride      = (uint)elementStride ;
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

        /// <summary>
        /// Binds a <see cref="BindlessUAVTexture"/> to an unbounded RWTexture2D[] variable.
        /// Call again after BindlessUAVTexture.Resize() to rebind the new descriptor range.
        /// <para><paramref name="baseResource"/>: native ptr of the base texture resource used
        /// for resource-state tracking (pass tex.GetNativeTexturePtr()).</para>
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
        /// Pushes inline 32-bit constants directly into the root signature (no GPU buffer needed).
        /// The CBV binding named <paramref name="name"/> must have been declared as a
        /// <see cref="RootConstantsHint"/> when the pipeline was created.
        /// <paramref name="dataPtr"/>: pointer to the source data. The data is <b>copied immediately</b>
        /// into an internal staging buffer so the pointer need not remain valid after this call returns.
        /// <paramref name="count32"/>: number of 32-bit values to upload (0 = derive from sizeof(T)).
        /// <paramref name="destOffset32"/>: destination offset in 32-bit values within the root constants slot.
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
            _stagingSlots[i].objectPtr   = 0; // resolved to pinned ring ptr at Snapshot time
            _stagingSlots[i].count       = count32;
            _stagingSlots[i].stride      = destOffset32;
            _stagingSlots[i].objectKind  = ObjKindRootConstants;
        }

        /// <summary>
        /// Pushes inline 32-bit constants from a <see cref="NativeArray{T}"/>.
        /// The data is <b>copied immediately</b> into an internal staging buffer.
        /// See <see cref="SetRootConstants{T}(string,T*,uint,uint)"/> for parameter documentation.
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
            _stagingSlots[i].objectPtr   = 0; // resolved to pinned ring ptr at Snapshot time
            _stagingSlots[i].count       = count32;
            _stagingSlots[i].stride      = destOffset32;
            _stagingSlots[i].objectKind  = ObjKindRootConstants;
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
            uint threadGroupX, uint threadGroupY, uint threadGroupZ)
        {
            if (_slotRing == null) return IntPtr.Zero;

            int ring = _ringIdx % RingSize;
            _ringIdx++;

            // Copy staging → pinned ring slot
            var slotArray = _slotRing[ring];
            for (int k = 0; k < (int)_slotCount; k++)
                slotArray[k] = _stagingSlots[k];

            // Copy root-constants staging data → pinned constants ring and fix up objectPtrs.
            // This ensures the render thread always reads from stable, pinned memory.
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

            var header = new NativeRenderPlugin.CS_RenderEventData
            {
                descriptorSetHandle = _descriptorSetHandle,
                threadGroupX        = threadGroupX,
                threadGroupY        = threadGroupY,
                threadGroupZ        = threadGroupZ,
                bindingCount        = _slotCount,
                bindingSlotsPtr     = (ulong)slotArray.GetUnsafePtr(),
            };
            _headerRing[ring][0] = header;

            return (IntPtr)_headerRing[ring].GetUnsafePtr();
        }
    }
}