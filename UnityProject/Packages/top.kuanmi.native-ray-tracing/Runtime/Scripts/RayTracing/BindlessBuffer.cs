using System;
using UnityEngine;

namespace NativeRender
{
    /// <summary>
    /// A GPU-visible array of ByteAddressBuffer SRV descriptors backed by a contiguous range in the
    /// global DescriptorHeapAllocator.  Mirrors <see cref="BindlessTexture"/> but targets buffer
    /// resources (ComputeBuffer / GraphicsBuffer).
    ///
    /// Usage:
    ///   var bb = new BindlessBuffer(4);
    ///   bb[0] = myComputeBuffer;
    ///   shader.SetBindlessBuffer("MyBuffers", bb);
    ///   // each frame: shader.Dispatch(...)
    ///
    ///   // To grow/shrink the array at runtime:
    ///   bb.Resize(8);
    ///   shader.SetBindlessBuffer("MyBuffers", bb);  // rebind after resize!
    ///
    ///   bb.Dispose();
    ///
    /// HLSL declaration:
    ///   ByteAddressBuffer MyBuffers[] : register(t0);
    ///   // load: MyBuffers[index].Load(byteOffset)
    ///
    /// Notes:
    ///   • Null entries (unassigned slots) read as zero.
    ///   • After Resize() the GPU handle changes; re-bind to all shaders that reference it.
    ///   • Buffers are non-owning: the BindlessBuffer does not prevent Unity from
    ///     destroying the underlying GPU resource.
    /// </summary>
    public sealed class BindlessBuffer : IDisposable
    {
        private ulong   _handle;
        private IntPtr[] _buffers;   // tracks native buffer pointers for non-owning reference
        private bool    _disposed;

        /// <summary>Opaque native handle. Used internally by RayTraceShader.SetBindlessBuffer.</summary>
        public ulong Handle => _handle;

        /// <summary>Current number of buffer slots in the array.</summary>
        public int Capacity => _buffers.Length;

        public bool IsValid => _handle != 0 && !_disposed;

        // -------------------------------------------------------------------
        // Construction
        // -------------------------------------------------------------------

        /// <summary>
        /// Creates a BindlessBuffer with |capacity| slots. All slots start as null.
        /// </summary>
        /// <param name="capacity">Number of buffer slots. Must be >= 1.</param>
        public BindlessBuffer(int capacity)
        {
            if (capacity < 1) capacity = 1;
            _handle  = NativeRenderPlugin.NR_CreateBindlessBuffer((uint)capacity);
            _buffers = new IntPtr[capacity];
            if (_handle == 0)
                Debug.LogError("[BindlessBuffer] NR_CreateBindlessBuffer failed");
        }

        // -------------------------------------------------------------------
        // Indexer — set individual buffer slots via ComputeBuffer / GraphicsBuffer
        // -------------------------------------------------------------------

        /// <summary>
        /// Sets the ComputeBuffer at |index|. Pass null to clear the slot (writes a null SRV).
        /// </summary>
        public void Set(int index, ComputeBuffer buffer)
        {
            SetNativePtrInternal(index, buffer != null ? buffer.GetNativeBufferPtr() : IntPtr.Zero);
        }

        /// <summary>
        /// Sets the GraphicsBuffer at |index|. Pass null to clear the slot (writes a null SRV).
        /// </summary>
        public void Set(int index, GraphicsBuffer buffer)
        {
            SetNativePtrInternal(index, buffer != null ? buffer.GetNativeBufferPtr() : IntPtr.Zero);
        }

        /// <summary>Sets a buffer slot directly from a native resource pointer (e.g. from Mesh.GetNativeVertexBufferPtr).</summary>
        public void SetNativePtr(int index, IntPtr ptr) => SetNativePtrInternal(index, ptr);

        private void SetNativePtrInternal(int index, IntPtr ptr)
        {
            if (index < 0 || index >= _buffers.Length)
                throw new ArgumentOutOfRangeException(nameof(index));
            _buffers[index] = ptr;
            if (IsValid)
                NativeRenderPlugin.NR_BB_SetBuffer(_handle, (uint)index, ptr);
        }

        // -------------------------------------------------------------------
        // Resize
        // -------------------------------------------------------------------

        /// <summary>
        /// Resizes the buffer array to |newCapacity| slots.
        /// <para>
        /// On grow: new slots are null. Existing buffers at indices below the old capacity
        /// are re-registered in the new descriptor range automatically.
        /// </para>
        /// <para>
        /// On shrink: buffers beyond |newCapacity| are dropped from the array.
        /// </para>
        /// <para>
        /// IMPORTANT: After resize the GPU descriptor base address changes.
        /// Call shader.SetBindlessBuffer(name, this) to rebind before the next dispatch.
        /// </para>
        /// </summary>
        public void Resize(int newCapacity)
        {
            if (newCapacity < 1) newCapacity = 1;
            if (!IsValid) return;

            // Resize the native heap range (the C++ side re-writes all existing SRVs)
            NativeRenderPlugin.NR_BB_Resize(_handle, (uint)newCapacity);

            // Resize the C# tracking array
            Array.Resize(ref _buffers, newCapacity);
        }

        // -------------------------------------------------------------------
        // IDisposable
        // -------------------------------------------------------------------

        public void Dispose()
        {
            if (!_disposed && _handle != 0)
            {
                NativeRenderPlugin.NR_DestroyBindlessBuffer(_handle);
                _handle = 0;
            }
            _disposed = true;
        }
    }
}
