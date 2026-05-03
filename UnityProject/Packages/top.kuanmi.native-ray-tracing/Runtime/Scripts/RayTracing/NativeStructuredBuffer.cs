using System;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Rendering;

namespace NativeRender
{
    /// <summary>
    /// Single D3D12 upload-heap structured buffer managed by the native plugin.
    /// Persistently mapped; CPU writes are visible to the GPU on the same command-list
    /// submission (no triple-buffering required because writes precede submission).
    ///
    /// Usage:
    ///   var nsb = new NativeStructuredBuffer(capacity, Marshal.SizeOf&lt;MyStruct&gt;());
    ///   // each frame (or when data changes):
    ///   nsb.UploadRange(array, dstOffset: 0, count: array.Length);
    ///   pipeline.SetStructuredBuffer("gIn_Data", nsb);
    ///   // cleanup:
    ///   nsb.Dispose();
    /// </summary>
    public sealed class NativeStructuredBuffer : IDisposable
    {
        /// <summary>Opaque plugin handle (NativeStructuredBuffer*).</summary>
        public ulong Handle { get; private set; }

        /// <summary>Current element capacity of the underlying D3D12 buffer.</summary>
        public int Capacity => (int)NativeRenderPlugin.NR_NSB_GetCapacity(Handle);

        /// <summary>Element stride in bytes (fixed at construction).</summary>
        public int Stride { get; }

        private bool _disposed;

        /// <summary>Allocates an upload-heap structured buffer with <paramref name="capacity"/> elements.</summary>
        public NativeStructuredBuffer(int capacity, int elementStride)
        {
            if (capacity    <= 0) throw new ArgumentOutOfRangeException(nameof(capacity));
            if (elementStride <= 0) throw new ArgumentOutOfRangeException(nameof(elementStride));
            Stride = elementStride;
            Handle = NativeRenderPlugin.NR_CreateNativeStructuredBuffer((uint)capacity, (uint)elementStride);
            if (Handle == 0)
                throw new InvalidOperationException("NR_CreateNativeStructuredBuffer failed (renderer not ready?)");
        }

        /// <summary>
        /// Copies <paramref name="count"/> elements from <paramref name="data"/>
        /// into the buffer starting at element index <paramref name="dstOffset"/>.
        /// </summary>
        public unsafe void UploadRange<T>(T[] data, int dstOffset, int count) where T : unmanaged
        {
            if (_disposed) throw new ObjectDisposedException(nameof(NativeStructuredBuffer));
            if (data == null) throw new ArgumentNullException(nameof(data));
            
            fixed (T* ptr = data)
                NativeRenderPlugin.NR_NSB_UploadRange(Handle, ptr + dstOffset, (uint)dstOffset, (uint)count);
        }

        /// <summary>
        /// Copies <paramref name="count"/> elements from a <see cref="NativeArray{T}"/>
        /// into the buffer starting at element index <paramref name="dstOffset"/>.
        /// </summary>
        public unsafe void UploadRange<T>(NativeArray<T> data, int dstOffset, int count) where T : unmanaged
        {
            if (_disposed) throw new ObjectDisposedException(nameof(NativeStructuredBuffer));
            NativeRenderPlugin.NR_NSB_UploadRange(
                Handle,
                NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(data),
                (uint)dstOffset,
                (uint)count);
        }

        /// <summary>
        /// Grows the buffer to at least <paramref name="newCapacity"/> elements,
        /// preserving existing data. No-op if already large enough.
        /// </summary>
        public void Grow(int newCapacity)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(NativeStructuredBuffer));
            NativeRenderPlugin.NR_NSB_Grow(Handle, (uint)newCapacity);
        }

        /// <summary>Returns the ID3D12Resource* as IntPtr for SRV binding.</summary>
        public IntPtr NativePtr => NativeRenderPlugin.NR_NSB_GetNativePtr(Handle);

        /// <summary>
        /// Records a <c>CopyBufferRegion</c> command (with surrounding resource barriers)
        /// into <paramref name="cmd"/> to upload this frame's staged data into the
        /// GPU-resident DEFAULT-heap buffer. Must be called before the buffer is used as
        /// an SRV in the same command buffer submission.
        /// </summary>
        public void FlushPendingCopies(CommandBuffer cmd)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(NativeStructuredBuffer));
            cmd.IssuePluginEventAndData(
                NativeRenderPlugin.NR_NSB_GetFlushEventFunc(),
                1,
                (IntPtr)Handle);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (Handle != 0)
            {
                NativeRenderPlugin.NR_DestroyNativeStructuredBuffer(Handle);
                Handle = 0;
            }
        }
    }
}
