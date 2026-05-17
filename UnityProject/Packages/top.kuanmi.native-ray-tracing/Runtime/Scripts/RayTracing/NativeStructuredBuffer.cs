using System;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Rendering;

namespace NativeRender
{
    /// <summary>
    /// GPU-resident (DEFAULT heap) structured buffer backed by a triple-buffered UPLOAD
    /// staging layer and a native plugin render-thread flush mechanism.
    ///
    /// Write path  : UploadRange() → EnqueueUpload (main thread, deep-copies data)
    /// Flush path  : FlushPendingCopies(cmd) → Drain event (RT writes staging[g_frameIndex])
    ///                                        → Flush event (RT: CopyBufferRegion → GPU buffer)
    /// The Drain+Flush pair ensures g_frameIndex is read on the render thread for both
    /// operations, eliminating the main-thread/render-thread race on g_frameIndex.
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
        /// Thread-safe (main thread): enqueues a deep-copy of <paramref name="count"/> elements
        /// from <paramref name="data"/> starting at <paramref name="dstOffset"/>.
        /// The actual staging write happens on the render thread when <see cref="FlushPendingCopies"/> fires.
        /// </summary>
        public unsafe void UploadRange<T>(T[] data, int dstOffset, int count) where T : unmanaged
        {
            if (_disposed) throw new ObjectDisposedException(nameof(NativeStructuredBuffer));
            if (data == null) throw new ArgumentNullException(nameof(data));

            fixed (T* ptr = data)
                NativeRenderPlugin.NR_NSB_EnqueueUpload(Handle, ptr + dstOffset, (uint)dstOffset, (uint)count);
        }

        /// <summary>
        /// Thread-safe (main thread): enqueues a deep-copy of <paramref name="count"/> elements
        /// from a <see cref="NativeArray{T}"/> starting at <paramref name="dstOffset"/>.
        /// </summary>
        public unsafe void UploadRange<T>(NativeArray<T> data, int dstOffset, int count) where T : unmanaged
        {
            if (_disposed) throw new ObjectDisposedException(nameof(NativeStructuredBuffer));
            NativeRenderPlugin.NR_NSB_EnqueueUpload(
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
        public IntPtr NativePtr
        {
            get
            {
                var var = NativeRenderPlugin.NR_NSB_GetNativePtr(Handle);
                // Debug.Log($"NativeStructuredBuffer: Capacity={Capacity}, Stride={Stride}, NativePtr=0x{var.ToString("X")}");
                return var;
            }
        }

        /// <summary>
        /// Records two render-thread events into <paramref name="cmd"/>:
        /// <list type="number">
        ///   <item>Drain: flushes the enqueue queue into staging[g_frameIndex].</item>
        ///   <item>Flush: copies the dirty staging range into the GPU-resident DEFAULT-heap buffer.</item>
        /// </list>
        /// Must be called before the buffer is used as an SRV in the same command buffer submission.
        /// </summary>
        public void FlushPendingCopies(CommandBuffer cmd)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(NativeStructuredBuffer));
            // Drain must precede Flush so that the queued data is in staging before the copy.
            cmd.IssuePluginEventAndData(
                NativeRenderPlugin.NR_NSB_GetDrainEventFunc(),
                1,
                (IntPtr)Handle);
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
