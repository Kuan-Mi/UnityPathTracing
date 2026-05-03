using System;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;

namespace NativeRender
{
    /// <summary>
    /// Triple-buffered D3D12 upload-heap constant buffer managed by the native plugin.
    ///
    /// Usage:
    ///   var nb = new NativeBuffer(Marshal.SizeOf&lt;MyConstants&gt;());
    ///   // each frame before Dispatch:
    ///   nb.Upload(myConstants);
    ///   descriptorSet.SetNativeBuffer("g_Constants", nb);
    ///   descriptorSet.Dispatch(...);
    ///   // cleanup:
    ///   nb.Dispose();
    /// </summary>
    public sealed class NativeBuffer : IDisposable
    {
        /// <summary>Opaque plugin handle (NativeBuffer*).</summary>
        public ulong Handle { get; private set; }

        private bool _disposed;

        /// <summary>Allocates a triple-buffered upload buffer of <paramref name="sizeInBytes"/>.</summary>
        public NativeBuffer(int sizeInBytes)
        {
            if (sizeInBytes <= 0) throw new ArgumentOutOfRangeException(nameof(sizeInBytes));
            Handle = NativeRenderPlugin.NR_CreateNativeBuffer((uint)sizeInBytes);
            if (Handle == 0)
                throw new InvalidOperationException("NR_CreateNativeBuffer failed (renderer not ready?)");
        }

        /// <summary>
        /// Copies an unmanaged struct into the current frame's mapped slot.
        /// Call each frame on the main thread before Dispatch.
        /// </summary>
        public unsafe void Upload<T>(in T data) where T : unmanaged
        {
            if (_disposed) throw new ObjectDisposedException(nameof(NativeBuffer));
            fixed (T* ptr = &data)
                NativeRenderPlugin.NR_NB_Upload(Handle, ptr, (uint)sizeof(T));
        }

        /// <summary>
        /// Copies raw bytes from a <see cref="NativeArray{T}"/> slice into the current frame's slot.
        /// </summary>
        public unsafe void Upload<T>(NativeArray<T> array) where T : unmanaged
        {
            if (_disposed) throw new ObjectDisposedException(nameof(NativeBuffer));
            NativeRenderPlugin.NR_NB_Upload(
                Handle,
                NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(array),
                (uint)(array.Length * UnsafeUtility.SizeOf<T>()));
        }

        /// <summary>
        /// Returns the current frame's ID3D12Resource* as IntPtr.
        /// Provided as a compatibility path; prefer <see cref="NativeComputeDescriptorSet.SetNativeBuffer"/>.
        /// </summary>
        public IntPtr NativePtr => NativeRenderPlugin.NR_NB_GetNativePtr(Handle);

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (Handle != 0)
            {
                NativeRenderPlugin.NR_DestroyNativeBuffer(Handle);
                Handle = 0;
            }
        }
    }
}
