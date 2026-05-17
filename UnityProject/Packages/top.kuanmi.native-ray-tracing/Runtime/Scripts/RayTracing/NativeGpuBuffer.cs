using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace NativeRender
{
    /// <summary>
    /// GPU-resident (DEFAULT heap) buffer with UAV support.
    /// Can be bound as a typed/structured SRV (<see cref="NativeDescriptorSetBase.SetTypedBuffer(string, NativeGpuBuffer, int, uint)"/>)
    /// or as a typed/structured UAV (<see cref="NativeDescriptorSetBase.SetRWTypedBuffer(string, NativeGpuBuffer, int, uint)"/>).
    ///
    /// Unlike <see cref="NativeBuffer"/>, this buffer lives entirely on the GPU (DEFAULT heap)
    /// and is written/read exclusively by compute shaders. No CPU upload is supported.
    ///
    /// CPU-side zeroing is available via <see cref="Clear"/>, which uses
    /// <c>ID3D12GraphicsCommandList::ClearUnorderedAccessViewUint</c> on the render thread.
    /// </summary>
    public sealed class NativeGpuBuffer : IDisposable
    {
        public ulong Handle { get; private set; }
        private bool _disposed;

        /// <param name="sizeInBytes">Size of the buffer in bytes.</param>
        public NativeGpuBuffer(uint sizeInBytes)
        {
            if (sizeInBytes == 0) throw new ArgumentOutOfRangeException(nameof(sizeInBytes));
            Handle = NativeRenderPlugin.NR_CreateNativeGpuBuffer(sizeInBytes);
            if (Handle == 0)
                Debug.LogError("[NativeGpuBuffer] NR_CreateNativeGpuBuffer failed.");
        }

        /// <summary>
        /// Enqueues a GPU-side zero-clear of the entire buffer on the render thread.
        /// Uses <c>ClearUnorderedAccessViewUint</c> — no CPU-GPU stall required.
        ///
        /// Call this inside a render pass's <c>SetRenderFunc</c> (via <see cref="UnsafeCommandBuffer"/>).
        /// The clear is ordered relative to surrounding commands by Unity's barrier system.
        /// </summary>
        public void Clear(UnsafeCommandBuffer cmd)
        {
            if (_disposed || Handle == 0) return;
            cmd.IssuePluginEventAndData(
                NativeRenderPlugin.NR_GetClearNativeGpuBufferCallbackPtr(),
                0,
                 (IntPtr)Handle);
        }

        public void Dispose()
        {
            if (_disposed) return;
            if (Handle != 0)
            {
                NativeRenderPlugin.NR_DestroyNativeGpuBuffer(Handle);
                Handle = 0;
            }
            _disposed = true;
        }
    }
}
