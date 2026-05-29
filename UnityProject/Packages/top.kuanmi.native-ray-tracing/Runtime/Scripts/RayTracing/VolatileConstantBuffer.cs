using System;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace NativeRender
{
    /// <summary>
    /// Volatile (dynamic) constant buffer — the nvrhi volatile-constant-buffer model.
    ///
    /// There is no persistent GPU resource and no CPU-side multi-buffering. Each upload
    /// packs the data into a self-contained, plugin-owned blob ([handle][bytes][payload]),
    /// hands the pointer to a single render-thread event, and the plugin copies it into a
    /// fresh suballocation of the shared UPLOAD pool (fence-recycled) and frees the blob.
    /// Because nothing persistent is shared between the main and render threads, the old
    /// triple-buffered source arrays are gone.
    ///
    /// The buffer is bound as a root CBV by GPU VA; it must be uploaded every frame before
    /// it is bound (volatile semantics).
    /// </summary>
    public sealed class VolatileConstantBuffer : IDisposable
    {
        public  ulong Handle { get; private set; }
        private bool  _disposed;

        private readonly int _sizeInBytes;

        // Blob layout (must match NbUploadHeader in Plugin.cpp):
        //   [ ulong bufferHandle ][ uint bytes ][ uint pad ][ payload bytes ]
        private const int HeaderSize = 16;

        private readonly InternalUploadPass _internalPass;

        public VolatileConstantBuffer(int sizeInBytes)
        {
            if (sizeInBytes <= 0) throw new ArgumentOutOfRangeException(nameof(sizeInBytes));

            _sizeInBytes = (sizeInBytes + 255) & ~255;
            // Volatile constant buffer: no backing resource, suballocated per upload.
            Handle = NativeRenderPlugin.NR_CreateNativeBuffer(
                (ulong)_sizeInBytes, 0u, 1u,
                /*canHaveUAVs*/ 0u, /*isConstantBuffer*/ 1u, /*isVolatile*/ 1u);

            _internalPass = new InternalUploadPass(this);
        }

        /// <summary>
        /// Builds a plugin-owned upload blob holding <paramref name="data"/>. The render-thread
        /// callback copies it into the upload pool and frees it. Returns IntPtr.Zero on failure.
        /// </summary>
        private unsafe IntPtr BuildBlob<T>(T data) where T : unmanaged
        {
            int bytes = Math.Min(sizeof(T), _sizeInBytes);

            IntPtr blob = NativeRenderPlugin.NR_NSB_AllocFlushBuffer((uint)(HeaderSize + bytes));
            if (blob == IntPtr.Zero) return IntPtr.Zero;

            byte* p = (byte*)blob;
            *(ulong*)(p + 0) = Handle;
            *(uint*)(p + 8)  = (uint)bytes;
            *(uint*)(p + 12) = 0u;
            UnsafeUtility.MemCpy(p + HeaderSize, UnsafeUtility.AddressOf(ref data), bytes);
            return blob;
        }

        /// <summary>
        /// Records an upload of <paramref name="data"/> by enqueuing a BeforeRendering pass that
        /// issues the plugin event. Call once per frame before the buffer is bound.
        /// </summary>
        public void Upload<T>(ScriptableRenderer renderer, T data) where T : unmanaged
        {
            if (_disposed) return;

            IntPtr blob = BuildBlob(data);
            if (blob == IntPtr.Zero) return;

            _internalPass.Setup(blob);
            renderer.EnqueuePass(_internalPass);
        }

        /// <summary>
        /// Records an upload of <paramref name="data"/> directly onto <paramref name="cmd"/>.
        /// </summary>
        public void UploadDirect<T>(UnsafeCommandBuffer cmd, T data) where T : unmanaged
        {
            if (_disposed) return;

            IntPtr blob = BuildBlob(data);
            if (blob == IntPtr.Zero) return;

            cmd.IssuePluginEventAndData(NativeRenderPlugin.GetNativeBufferUploadCallbackPtr(), 0x01, blob);
        }

        private class InternalUploadPass : ScriptableRenderPass
        {
            private readonly VolatileConstantBuffer _owner;
            private          IntPtr       _blob;

            public void Setup(IntPtr blob) => _blob = blob;

            public InternalUploadPass(VolatileConstantBuffer owner)
            {
                _owner          = owner;
                renderPassEvent = RenderPassEvent.BeforeRendering;
            }

            public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
            {
                using var builder = renderGraph.AddUnsafePass<UploadPassData>("NativeBufferUpload", out var passData);

                passData.Blob = _blob;

                builder.AllowPassCulling(false);
                builder.SetRenderFunc((UploadPassData data, UnsafeGraphContext context) =>
                {
                    context.cmd.IssuePluginEventAndData(
                        NativeRenderPlugin.GetNativeBufferUploadCallbackPtr(), 0x01, data.Blob);
                });
            }

            class UploadPassData
            {
                public IntPtr Blob;
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            if (Handle != 0)
            {
                NativeRenderPlugin.NR_DestroyNativeBuffer(Handle);
                Handle = 0;
            }
            _disposed = true;
        }
    }
}
