using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine.Rendering;

namespace NativeRender
{
    /// <summary>
    /// GPU-resident (DEFAULT heap) structured buffer of fixed capacity.
    ///
    /// The allocation is immutable in size — to change capacity the owner disposes
    /// this instance and constructs a new one (the old D3D12 resource is deferred-
    /// deleted in the plugin after the GPU is done), mirroring nvrhi's Buffer model
    /// and Unity's GraphicsBuffer release-and-recreate idiom.
    ///
    /// Write path : SetData() — pure managed; copies bytes into a C# accumulator and
    ///              records the range. No native call.
    /// Flush path : Flush(cmd) — packs the accumulated ranges into a snapshot blob,
    ///              hands its pointer to a single IssuePluginEventAndData event, and
    ///              resets the accumulator. The render thread parses the blob, records
    ///              the staged CopyBufferRegion(s), and frees the blob. Issued only when
    ///              data actually changed, so static buffers are uploaded once and stay
    ///              resident (no per-frame re-upload).
    /// </summary>
    public sealed class NativeStructuredBuffer : IDisposable
    {
        // Snapshot blob layout (must match NsbFlushHeader / NsbFlushRange in NativeStructuredBuffer.h).
        private const int HeaderSize = 16; // uint64 handle + uint32 stride + uint32 rangeCount
        private const int RangeSize  = 12; // uint32 elementOffset + uint32 elementCount + uint32 payloadByteOffset

        /// <summary>Opaque plugin handle (NativeStructuredBuffer*).</summary>
        public ulong Handle { get; private set; }

        /// <summary>Fixed element capacity of the underlying D3D12 buffer.</summary>
        public int Capacity => (int)NativeRenderPlugin.NR_NSB_GetCapacity(Handle);

        /// <summary>Element stride in bytes (fixed at construction).</summary>
        public int Stride { get; }

        private bool _disposed;

        private readonly struct Range
        {
            public readonly int ElementOffset;
            public readonly int ElementCount;
            public readonly int PayloadByteOffset;
            public Range(int elementOffset, int elementCount, int payloadByteOffset)
            {
                ElementOffset     = elementOffset;
                ElementCount      = elementCount;
                PayloadByteOffset = payloadByteOffset;
            }
        }

        // Managed accumulation of writes since the last Flush. _payload holds packed bytes;
        // _ranges describes where each chunk lands in the GPU buffer. Capacity is reused across
        // flushes. When _ranges is empty the buffer is clean and Flush issues no event.
        private readonly List<Range> _ranges = new List<Range>();
        private byte[] _payload = Array.Empty<byte>();
        private int    _payloadLen;

        /// <summary>Allocates a fixed-capacity structured buffer with <paramref name="capacity"/> elements.</summary>
        public NativeStructuredBuffer(int capacity, int elementStride)
        {
            if (capacity      <= 0) throw new ArgumentOutOfRangeException(nameof(capacity));
            if (elementStride <= 0) throw new ArgumentOutOfRangeException(nameof(elementStride));
            Stride = elementStride;
            Handle = NativeRenderPlugin.NR_CreateNativeStructuredBuffer((uint)capacity, (uint)elementStride);
            if (Handle == 0)
                throw new InvalidOperationException("NR_CreateNativeStructuredBuffer failed (renderer not ready?)");
        }

        /// <summary>
        /// Main thread: copies <paramref name="count"/> elements from <paramref name="data"/> (starting
        /// at element <paramref name="dstOffset"/>) into the pending accumulator, to be written at the
        /// same offset in the GPU buffer by the next <see cref="Flush"/>. Pure managed — no native call.
        /// </summary>
        public unsafe void SetData<T>(T[] data, int dstOffset, int count) where T : unmanaged
        {
            if (_disposed) throw new ObjectDisposedException(nameof(NativeStructuredBuffer));
            if (data == null) throw new ArgumentNullException(nameof(data));
            if (count <= 0) return;

            fixed (T* src = &data[dstOffset])
                Append(src, dstOffset, count);
        }

        /// <summary>
        /// Main thread: copies <paramref name="count"/> elements from a <see cref="NativeArray{T}"/>
        /// (starting at element <paramref name="dstOffset"/>) into the pending accumulator.
        /// </summary>
        public unsafe void SetData<T>(NativeArray<T> data, int dstOffset, int count) where T : unmanaged
        {
            if (_disposed) throw new ObjectDisposedException(nameof(NativeStructuredBuffer));
            if (count <= 0) return;

            byte* basePtr = (byte*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(data);
            Append(basePtr + (long)dstOffset * Stride, dstOffset, count);
        }

        private unsafe void Append(void* src, int dstElementOffset, int count)
        {
            int byteCount = count * Stride;
            EnsurePayloadCapacity(_payloadLen + byteCount);

            fixed (byte* dst = &_payload[_payloadLen])
                Buffer.MemoryCopy(src, dst, byteCount, byteCount);

            _ranges.Add(new Range(dstElementOffset, count, _payloadLen));
            _payloadLen += byteCount;
        }

        private void EnsurePayloadCapacity(int needed)
        {
            if (_payload.Length >= needed) return;
            int newCap = Math.Max(needed, Math.Max(_payload.Length * 2, 1024));
            Array.Resize(ref _payload, newCap);
        }

        /// <summary>Returns the ID3D12Resource* as IntPtr for SRV binding.</summary>
        public IntPtr NativePtr => NativeRenderPlugin.NR_NSB_GetNativePtr(Handle);

        /// <summary>
        /// Packs the accumulated writes into a snapshot blob and issues a single render-thread event
        /// (carrying the blob as the event data) that copies them into the GPU-resident buffer. A no-op
        /// when no <see cref="SetData"/> has occurred since the last flush. Must be called before the
        /// buffer is read as an SRV in the same command-buffer submission.
        /// </summary>
        public unsafe void Flush(CommandBuffer cmd)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(NativeStructuredBuffer));
            if (_ranges.Count == 0) return; // clean — buffer keeps its resident data

            int rangeCount     = _ranges.Count;
            int rangeTableSize = rangeCount * RangeSize;
            int totalSize      = HeaderSize + rangeTableSize + _payloadLen;

            IntPtr blob = NativeRenderPlugin.NR_NSB_AllocFlushBuffer((uint)totalSize);
            if (blob == IntPtr.Zero) { ResetAccumulator(); return; }

            byte* p = (byte*)blob;
            *(ulong*)(p + 0) = Handle;
            *(uint*)(p + 8)  = (uint)Stride;
            *(uint*)(p + 12) = (uint)rangeCount;

            byte* table = p + HeaderSize;
            for (int i = 0; i < rangeCount; i++)
            {
                Range r = _ranges[i];
                byte* e = table + i * RangeSize;
                *(uint*)(e + 0) = (uint)r.ElementOffset;
                *(uint*)(e + 4) = (uint)r.ElementCount;
                *(uint*)(e + 8) = (uint)r.PayloadByteOffset;
            }

            if (_payloadLen > 0)
            {
                byte* payloadDst = p + HeaderSize + rangeTableSize;
                fixed (byte* payloadSrc = _payload)
                    Buffer.MemoryCopy(payloadSrc, payloadDst, _payloadLen, _payloadLen);
            }

            cmd.IssuePluginEventAndData(NativeRenderPlugin.NR_NSB_GetFlushEventFunc(), 1, blob);
            ResetAccumulator();
        }

        private void ResetAccumulator()
        {
            _ranges.Clear();
            _payloadLen = 0; // keep _payload capacity for reuse
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
