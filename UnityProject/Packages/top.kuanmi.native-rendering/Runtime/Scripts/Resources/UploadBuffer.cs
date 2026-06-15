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
    /// Write path : SetData() — pure managed; records the write. No native call.
    /// Flush path : Flush(cmd) — packs the pending writes into a snapshot blob, hands
    ///              its pointer to a single IssuePluginEventAndData event, and resets.
    ///              The render thread parses the blob, records the staged
    ///              CopyBufferRegion(s), and frees the blob. Issued only when data
    ///              actually changed, so static buffers are uploaded once and stay
    ///              resident (no per-frame re-upload).
    ///
    /// Upload granularity is fixed at construction via <see cref="UploadMode"/>.
    /// </summary>
    public sealed class UploadBuffer : IDisposable
    {
        /// <summary>How accumulated writes are turned into GPU copies at <see cref="Flush"/>.</summary>
        public enum UploadMode
        {
            /// <summary>Each SetData write becomes its own CopyBufferRegion. No CPU mirror.</summary>
            Ranges,

            /// <summary>
            /// Writes land in a full CPU mirror; Flush emits a single CopyBufferRegion spanning the
            /// dirty [min,max) element range. Gaps between disjoint writes stay correct because the
            /// mirror retains previously-written content.
            /// </summary>
            Whole,
        }

        // Snapshot blob layout (must match NsbFlushHeader / NsbFlushRange in NativeBuffer.h).
        private const int HeaderSize = 16; // uint64 handle + uint32 stride + uint32 rangeCount
        private const int RangeSize  = 12; // uint32 elementOffset + uint32 elementCount + uint32 payloadByteOffset

        /// <summary>Opaque plugin handle (NativeBuffer*).</summary>
        public ulong Handle { get; private set; }

        /// <summary>Fixed element capacity of the underlying D3D12 buffer.</summary>
        public int count { get; }

        /// <summary>Element stride in bytes (fixed at construction).</summary>
        public int stride { get; }

        /// <summary>Upload granularity (fixed at construction).</summary>
        public UploadMode Mode { get; }

        /// <summary>True when the buffer was created with UAV support (bindable as RWStructuredBuffer).</summary>
        public bool AllowsUAV { get; }

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

        // --- Ranges mode --- packed bytes + one range entry per SetData. Capacity reused across flushes.
        private readonly List<Range> _ranges;
        private byte[] _payload = Array.Empty<byte>();
        private int    _payloadLen;

        // --- Whole mode --- full CPU mirror + dirty [min,max) element span.
        private readonly byte[] _mirror;
        private int  _dirtyMinElem;
        private int  _dirtyMaxElem; // half-open
        private bool _wholeDirty;

        /// <summary>
        /// Allocates a fixed-capacity structured buffer with <paramref name="capacity"/> elements.
        /// Set <paramref name="allowUAV"/> when the buffer must also be GPU-writable (bound as a
        /// RWStructuredBuffer) — e.g. a buffer that mixes CPU uploads with compute-shader writes.
        /// </summary>
        public UploadBuffer(int capacity, int elementStride, UploadMode mode = UploadMode.Ranges, bool allowUAV = false, string debugName = null)
        {
            if (capacity      <= 0) throw new ArgumentOutOfRangeException(nameof(capacity));
            if (elementStride <= 0) throw new ArgumentOutOfRangeException(nameof(elementStride));
            count     = capacity;
            stride    = elementStride;
            Mode      = mode;
            AllowsUAV = allowUAV;
            // GPU-resident DEFAULT-heap structured buffer fed by the staged-copy snapshot path.
            Handle = NativeRenderPlugin.NR_CreateNativeBuffer(
                (ulong)capacity * (ulong)elementStride, (uint)elementStride, 1u,
                /*canHaveUAVs*/ allowUAV ? 1u : 0u, /*isConstantBuffer*/ 0u, /*isVolatile*/ 0u, debugName);
            if (Handle == 0)
                throw new InvalidOperationException("NR_CreateNativeBuffer failed (renderer not ready?)");

            if (mode == UploadMode.Whole)
                _mirror = new byte[(long)capacity * elementStride];
            else
                _ranges = new List<Range>();
        }

        /// <summary>
        /// Main thread: records a write of <paramref name="count"/> elements from <paramref name="data"/>
        /// (starting at element <paramref name="dstOffset"/>) to the same offset in the GPU buffer, applied
        /// by the next <see cref="Flush"/>. Pure managed — no native call.
        /// </summary>
        public unsafe void SetData<T>(T[] data, int dstOffset, int count) where T : unmanaged
        {
            if (_disposed) throw new ObjectDisposedException(nameof(UploadBuffer));
            if (data == null) throw new ArgumentNullException(nameof(data));
            if (count <= 0) return;

            fixed (T* src = &data[dstOffset])
                Append(src, dstOffset, count);
        }

        /// <summary>
        /// Main thread: records a write of <paramref name="count"/> elements read from
        /// <paramref name="data"/> starting at element <paramref name="srcOffset"/>, into the GPU
        /// buffer starting at element <paramref name="dstOffset"/>. Mirrors GraphicsBuffer.SetData's
        /// (managedBufferStartIndex, graphicsBufferStartIndex, count) form for the case where the
        /// source and destination offsets differ. Pure managed — no native call.
        /// </summary>
        public unsafe void SetData<T>(T[] data, int srcOffset, int dstOffset, int count) where T : unmanaged
        {
            if (_disposed) throw new ObjectDisposedException(nameof(UploadBuffer));
            if (data == null) throw new ArgumentNullException(nameof(data));
            if (count <= 0) return;

            fixed (T* src = &data[srcOffset])
                Append(src, dstOffset, count);
        }

        /// <summary>
        /// Main thread: records a write of <paramref name="count"/> elements from a
        /// <see cref="NativeArray{T}"/> (starting at element <paramref name="dstOffset"/>).
        /// </summary>
        public unsafe void SetData<T>(NativeArray<T> data, int dstOffset, int count) where T : unmanaged
        {
            if (_disposed) throw new ObjectDisposedException(nameof(UploadBuffer));
            if (count <= 0) return;

            byte* basePtr = (byte*)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(data);
            Append(basePtr + (long)dstOffset * stride, dstOffset, count);
        }

        /// <summary>
        /// Records a raw byte write from <paramref name="count"/> source elements into this buffer.
        /// The destination offset is still expressed in this buffer's stride-sized elements.
        /// Use this for raw buffers whose logical payload type is wider than the buffer stride
        /// (for example writing 32-byte task structs into a 4-byte-addressed RWByteAddressBuffer).
        /// </summary>
        public unsafe void SetRawData<T>(T[] data, int srcOffset, int dstOffset, int count) where T : unmanaged
        {
            if (_disposed) throw new ObjectDisposedException(nameof(UploadBuffer));
            if (data == null) throw new ArgumentNullException(nameof(data));
            if (count <= 0) return;

            int byteCount = count * sizeof(T);
            if ((byteCount % stride) != 0)
                throw new ArgumentException("Raw upload byte count must be a multiple of the buffer stride.", nameof(count));

            fixed (T* src = &data[srcOffset])
                AppendBytes(src, dstOffset, byteCount);
        }

        private unsafe void Append(void* src, int dstElementOffset, int count)
        {
            AppendBytes(src, dstElementOffset, count * stride);
        }

        private unsafe void AppendBytes(void* src, int dstElementOffset, int byteCount)
        {
            int elementCount = byteCount / stride;

            if (Mode == UploadMode.Whole)
            {
                fixed (byte* dst = &_mirror[(long)dstElementOffset * stride])
                    Buffer.MemoryCopy(src, dst, byteCount, byteCount);

                int end = dstElementOffset + elementCount;
                if (!_wholeDirty)
                {
                    _dirtyMinElem = dstElementOffset;
                    _dirtyMaxElem = end;
                    _wholeDirty   = true;
                }
                else
                {
                    if (dstElementOffset < _dirtyMinElem) _dirtyMinElem = dstElementOffset;
                    if (end > _dirtyMaxElem)              _dirtyMaxElem = end;
                }
            }
            else
            {
                EnsurePayloadCapacity(_payloadLen + byteCount);
                fixed (byte* dst = &_payload[_payloadLen])
                    Buffer.MemoryCopy(src, dst, byteCount, byteCount);

                _ranges.Add(new Range(dstElementOffset, elementCount, _payloadLen));
                _payloadLen += byteCount;
            }
        }

        private void EnsurePayloadCapacity(int needed)
        {
            if (_payload.Length >= needed) return;
            int newCap = Math.Max(needed, Math.Max(_payload.Length * 2, 1024));
            Array.Resize(ref _payload, newCap);
        }


        /// <summary>
        /// Packs the pending writes into a snapshot blob and issues a single render-thread event
        /// (carrying the blob as the event data) that copies them into the GPU-resident buffer. A no-op
        /// when no <see cref="SetData"/> has occurred since the last flush. Must be called before the
        /// buffer is read as an SRV in the same command-buffer submission.
        /// </summary>
        public unsafe void Flush(CommandBuffer cmd)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(UploadBuffer));

            bool whole = Mode == UploadMode.Whole;
            if (whole ? !_wholeDirty : _ranges.Count == 0) return; // clean — buffer keeps its resident data

            int rangeCount = whole ? 1 : _ranges.Count;
            int payloadLen = whole ? (_dirtyMaxElem - _dirtyMinElem) * stride : _payloadLen;

            int rangeTableSize = rangeCount * RangeSize;
            int totalSize      = HeaderSize + rangeTableSize + payloadLen;

            IntPtr blob = NativeRenderPlugin.NR_NSB_AllocFlushBuffer((uint)totalSize);
            if (blob == IntPtr.Zero) { ResetAccumulator(); return; }

            byte* p = (byte*)blob;
            *(ulong*)(p + 0) = Handle;
            *(uint*)(p + 8)  = (uint)stride;
            *(uint*)(p + 12) = (uint)rangeCount;

            byte* table      = p + HeaderSize;
            byte* payloadDst = p + HeaderSize + rangeTableSize;

            if (whole)
            {
                *(uint*)(table + 0) = (uint)_dirtyMinElem;
                *(uint*)(table + 4) = (uint)(_dirtyMaxElem - _dirtyMinElem);
                *(uint*)(table + 8) = 0u;

                if (payloadLen > 0)
                    fixed (byte* src = &_mirror[(long)_dirtyMinElem * stride])
                        Buffer.MemoryCopy(src, payloadDst, payloadLen, payloadLen);
            }
            else
            {
                for (int i = 0; i < rangeCount; i++)
                {
                    Range r = _ranges[i];
                    byte* e = table + i * RangeSize;
                    *(uint*)(e + 0) = (uint)r.ElementOffset;
                    *(uint*)(e + 4) = (uint)r.ElementCount;
                    *(uint*)(e + 8) = (uint)r.PayloadByteOffset;
                }

                if (payloadLen > 0)
                    fixed (byte* src = _payload)
                        Buffer.MemoryCopy(src, payloadDst, payloadLen, payloadLen);
            }

            cmd.IssuePluginEventAndData(NativeRenderPlugin.NR_NSB_GetFlushEventFunc(), 1, blob);
            ResetAccumulator();
        }

        private void ResetAccumulator()
        {
            if (Mode == UploadMode.Whole)
            {
                _wholeDirty = false; // keep mirror content for the next partial update
            }
            else
            {
                _ranges.Clear();
                _payloadLen = 0; // keep _payload capacity for reuse
            }
        }

        // Readback request blob layout (must match NsbReadbackRequest in NativeBuffer.h).
        private const int ReadbackRequestSize = 24; // ulong handle + ulong srcByteOffset + ulong bytes

        /// <summary>
        /// Records a GPU->CPU readback of <paramref name="elementCount"/> elements starting at element
        /// <paramref name="srcElementOffset"/>. Async: the copy is recorded on <paramref name="cmd"/>
        /// into a slot of a native staging ring and completes a frame or two later — poll
        /// <see cref="TryGetReadback{T}"/>, which returns the newest completed request. Safe to issue
        /// every frame: the ring is deep enough that an unread copy is never overwritten while a
        /// newer one is still in flight (mirrors RTXPT's cReadbackLag staging ring).
        /// </summary>
        public unsafe void RequestReadback(CommandBuffer cmd, int srcElementOffset, int elementCount)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(UploadBuffer));
            if (elementCount <= 0) return;
            if (srcElementOffset < 0 || srcElementOffset + elementCount > count)
                throw new ArgumentOutOfRangeException(nameof(elementCount),
                    $"Readback range [{srcElementOffset}, {srcElementOffset + elementCount}) exceeds buffer capacity {count}.");

            IntPtr blob = NativeRenderPlugin.NR_NSB_AllocFlushBuffer(ReadbackRequestSize);
            if (blob == IntPtr.Zero) return;

            ulong* p = (ulong*)blob;
            p[0] = Handle;
            p[1] = (ulong)((long)srcElementOffset * stride);
            p[2] = (ulong)((long)elementCount * stride);

            cmd.IssuePluginEventAndData(NativeRenderPlugin.NR_NSB_GetReadbackEventFunc(), 1, blob);
        }

        /// <summary>
        /// Polls the pending <see cref="RequestReadback"/> requests. Returns true and fills the
        /// first <paramref name="count"/> elements of <paramref name="dst"/> with the newest
        /// request whose GPU copy has completed (retiring it and any older ones); returns false
        /// while all requests are still in flight or none is pending.
        /// </summary>
        public unsafe bool TryGetReadback<T>(T[] dst, int count) where T : unmanaged
        {
            if (_disposed) throw new ObjectDisposedException(nameof(UploadBuffer));
            if (dst == null || count <= 0) return false;
            if (count > dst.Length)
                throw new ArgumentOutOfRangeException(nameof(count),
                    $"count ({count}) exceeds dst.Length ({dst.Length}); the native copy would overrun the array.");

            ulong dstBytes = (ulong)count * (ulong)sizeof(T);
            fixed (T* p = dst)
                return NativeRenderPlugin.NR_NSB_TryReadback(Handle, (IntPtr)p, dstBytes) != 0;
        }

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
