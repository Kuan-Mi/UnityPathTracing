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
    public sealed class NativeStructuredBuffer : IDisposable
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

        // Snapshot blob layout (must match NsbFlushHeader / NsbFlushRange in NativeStructuredBuffer.h).
        private const int HeaderSize = 16; // uint64 handle + uint32 stride + uint32 rangeCount
        private const int RangeSize  = 12; // uint32 elementOffset + uint32 elementCount + uint32 payloadByteOffset

        /// <summary>Opaque plugin handle (NativeStructuredBuffer*).</summary>
        public ulong Handle { get; private set; }

        /// <summary>Fixed element capacity of the underlying D3D12 buffer.</summary>
        public int Capacity => (int)NativeRenderPlugin.NR_NSB_GetCapacity(Handle);

        /// <summary>Element stride in bytes (fixed at construction).</summary>
        public int Stride { get; }

        /// <summary>Upload granularity (fixed at construction).</summary>
        public UploadMode Mode { get; }

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

        /// <summary>Allocates a fixed-capacity structured buffer with <paramref name="capacity"/> elements.</summary>
        public NativeStructuredBuffer(int capacity, int elementStride, UploadMode mode = UploadMode.Ranges)
        {
            if (capacity      <= 0) throw new ArgumentOutOfRangeException(nameof(capacity));
            if (elementStride <= 0) throw new ArgumentOutOfRangeException(nameof(elementStride));
            Stride = elementStride;
            Mode   = mode;
            Handle = NativeRenderPlugin.NR_CreateNativeStructuredBuffer((uint)capacity, (uint)elementStride);
            if (Handle == 0)
                throw new InvalidOperationException("NR_CreateNativeStructuredBuffer failed (renderer not ready?)");

            // The DEFAULT-heap resource pointer is fixed for the buffer's lifetime (size is
            // immutable), so resolve it once instead of P/Invoking on every access.
            NativePtr = NativeRenderPlugin.NR_NSB_GetNativePtr(Handle);

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
            if (_disposed) throw new ObjectDisposedException(nameof(NativeStructuredBuffer));
            if (data == null) throw new ArgumentNullException(nameof(data));
            if (count <= 0) return;

            fixed (T* src = &data[dstOffset])
                Append(src, dstOffset, count);
        }

        /// <summary>
        /// Main thread: records a write of <paramref name="count"/> elements from a
        /// <see cref="NativeArray{T}"/> (starting at element <paramref name="dstOffset"/>).
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

            if (Mode == UploadMode.Whole)
            {
                fixed (byte* dst = &_mirror[(long)dstElementOffset * Stride])
                    Buffer.MemoryCopy(src, dst, byteCount, byteCount);

                int end = dstElementOffset + count;
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

                _ranges.Add(new Range(dstElementOffset, count, _payloadLen));
                _payloadLen += byteCount;
            }
        }

        private void EnsurePayloadCapacity(int needed)
        {
            if (_payload.Length >= needed) return;
            int newCap = Math.Max(needed, Math.Max(_payload.Length * 2, 1024));
            Array.Resize(ref _payload, newCap);
        }

        /// <summary>ID3D12Resource* as IntPtr for SRV binding. Stable for the buffer's lifetime.</summary>
        public IntPtr NativePtr { get; }

        /// <summary>
        /// Packs the pending writes into a snapshot blob and issues a single render-thread event
        /// (carrying the blob as the event data) that copies them into the GPU-resident buffer. A no-op
        /// when no <see cref="SetData"/> has occurred since the last flush. Must be called before the
        /// buffer is read as an SRV in the same command-buffer submission.
        /// </summary>
        public unsafe void Flush(CommandBuffer cmd)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(NativeStructuredBuffer));

            bool whole = Mode == UploadMode.Whole;
            if (whole ? !_wholeDirty : _ranges.Count == 0) return; // clean — buffer keeps its resident data

            int rangeCount = whole ? 1 : _ranges.Count;
            int payloadLen = whole ? (_dirtyMaxElem - _dirtyMinElem) * Stride : _payloadLen;

            int rangeTableSize = rangeCount * RangeSize;
            int totalSize      = HeaderSize + rangeTableSize + payloadLen;

            IntPtr blob = NativeRenderPlugin.NR_NSB_AllocFlushBuffer((uint)totalSize);
            if (blob == IntPtr.Zero) { ResetAccumulator(); return; }

            byte* p = (byte*)blob;
            *(ulong*)(p + 0) = Handle;
            *(uint*)(p + 8)  = (uint)Stride;
            *(uint*)(p + 12) = (uint)rangeCount;

            byte* table      = p + HeaderSize;
            byte* payloadDst = p + HeaderSize + rangeTableSize;

            if (whole)
            {
                *(uint*)(table + 0) = (uint)_dirtyMinElem;
                *(uint*)(table + 4) = (uint)(_dirtyMaxElem - _dirtyMinElem);
                *(uint*)(table + 8) = 0u;

                if (payloadLen > 0)
                    fixed (byte* src = &_mirror[(long)_dirtyMinElem * Stride])
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
