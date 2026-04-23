using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;

namespace NativeRender
{
    /// <summary>
    /// Best-fit free-list allocator for slots inside <c>_primitiveDataBuf</c>.
    ///
    /// Manages a flat array of <c>PrimitiveDataNRD</c> elements by tracking which
    /// index ranges are free. All methods operate on element indices (not byte offsets).
    /// GPU buffer creation and NativeArray management are the caller's responsibility.
    ///
    /// Typical usage (separate-BLAS incremental path):
    /// <code>
    ///   // Full rebuild — reset to match actual allocation.
    ///   _primAlloc.Reset(totalTriCount);
    ///
    ///   // Add object.
    ///   uint offset = _primAlloc.Allocate(triCount);
    ///   if (offset == PrimitiveSlotAllocator.InvalidOffset)
    ///   {
    ///       _primAlloc.GrowTo(_primAlloc.Capacity * 2);
    ///       // ... grow GPU buffer ...
    ///       offset = _primAlloc.Allocate(triCount);
    ///   }
    ///
    ///   // Remove object.
    ///   _primAlloc.Free(primitiveOffset, triCount);
    ///
    ///   // Optionally trigger full rebuild if too fragmented.
    ///   if (_primAlloc.FragmentationRatio > 0.5f) MarkRebuildDirty();
    /// </code>
    /// </summary>
    internal sealed class PrimitiveSlotAllocator
    {
        /// <summary>Sentinel value returned by <see cref="Allocate"/> when no block fits.</summary>
        public const uint InvalidOffset = uint.MaxValue;

        // ------------------------------------------------------------------ //
        // Logging
        // ------------------------------------------------------------------ //

        /// <summary>Verbosity levels for per-instance diagnostic logging.</summary>
        public enum Verbosity
        {
            /// <summary>Silent — no output (default, zero runtime cost).</summary>
            None = 0,
            /// <summary>Log only errors and capacity changes (Reset, GrowTo).</summary>
            Structural,
            /// <summary>Log every Allocate and Free call as well.</summary>
            Verbose,
        }

        /// <summary>
        /// Controls how much this allocator logs via <c>UnityEngine.Debug.Log</c>.
        /// Set to <see cref="Verbosity.Verbose"/> when debugging slot management issues.
        /// Defaults to <see cref="Verbosity.None"/>.
        /// </summary>
        public Verbosity LogLevel { get; set; } = Verbosity.Verbose;

        /// <summary>
        /// Optional label prepended to all log messages (e.g. "instance" or "primitive")
        /// so you can tell multiple allocators apart in the console.
        /// </summary>
        public string Name { get; set; } = "PrimitiveSlotAllocator";

        private string Prefix => $"[{Name}]";

        // Free blocks sorted ascending by offset. Adjacent free blocks are always
        // merged immediately (eager coalescing) so this list stays as short as possible.
        private readonly List<FreeBlock> _freeBlocks = new List<FreeBlock>();

        private struct FreeBlock
        {
            public uint Offset;
            public uint Count;
        }

        // ------------------------------------------------------------------ //
        // Public state
        // ------------------------------------------------------------------ //

        /// <summary>Total element capacity managed by this allocator.</summary>
        public uint Capacity { get; private set; }

        /// <summary>Number of currently allocated elements.</summary>
        public uint UsedCount { get; private set; }

        /// <summary>Total number of free elements across all free blocks.</summary>
        public uint TotalFreeCount
        {
            get
            {
                uint total = 0;
                for (int i = 0; i < _freeBlocks.Count; i++)
                    total += _freeBlocks[i].Count;
                return total;
            }
        }

        /// <summary>Size of the largest contiguous free block (0 if none).</summary>
        public uint LargestFreeBlock
        {
            get
            {
                uint best = 0;
                for (int i = 0; i < _freeBlocks.Count; i++)
                    if (_freeBlocks[i].Count > best) best = _freeBlocks[i].Count;
                return best;
            }
        }

        /// <summary>
        /// Fragmentation ratio in [0, 1].
        /// 0 = all free space is one contiguous block (no fragmentation).
        /// Approaches 1 when free space is spread across many tiny blocks.
        /// Returns 0 when there is no free space.
        /// </summary>
        public float FragmentationRatio
        {
            get
            {
                uint free = TotalFreeCount;
                if (free == 0) return 0f;
                uint largest = LargestFreeBlock;
                return 1f - (float)largest / free;
            }
        }

        // ------------------------------------------------------------------ //
        // Construction / reset
        // ------------------------------------------------------------------ //

        /// <summary>Creates an allocator with zero capacity. Call <see cref="Reset"/> or <see cref="GrowTo"/> before use.</summary>
        public PrimitiveSlotAllocator() { }

        /// <summary>
        /// Resets the allocator to a single fully-free block of <paramref name="capacity"/> elements.
        /// All previous allocation state is discarded.
        /// </summary>
        public void Reset(int capacity)
        {
            if (capacity < 0) throw new ArgumentOutOfRangeException(nameof(capacity));
            _freeBlocks.Clear();
            Capacity  = (uint)capacity;
            UsedCount = 0;
            if (capacity > 0)
                _freeBlocks.Add(new FreeBlock { Offset = 0, Count = (uint)capacity });
            if (LogLevel >= Verbosity.Structural)
                Debug.Log($"{Prefix} Reset  capacity={capacity}  all slots free");
        }

        /// <summary>
        /// Resets the allocator to a fully-allocated (no free blocks) state of
        /// <paramref name="capacity"/> elements.  Use this after a full scene rebuild
        /// to record that every slot [0, capacity) is already in use.
        /// </summary>
        public void ResetFullyAllocated(int capacity)
        {
            if (capacity < 0) throw new ArgumentOutOfRangeException(nameof(capacity));
            _freeBlocks.Clear();
            Capacity  = (uint)capacity;
            UsedCount = (uint)capacity;
            if (LogLevel >= Verbosity.Structural)
                Debug.Log($"{Prefix} ResetFullyAllocated  capacity={capacity}  all slots used");
        }

        // ------------------------------------------------------------------ //
        // Allocation
        // ------------------------------------------------------------------ //

        /// <summary>
        /// Allocates <paramref name="count"/> contiguous elements using a best-fit strategy.
        /// </summary>
        /// <returns>
        /// The starting element index of the allocated range, or <see cref="InvalidOffset"/>
        /// if no sufficiently large free block exists (caller should <see cref="GrowTo"/> and retry).
        /// </returns>
        public uint Allocate(int count)
        {
            if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count), "count must be > 0");
            uint need = (uint)count;

            // Best-fit: find the smallest block that is large enough.
            int  bestIndex = -1;
            uint bestCount = uint.MaxValue;
            for (int i = 0; i < _freeBlocks.Count; i++)
            {
                uint bc = _freeBlocks[i].Count;
                if (bc >= need && bc < bestCount)
                {
                    bestCount = bc;
                    bestIndex = i;
                }
            }

            if (bestIndex < 0)
            {
                if (LogLevel >= Verbosity.Verbose)
                    Debug.Log($"{Prefix} Allocate({count}) FAILED — no block large enough" +
                        $"  capacity={Capacity}  used={UsedCount}  freeBlocks={_freeBlocks.Count}" +
                        $"  largestFree={LargestFreeBlock}");
                return InvalidOffset;
            }

            FreeBlock blk    = _freeBlocks[bestIndex];
            uint      offset = blk.Offset;
            uint      rem    = blk.Count - need;

            if (rem == 0)
            {
                _freeBlocks.RemoveAt(bestIndex);
            }
            else
            {
                _freeBlocks[bestIndex] = new FreeBlock { Offset = blk.Offset + need, Count = rem };
            }

            UsedCount += need;
            if (LogLevel >= Verbosity.Verbose)
                Debug.Log($"{Prefix} Allocate({count}) → offset={offset}" +
                    $"  used={UsedCount}/{Capacity}  freeBlocks={_freeBlocks.Count}");
            return offset;
        }

        // ------------------------------------------------------------------ //
        // Deallocation
        // ------------------------------------------------------------------ //

        /// <summary>
        /// Returns a previously allocated range back to the free pool.
        /// Adjacent free blocks are merged immediately (eager coalescing).
        /// </summary>
        /// <param name="offset">Starting element index (as returned by <see cref="Allocate"/>).</param>
        /// <param name="count">Number of elements to free (must match the original allocation size).</param>
        public void Free(uint offset, int count)
        {
            if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count), "count must be > 0");
            uint ucount = (uint)count;

#if UNITY_ASSERTIONS
            if (offset + ucount > Capacity)
                throw new ArgumentException($"[PrimitiveSlotAllocator] Free({offset},{count}) exceeds Capacity({Capacity})");
#endif

            // Binary search for the insertion point (first free block with Offset > offset).
            int lo = 0, hi = _freeBlocks.Count;
            while (lo < hi)
            {
                int mid = (lo + hi) >> 1;
                if (_freeBlocks[mid].Offset <= offset) lo = mid + 1;
                else hi = mid;
            }
            int insertAt = lo; // _freeBlocks[insertAt].Offset > offset (or insertAt == Count)

            // Merge with predecessor.
            bool mergedPrev = false;
            if (insertAt > 0)
            {
                FreeBlock prev = _freeBlocks[insertAt - 1];
                if (prev.Offset + prev.Count == offset)
                {
                    // Extend prev block to cover the freed range.
                    _freeBlocks[insertAt - 1] = new FreeBlock { Offset = prev.Offset, Count = prev.Count + ucount };
                    offset  = prev.Offset;
                    ucount += prev.Count;
                    mergedPrev = true;
                    insertAt--;
                }
            }

            if (!mergedPrev)
            {
                // Insert new free block at insertAt.
                _freeBlocks.Insert(insertAt, new FreeBlock { Offset = offset, Count = ucount });
            }

            // Merge with successor.
            if (insertAt + 1 < _freeBlocks.Count)
            {
                FreeBlock next = _freeBlocks[insertAt + 1];
                if (offset + ucount == next.Offset)
                {
                    _freeBlocks[insertAt] = new FreeBlock { Offset = offset, Count = ucount + next.Count };
                    _freeBlocks.RemoveAt(insertAt + 1);
                }
            }

            UsedCount -= (uint)count;
            if (LogLevel >= Verbosity.Verbose)
                Debug.Log($"{Prefix} Free(offset={offset - (mergedPrev ? _freeBlocks[insertAt].Count - ucount : 0)},count={count})" +
                    $"  used={UsedCount}/{Capacity}  freeBlocks={_freeBlocks.Count}");
        }

        // ------------------------------------------------------------------ //
        // Capacity growth
        // ------------------------------------------------------------------ //

        /// <summary>
        /// Expands the allocator's managed capacity to <paramref name="newCapacity"/> elements.
        /// The new range <c>[oldCapacity, newCapacity)</c> is added as a free block (and merged
        /// with the existing tail free block if contiguous).
        /// Does nothing if <paramref name="newCapacity"/> &lt;= current <see cref="Capacity"/>.
        /// The caller is responsible for growing the backing <c>NativeArray</c> and
        /// <c>GraphicsBuffer</c> to match.
        /// </summary>
        public void GrowTo(int newCapacity)
        {
            if (newCapacity <= (int)Capacity) return;
            uint oldCap = Capacity;
            Capacity = (uint)newCapacity;
            // Add the new range directly as a free block without touching UsedCount
            // (calling Free() would underflow UsedCount since the new range was never allocated).
            uint delta = (uint)newCapacity - oldCap;
            // Merge with existing tail free block if contiguous, otherwise insert.
            int last = _freeBlocks.Count - 1;
            if (last >= 0 && _freeBlocks[last].Offset + _freeBlocks[last].Count == oldCap)
            {
                var tail = _freeBlocks[last];
                _freeBlocks[last] = new FreeBlock { Offset = tail.Offset, Count = tail.Count + delta };
            }
            else
            {
                _freeBlocks.Add(new FreeBlock { Offset = oldCap, Count = delta });
            }
            if (LogLevel >= Verbosity.Structural)
                Debug.Log($"{Prefix} GrowTo  {oldCap} → {newCapacity}" +
                    $"  used={UsedCount}  freeBlocks={_freeBlocks.Count}  largestFree={LargestFreeBlock}");
        }

        // ------------------------------------------------------------------ //
        // Diagnostics
        // ------------------------------------------------------------------ //

        /// <summary>Returns a human-readable one-line summary of allocator state.</summary>
        public override string ToString() =>
            $"{Name}  capacity={Capacity}  used={UsedCount}  free={TotalFreeCount}  " +
            $"blocks={_freeBlocks.Count}  largestFree={LargestFreeBlock}  fragRatio={FragmentationRatio:F3}";

        /// <summary>
        /// Logs a full dump of all free blocks via <c>UnityEngine.Debug.Log</c>.
        /// Use this when you need to inspect fragmentation in detail.
        /// </summary>
        public void DumpFreeBlocks()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"{Prefix} DumpFreeBlocks  {this}");
            if (_freeBlocks.Count == 0)
            {
                sb.AppendLine("  (no free blocks)");
            }
            else
            {
                for (int i = 0; i < _freeBlocks.Count; i++)
                {
                    var b = _freeBlocks[i];
                    sb.AppendLine($"  [{i,4}]  offset={b.Offset,8}  count={b.Count,8}" +
                        $"  end={b.Offset + b.Count - 1,8}");
                }
            }
            Debug.Log(sb.ToString());
        }
    }
}
