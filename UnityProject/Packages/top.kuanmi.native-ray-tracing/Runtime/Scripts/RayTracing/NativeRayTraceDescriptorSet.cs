using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine.Rendering;

namespace NativeRender
{
    /// <summary>
    /// Holds per-dispatch resource bindings for a <see cref="RayTracePipeline"/>.
    ///
    /// Mirrors <see cref="NativeComputeDescriptorSet"/> but targets DXR
    /// (<c>DispatchRays</c>) instead of <c>Dispatch</c>.
    ///
    /// Create one (or more) per pass; call <c>Set*</c> each frame to update
    /// bindings, then hand the descriptor set to
    /// <see cref="RayTracePipeline.Dispatch(CommandBuffer,NativeRayTraceDescriptorSet,uint,uint)"/>.
    ///
    /// Owns a ring buffer of pinned <see cref="NativeArray{T}"/> allocations so
    /// main-thread writes and render-thread reads are safely decoupled.
    ///
    /// Lifetime: must be explicitly disposed via <see cref="Dispose"/>.
    /// </summary>
    public sealed class NativeRayTraceDescriptorSet : NativeDescriptorSetBase
    {
        private readonly RayTracePipeline _pipeline;

        // Typed header ring for ray-trace dispatch events.
        private NativeArray<NativeRenderPlugin.RTS_RenderEventData>[] _headerRing;

        // -------------------------------------------------------------------
        // Construction
        // -------------------------------------------------------------------

        /// <summary>
        /// Creates a descriptor set tied to <paramref name="pipeline"/>.
        /// Subscribes to <see cref="RayTracePipeline.OnRebuilt"/> for automatic
        /// re-creation after shader hot-reload.
        /// </summary>
        public NativeRayTraceDescriptorSet(RayTracePipeline pipeline)
        {
            if (pipeline == null)
                throw new ArgumentNullException(nameof(pipeline));

            _pipeline = pipeline;
            CopySlotLayout(pipeline);
            AllocateRingBuffers();
            _descriptorSetHandle = NativeRenderPlugin.NR_RTS_CreateDescriptorSet(pipeline.Handle);
            pipeline.OnRebuilt  += OnPipelineRebuilt;
        }

        private void CopySlotLayout(RayTracePipeline pipeline)
        {
            CopySlotLayoutCommon(pipeline.SlotCount, pipeline.NameToSlot);
        }

        private void AllocateRingBuffers()
        {
            AllocateRingBuffersCommon();
            _headerRing = new NativeArray<NativeRenderPlugin.RTS_RenderEventData>[RingSize];
            for (int i = 0; i < RingSize; i++)
                _headerRing[i] = new NativeArray<NativeRenderPlugin.RTS_RenderEventData>(1, Allocator.Persistent);
        }

        private void FreeRingBuffers()
        {
            FreeRingBuffersCommon();
            if (_headerRing == null) return;
            for (int i = 0; i < RingSize; i++)
                if (_headerRing[i].IsCreated) _headerRing[i].Dispose();
            _headerRing = null;
        }

        private void OnPipelineRebuilt(RayTracePipeline pipeline)
        {
            if (_descriptorSetHandle != 0)
            {
                NativeRenderPlugin.NR_RTS_DestroyDescriptorSet(_descriptorSetHandle);
                _descriptorSetHandle = 0;
            }
            FreeRingBuffers();
            CopySlotLayout(pipeline);
            AllocateRingBuffers();
            _descriptorSetHandle = NativeRenderPlugin.NR_RTS_CreateDescriptorSet(pipeline.Handle);
        }

        // -------------------------------------------------------------------
        // IDisposable
        // -------------------------------------------------------------------

        public override void Dispose()
        {
            _pipeline.OnRebuilt -= OnPipelineRebuilt;
            if (_descriptorSetHandle != 0)
            {
                NativeRenderPlugin.NR_RTS_DestroyDescriptorSet(_descriptorSetHandle);
                _descriptorSetHandle = 0;
            }
            FreeRingBuffers();
        }

        // -------------------------------------------------------------------
        // Internal dispatch helper (called by RayTracePipeline.Dispatch)
        // -------------------------------------------------------------------

        /// <summary>
        /// Snapshots the current staging bindings into the next ring-buffer slot,
        /// fills the event-data header, and returns a pointer to it for
        /// <c>IssuePluginEventAndData</c>.  Returns <see cref="IntPtr.Zero"/> on failure.
        /// </summary>
        internal unsafe IntPtr SnapshotAndBuildHeader(uint width, uint height)
        {
            if (_descriptorSetHandle == 0) return IntPtr.Zero;

            int ring = CopyStagingToRing();
            if (ring < 0) return IntPtr.Zero;

            var header = new NativeRenderPlugin.RTS_RenderEventData
            {
                descriptorSetHandle = _descriptorSetHandle,
                bindingSlotsPtr     = (ulong)_slotRing[ring].GetUnsafePtr(),
                bindingCount        = _slotCount,
                width               = width,
                height              = height,
                _pad                = 0,
            };
            _headerRing[ring][0] = header;

            return (IntPtr)_headerRing[ring].GetUnsafePtr();
        }
    }
}