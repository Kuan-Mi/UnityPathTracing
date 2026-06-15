using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine.Rendering;

namespace NativeRender
{
    /// <summary>
    /// Holds resource bindings (descriptor set) for a <see cref="NativeComputePipeline"/>.
    ///
    /// Each pass creates one (or more) <see cref="NativeComputeDescriptorSet"/> instances
    /// tied to a <see cref="NativeComputePipeline"/> and calls <c>Set*</c> methods each frame
    /// before passing the descriptor set to
    /// <see cref="NativeComputePipeline.Dispatch(CommandBuffer, NativeComputeDescriptorSet, uint, uint, uint)"/>.
    ///
    /// Owns a ring buffer of pinned <see cref="NativeArray{T}"/> allocations so that
    /// main-thread bindings and render-thread execution are fully decoupled.
    /// The same descriptor set can be dispatched multiple times per frame.
    ///
    /// Lifetime: must be explicitly disposed via <see cref="Dispose"/>.
    /// </summary>
    public sealed class NativeComputeDescriptorSet : NativeDescriptorSetBase
    {
        private readonly NativeComputePipeline _pipeline;

        // Typed header ring for compute dispatch events.
        private NativeArray<NativeRenderPlugin.CS_RenderEventData>[] _headerRing;

        // -------------------------------------------------------------------
        // Construction
        // -------------------------------------------------------------------

        /// <summary>
        /// Creates a descriptor set for the given pipeline.
        /// The descriptor set subscribes to the pipeline's <c>OnRebuilt</c> event so
        /// that it automatically reallocates its ring buffer after a hot-reload.
        /// </summary>
        public NativeComputeDescriptorSet(NativeComputePipeline pipeline)
        {
            if (pipeline == null)
                throw new ArgumentNullException(nameof(pipeline));

            _pipeline = pipeline;
            CopySlotLayout(pipeline);
            AllocateRingBuffers();
            _descriptorSetHandle =  NativeRenderPlugin.NR_CS_CreateDescriptorSet(pipeline.Handle);
            pipeline.OnRebuilt   += OnPipelineRebuilt;
        }

        private void CopySlotLayout(NativeComputePipeline pipeline)
        {
            CopySlotLayoutCommon(pipeline.SlotCount, pipeline.NameToSlot);
        }

        private void AllocateRingBuffers()
        {
            AllocateRingBuffersCommon();
            _headerRing = new NativeArray<NativeRenderPlugin.CS_RenderEventData>[RingSize];
            for (int i = 0; i < RingSize; i++)
                _headerRing[i] = new NativeArray<NativeRenderPlugin.CS_RenderEventData>(1, Allocator.Persistent);
        }

        private void FreeRingBuffers()
        {
            FreeRingBuffersCommon();
            if (_headerRing == null) return;
            for (int i = 0; i < RingSize; i++)
                if (_headerRing[i].IsCreated) _headerRing[i].Dispose();
            _headerRing = null;
        }

        private void OnPipelineRebuilt(NativeComputePipeline pipeline)
        {
            if (_descriptorSetHandle != 0)
            {
                NativeRenderPlugin.NR_CS_DestroyDescriptorSet(_descriptorSetHandle);
                _descriptorSetHandle = 0;
            }
            FreeRingBuffers();
            CopySlotLayout(pipeline);
            AllocateRingBuffers();
            _descriptorSetHandle = NativeRenderPlugin.NR_CS_CreateDescriptorSet(pipeline.Handle);
        }

        // -------------------------------------------------------------------
        // IDisposable
        // -------------------------------------------------------------------

        public override void Dispose()
        {
            _pipeline.OnRebuilt -= OnPipelineRebuilt;
            if (_descriptorSetHandle != 0)
            {
                NativeRenderPlugin.NR_CS_DestroyDescriptorSet(_descriptorSetHandle);
                _descriptorSetHandle = 0;
            }
            FreeRingBuffers();
        }

        // -------------------------------------------------------------------
        // Internal dispatch helper (called by NativeComputePipeline.Dispatch)
        // -------------------------------------------------------------------

        /// <summary>
        /// Snapshots the current staging bindings into the next ring-buffer slot,
        /// fills the event-data header and returns a pointer to it for
        /// <c>IssuePluginEventAndData</c>.  Returns <see cref="IntPtr.Zero"/> on failure.
        /// </summary>
        internal unsafe IntPtr SnapshotAndBuildHeader(
            uint threadGroupX, uint threadGroupY, uint threadGroupZ)
        {
            int ring = CopyStagingToRing();
            if (ring < 0) return IntPtr.Zero;

            var header = new NativeRenderPlugin.CS_RenderEventData
            {
                descriptorSetHandle = _descriptorSetHandle,
                threadGroupX        = threadGroupX,
                threadGroupY        = threadGroupY,
                threadGroupZ        = threadGroupZ,
                bindingCount        = _slotCount,
                bindingSlotsPtr     = (ulong)_slotRing[ring].GetUnsafePtr(),
            };
            _headerRing[ring][0] = header;

            return (IntPtr)_headerRing[ring].GetUnsafePtr();
        }
    }
}