using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine.Rendering;

namespace NativeRender
{
    /// <summary>
    /// Holds resource bindings (SRV / CBV / sampler inputs) for a <see cref="NativeRasterPipeline"/>.
    /// Render targets are NOT bound here — they are passed per draw via <see cref="RasterDrawDesc"/>.
    ///
    /// Mirrors <see cref="NativeComputeDescriptorSet"/>: owns a ring of pinned header / slot buffers
    /// so main-thread binding and render-thread execution are decoupled.
    /// Lifetime: must be explicitly disposed via <see cref="Dispose"/>.
    /// </summary>
    public sealed class NativeRasterDescriptorSet : NativeDescriptorSetBase
    {
        private readonly NativeRasterPipeline _pipeline;
        private NativeArray<NativeRenderPlugin.RAS_RenderEventData>[] _headerRing;

        public NativeRasterDescriptorSet(NativeRasterPipeline pipeline)
        {
            if (pipeline == null) throw new ArgumentNullException(nameof(pipeline));
            _pipeline = pipeline;
            CopySlotLayout(pipeline);
            AllocateRingBuffers();
            _descriptorSetHandle = NativeRenderPlugin.NR_RAS_CreateDescriptorSet(pipeline.Handle);
            pipeline.OnRebuilt  += OnPipelineRebuilt;
        }

        private void CopySlotLayout(NativeRasterPipeline pipeline)
            => CopySlotLayoutCommon(pipeline.SlotCount, pipeline.NameToSlot);

        private void AllocateRingBuffers()
        {
            AllocateRingBuffersCommon();
            _headerRing = new NativeArray<NativeRenderPlugin.RAS_RenderEventData>[RingSize];
            for (int i = 0; i < RingSize; i++)
                _headerRing[i] = new NativeArray<NativeRenderPlugin.RAS_RenderEventData>(1, Allocator.Persistent);
        }

        private void FreeRingBuffers()
        {
            FreeRingBuffersCommon();
            if (_headerRing == null) return;
            for (int i = 0; i < RingSize; i++)
                if (_headerRing[i].IsCreated) _headerRing[i].Dispose();
            _headerRing = null;
        }

        private void OnPipelineRebuilt(NativeRasterPipeline pipeline)
        {
            if (_descriptorSetHandle != 0)
            {
                NativeRenderPlugin.NR_RAS_DestroyDescriptorSet(_descriptorSetHandle);
                _descriptorSetHandle = 0;
            }
            FreeRingBuffers();
            CopySlotLayout(pipeline);
            AllocateRingBuffers();
            _descriptorSetHandle = NativeRenderPlugin.NR_RAS_CreateDescriptorSet(pipeline.Handle);
        }

        public override void Dispose()
        {
            _pipeline.OnRebuilt -= OnPipelineRebuilt;
            if (_descriptorSetHandle != 0)
            {
                NativeRenderPlugin.NR_RAS_DestroyDescriptorSet(_descriptorSetHandle);
                _descriptorSetHandle = 0;
            }
            FreeRingBuffers();
        }

        // -------------------------------------------------------------------
        // Internal draw helper (called by NativeRasterPipeline.Draw)
        // -------------------------------------------------------------------

        internal unsafe IntPtr SnapshotAndBuildHeader(in RasterDrawDesc draw)
        {
            int ring = CopyStagingToRing();
            if (ring < 0) return IntPtr.Zero;

            var hdr = (NativeRenderPlugin.RAS_RenderEventData*)_headerRing[ring].GetUnsafePtr();
            *hdr = default;
            hdr->descriptorSetHandle = _descriptorSetHandle;
            hdr->bindingSlotsPtr     = (ulong)_slotRing[ring].GetUnsafePtr();
            hdr->bindingCount        = _slotCount;

            uint numRT = draw.numRenderTargets;
            if (numRT > 8) numRT = 8;
            hdr->numRenderTargets = numRT;
            for (uint i = 0; i < numRT; i++)
            {
                hdr->rtvResources[i] = (draw.colorResources != null && i < draw.colorResources.Length)
                    ? (ulong)draw.colorResources[i] : 0;
                hdr->rtvFormats[i] = (draw.colorFormats != null && i < draw.colorFormats.Length)
                    ? draw.colorFormats[i] : 0;
            }

            hdr->dsvResource = (ulong)draw.depthResource;
            hdr->dsvFormat   = draw.depthFormat;

            uint clearFlags = 0;
            if (draw.clearColor) clearFlags |= 0x1u;
            if (draw.clearDepth) clearFlags |= 0x2u;
            hdr->clearFlags    = clearFlags;
            hdr->clearColor[0] = draw.clearColorValue.r;
            hdr->clearColor[1] = draw.clearColorValue.g;
            hdr->clearColor[2] = draw.clearColorValue.b;
            hdr->clearColor[3] = draw.clearColorValue.a;
            hdr->clearDepth    = draw.clearDepthValue;

            hdr->viewportX = draw.viewport.x;
            hdr->viewportY = draw.viewport.y;
            hdr->viewportW = draw.viewport.width;
            hdr->viewportH = draw.viewport.height;

            hdr->vertexCount   = draw.vertexCount;
            hdr->instanceCount = draw.instanceCount;
            hdr->blendFactor   = draw.blendFactor;

            return (IntPtr)hdr;
        }
    }
}
