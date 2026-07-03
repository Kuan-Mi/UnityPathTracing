using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace NativeRender
{
    /// <summary>
    /// P/Invoke wrapper for the native DX12 ray tracing plugin.
    /// </summary>
    public static class NativeRenderPlugin
    {
        private const string DllName = "NativeRenderPlugin";

        // -------------------------------------------------------------------
        // Frame lifecycle
        // -------------------------------------------------------------------

        // -------------------------------------------------------------------
        // Acceleration Structure API
        // -------------------------------------------------------------------

        /// <summary>
        /// Creates a new acceleration structure. Returns an opaque uint64 handle.
        /// Caller must call NR_DestroyAccelerationStructure when done.
        /// </summary>
        [DllImport(DllName)]
        public static extern ulong NR_CreateAccelerationStructure();

        [DllImport(DllName)]
        public static extern ulong NR_CreateAccelerationStructureEx([MarshalAs(UnmanagedType.I1)] bool useRtxmu);

        [DllImport(DllName)]
        public static extern ulong NR_CreateAccelerationStructureEx2(
            [MarshalAs(UnmanagedType.I1)] bool useRtxmu,
            [MarshalAs(UnmanagedType.I1)] bool useCompaction);

        /// <summary>Destroys an acceleration structure created by NR_CreateAccelerationStructure.</summary>
        [DllImport(DllName)]
        public static extern void NR_DestroyAccelerationStructure(ulong handle);

        /// <summary>Removes all instances from an acceleration structure.</summary>
        [DllImport(DllName)]
        public static extern void NR_AS_Clear(ulong handle);

        /// <summary>Per-submesh descriptor. Must match C++ NR_SubmeshDesc exactly.</summary>
        public const uint SUBMESH_FLAG_GEOMETRY_OPAQUE = 0x1u;

        public struct SubmeshDesc
        {
            public uint indexCount;
            public uint indexByteOffset;
            public uint baseVertex; // Unity SubMeshDescriptor.baseVertex
            public uint flags; // NR_SUBMESH_FLAG_* bitmask (bit 0 = GEOMETRY_OPAQUE)
        }

        /// <summary>
        /// Per-submesh pre-baked OMM descriptor passed inline to NR_AS_AddInstance.
        /// Must match C++ NR_SubmeshOMMDesc exactly (pointers first, then u32s — no pad needed).
        /// Set arrayData = IntPtr.Zero to skip OMM for this submesh.
        /// All pointers must remain pinned for the duration of the NR_AS_AddInstance call.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct SubmeshOMMDesc
        {
            public IntPtr arrayData; // nullable
            public IntPtr descArray;
            public IntPtr indexBuffer;
            public IntPtr histogramFlat; // uint32[] of {count, subdivLevel, format} * histogramCount

            public uint arrayDataSize;
            public uint descArrayCount;
            public uint indexCount;
            public uint indexStride;
            public uint histogramCount;
        }

        /// <summary>
        /// All per-instance parameters for NR_AS_AddInstance (excluding the AS handle).
        /// Must match the C++ NR_AddInstanceDesc struct layout exactly
        /// (pointers first, then u32s — no pad needed).
        /// submeshDescs and ommDescs must remain pinned for the duration of the call.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct AddInstanceDesc
        {
            public IntPtr vertexBufferNativePtr;
            public IntPtr indexBufferNativePtr;
            public IntPtr submeshDescs; // NR_SubmeshDesc*
            public IntPtr ommDescs; // NR_SubmeshOMMDesc* or IntPtr.Zero

            public uint instanceHandle; // unique handle (e.g. MeshRenderer.GetInstanceID())
            public uint vertexCount;
            public uint vertexStride;
            public uint indexStride;
            public uint submeshCount;
            public uint isDynamic; // 1 = SkinnedMeshRenderer (BLAS rebuilt every frame)
            public uint hitGroupContribution; // InstanceContributionToHitGroupIndex, computed by C#
            public uint _pad; // explicit padding to match C++ 8-byte struct alignment (total 64 bytes)
        }

        /// <summary>
        /// Adds one instance (all submeshes at once) to an acceleration structure.
        /// desc.submeshDescs and desc.ommDescs must be pinned for the duration of the call.
        /// Returns true on success.
        /// </summary>
        [DllImport(DllName)]
        public static extern bool NR_AS_AddInstance(ulong handle, ref AddInstanceDesc desc);

        /// <summary>
        /// Updates the world transform of an existing instance.
        /// transform3x4 must point to 12 floats (row-major 3x4 object-to-world).
        /// </summary>
        [DllImport(DllName)]
        public static extern void NR_AS_SetInstanceTransform(ulong handle, uint instanceIndex, IntPtr transform3x4);

        /// <summary>Sets the per-instance visibility mask (8 bits). Default = 0xFF.</summary>
        [DllImport(DllName)]
        public static extern void NR_AS_SetInstanceMask(ulong handle, uint instanceIndex, byte mask);

        /// <summary>Sets the custom InstanceID returned by InstanceID() in HLSL shaders.
        /// instanceHandle is the value passed to NR_AS_AddInstance (e.g. MeshRenderer.GetInstanceID()).</summary>
        [DllImport(DllName)]
        public static extern void NR_AS_SetInstanceID(ulong handle, uint instanceHandle, uint id);

        /// <summary>Sets the TLAS emission order for an instance. The native build emits TLAS
        /// instances sorted ascending by this value (stable; default 0xFFFFFFFF = legacy slot
        /// order). Required when shaders index per-instance buffers via InstanceIndex().</summary>
        [DllImport(DllName)]
        public static extern void NR_AS_SetInstanceOrderIndex(ulong handle, uint instanceHandle, uint order);

        /// <summary>Updates InstanceContributionToHitGroupIndex for an existing instance (its
        /// base offset in the flat shader table, which shifts on scene layout changes).</summary>
        [DllImport(DllName)]
        public static extern void NR_AS_SetInstanceHitGroupContribution(ulong handle, uint instanceHandle, uint contribution);

        /// <summary>
        /// Removes an instance previously added via NR_AS_AddInstance.
        /// instanceHandle is the value passed to NR_AS_AddInstance.
        /// Decrements the BLAS ref-count; GPU resources are freed after 3 frames.
        /// </summary>
        [DllImport(DllName)]
        public static extern void NR_AS_RemoveInstance(ulong handle, uint instanceHandle);

        /// <summary>
        /// For SkinnedMeshRenderer instances: updates the GPU vertex buffer pointer
        /// to the current-frame skinned result. Discards the stale BLAS (deferred
        /// 3-frame GPU delete) and schedules a BLAS rebuild on the next BuildOrUpdate.
        /// vbPtr must be GraphicsBuffer.GetNativeBufferPtr() for the skinned VB.
        /// </summary>
        [DllImport(DllName)]
        public static extern void NR_AS_UpdateDynamicVertexBuffer(
            ulong handle, uint instanceHandle, IntPtr vbPtr);

        // -------------------------------------------------------------------
        // RayTraceShader API  (multi-shader, per-instance DXR pipelines)
        // -------------------------------------------------------------------

        /// <summary>Destroys a RayTraceShader created by NR_CreateRayTraceShaderFromBytes.</summary>
        [DllImport(DllName)]
        public static extern void NR_DestroyRayTraceShader(ulong handle);

        /// <summary>
        /// Builds a DXR pipeline from pre-compiled DXIL bytes.  Returns an opaque handle
        /// on success, 0 on failure.  The byte array is copied internally; the caller does
        /// not need to keep it alive after this call returns.
        /// name is used as the D3D12 debug name visible in PIX / RenderDoc (optional, can be null).
        /// flags: bit 0 = D3D12_RAYTRACING_PIPELINE_FLAG_ALLOW_OPACITY_MICROMAPS (only set when target profile >= lib_6_9).
        /// maxPayloadSizeInBytes: MaxPayloadSizeInBytes for D3D12_RAYTRACING_SHADER_CONFIG (must match the largest payload struct in the shader).
        /// rayGenName: Name of the RayGeneration entry point to use for DispatchRays. Null or empty = use first discovered.
        /// </summary>
        [DllImport(DllName)]
        public static extern ulong NR_CreateRayTraceShaderFromBytes(byte[] dxilBytes, uint size, string name, uint flags = 0, uint maxPayloadSizeInBytes = 4, string rayGenName = null);

        [DllImport(DllName)]
        public static extern ulong NR_CreateRayTraceShaderFromBytesEx(
            byte[] dxilBytes,
            uint size,
            string name,
            uint flags,
            uint maxPayloadSizeInBytes,
            string rayGenName,
            [In] ulong[] layoutHandles,
            uint layoutHandleCount);

        /// <summary>
        /// Builds a DXR pipeline from N pre-compiled DXIL blobs merged into one RTPSO.
        /// blobDataPtrs[0] must contain raygen + miss shaders.
        /// blobDataPtrs[1..N-1] contain additional hit-group blobs (per-material permutations).
        /// All blobs are pinned by the caller for the duration of this call.
        /// Returns an opaque handle on success, 0 on failure.
        /// </summary>
        [DllImport(DllName)]
        public static extern ulong NR_CreateRayTracePipelineFromBlobs(
            [In] IntPtr[] blobDataPtrs,
            [In] uint[] blobSizes,
            uint blobCount,
            string name,
            uint flags,
            uint maxPayloadSizeInBytes,
            string rayGenName);

        [DllImport(DllName)]
        public static extern ulong NR_CreateRayTracePipelineFromBlobsEx(
            [In] IntPtr[] blobDataPtrs,
            [In] uint[] blobSizes,
            uint blobCount,
            string name,
            uint flags,
            uint maxPayloadSizeInBytes,
            string rayGenName,
            [In] ulong[] layoutHandles,
            uint layoutHandleCount,
            [In] byte[] hitGroupAnyHit,
            uint hitGroupAnyHitCount);


        /// <summary>Creates a RayTraceDescriptorSet bound to the given shader handle. Returns 0 on failure.</summary>
        [DllImport(DllName)]
        public static extern ulong NR_RTS_CreateDescriptorSet(ulong shaderHandle);

        /// <summary>Destroys a RayTraceDescriptorSet created by NR_RTS_CreateDescriptorSet (deferred).</summary>
        [DllImport(DllName)]
        public static extern void NR_RTS_DestroyDescriptorSet(ulong handle);

        /// <summary>Returns the layout slot count (one slot per binding-layout item).
        /// The name→slot mapping is built C#-side from the layout + import-time reflection.</summary>
        [DllImport(DllName)]
        public static extern uint NR_RTS_GetBindingCount(ulong shaderHandle);

        /// <summary>Returns the render event callback pointer for per-descriptor-set dispatches.</summary>
        [DllImport(DllName)]
        public static extern IntPtr NR_RTS_GetRenderEventFunc();

        /// <summary>Returns sizeof(RTS_RenderEventData) for buffer allocation.</summary>
        [DllImport(DllName)]
        public static extern uint NR_RTS_GetRenderEventDataSize();

        /// <summary>Returns the render event callback pointer for AS BuildOrUpdate.</summary>
        [DllImport(DllName)]
        public static extern IntPtr NR_AS_GetBuildRenderEventFunc();

        [DllImport(DllName)]
        public static extern IntPtr NR_GetFrameTickEventFunc();

        /// <summary>
        /// Returns the render-event function pointer for GPU flush + signal.
        /// Issue via <c>cmd.IssuePluginEvent(NR_GetGpuFlushEventFunc(), 0)</c> then
        /// <c>Graphics.ExecuteCommandBuffer(cmd)</c>, then call <see cref="NR_WaitForGpuFlush"/>
        /// on the main thread to block until the render thread has finished and the GPU is idle.
        /// </summary>
        [DllImport(DllName)]
        public static extern IntPtr NR_GetGpuFlushEventFunc();

        /// <summary>
        /// Blocks the calling (main) thread until the render thread has executed
        /// <c>FlushGpuAndWait</c> via the event issued by <see cref="NR_GetGpuFlushEventFunc"/>.
        /// Must be called after <c>Graphics.ExecuteCommandBuffer</c> with that event.
        /// Timeout: 15 seconds.
        /// </summary>
        [DllImport(DllName)]
        public static extern void NR_WaitForGpuFlush();

        /// <summary>Returns sizeof(AS_BuildEventData) for buffer allocation.</summary>
        [DllImport(DllName)]
        public static extern uint NR_AS_GetBuildEventDataSize();

        /// <summary>Returns the native ID3D12Resource* pointer of the TLAS. Used to bind via SetAccelerationStructure.</summary>
        [DllImport(DllName)]
        public static extern IntPtr NR_AS_GetTLASNativePtr(ulong asHandle);

        /// <summary>
        /// Event data for NR_RTS_GetRenderEventFunc dispatches.
        /// Must match C++ RTS_RenderEventData exactly (Pack=4, 32 bytes).
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        public struct RTS_RenderEventData
        {
            public ulong descriptorSetHandle; // +0  RayTraceDescriptorSet*
            public ulong bindingSlotsPtr; // +8  CS_BindingSlot*
            public uint  bindingCount; // +16
            public uint  width; // +20
            public uint  height; // +24
            public uint  _pad; // +28
        }

        /// <summary>Event data for NR_AS_GetBuildRenderEventFunc. Must match C++ AS_BuildEventData (Pack=4, 8 bytes).</summary>
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        public struct AS_BuildEventData
        {
            public ulong asHandle;
        }

        /// <summary>
        /// Event data for NR_RTS_GetRebuildHitGroupTableEventFunc.
        /// C# pre-computes the flat variant-index array; native side writes shader identifiers.
        /// Must match C++ ShtRebuildEventData (Pack=4, 24 bytes).
        /// variantIndicesPtr must point to a pinned/NativeArray of uint32 for the duration of GPU execution.
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        public struct ShtRebuildEventData
        {
            public ulong  shaderHandle; // RayTraceShader*
            public IntPtr variantIndicesPtr; // const uint32_t* — per-geometry variant index array
            public uint   count; // number of geometries
            public uint   _pad;
        }

        /// <summary>Returns the render event function for shader hit-group table rebuild.
        /// Issue via CommandBuffer.IssuePluginEventAndData after the AS build event.</summary>
        [DllImport(DllName)]
        public static extern IntPtr NR_RTS_GetRebuildHitGroupTableEventFunc();

        /// <summary>Returns sizeof(ShtRebuildEventData) for buffer allocation.</summary>
        [DllImport(DllName)]
        public static extern uint NR_RTS_GetRebuildHitGroupTableEventDataSize();

        // -------------------------------------------------------------------
        // BindlessTexture API  (independent GPU-visible texture array)
        // -------------------------------------------------------------------

        /// <summary>
        /// Creates a BindlessTexture with |capacity| shader-visible SRV slots.
        /// Returns an opaque handle; caller must call NR_DestroyBindlessTexture when done.
        /// </summary>
        [DllImport(DllName)]
        public static extern ulong NR_CreateBindlessTexture(uint capacity);

        /// <summary>Destroys a BindlessTexture created by NR_CreateBindlessTexture.</summary>
        [DllImport(DllName)]
        public static extern void NR_DestroyBindlessTexture(ulong handle);

        /// <summary>
        /// Sets the texture at |index| within the BindlessTexture array.
        /// Pass IntPtr.Zero to write a null SRV at that slot.
        /// Returns 1 on success, 0 if index is out of range.
        /// </summary>
        [DllImport(DllName)]
        public static extern int NR_BT_SetTexture(ulong handle, uint index, IntPtr d3d12ResourcePtr);

        /// <summary>
        /// Resizes the BindlessTexture to |newCapacity| slots.
        /// After resize, re-bind to any shader that uses it via NR_RTS_SetBindlessTexture.
        /// </summary>
        [DllImport(DllName)]
        public static extern void NR_BT_Resize(ulong handle, uint newCapacity);

        /// <summary>Returns the current capacity of the BindlessTexture.</summary>
        [DllImport(DllName)]
        public static extern uint NR_BT_GetCapacity(ulong handle);

        // -------------------------------------------------------------------
        // BindlessBuffer API  (independent GPU-visible buffer array)
        // -------------------------------------------------------------------

        /// <summary>
        /// Binds a BindlessBuffer to an unbounded ByteAddressBuffer[] variable by name.
        /// Returns 1 on success, 0 if the variable is not found or is not an array type.
        /// After BindlessBuffer.Resize() call this again to rebind the new descriptor range.
        /// </summary>
        [DllImport(DllName)]
        public static extern int NR_RTS_SetBindlessBuffer(ulong shaderHandle,
            [MarshalAs(UnmanagedType.LPStr)] string name, ulong bbHandle);

        /// <summary>
        /// Creates a BindlessBuffer with |capacity| shader-visible SRV slots (ByteAddressBuffer).
        /// Returns an opaque handle; caller must call NR_DestroyBindlessBuffer when done.
        /// </summary>
        [DllImport(DllName)]
        public static extern ulong NR_CreateBindlessBuffer(uint capacity);

        /// <summary>Destroys a BindlessBuffer created by NR_CreateBindlessBuffer.</summary>
        [DllImport(DllName)]
        public static extern void NR_DestroyBindlessBuffer(ulong handle);

        /// <summary>
        /// Sets the buffer at |index| within the BindlessBuffer array.
        /// Pass IntPtr.Zero to write a null SRV at that slot.
        /// Returns 1 on success, 0 if index is out of range.
        /// </summary>
        [DllImport(DllName)]
        public static extern int NR_BB_SetBuffer(ulong handle, uint index, IntPtr d3d12ResourcePtr);

        /// <summary>
        /// Resizes the BindlessBuffer to |newCapacity| slots.
        /// After resize, re-bind to any shader that uses it via NR_RTS_SetBindlessBuffer.
        /// </summary>
        [DllImport(DllName)]
        public static extern void NR_BB_Resize(ulong handle, uint newCapacity);

        /// <summary>Returns the current capacity of the BindlessBuffer.</summary>
        [DllImport(DllName)]
        public static extern uint NR_BB_GetCapacity(ulong handle);

        // -------------------------------------------------------------------
        // BindlessUAVTexture API  (independent GPU-visible RWTexture2D[] UAV array)
        // -------------------------------------------------------------------

        /// <summary>
        /// Creates a BindlessUAVTexture with |capacity| shader-visible UAV slots (RWTexture2D[]).
        /// Returns an opaque handle; caller must call NR_DestroyBindlessUAVTexture when done.
        /// </summary>
        [DllImport(DllName)]
        public static extern ulong NR_CreateBindlessUAVTexture(uint capacity);

        /// <summary>Destroys a BindlessUAVTexture created by NR_CreateBindlessUAVTexture.</summary>
        [DllImport(DllName)]
        public static extern void NR_DestroyBindlessUAVTexture(ulong handle);

        /// <summary>
        /// Sets the UAV descriptor at |index| within the BindlessUAVTexture array.
        /// Pass IntPtr.Zero to write a null UAV at that slot.
        /// <paramref name="mipSlice"/>: which mip level to expose (0-based).
        /// <paramref name="dxgiFormat"/>: DXGI_FORMAT for the view; 0 = derive from resource.
        /// Returns 1 on success, 0 if index is out of range.
        /// </summary>
        [DllImport(DllName)]
        public static extern int NR_BUAV_SetTexture(ulong handle, uint index,
            IntPtr d3d12ResourcePtr, uint mipSlice, uint dxgiFormat);

        /// <summary>
        /// Resizes the BindlessUAVTexture to |newCapacity| slots.
        /// After resize, re-bind to any shader that uses it.
        /// </summary>
        [DllImport(DllName)]
        public static extern void NR_BUAV_Resize(ulong handle, uint newCapacity);

        /// <summary>Returns the current capacity of the BindlessUAVTexture.</summary>
        [DllImport(DllName)]
        public static extern uint NR_BUAV_GetCapacity(ulong handle);

        // -------------------------------------------------------------------
        // NativeBuffer API  (unified plugin buffer — nvrhi Buffer analog)
        //
        // One create/destroy pair covers the three former buffer roles; the desc
        // flags select the heap and the active upload/readback path:
        //   * Volatile constant/dynamic buffer:
        //         isVolatile = 1, maxVersions = N (>1)  -> per-frame UPLOAD ring,
        //         written via the upload callback (GetNativeBufferUploadCallbackPtr).
        //   * GPU-resident structured buffer with staged CPU writes:
        //         structStride > 0  -> DEFAULT heap fed by the NR_NSB_* snapshot path.
        //   * GPU-only UAV buffer:
        //         canHaveUAVs = 1  -> DEFAULT heap with ALLOW_UNORDERED_ACCESS.
        // -------------------------------------------------------------------

        /// <summary>
        /// Allocates a unified plugin buffer. Returns an opaque handle; caller must call
        /// <see cref="NR_DestroyNativeBuffer"/> when done.
        /// </summary>
        [DllImport(DllName)]
        public static extern ulong NR_CreateNativeBuffer(
            ulong byteSize, uint structStride, uint maxVersions,
            uint canHaveUAVs, uint isConstantBuffer, uint isVolatile,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string debugName);

        /// <summary>Enqueues destruction after a GPU fence delay.</summary>
        [DllImport(DllName)]
        public static extern void NR_DestroyNativeBuffer(ulong handle);

        /// <summary>
        /// Returns the render-event callback pointer for zeroing a DEFAULT-heap buffer via
        /// ClearUnorderedAccessViewUint. Pass the result to <c>cmd.IssuePluginEventAndData</c>
        /// with the buffer Handle as data.
        /// </summary>
        [DllImport(DllName)]
        public static extern IntPtr NR_GetClearNativeGpuBufferCallbackPtr();

        /// <summary>
        /// Returns the render-event callback pointer that clears a UAV-capable D3D12 texture via
        /// ClearUnorderedAccessViewFloat (nvrhi clearTextureFloat equivalent — no pipeline, no
        /// draw). Pass a <c>ClearTextureUavFloatEventData</c> blob (allocated via
        /// <see cref="NR_NSB_AllocFlushBuffer"/>, see <see cref="NativeTextureClear"/>) as the data
        /// argument of <c>cmd.IssuePluginEventAndData</c>; the callback frees it.
        /// </summary>
        [DllImport(DllName)]
        public static extern IntPtr NR_GetClearTextureUavFloatCallbackPtr();

        /// <summary>Returns the current element capacity of the buffer.</summary>
        [DllImport(DllName)]
        public static extern uint NR_NSB_GetCapacity(ulong handle);

        /// <summary>
        /// Returns the <c>UnityRenderingEventAndData</c> function pointer that drains the pending
        /// queue and records the staged <c>CopyBufferRegion</c> into the GPU-resident buffer.
        /// Pass the buffer handle as the data pointer to <c>CommandBuffer.IssuePluginEventAndData</c>.
        /// </summary>
        [DllImport(DllName)]
        public static extern IntPtr NR_NSB_GetFlushEventFunc();

        /// <summary>
        /// Allocates a flush snapshot blob of <paramref name="sizeBytes"/> bytes. C# fills it with
        /// the header + range table + packed payload and passes the pointer as the data argument of
        /// <c>CommandBuffer.IssuePluginEventAndData</c>; the render-thread flush callback parses and
        /// frees it. Allocated and freed by the plugin so the allocator matches across the boundary.
        /// </summary>
        [DllImport(DllName)]
        public static extern IntPtr NR_NSB_AllocFlushBuffer(uint sizeBytes);

        /// <summary>
        /// Returns the <c>UnityRenderingEventAndData</c> function pointer that records a GPU->CPU
        /// readback copy. Pass an <c>NsbReadbackRequest</c> blob (allocated via
        /// <see cref="NR_NSB_AllocFlushBuffer"/>) as the data argument of
        /// <c>CommandBuffer.IssuePluginEventAndData</c>; the callback frees it.
        /// </summary>
        [DllImport(DllName)]
        public static extern IntPtr NR_NSB_GetReadbackEventFunc();

        /// <summary>
        /// Main thread: copies a completed readback into <paramref name="dst"/> (up to
        /// <paramref name="dstBytes"/>) and returns 1; returns 0 while the copy is still in flight
        /// on the GPU or when no readback is pending.
        /// </summary>
        [DllImport(DllName)]
        public static extern uint NR_NSB_TryReadback(ulong handle, IntPtr dst, ulong dstBytes);

        // -------------------------------------------------------------------
        // ComputeShader API  (generic compute pipeline, cs_6_x)
        // -------------------------------------------------------------------

        /// <summary>
        /// Builds a compute pipeline from pre-compiled DXIL bytes (cs_6_x).
        /// Returns an opaque handle on success, 0 on failure.
        /// name is used as the D3D12 debug name visible in PIX / RenderDoc (optional, can be null).
        /// </summary>
        [DllImport(DllName)]
        public static extern ulong NR_CreateComputeShader(byte[] dxilBytes, uint size, string name);

        /// <summary>
        /// Like NR_CreateComputeShader, but accepts native binding layout handles
        /// that shape the native root signature.
        /// Must be called before resource bindings are set up.
        /// Returns an opaque handle on success, 0 on failure.
        /// </summary>
        [DllImport(DllName)]
        public static extern ulong NR_CreateComputeShaderEx(
            byte[] dxilBytes, uint size,
            [MarshalAs(UnmanagedType.LPStr)] string name,
            [In] ulong[] layoutHandles,
            uint layoutHandleCount);

        /// <summary>Destroys a ComputeShader created by NR_CreateComputeShader.</summary>
        [DllImport(DllName)]
        public static extern void NR_DestroyComputeShader(ulong handle);

        /// <summary>Returns the layout slot count (one slot per binding-layout item).
        /// The name→slot mapping is built C#-side from the layout + import-time reflection.</summary>
        [DllImport(DllName)]
        public static extern uint NR_CS_GetBindingCount(ulong handle);

        /// <summary>Returns the render event callback pointer for compute dispatches.</summary>
        [DllImport(DllName)]
        public static extern IntPtr NR_CS_GetRenderEventFunc();

        /// <summary>Returns sizeof(CS_RenderEventData) for buffer allocation.</summary>
        [DllImport(DllName)]
        public static extern uint NR_CS_GetRenderEventDataSize();

        /// <summary>Creates a ComputeDescriptorSet for the given ComputeShader handle.
        /// Each C# NativeComputeDescriptorSet must own one of these so multiple
        /// dispatches of the same shader per frame use independent heap slices.</summary>
        [DllImport(DllName)]
        public static extern ulong NR_CS_CreateDescriptorSet(ulong shaderHandle);

        /// <summary>Destroys a ComputeDescriptorSet (deferred until GPU is done).</summary>
        [DllImport(DllName)]
        public static extern void NR_CS_DestroyDescriptorSet(ulong handle);

        // -----------------------------------------------------------------------
        // CS data structures passed through IssuePluginEventAndData
        // -----------------------------------------------------------------------

        /// <summary>
        /// One slot per binding-layout item. Filled on the main thread; read on the render thread.
        /// Must match C++ BindingSlot exactly (Pack=4, 24 bytes).
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        public struct CS_BindingSlot
        {
            public ulong objectPtr; // resource/wrapper/payload pointer, interpreted per objectKind
            public uint  count; // element count  (StructuredBuffer or typed buffer)
            public uint  stride; // element stride (StructuredBuffer; 0 = raw/typed)
            public uint  objectKind; // 0=None(raw ID3D12Resource*),1=AccelStruct,2=BindlessTex,3=BindlessBuf,4=RootConstants,5=NativeBuffer,6=BindlessUAVTex,7=Sampler
            public uint  format; // DXGI_FORMAT for typed buffer UAV (0 = raw/structured)
        }

        /// <summary>One nvrhi-style binding layout descriptor sent when creating a native binding layout.</summary>
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        public struct NR_BindingLayoutDesc
        {
            public uint visibility;
            public uint registerSpace;
            public IntPtr bindings;
            public uint bindingCount;
        }

        /// <summary>One nvrhi-style binding-layout item sent when creating a native binding layout.</summary>
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        public struct NR_BindingLayoutItem
        {
            public uint type;
            public uint slot;
            public uint size;
        }

        /// <summary>One nvrhi-style bindless layout descriptor.</summary>
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        public struct NR_BindlessLayoutDesc
        {
            public uint visibility;
            public uint firstSlot;
            public uint maxCapacity;
            public IntPtr registerSpaces;
            public uint registerSpaceCount;
        }

        /// <summary>One explicit static-sampler definition sent when creating a native binding layout.</summary>
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        public struct NR_StaticSampler
        {
            public uint reg;
            public uint space;
            public uint filter;
            public uint addressU;
            public uint addressV;
            public uint addressW;
            public uint mips;
            public uint maxAnisotropy;
        }

        [DllImport(DllName)]
        public static extern ulong NR_CreateBindingLayout(
            [In] ref NR_BindingLayoutDesc layoutDesc,
            [In] NR_StaticSampler[] staticSamplers,
            uint staticSamplerCount);

        [DllImport(DllName)]
        public static extern ulong NR_CreateBindlessLayout(
            [In] ref NR_BindlessLayoutDesc bindlessDesc);

        [DllImport(DllName)]
        public static extern void NR_DestroyBindingLayout(ulong handle);

        /// <summary>
        /// Event data for NR_CS_GetRenderEventFunc dispatches.
        /// Must match C++ CS_RenderEventData exactly (Pack=4, 32 bytes).
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        public struct CS_RenderEventData
        {
            public ulong descriptorSetHandle; // pointer to ComputeDescriptorSet
            public uint  threadGroupX;
            public uint  threadGroupY;
            public uint  threadGroupZ;
            public uint  bindingCount;
            public ulong bindingSlotsPtr; // pointer to CS_BindingSlot[] in pinned NativeArray
        }

        // -------------------------------------------------------------------
        // BC6H cubemap API  (plugin-owned compressed env cube + reinterpret copy)
        // -------------------------------------------------------------------

        /// <summary>Creates a BC6H_UFLOAT TextureCube (6 faces, <paramref name="mipLevels"/> mips)
        /// committed resource. Returns the raw ID3D12Resource* (bindable like a Unity texture ptr),
        /// or IntPtr.Zero on failure. Release via <see cref="NR_DestroyBC6HCube"/>.</summary>
        [DllImport(DllName)]
        public static extern IntPtr NR_CreateBC6HCube(uint dim, uint mipLevels);

        /// <summary>Defer-releases a BC6H cube created by <see cref="NR_CreateBC6HCube"/>.</summary>
        [DllImport(DllName)]
        public static extern void NR_DestroyBC6HCube(IntPtr handle);

        /// <summary>Returns the render event callback that reinterpret-copies the RGBA32_UINT scratch
        /// cube into the BC6H cube (per subresource). Pass a pinned <see cref="BC6HCopy_RenderEventData"/>
        /// to <c>CommandBuffer.IssuePluginEventAndData</c>.</summary>
        [DllImport(DllName)]
        public static extern IntPtr NR_GetBC6HCopyEventFunc();

        /// <summary>Returns sizeof(BC6HCopy_RenderEventData) for buffer allocation.</summary>
        [DllImport(DllName)]
        public static extern uint NR_GetBC6HCopyEventDataSize();

        /// <summary>
        /// Event data for <see cref="NR_GetBC6HCopyEventFunc"/>. Must match C++
        /// BC6HCopy_RenderEventData exactly (Pack=4, 24 bytes).
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        public struct BC6HCopy_RenderEventData
        {
            public ulong srcScratch; // ID3D12Resource* RGBA32_UINT scratch cube
            public ulong dstBC6H; // ID3D12Resource* BC6H cube
            public uint  mipLevels;
            public uint  arraySize; // 6 for a cube
        }

        // -------------------------------------------------------------------
        // RasterShader API  (generic graphics pipeline, vs_6_x + ps_6_x)
        // -------------------------------------------------------------------

        /// <summary>
        /// Fixed-function graphics-pipeline state. Must match C++ RasterPipelineStateDesc
        /// exactly (Pack=4). Enum fields use raw D3D12 enum values; 0 selects a default
        /// (cull NONE, fill SOLID, depth func LESS_EQUAL, topology TRIANGLE).
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        public struct RasterPipelineStateDesc
        {
            public uint numRenderTargets; // 0..8

            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
            public uint[] rtvFormats; // DXGI_FORMAT per RT

            public uint dsvFormat; // DXGI_FORMAT, 0 = none
            public uint cullMode; // D3D12_CULL_MODE; 0 => NONE
            public uint fillMode; // D3D12_FILL_MODE; 0 => SOLID
            public uint depthTestEnable;
            public uint depthWriteEnable;
            public uint depthFunc; // D3D12_COMPARISON_FUNC; 0 => LESS_EQUAL
            public uint blendMode; // 0=opaque,1=alpha,2=additive,3=premultiplied
            public uint primitiveTopology; // D3D_PRIMITIVE_TOPOLOGY (TRIANGLELIST=4, TRIANGLESTRIP=5, …); 0 => TRIANGLELIST
            public uint frontCounterClockwise;
            public uint sampleCount; // 0 => 1

            // D3D_PRIMITIVE_TOPOLOGY values.
            public const uint TopologyTriangleList  = 4;
            public const uint TopologyTriangleStrip = 5;

            // blendMode values (see C++ MakeBlend).
            public const uint BlendModeOpaque        = 0;
            public const uint BlendModeAlpha         = 1;
            public const uint BlendModeAdditive      = 2;
            public const uint BlendModePremultiplied = 3;
            public const uint BlendModeConstantColor = 4; // src*C + dst*(1-C); C = RasterDrawDesc.blendFactor

            /// <summary>
            /// Single-RTV opaque fullscreen-pass default for the given color format. Uses a triangle
            /// STRIP (the donut fullscreen_vs.hlsl quad) — draw with vertexCount = 4. Pass
            /// <paramref name="primitiveTopology"/> = <see cref="TopologyTriangleList"/> for a 3-vertex
            /// triangle-list fullscreen VS instead.
            /// </summary>
            public static RasterPipelineStateDesc FullscreenOpaque(uint rtvFormat, uint primitiveTopology = TopologyTriangleStrip)
            {
                var d = new RasterPipelineStateDesc
                {
                    numRenderTargets      = 1,
                    rtvFormats            = new uint[8],
                    dsvFormat             = 0,
                    cullMode              = 1, // D3D12_CULL_MODE_NONE
                    fillMode              = 3, // D3D12_FILL_MODE_SOLID
                    depthTestEnable       = 0,
                    depthWriteEnable      = 0,
                    depthFunc             = 0,
                    blendMode             = 0,
                    primitiveTopology     = primitiveTopology,
                    frontCounterClockwise = 0,
                    sampleCount           = 1,
                };
                d.rtvFormats[0] = rtvFormat;
                return d;
            }
        }

        /// <summary>
        /// Builds a graphics pipeline from pre-compiled VS + PS DXIL blobs and a fixed-function
        /// state desc. Returns an opaque handle on success, 0 on failure.
        /// </summary>
        [DllImport(DllName)]
        public static extern ulong NR_CreateRasterShader(
            byte[] vsDxil, uint vsSize, byte[] psDxil, uint psSize,
            ref RasterPipelineStateDesc state,
            [MarshalAs(UnmanagedType.LPStr)] string name);

        /// <summary>Like NR_CreateRasterShader, with explicit native binding layout handles.</summary>
        [DllImport(DllName)]
        public static extern ulong NR_CreateRasterShaderEx(
            byte[] vsDxil, uint vsSize, byte[] psDxil, uint psSize,
            ref RasterPipelineStateDesc state,
            [MarshalAs(UnmanagedType.LPStr)] string name,
            [In] ulong[] layoutHandles,
            uint layoutHandleCount);

        /// <summary>Destroys a RasterShader created by NR_CreateRasterShader(Ex) (deferred).</summary>
        [DllImport(DllName)]
        public static extern void NR_DestroyRasterShader(ulong handle);

        /// <summary>Creates a RasterDescriptorSet for the given RasterShader handle. Returns 0 on failure.</summary>
        [DllImport(DllName)]
        public static extern ulong NR_RAS_CreateDescriptorSet(ulong shaderHandle);

        /// <summary>Destroys a RasterDescriptorSet (deferred until the GPU is done).</summary>
        [DllImport(DllName)]
        public static extern void NR_RAS_DestroyDescriptorSet(ulong handle);

        /// <summary>Returns the layout slot count (one slot per binding-layout item).
        /// The name→slot mapping is built C#-side from the layout + import-time reflection.</summary>
        [DllImport(DllName)]
        public static extern uint NR_RAS_GetBindingCount(ulong handle);

        /// <summary>Returns the render event callback pointer for per-descriptor-set draws.</summary>
        [DllImport(DllName)]
        public static extern IntPtr NR_RAS_GetRenderEventFunc();

        /// <summary>Returns sizeof(RAS_RenderEventData) for buffer allocation.</summary>
        [DllImport(DllName)]
        public static extern uint NR_RAS_GetRenderEventDataSize();

        /// <summary>
        /// Event data for NR_RAS_GetRenderEventFunc draws.
        /// Must match C++ RAS_RenderEventData exactly (Pack=4).
        /// Blittable (fixed buffers, no managed arrays) so it can live in a pinned
        /// <see cref="Unity.Collections.NativeArray{T}"/> ring like CS_RenderEventData.
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        public unsafe struct RAS_RenderEventData
        {
            public       ulong descriptorSetHandle; // RasterDescriptorSet*
            public       ulong bindingSlotsPtr; // CS_BindingSlot*
            public       uint  bindingCount;
            public       uint  numRenderTargets;
            public fixed ulong rtvResources[8]; // ID3D12Resource*
            public fixed uint  rtvFormats[8]; // DXGI_FORMAT
            public       ulong dsvResource; // ID3D12Resource* or 0
            public       uint  dsvFormat; // DXGI_FORMAT
            public       uint  clearFlags; // bit0 = clear color, bit1 = clear depth
            public fixed float clearColor[4];
            public       float clearDepth;
            public       float viewportX, viewportY, viewportW, viewportH;
            public       uint  vertexCount;
            public       uint  instanceCount;
            public       float blendFactor; // OMSetBlendFactor constant (blendMode==4 constant-color); was _pad
        }

        // -----------------------------------------------------------------------
        // ShaderCompilerPlugin — standalone HLSL-to-DXIL compiler DLL.
        // No Unity runtime dependency.
        // -----------------------------------------------------------------------
        public static class ShaderCompilerPlugin
        {
            private const string DllName = "ShaderCompilerPlugin";

            /// <summary>
            /// Directory where DXC writes separate shader PDBs (debug symbols). Compiling with
            /// debug info (-Zi) but without -Qembed_debug keeps the DXIL blob — and therefore the
            /// serialized asset and the player build — small, while PIX still resolves symbols from
            /// the PDBs in this folder (add it to PIX's symbol search path). Compilation only runs
            /// at editor import time, so this resolves relative to the project root.
            /// </summary>
            public static string PdbOutputDir =>
                System.IO.Path.GetFullPath(System.IO.Path.Combine(Application.dataPath, "..", "ShaderPDB"));

            /// <summary>
            /// Compiles the HLSL file at <paramref name="hlslPath"/> to DXIL bytecode.
            /// On success returns true and sets <paramref name="outBytes"/> / <paramref name="outSize"/>;
            /// the caller must free the buffer with <see cref="NR_SC_Free"/>.
            /// <paramref name="includeDirs"/> may be null or semicolon-separated paths.
            /// <paramref name="extraArgs"/> may be null or semicolon-separated additional DXC arguments
            /// (e.g. "-disable-payload-qualifiers").
            /// <paramref name="pdbOutDir"/> may be null/empty; when set, the separate PDB is written there.
            /// </summary>
            [DllImport(DllName)]
            public static extern bool NR_SC_Compile(
                [MarshalAs(UnmanagedType.LPStr)] string hlslPath,
                [MarshalAs(UnmanagedType.LPStr)] string targetProfile,
                [MarshalAs(UnmanagedType.LPStr)] string includeDirs,
                [MarshalAs(UnmanagedType.LPStr)] string defines,
                [MarshalAs(UnmanagedType.LPStr)] string extraArgs,
                [MarshalAs(UnmanagedType.LPStr)] string pdbOutDir,
                out IntPtr outBytes,
                out uint outSize);

            /// <summary>Frees the buffer allocated by <see cref="NR_SC_Compile"/>, <see cref="NR_SC_CompileCS"/>, <see cref="NR_SC_ReflectCS"/>, or <see cref="NR_SC_ReflectLib"/>.</summary>
            [DllImport(DllName)]
            public static extern void NR_SC_Free(IntPtr ptr);

            /// <summary>
            /// Reflects a compiled DXIL compute shader blob and returns a JSON string describing
            /// bound resources (SRV / UAV / CBV / Sampler) and the numthreads declaration.
            /// On success returns true and sets <paramref name="outJson"/> / <paramref name="outJsonLen"/>;
            /// the caller must free the buffer with <see cref="NR_SC_Free"/>.
            /// </summary>
            [DllImport(DllName)]
            public static extern bool NR_SC_ReflectCS(
                byte[] dxilBytes,
                uint size,
                out IntPtr outJson,
                out uint outJsonLen);

            /// <summary>
            /// Reflects a compiled DXIL *library* blob (lib_6_x, used by DXR ray tracing shaders) and
            /// returns a JSON string describing bound resources across all exported functions.
            /// JSON shape: { "bindings": [ { "name", "type", "space", "reg", "dim", "retType" }, ... ] }
            /// On success returns true; the caller must free the buffer with <see cref="NR_SC_Free"/>.
            /// </summary>
            [DllImport(DllName)]
            public static extern bool NR_SC_ReflectLib(
                byte[] dxilBytes,
                uint size,
                out IntPtr outJson,
                out uint outJsonLen);

            /// <summary>
            /// Compiles a compute shader HLSL file to DXIL bytecode with a specified entry point and target profile.
            /// On success returns true and sets <paramref name="outBytes"/> / <paramref name="outSize"/>;
            /// the caller must free the buffer with <see cref="NR_SC_Free"/>.
            /// </summary>
            [DllImport(DllName)]
            public static extern bool NR_SC_CompileCS(
                [MarshalAs(UnmanagedType.LPStr)] string hlslPath,
                [MarshalAs(UnmanagedType.LPStr)] string entryPoint,
                [MarshalAs(UnmanagedType.LPStr)] string target,
                [MarshalAs(UnmanagedType.LPStr)] string includeDirs,
                [MarshalAs(UnmanagedType.LPStr)] string defines,
                [MarshalAs(UnmanagedType.LPStr)] string extraArgs,
                [MarshalAs(UnmanagedType.LPStr)] string pdbOutDir,
                out IntPtr outBytes,
                out uint outSize);
        }

        [DllImport(DllName)]
        public static extern IntPtr GetNativeBufferUploadCallbackPtr();
    }
}
