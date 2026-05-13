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
            public uint baseVertex;      // Unity SubMeshDescriptor.baseVertex
            public uint flags;           // NR_SUBMESH_FLAG_* bitmask (bit 0 = GEOMETRY_OPAQUE)
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
            public IntPtr arrayData;        // nullable
            public IntPtr descArray;
            public IntPtr indexBuffer;
            public IntPtr histogramFlat;    // uint32[] of {count, subdivLevel, format} * histogramCount

            public uint   arrayDataSize;
            public uint   descArrayCount;
            public uint   indexCount;
            public uint   indexStride;
            public uint   histogramCount;
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
            public IntPtr submeshDescs;         // NR_SubmeshDesc*
            public IntPtr ommDescs;             // NR_SubmeshOMMDesc* or IntPtr.Zero

            public uint   instanceHandle;       // unique handle (e.g. MeshRenderer.GetInstanceID())
            public uint   vertexCount;
            public uint   vertexStride;
            public uint   indexStride;
            public uint   submeshCount;
            public uint   isDynamic;             // 1 = SkinnedMeshRenderer (BLAS rebuilt every frame)
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
            ulong handle, uint instanceHandle, IntPtr vbPtr, uint vertexCount, uint vertexStride);

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
        /// </summary>
        [DllImport(DllName)]
        public static extern ulong NR_CreateRayTraceShaderFromBytes(byte[] dxilBytes, uint size, string name);

        /// <summary>Binds a raw/structured buffer (SRV) by HLSL variable name. Returns 1 on success.</summary>
        [DllImport(DllName)]
        public static extern int NR_RTS_SetBuffer(ulong handle,
            [MarshalAs(UnmanagedType.LPStr)] string name, IntPtr d3d12ResourcePtr);

        /// <summary>Binds an RW buffer (UAV) by HLSL variable name. Returns 1 on success.</summary>
        [DllImport(DllName)]
        public static extern int NR_RTS_SetRWBuffer(ulong handle,
            [MarshalAs(UnmanagedType.LPStr)] string name, IntPtr d3d12ResourcePtr);

        /// <summary>Binds an RWStructuredBuffer (UAV) by HLSL variable name, with explicit element count and stride. Returns 1 on success.</summary>
        [DllImport(DllName)]
        public static extern int NR_RTS_SetRWStructuredBuffer(ulong handle,
            [MarshalAs(UnmanagedType.LPStr)] string name, IntPtr d3d12ResourcePtr,
            uint elementCount, uint elementStride);

        /// <summary>Binds a texture (SRV) by HLSL variable name. Returns 1 on success.</summary>
        [DllImport(DllName)]
        public static extern int NR_RTS_SetTexture(ulong handle,
            [MarshalAs(UnmanagedType.LPStr)] string name, IntPtr d3d12ResourcePtr);

        /// <summary>Binds an RW texture (UAV) by HLSL variable name. Returns 1 on success.</summary>
        [DllImport(DllName)]
        public static extern int NR_RTS_SetRWTexture(ulong handle,
            [MarshalAs(UnmanagedType.LPStr)] string name, IntPtr d3d12ResourcePtr);

        /// <summary>Binds a constant buffer (CBV) by HLSL variable name. Returns 1 on success.</summary>
        [DllImport(DllName)]
        public static extern int NR_RTS_SetConstantBuffer(ulong handle,
            [MarshalAs(UnmanagedType.LPStr)] string name, IntPtr d3d12ResourcePtr);

        /// <summary>Binds a StructuredBuffer by HLSL variable name, with explicit element count and stride. Returns 1 on success.</summary>
        [DllImport(DllName)]
        public static extern int NR_RTS_SetStructuredBuffer(ulong handle,
            [MarshalAs(UnmanagedType.LPStr)] string name, IntPtr d3d12ResourcePtr,
            uint elementCount, uint elementStride);

        /// <summary>Binds a RaytracingAccelerationStructure (TLAS) by HLSL variable name. Returns 1 on success.</summary>
        [DllImport(DllName)]
        public static extern int NR_RTS_SetAccelerationStructure(ulong handle,
            [MarshalAs(UnmanagedType.LPStr)] string name, IntPtr tlasd3d12Ptr);

        /// <summary>
        /// Preferred: binds by AccelerationStructure handle. The TLAS pointer is resolved dynamically at
        /// Dispatch time, so a full TLAS rebuild (new buffer) is always picked up in the same frame.
        /// </summary>
        [DllImport(DllName)]
        public static extern int NR_RTS_SetAccelerationStructureHandle(ulong shaderHandle,
            [MarshalAs(UnmanagedType.LPStr)] string name, ulong asHandle);

        /// <summary>
        /// Binds a BindlessTexture to an unbounded Texture2D[] variable (any space) by name.
        /// Returns 1 on success, 0 if the variable is not found or is not an array type.
        /// After BindlessTexture.Resize() call this again to rebind the new descriptor range.
        /// </summary>
        [DllImport(DllName)]
        public static extern int NR_RTS_SetBindlessTexture(ulong shaderHandle,
            [MarshalAs(UnmanagedType.LPStr)] string name, ulong btHandle);

        // -------------------------------------------------------------------
        // RayTraceDescriptorSet API
        // -------------------------------------------------------------------

        /// <summary>Creates a RayTraceDescriptorSet bound to the given shader handle. Returns 0 on failure.</summary>
        [DllImport(DllName)]
        public static extern ulong NR_RTS_CreateDescriptorSet(ulong shaderHandle);

        /// <summary>Destroys a RayTraceDescriptorSet created by NR_RTS_CreateDescriptorSet (deferred).</summary>
        [DllImport(DllName)]
        public static extern void NR_RTS_DestroyDescriptorSet(ulong handle);

        /// <summary>Returns the number of resource bindings in the shader (slot count).</summary>
        [DllImport(DllName)]
        public static extern uint NR_RTS_GetBindingCount(ulong shaderHandle);

        /// <summary>Returns the slot index for the named binding, or uint.MaxValue if not found.</summary>
        [DllImport(DllName)]
        public static extern uint NR_RTS_GetSlotIndex(ulong shaderHandle,
            [MarshalAs(UnmanagedType.LPStr)] string name);

        /// <summary>Returns the HLSL variable name for the binding at the given index, or null.</summary>
        [DllImport(DllName)]
        public static extern IntPtr NR_RTS_GetBindingName(ulong shaderHandle, uint index);

        /// <summary>
        /// Hints that the named CBV binding should be treated as 32-bit root constants.
        /// Must be called BEFORE NR_CreateRayTraceShaderFromBytes.
        /// </summary>
        [DllImport(DllName)]
        public static extern void NR_RTS_SetRootConstantsHint(ulong shaderHandle,
            [MarshalAs(UnmanagedType.LPStr)] string name, uint num32BitValues);

        /// <summary>
        /// Hints that the named SRV or TLAS binding should be bound as an inline root SRV descriptor.
        /// Must be called BEFORE NR_CreateRayTraceShaderFromBytes.
        /// </summary>
        [DllImport(DllName)]
        public static extern void NR_RTS_SetRootSRVHint(ulong shaderHandle,
            [MarshalAs(UnmanagedType.LPStr)] string name);

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
            public ulong  descriptorSetHandle;  // +0  RayTraceDescriptorSet*
            public ulong  bindingSlotsPtr;       // +8  CS_BindingSlot*
            public uint   bindingCount;          // +16
            public uint   width;                 // +20
            public uint   height;                // +24
            public uint   _pad;                  // +28
        }

        /// <summary>Event data for NR_AS_GetBuildRenderEventFunc. Must match C++ AS_BuildEventData (Pack=4).</summary>
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        public struct AS_BuildEventData
        {
            public ulong asHandle;
        }

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
        // NativeBuffer API  (triple-buffered upload-heap constant buffer)
        // -------------------------------------------------------------------

        /// <summary>
        /// Allocates a triple-buffered upload-heap buffer of <paramref name="sizeInBytes"/>.
        /// Returns an opaque handle; caller must call NR_DestroyNativeBuffer when done.
        /// </summary>
        [DllImport(DllName)]
        public static extern ulong NR_CreateNativeBuffer(uint sizeInBytes);

        /// <summary>Enqueues destruction after a GPU fence delay.</summary>
        [DllImport(DllName)]
        public static extern void NR_DestroyNativeBuffer(ulong handle);

        /// <summary>
        /// Copies <paramref name="bytes"/> bytes from <paramref name="data"/> into the
        /// current frame's mapped slot.  Call each frame before issuing GPU work.
        /// </summary>
        [DllImport(DllName)]
        public static extern unsafe void NR_NB_Upload(ulong handle, void* data, uint bytes);

        /// <summary>
        /// Returns the ID3D12Resource* for the current frame slot as IntPtr.
        /// Compatibility/debug path; prefer passing the handle as objectPtr with
        /// objectKind = NativeBuffer in CS_BindingSlot.
        /// </summary>
        [DllImport(DllName)]
        public static extern IntPtr NR_NB_GetNativePtr(ulong handle);

        // -------------------------------------------------------------------
        // NativeStructuredBuffer API  (single upload-heap SRV buffer)
        // -------------------------------------------------------------------

        /// <summary>Allocates an upload-heap structured buffer with <paramref name="capacity"/> elements.</summary>
        [DllImport(DllName)]
        public static extern ulong NR_CreateNativeStructuredBuffer(uint capacity, uint elementStride);

        /// <summary>Enqueues destruction after a GPU fence delay.</summary>
        [DllImport(DllName)]
        public static extern void NR_DestroyNativeStructuredBuffer(ulong handle);

        /// <summary>Copies <paramref name="elementCount"/> elements from <paramref name="data"/> starting at <paramref name="elementOffset"/>.</summary>
        [DllImport(DllName)]
        public static extern unsafe void NR_NSB_UploadRange(ulong handle, void* data, uint elementOffset, uint elementCount);

        /// <summary>Grows the buffer to at least <paramref name="newCapacity"/> elements, preserving existing data.</summary>
        [DllImport(DllName)]
        public static extern void NR_NSB_Grow(ulong handle, uint newCapacity);

        /// <summary>Returns the ID3D12Resource* as IntPtr for binding as an SRV.</summary>
        [DllImport(DllName)]
        public static extern IntPtr NR_NSB_GetNativePtr(ulong handle);

        /// <summary>Returns the current element capacity of the buffer.</summary>
        [DllImport(DllName)]
        public static extern uint NR_NSB_GetCapacity(ulong handle);

        /// <summary>
        /// Returns the <c>UnityRenderingEventAndData</c> function pointer for flushing staged
        /// instance data into the GPU-resident buffer via <c>CopyBufferRegion</c>.
        /// Pass the buffer handle as the data pointer to <c>CommandBuffer.IssuePluginEventAndData</c>.
        /// </summary>
        [DllImport(DllName)]
        public static extern IntPtr NR_NSB_GetFlushEventFunc();

        /// <summary>
        /// Returns the <c>UnityRenderingEventAndData</c> function pointer for draining the
        /// main-thread pending-upload queue into staging[g_frameIndex] on the render thread.
        /// Must be issued BEFORE <see cref="NR_NSB_GetFlushEventFunc"/> in the same CommandBuffer.
        /// </summary>
        [DllImport(DllName)]
        public static extern IntPtr NR_NSB_GetDrainEventFunc();

        /// <summary>
        /// Thread-safe (main thread): deep-copies <paramref name="elementCount"/> elements from
        /// <paramref name="data"/> into the pending-upload queue. The actual staging write
        /// happens on the render thread when the Drain event fires.
        /// </summary>
        [DllImport(DllName)]
        public static unsafe extern void NR_NSB_EnqueueUpload(ulong handle, void* data, uint elementOffset, uint elementCount);

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
        /// Like NR_CreateComputeShader, but accepts a hintsJson string that promotes selected
        /// CBV bindings to root 32-bit constants (SetComputeRoot32BitConstants).
        /// hintsJson format: [{"name":"MyConstants","count":4}, ...]
        /// Must be called before resource bindings are set up.
        /// Returns an opaque handle on success, 0 on failure.
        /// </summary>
        [DllImport(DllName)]
        public static extern ulong NR_CreateComputeShaderEx(
            byte[] dxilBytes, uint size,
            [MarshalAs(UnmanagedType.LPStr)] string name,
            [MarshalAs(UnmanagedType.LPStr)] string hintsJson);

        /// <summary>Destroys a ComputeShader created by NR_CreateComputeShader.</summary>
        [DllImport(DllName)]
        public static extern void NR_DestroyComputeShader(ulong handle);

        /// <summary>Returns the total number of reflected bindings for a compute shader handle.</summary>
        [DllImport(DllName)]
        public static extern uint NR_CS_GetBindingCount(ulong handle);

        /// <summary>
        /// Returns the slot index (0-based) for a given HLSL variable name.
        /// Returns uint.MaxValue if the name is not found.
        /// </summary>
        [DllImport(DllName)]
        public static extern uint NR_CS_GetSlotIndex(ulong handle,
            [MarshalAs(UnmanagedType.LPStr)] string name);

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
        /// One slot per reflected binding. Filled on the main thread; read on the render thread.
        /// Must match C++ CS_BindingSlot exactly (Pack=4, 32 bytes).
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        public struct CS_BindingSlot
        {
            public ulong  resourcePtr;   // ID3D12Resource* (may be 0)
            public ulong  objectPtr;     // AccelerationStructure* | BindlessTexture* | BindlessBuffer*
            public uint   count;         // element count  (StructuredBuffer or typed buffer)
            public uint   stride;        // element stride (StructuredBuffer; 0 = raw/typed)
            public uint   objectKind;    // 0=none, 1=AccelStruct, 2=BindlessTexture, 3=BindlessBuffer
            public uint   format;        // DXGI_FORMAT for typed buffer UAV (0 = raw/structured)
        }

        /// <summary>
        /// Event data for NR_CS_GetRenderEventFunc dispatches.
        /// Must match C++ CS_RenderEventData exactly (Pack=4, 32 bytes).
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        public struct CS_RenderEventData
        {
            public ulong  descriptorSetHandle; // pointer to ComputeDescriptorSet
            public uint   threadGroupX;
            public uint   threadGroupY;
            public uint   threadGroupZ;
            public uint   bindingCount;
            public ulong  bindingSlotsPtr; // pointer to CS_BindingSlot[] in pinned NativeArray
        }

        // -----------------------------------------------------------------------
        // ShaderCompilerPlugin — standalone HLSL-to-DXIL compiler DLL.
        // No Unity runtime dependency.
        // -----------------------------------------------------------------------
        public static class ShaderCompilerPlugin
        {
        private const string DllName = "ShaderCompilerPlugin";

        /// <summary>
        /// Compiles the HLSL file at <paramref name="hlslPath"/> to DXIL bytecode.
        /// On success returns true and sets <paramref name="outBytes"/> / <paramref name="outSize"/>;
        /// the caller must free the buffer with <see cref="NR_SC_Free"/>.
        /// <paramref name="includeDirs"/> may be null or semicolon-separated paths.
        /// <paramref name="extraArgs"/> may be null or semicolon-separated additional DXC arguments
        /// (e.g. "-disable-payload-qualifiers").
        /// </summary>
        [DllImport(DllName)]
        public static extern bool NR_SC_Compile(
            [MarshalAs(UnmanagedType.LPStr)] string hlslPath,
            [MarshalAs(UnmanagedType.LPStr)] string includeDirs,
            [MarshalAs(UnmanagedType.LPStr)] string defines,
            [MarshalAs(UnmanagedType.LPStr)] string extraArgs,
            out IntPtr outBytes,
            out uint   outSize);

        /// <summary>Frees the buffer allocated by <see cref="NR_SC_Compile"/>, <see cref="NR_SC_CompileCS"/>, or <see cref="NR_SC_ReflectCS"/>.</summary>
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
            byte[]     dxilBytes,
            uint       size,
            out IntPtr outJson,
            out uint   outJsonLen);

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
            out IntPtr outBytes,
            out uint   outSize);
        }    }
}