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

        /// <summary>Per-submesh descriptor. Must match C++ NR_SubmeshDesc exactly (Pack=4).</summary>
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        public struct SubmeshDesc
        {
            public uint indexCount;
            public uint indexByteOffset;
            public uint materialIndex;
        }

        /// <summary>
        /// Per-submesh pre-baked OMM descriptor passed inline to NR_AS_AddInstance.
        /// Must match C++ NR_SubmeshOMMDesc exactly (natural 8-byte pointer alignment).
        /// Set arrayData = IntPtr.Zero to skip OMM for this submesh.
        /// All pointers must remain pinned for the duration of the NR_AS_AddInstance call.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct SubmeshOMMDesc
        {
            public IntPtr arrayData;        // nullable
            public uint   arrayDataSize;
            public uint   _pad0;
            public IntPtr descArray;
            public uint   descArrayCount;
            public uint   _pad1;
            public IntPtr indexBuffer;
            public uint   indexCount;
            public uint   indexStride;
            public IntPtr histogramFlat;    // uint32[] of {count, subdivLevel, format} * histogramCount
            public uint   histogramCount;
            public uint   _pad2;
        }

        /// <summary>
        /// Adds one instance (all submeshes at once) to an acceleration structure.
        /// submeshDescs: pinned pointer to SubmeshDesc[submeshCount].
        /// ommDescs:     pinned pointer to SubmeshOMMDesc[submeshCount], or IntPtr.Zero for no OMM.
        /// instanceHandle must be unique among active instances (e.g. MeshRenderer.GetInstanceID()).
        /// Returns true on success.
        /// </summary>
        [DllImport(DllName)]
        public static extern bool NR_AS_AddInstance(
            ulong  handle,
            uint   instanceHandle,
            IntPtr vertexBufferNativePtr,
            uint   vertexCount,
            uint   vertexStride,
            uint   positionOffset,
            uint   normalOffset,
            uint   texCoord1Offset,
            uint   tangentOffset,
            IntPtr indexBufferNativePtr,
            uint   indexStride,
            IntPtr submeshDescs,
            uint   submeshCount,
            IntPtr ommDescs);   // NR_SubmeshOMMDesc* or null

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
        /// </summary>
        [DllImport(DllName)]
        public static extern ulong NR_CreateRayTraceShaderFromBytes(byte[] dxilBytes, uint size);

        /// <summary>Binds a raw/structured buffer (SRV) by HLSL variable name. Returns 1 on success.</summary>
        [DllImport(DllName)]
        public static extern int NR_RTS_SetBuffer(ulong handle,
            [MarshalAs(UnmanagedType.LPStr)] string name, IntPtr d3d12ResourcePtr);

        /// <summary>Binds an RW buffer (UAV) by HLSL variable name. Returns 1 on success.</summary>
        [DllImport(DllName)]
        public static extern int NR_RTS_SetRWBuffer(ulong handle,
            [MarshalAs(UnmanagedType.LPStr)] string name, IntPtr d3d12ResourcePtr);

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

        /// <summary>Returns the render event callback pointer for per-shader dispatches.</summary>
        [DllImport(DllName)]
        public static extern IntPtr NR_RTS_GetRenderEventFunc();

        /// <summary>Returns sizeof(RTS_RenderEventData) for buffer allocation.</summary>
        [DllImport(DllName)]
        public static extern uint NR_RTS_GetRenderEventDataSize();

        /// <summary>Returns the render event callback pointer for AS BuildOrUpdate.</summary>
        [DllImport(DllName)]
        public static extern IntPtr NR_AS_GetBuildRenderEventFunc();

        /// <summary>Returns sizeof(AS_BuildEventData) for buffer allocation.</summary>
        [DllImport(DllName)]
        public static extern uint NR_AS_GetBuildEventDataSize();

        /// <summary>Returns the native ID3D12Resource* pointer of the TLAS. Used to bind via SetAccelerationStructure.</summary>
        [DllImport(DllName)]
        public static extern IntPtr NR_AS_GetTLASNativePtr(ulong asHandle);

        /// <summary>
        /// Event data for NR_RTS_GetRenderEventFunc dispatches.
        /// Must match C++ RTS_RenderEventData exactly (Pack=4).
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        public struct RTS_RenderEventData
        {
            public ulong  shaderHandle;
            public uint   width;
            public uint   height;
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
            [MarshalAs(UnmanagedType.LPStr)] string extraArgs,
            out IntPtr outBytes,
            out uint   outSize);

        /// <summary>Frees the buffer allocated by <see cref="NR_SC_Compile"/>.</summary>
        [DllImport(DllName)]
        public static extern void NR_SC_Free(IntPtr ptr);        }    }
}