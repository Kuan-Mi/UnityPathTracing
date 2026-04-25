using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

namespace NativeRender
{
    /// <summary>
    /// Managed wrapper around the native RayTracingAccelerationStructure.
    /// Owns texture/material registration and GPU-instance management keyed by
    /// <see cref="MeshRenderer.GetInstanceID"/>.
    /// </summary>
    public class RayTracingAccelerationStructure : IDisposable
    {
        private ulong _handle;

        /// <summary>Opaque native handle — passed to RenderEventData.accelerationStructureHandle.</summary>
        public ulong Handle => _handle;

        // Persisted event data for AS build dispatches
        private NativeArray<NativeRenderPlugin.AS_BuildEventData> _buildEventData;

        /// <summary>
        /// Issues a GPU command to build or update the acceleration structure.
        /// Must be called before the ray trace Dispatch each frame.
        /// </summary>
        public void BuildOrUpdate(CommandBuffer cmd)
        {
            if (_handle == 0) return;
            if (!_buildEventData.IsCreated)
            {
                _buildEventData    = new NativeArray<NativeRenderPlugin.AS_BuildEventData>(1, Allocator.Persistent);
                _buildEventData[0] = new NativeRenderPlugin.AS_BuildEventData { asHandle = _handle };
            }

            unsafe
            {
                cmd.IssuePluginEventAndData(
                    NativeRenderPlugin.NR_AS_GetBuildRenderEventFunc(),
                    0,
                    (IntPtr)Unity.Collections.LowLevel.Unsafe.NativeArrayUnsafeUtility.GetUnsafePtr(_buildEventData));
            }
        }

        /// <summary>Returns the native ID3D12Resource* of the TLAS for binding via SetAccelerationStructure.</summary>
        public IntPtr GetTLASNativePtr() => NativeRenderPlugin.NR_AS_GetTLASNativePtr(_handle);

        public RayTracingAccelerationStructure()
        {
            _handle = NativeRenderPlugin.NR_CreateAccelerationStructure();
            if (_handle == 0)
                throw new InvalidOperationException(
                    "[NativeRayTracing] NR_CreateAccelerationStructure returned null. " +
                    "Is the plugin loaded and the renderer initialised?");
        }

        public void Dispose()
        {
            if (_buildEventData.IsCreated) _buildEventData.Dispose();
            if (_handle != 0)
            {
                NativeRenderPlugin.NR_DestroyAccelerationStructure(_handle);
                _handle = 0;
            }
        }

        /// <summary>Remove all instances from the acceleration structure.</summary>
        public void Clear()
        {
            if (_handle != 0)
                NativeRenderPlugin.NR_AS_Clear(_handle);
        }

        /// <summary>
        /// Registers <paramref name="meshRenderer"/> as one BLAS instance.
        /// Uploads textures and registers materials automatically.
        /// <paramref name="ommCaches"/> may be null; if provided its length must match
        /// the mesh's subMeshCount.
        /// </summary>
        /// <returns>True on success.</returns>
        public unsafe bool AddInstance(MeshRenderer meshRenderer)
        {
            if (_handle == 0 || meshRenderer == null) return false;

            var meshFilter = meshRenderer.GetComponent<MeshFilter>();
            if (meshFilter == null || meshFilter.sharedMesh == null)
            {
                Debug.LogWarning($"[NativeRayTracing] '{meshRenderer.name}' has no mesh — skipping");
                return false;
            }

            Mesh mesh = meshFilter.sharedMesh;
            mesh.UploadMeshData(false);

            OMMCache[] ommCaches = null;
            var nativeTarget = meshFilter.GetComponent<NativeRayTracingTarget>();
            if (nativeTarget != null)
            {
                ommCaches = nativeTarget.ommCaches;
            }

            IntPtr vbPtr = mesh.GetNativeVertexBufferPtr(0);
            IntPtr ibPtr = mesh.GetNativeIndexBufferPtr();
            if (vbPtr == IntPtr.Zero || ibPtr == IntPtr.Zero)
            {
                Debug.LogError($"[NativeRayTracing] Failed to get GPU buffers for '{mesh.name}' — skipping");
                return false;
            }

            uint vertexCount  = (uint)mesh.vertexCount;
            uint vertexStride = (uint)mesh.GetVertexBufferStride(0);
            uint indexStride  = mesh.indexFormat == IndexFormat.UInt16 ? 2u : 4u;

            int subMeshCount = mesh.subMeshCount;

            // Build SubmeshDesc array.
            var                                 submeshDescs = new NativeRenderPlugin.SubmeshDesc[subMeshCount];
            NativeRenderPlugin.SubmeshOMMDesc[] ommDescs     = null;
            bool                                hasAnyOMM    = false;

            for (int s = 0; s < subMeshCount; s++)
            {
                SubMeshDescriptor sub = mesh.GetSubMesh(s);
                submeshDescs[s] = new NativeRenderPlugin.SubmeshDesc
                {
                    indexCount      = (uint)sub.indexCount,
                    indexByteOffset = (uint)sub.indexStart * indexStride,
                };

                if (ommCaches != null && s < ommCaches.Length && ommCaches[s] != null && ommCaches[s].IsValid)
                    hasAnyOMM = true;
            }

            if (hasAnyOMM)
                ommDescs = new NativeRenderPlugin.SubmeshOMMDesc[subMeshCount];

            var handles = new List<GCHandle>();
            try
            {
                if (ommDescs != null)
                {
                    for (int s = 0; s < subMeshCount; s++)
                    {
                        OMMCache cache = (ommCaches != null && s < ommCaches.Length) ? ommCaches[s] : null;
                        if (cache == null || !cache.IsValid) continue;

                        var hArray = GCHandle.Alloc(cache.bakedArrayData, GCHandleType.Pinned);
                        var hDesc  = GCHandle.Alloc(cache.bakedDescArray, GCHandleType.Pinned);
                        var hIdx   = GCHandle.Alloc(cache.bakedIndexBuffer, GCHandleType.Pinned);
                        var hHist  = GCHandle.Alloc(cache.histogramFlat, GCHandleType.Pinned);
                        handles.Add(hArray);
                        handles.Add(hDesc);
                        handles.Add(hIdx);
                        handles.Add(hHist);

                        ommDescs[s] = new NativeRenderPlugin.SubmeshOMMDesc
                        {
                            arrayData      = hArray.AddrOfPinnedObject(),
                            arrayDataSize  = (uint)cache.bakedArrayData.Length,
                            descArray      = hDesc.AddrOfPinnedObject(),
                            descArrayCount = cache.bakedDescArrayCount,
                            indexBuffer    = hIdx.AddrOfPinnedObject(),
                            indexCount     = cache.bakedIndexCount,
                            indexStride    = cache.bakedIndexStride,
                            histogramFlat  = hHist.AddrOfPinnedObject(),
                            histogramCount = (uint)cache.HistogramEntryCount,
                        };
                    }
                }

                uint instanceHandle = (uint)meshRenderer.GetInstanceID();
                fixed (NativeRenderPlugin.SubmeshDesc* pDescs = submeshDescs)
                {
                    bool ok;
                    if (ommDescs != null)
                    {
                        fixed (NativeRenderPlugin.SubmeshOMMDesc* pOMM = ommDescs)
                        {
                            var desc = new NativeRenderPlugin.AddInstanceDesc
                            {
                                instanceHandle        = instanceHandle,
                                vertexBufferNativePtr = vbPtr,
                                vertexCount           = vertexCount,
                                vertexStride          = vertexStride,
                                indexBufferNativePtr  = ibPtr,
                                indexStride           = indexStride,
                                submeshDescs          = (IntPtr)pDescs,
                                submeshCount          = (uint)subMeshCount,
                                ommDescs              = (IntPtr)pOMM,
                            };
                            ok = NativeRenderPlugin.NR_AS_AddInstance(_handle, ref desc);
                        }
                    }
                    else
                    {
                        var desc = new NativeRenderPlugin.AddInstanceDesc
                        {
                            instanceHandle        = instanceHandle,
                            vertexBufferNativePtr = vbPtr,
                            vertexCount           = vertexCount,
                            vertexStride          = vertexStride,
                            indexBufferNativePtr  = ibPtr,
                            indexStride           = indexStride,
                            submeshDescs          = (IntPtr)pDescs,
                            submeshCount          = (uint)subMeshCount,
                            ommDescs              = IntPtr.Zero,
                        };
                        ok = NativeRenderPlugin.NR_AS_AddInstance(_handle, ref desc);
                    }

                    if (!ok)
                    {
                        Debug.LogError($"[NativeRayTracing] AddInstance failed for '{mesh.name}'");
                        return false;
                    }
                }
            }
            finally
            {
                foreach (var h in handles) h.Free();
            }

            // Debug.Log($"[NativeRayTracing] Added '{mesh.name}' subMeshCount={subMeshCount} verts={vertexCount} → handle={(uint)meshRenderer.GetInstanceID()}");
            return true;
        }

        /// <summary>
        /// Updates the world transform of the instance associated with <paramref name="meshRenderer"/>.
        /// </summary>
        public unsafe void SetInstanceTransform(MeshRenderer meshRenderer, Matrix4x4 objectToWorld)
        {
            if (_handle == 0 || meshRenderer == null)
            {
                Debug.LogError($"[NativeRayTracing] SetInstanceTransform failed: invalid handle or meshRenderer");
                return;
            }
            uint idx = (uint)meshRenderer.GetInstanceID();

            float* m = stackalloc float[12];
            m[0]  = objectToWorld.m00;
            m[1]  = objectToWorld.m01;
            m[2]  = objectToWorld.m02;
            m[3]  = objectToWorld.m03;
            m[4]  = objectToWorld.m10;
            m[5]  = objectToWorld.m11;
            m[6]  = objectToWorld.m12;
            m[7]  = objectToWorld.m13;
            m[8]  = objectToWorld.m20;
            m[9]  = objectToWorld.m21;
            m[10] = objectToWorld.m22;
            m[11] = objectToWorld.m23;

            NativeRenderPlugin.NR_AS_SetInstanceTransform(_handle, idx, (IntPtr)m);
        }

        /// <summary>Sets the per-instance visibility mask (8 bits, default 0xFF).</summary>
        public void SetInstanceMask(MeshRenderer meshRenderer, byte mask)
        {
            if (_handle == 0 || meshRenderer == null) return;
            NativeRenderPlugin.NR_AS_SetInstanceMask(_handle, (uint)meshRenderer.GetInstanceID(), mask);
        }

        /// <summary>
        /// Sets the custom InstanceID returned by <c>InstanceID()</c> in HLSL shaders.
        /// Call this to align <c>InstanceID()</c> with an index into a structured buffer (e.g. t_InstanceData).
        /// Triggers a TLAS rebuild on the next <see cref="BuildOrUpdate"/> call.
        /// </summary>
        public void SetInstanceID(MeshRenderer meshRenderer, uint id)
        {
            if (_handle == 0 || meshRenderer == null) return;
            // Debug.Log($"[NativeRayTracing] SetInstanceID for '{meshRenderer.name}' to {id}");

            NativeRenderPlugin.NR_AS_SetInstanceID(_handle, (uint)meshRenderer.GetInstanceID(), id);
        }

        /// <summary>
        /// Removes the instance associated with <paramref name="meshRenderer"/>.
        /// Decrements the BLAS ref-count; the BLAS GPU resources are freed after 3 frames
        /// (once no other renderer shares the same mesh).
        /// Also decrements refcounts for all textures and materials used by this instance.
        /// </summary>
        public void RemoveInstance(MeshRenderer meshRenderer)
        {
            if (_handle == 0 || meshRenderer == null) return;
            uint instanceHandle = (uint)meshRenderer.GetInstanceID();
            NativeRenderPlugin.NR_AS_RemoveInstance(_handle, instanceHandle);
            // Debug.Log($"[NativeRayTracing] Removed instance handle={instanceHandle} for '{meshRenderer.name}'");
        }
        
        public void RemoveInstance(int meshRendererId)
        {
            if (_handle == 0 ) return;
            NativeRenderPlugin.NR_AS_RemoveInstance(_handle, (uint)meshRendererId);
        }

        // ===================================================================
        // SkinnedMeshRenderer support
        //   Each SkinnedMeshRenderer gets its own dedicated BLAS (not shared
        //   with other instances) that is rebuilt every frame to reflect the
        //   current GPU skinning result.
        //
        //   Usage (per-frame):
        //     1. AddInstance(smr)            — once, on enable
        //     2. UpdateSkinnedInstance(smr)  — every frame, before BuildOrUpdate
        //     3. RemoveInstance(smr)         — once, on disable
        // ===================================================================
 
        /// <summary>
        /// Registers a <see cref="SkinnedMeshRenderer"/> as a dynamic BLAS instance.
        /// The vertex buffer is obtained via <c>GetVertexBuffer()</c> each frame;
        /// the index buffer is taken from <c>sharedMesh</c> (static).
        /// </summary>
        public unsafe bool AddInstance(SkinnedMeshRenderer smr)
        {
            if (_handle == 0 || smr == null) return false;

            Mesh mesh = smr.sharedMesh;
            if (mesh == null)
            {
                Debug.LogWarning($"[NativeRayTracing] SkinnedMeshRenderer '{smr.name}' has no sharedMesh — skipping");
                return false;
            }

            // Ensure the vertex buffer is accessible as a raw GPU resource.
            smr.vertexBufferTarget |= GraphicsBuffer.Target.Raw;

            mesh.UploadMeshData(false);

            // Current-frame skinned vertex buffer
            // Note: After setting vertexBufferTarget, Unity may need to recreate the buffer.
            GraphicsBuffer skinnedVB = smr.GetVertexBuffer();
            if (skinnedVB == null)
            {
                // Try to force buffer creation
                smr.forceMatrixRecalculationPerRender = true;
                skinnedVB = smr.GetVertexBuffer();

                if (skinnedVB == null)
                {
                    Debug.LogWarning($"[NativeRayTracing] SkinnedMeshRenderer '{smr.name}': GetVertexBuffer() returned null. " +
                                   "This can happen when toggling the renderer. Will retry next frame.");
                    return false;
                }
            }

            IntPtr vbPtr = skinnedVB.GetNativeBufferPtr();
            IntPtr ibPtr = mesh.GetNativeIndexBufferPtr();
            if (vbPtr == IntPtr.Zero || ibPtr == IntPtr.Zero)
            {
                Debug.LogError($"[NativeRayTracing] SkinnedMeshRenderer '{smr.name}': failed to get GPU buffer pointers");
                return false;
            }

            uint vertexCount  = (uint)mesh.vertexCount;
            uint vertexStride = (uint)mesh.GetVertexBufferStride(0);
            uint indexStride  = mesh.indexFormat == IndexFormat.UInt16 ? 2u : 4u;
            int  subMeshCount = mesh.subMeshCount;

            var submeshDescs = new NativeRenderPlugin.SubmeshDesc[subMeshCount];
            for (int s = 0; s < subMeshCount; s++)
            {
                SubMeshDescriptor sub = mesh.GetSubMesh(s);
                submeshDescs[s] = new NativeRenderPlugin.SubmeshDesc
                {
                    indexCount      = (uint)sub.indexCount,
                    indexByteOffset = (uint)sub.indexStart * indexStride,
                };
            }

            uint instanceHandle = (uint)smr.GetInstanceID();
            fixed (NativeRenderPlugin.SubmeshDesc* pDescs = submeshDescs)
            {
                var desc = new NativeRenderPlugin.AddInstanceDesc
                {
                    instanceHandle        = instanceHandle,
                    vertexBufferNativePtr = vbPtr,
                    vertexCount           = vertexCount,
                    vertexStride          = vertexStride,
                    indexBufferNativePtr  = ibPtr,
                    indexStride           = indexStride,
                    submeshDescs          = (IntPtr)pDescs,
                    submeshCount          = (uint)subMeshCount,
                    ommDescs              = IntPtr.Zero,
                    isDynamic             = 1,
                };
                if (!NativeRenderPlugin.NR_AS_AddInstance(_handle, ref desc))
                {
                    Debug.LogError($"[NativeRayTracing] AddInstance (skinned) failed for '{smr.name}'");
                    return false;
                }
            }
            return true;
        }

        /// <summary>
        /// Updates the GPU vertex buffer for a skinned instance to the current frame's
        /// skinning result. Call every frame before <see cref="BuildOrUpdate"/>.
        /// </summary>
        public void UpdateSkinnedInstance(SkinnedMeshRenderer smr)
        {
            if (_handle == 0 || smr == null) return;

            GraphicsBuffer skinnedVB = smr.GetVertexBuffer();
            if (skinnedVB == null) return;

            IntPtr vbPtr = skinnedVB.GetNativeBufferPtr();
            if (vbPtr == IntPtr.Zero) return;

            Mesh mesh = smr.sharedMesh;
            uint vertexCount  = mesh != null ? (uint)mesh.vertexCount  : 0u;
            uint vertexStride = mesh != null ? (uint)mesh.GetVertexBufferStride(0) : 0u;

            NativeRenderPlugin.NR_AS_UpdateDynamicVertexBuffer(
                _handle, (uint)smr.GetInstanceID(), vbPtr, vertexCount, vertexStride);
        }

        /// <summary>Removes the skinned instance associated with <paramref name="smr"/>.</summary>
        public void RemoveInstance(SkinnedMeshRenderer smr)
        {
            if (_handle == 0 || smr == null) return;
            NativeRenderPlugin.NR_AS_RemoveInstance(_handle, (uint)smr.GetInstanceID());
        }

        /// <summary>Updates the world transform for a skinned instance.</summary>
        public unsafe void SetInstanceTransform(SkinnedMeshRenderer smr, Matrix4x4 objectToWorld)
        {
            if (_handle == 0 || smr == null) return;
            float* m = stackalloc float[12];
            m[0]  = objectToWorld.m00; m[1]  = objectToWorld.m01; m[2]  = objectToWorld.m02; m[3]  = objectToWorld.m03;
            m[4]  = objectToWorld.m10; m[5]  = objectToWorld.m11; m[6]  = objectToWorld.m12; m[7]  = objectToWorld.m13;
            m[8]  = objectToWorld.m20; m[9]  = objectToWorld.m21; m[10] = objectToWorld.m22; m[11] = objectToWorld.m23;
            NativeRenderPlugin.NR_AS_SetInstanceTransform(_handle, (uint)smr.GetInstanceID(), (IntPtr)m);
        }

        /// <summary>Sets the visibility mask for a skinned instance (default 0xFF).</summary>
        public void SetInstanceMask(SkinnedMeshRenderer smr, byte mask)
        {
            if (_handle == 0 || smr == null) return;
            NativeRenderPlugin.NR_AS_SetInstanceMask(_handle, (uint)smr.GetInstanceID(), mask);
        }

        /// <summary>Sets the custom InstanceID for a skinned instance.</summary>
        public void SetInstanceID(SkinnedMeshRenderer smr, uint id)
        {
            if (_handle == 0 || smr == null) return;
            NativeRenderPlugin.NR_AS_SetInstanceID(_handle, (uint)smr.GetInstanceID(), id);
        }
    }
}