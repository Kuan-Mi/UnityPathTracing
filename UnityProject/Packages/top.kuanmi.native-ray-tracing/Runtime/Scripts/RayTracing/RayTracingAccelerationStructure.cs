using System;
using System.Collections.Concurrent;
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
        
        
        private List<SkinnedMeshRenderer> skinnedMeshRenderers = new List<SkinnedMeshRenderer>();
        private List<MeshRenderer> meshRenderers = new List<MeshRenderer>();

        // Thread-safe collections for pending SkinnedMeshRenderers
        // Key: SkinnedMeshRenderer.GetInstanceID()
        // (vertex buffer not ready on first frame, will retry on subsequent frames)
        private readonly ConcurrentDictionary<int, SkinnedMeshRenderer> _pendingSkinnedInstances = new();
        private readonly ConcurrentDictionary<int, PendingSkinnedSetup> _pendingSetups           = new();
        private readonly ConcurrentDictionary<int, int>                 _pendingRetryCount       = new();


        private readonly Dictionary<uint, IntPtr> skinedPtr     = new();
        private readonly Dictionary<uint, IntPtr> skinedIPtr     = new();
        private const    int                      MaxRetryCount = 60; // Max 60 frames (~1 second at 60fps)

        private class PendingSkinnedSetup
        {
            public Matrix4x4? transform;
            public byte?      mask;
            public uint?      instanceID;
        }

        /// <summary>
        /// Issues a GPU command to build or update the acceleration structure.
        /// Must be called before the ray trace Dispatch each frame.
        /// Also retries adding any pending SkinnedMeshRenderers that failed on previous frames.
        /// NOTE: This method is called during CommandBuffer recording (main thread), not execution.
        /// RetryPendingSkinnedInstances runs on the main thread, so no thread safety issues.
        /// </summary>
        public void BuildOrUpdate(CommandBuffer cmd)
        {
            if (_handle == 0) return;

            // Retry adding pending SkinnedMeshRenderers (runs on main thread during cmd recording)
            RetryPendingSkinnedInstances();

            foreach (var smr in skinnedMeshRenderers)
            {
                UpdateSkinnedInstance(smr); // 强制更新buffer指针
            }
            // foreach (var mr in meshRenderers)
            // {
            //     UpdateInstance(mr); // 强制更新buffer指针
            // }

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

            OMMCache[] ommCaches    = null;
            var        nativeTarget = meshFilter.GetComponent<NativeRayTracingTarget>();
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
            
            meshRenderers.Add(meshRenderer);

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
            meshRenderers.Remove(meshRenderer);
            // Debug.Log($"[NativeRayTracing] Removed instance handle={instanceHandle} for '{meshRenderer.name}'");
        }

        public void RemoveInstance(int meshRendererId)
        {
            if (_handle == 0) return;
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
        /// If the vertex buffer is not ready (e.g., first frame), the instance is queued
        /// and will be retried automatically on subsequent frames.
        /// Returns true if successfully added OR queued for retry (allowing caller to proceed with setup).
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
            int            id        = smr.GetInstanceID();
            if (skinnedVB == null)
            {
                // Try to force buffer creation
                smr.forceMatrixRecalculationPerRender = true;
                skinnedVB                             = smr.GetVertexBuffer();

                if (skinnedVB == null)
                {
                    // Queue for retry on next frame
                    if (_pendingSkinnedInstances.TryAdd(id, smr))
                    {
                        _pendingSetups.TryAdd(id, new PendingSkinnedSetup());
                        _pendingRetryCount.TryAdd(id, 0);
                        Debug.Log($"[NativeRayTracing] SkinnedMeshRenderer '{smr.name}': vertex buffer not ready, queued for retry.");
                    }

                    // Return true so caller can proceed with buffer allocation and setup
                    return true;
                }
            }

            IntPtr vbPtr = skinnedVB.GetNativeBufferPtr();
            IntPtr ibPtr = mesh.GetNativeIndexBufferPtr();
            if (vbPtr == IntPtr.Zero || ibPtr == IntPtr.Zero)
            {
                Debug.LogError($"[NativeRayTracing] SkinnedMeshRenderer '{smr.name}': failed to get GPU buffer pointers");
                return false;
            }

            vbPtr = mesh.GetVertexBuffer(0).GetNativeBufferPtr();

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

            // Successfully added, remove from pending collections and apply cached setup
            _pendingSkinnedInstances.TryRemove(id, out _);
            _pendingRetryCount.TryRemove(id, out _);

            // Retrieve and remove cached setup BEFORE calling Set methods
            // (otherwise Set methods will see it's still pending and cache again)
            if (_pendingSetups.TryRemove(id, out var setup))
            {
                // Apply the cached setup
                if (setup.transform.HasValue)
                    SetInstanceTransform(smr, setup.transform.Value);
                if (setup.mask.HasValue)
                    SetInstanceMask(smr, setup.mask.Value);
                if (setup.instanceID.HasValue)
                    SetInstanceID(smr, setup.instanceID.Value);
            }
            
            if(!skinnedMeshRenderers.Contains(smr))
                skinnedMeshRenderers.Add(smr);
            return true;
        }

        /// <summary>
        /// Checks if a SkinnedMeshRenderer is queued for retry (vertex buffer not ready yet).
        /// </summary>
        public bool IsSkinnedInstancePending(SkinnedMeshRenderer smr)
        {
            if (smr == null) return false;
            return _pendingSkinnedInstances.ContainsKey(smr.GetInstanceID());
        }

        /// <summary>
        /// Updates the GPU vertex buffer for a skinned instance to the current frame's
        /// skinning result. Call every frame before <see cref="BuildOrUpdate"/>.
        /// </summary>
        public void UpdateSkinnedInstance(SkinnedMeshRenderer smr)
        {
            if (_handle == 0 || smr == null) return;

            GraphicsBuffer skinnedVB = smr.GetVertexBuffer();
            skinnedVB.name = $"{smr.name}_SkinnedVB";
            if (skinnedVB == null)
            {
                Debug.LogError($"[NativeRayTracing] UpdateSkinnedInstance: vertex buffer not available for '{smr.name}'");
                return;
            }

            if (!skinnedVB.IsValid())
            {
                Debug.LogError($"[NativeRayTracing] UpdateSkinnedInstance: vertex buffer is not valid for '{smr.name}'");
                return;
            }

            IntPtr vbPtr = skinnedVB.GetNativeBufferPtr();
            if (vbPtr == IntPtr.Zero)
            {
                Debug.LogError($"[NativeRayTracing] UpdateSkinnedInstance: failed to get vertex buffer pointer for '{smr.name}'");
                return;
            }

            var indexBuffer = smr.sharedMesh.GetIndexBuffer();
            indexBuffer.name = $"{smr.sharedMesh.name}_IndexBuffer";

            vbPtr = smr.sharedMesh.GetVertexBuffer(0).GetNativeBufferPtr();
            
            var ibPtr          = smr.sharedMesh != null ? indexBuffer.GetNativeBufferPtr() : IntPtr.Zero;
            if (ibPtr == IntPtr.Zero)            {
                Debug.LogError($"[NativeRayTracing] UpdateSkinnedInstance: failed to get index buffer pointer for '{smr.name}'");
                return;
            }


            Mesh mesh         = smr.sharedMesh;
            uint vertexCount  = mesh != null ? (uint)mesh.vertexCount : 0u;
            uint vertexStride = mesh != null ? (uint)mesh.GetVertexBufferStride(0) : 0u;

            if (skinedIPtr.TryGetValue((uint)smr.GetInstanceID(), out var existingIBPtr2))
            {
                if(existingIBPtr2 != ibPtr)
                {
                    Debug.LogError($"[NativeRayTracing] UpdateSkinnedInstance: index buffer pointer changed for '{smr.name}' (was {existingIBPtr2}, now {ibPtr})");
                }
            }


            if (skinedPtr.TryGetValue((uint)smr.GetInstanceID(), out var existingPtr) && existingPtr == vbPtr)
            {
            }
            else
            { 
                Debug.Log($"Old VB Ptr: {(existingPtr != IntPtr.Zero ? existingPtr.ToString("X") : "null")}, New VB Ptr: {vbPtr.ToString("X")} for '{smr.name}'");
                
                // 这里验证了 IB是不会改变的
                Debug.Log($"Old IB Ptr: {(existingIBPtr2 != IntPtr.Zero ? existingIBPtr2.ToString("X") : "null")}, New IB Ptr: {ibPtr.ToString("X")} for '{smr.name}'");
                
                skinedPtr[(uint)smr.GetInstanceID()] = vbPtr;
                skinedIPtr[(uint)smr.GetInstanceID()] = ibPtr;
                NativeRenderPlugin.NR_AS_UpdateDynamicVertexBuffer(
                    _handle, (uint)smr.GetInstanceID(), vbPtr, vertexCount, vertexStride);
            }
        }

        /// <summary>Removes the skinned instance associated with <paramref name="smr"/>.</summary>
        public void RemoveInstance(SkinnedMeshRenderer smr)
        {
            if (_handle == 0 || smr == null) return;
            int id = smr.GetInstanceID();
            NativeRenderPlugin.NR_AS_RemoveInstance(_handle, (uint)id);

            // Also remove from pending collections if it was queued
            _pendingSkinnedInstances.TryRemove(id, out _);
            _pendingSetups.TryRemove(id, out _);
            _pendingRetryCount.TryRemove(id, out _);

            skinnedMeshRenderers.Remove(smr);
        }

        /// <summary>Updates the world transform for a skinned instance.</summary>
        public unsafe void SetInstanceTransform(SkinnedMeshRenderer smr, Matrix4x4 objectToWorld)
        {
            if (_handle == 0 || smr == null) return;

            int id = smr.GetInstanceID();
            // If instance is pending, cache the transform for later
            if (_pendingSetups.TryGetValue(id, out var setup))
            {
                setup.transform = objectToWorld;
                return;
            }

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
            NativeRenderPlugin.NR_AS_SetInstanceTransform(_handle, (uint)id, (IntPtr)m);
        }

        /// <summary>Sets the visibility mask for a skinned instance (default 0xFF).</summary>
        public void SetInstanceMask(SkinnedMeshRenderer smr, byte mask)
        {
            if (_handle == 0 || smr == null) return;

            int id = smr.GetInstanceID();
            // If instance is pending, cache the mask for later
            if (_pendingSetups.TryGetValue(id, out var setup))
            {
                setup.mask = mask;
                return;
            }

            NativeRenderPlugin.NR_AS_SetInstanceMask(_handle, (uint)id, mask);
        }

        /// <summary>Sets the custom InstanceID for a skinned instance.</summary>
        public void SetInstanceID(SkinnedMeshRenderer smr, uint id)
        {
            if (_handle == 0 || smr == null) return;

            int instanceId = smr.GetInstanceID();
            // If instance is pending, cache the instanceID for later
            if (_pendingSetups.TryGetValue(instanceId, out var setup))
            {
                setup.instanceID = id;
                return;
            }

            NativeRenderPlugin.NR_AS_SetInstanceID(_handle, (uint)instanceId, id);
        }

        /// <summary>
        /// Retries adding SkinnedMeshRenderers that failed on previous frames
        /// (typically because their vertex buffer wasn't ready yet).
        /// Called automatically by <see cref="BuildOrUpdate"/>.
        /// </summary>
        private void RetryPendingSkinnedInstances()
        {
            if (_pendingSkinnedInstances.IsEmpty) return;

            // Iterate over pending instances (thread-safe snapshot)
            foreach (var kvp in _pendingSkinnedInstances)
            {
                int                 id  = kvp.Key;
                SkinnedMeshRenderer smr = kvp.Value;

                // Skip if the renderer was destroyed
                if (smr == null)
                {
                    _pendingSkinnedInstances.TryRemove(id, out _);
                    _pendingSetups.TryRemove(id, out _);
                    _pendingRetryCount.TryRemove(id, out _);
                    continue;
                }

                // Check retry count
                int retryCount = _pendingRetryCount.GetOrAdd(id, 0);
                if (retryCount >= MaxRetryCount)
                {
                    Debug.LogWarning($"[NativeRayTracing] SkinnedMeshRenderer '{smr.name}' failed to add after {MaxRetryCount} retries. " +
                                     $"Vertex buffer may not be available. Removing from retry queue.");
                    _pendingSkinnedInstances.TryRemove(id, out _);
                    _pendingSetups.TryRemove(id, out _);
                    _pendingRetryCount.TryRemove(id, out _);
                    continue;
                }

                // Increment retry count
                _pendingRetryCount[id] = retryCount + 1;

                Debug.Log($"[NativeRayTracing] Retrying SkinnedMeshRenderer '{smr.name}' (retry {retryCount + 1}/{MaxRetryCount})");
                // Try adding again (AddInstance will remove from pending collections if successful)
                AddInstance(smr);
            }
        }
    }
}