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
                _buildEventData = new NativeArray<NativeRenderPlugin.AS_BuildEventData>(1, Allocator.Persistent);
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
        public unsafe bool AddInstance(MeshRenderer meshRenderer, OMMCache[] ommCaches = null)
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

            uint posOff  = mesh.HasVertexAttribute(VertexAttribute.Position) ? (uint)mesh.GetVertexAttributeOffset(VertexAttribute.Position) : 0u;
            uint normOff = mesh.HasVertexAttribute(VertexAttribute.Normal) ? (uint)mesh.GetVertexAttributeOffset(VertexAttribute.Normal) : 0xFFFFFFFFu;
            uint uvOff   = mesh.HasVertexAttribute(VertexAttribute.TexCoord0) ? (uint)mesh.GetVertexAttributeOffset(VertexAttribute.TexCoord0) : 0xFFFFFFFFu;
            uint tanOff  = mesh.HasVertexAttribute(VertexAttribute.Tangent) ? (uint)mesh.GetVertexAttributeOffset(VertexAttribute.Tangent) : 0xFFFFFFFFu;

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
                    materialIndex   = 0, // material indexing managed by C# scene GPU data
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
                            ok = NativeRenderPlugin.NR_AS_AddInstance(
                                _handle, instanceHandle, vbPtr, vertexCount, vertexStride,
                                posOff, normOff, uvOff, tanOff,
                                ibPtr, indexStride,
                                (IntPtr)pDescs, (uint)subMeshCount, (IntPtr)pOMM);
                        }
                    }
                    else
                    {
                        ok = NativeRenderPlugin.NR_AS_AddInstance(
                            _handle, instanceHandle, vbPtr, vertexCount, vertexStride,
                            posOff, normOff, uvOff, tanOff,
                            ibPtr, indexStride,
                            (IntPtr)pDescs, (uint)subMeshCount, IntPtr.Zero);
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
            if (_handle == 0 || meshRenderer == null) return;
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
    }
}