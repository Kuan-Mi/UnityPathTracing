using System;
using System.Collections.Generic;
using NativeRender;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;
using RayTracingAccelerationStructure = NativeRender.RayTracingAccelerationStructure;

namespace PathTracing
{
    /// <summary>
    /// Incremental TLAS registration for the RTXPT scene, mirroring the original RTXPT (donut)
    /// model: BLASes are persistent per mesh; a topology change only edits the TLAS instance
    /// list. <see cref="Sync"/> diffs the new <see cref="RtxptSceneLayout"/> against the live
    /// registrations — survivors keep their BLAS, changed instances are re-created, stale ones
    /// are removed by raw handle (which also covers renderers that have since been destroyed).
    ///
    /// Ordering invariant: RTXPT shaders fetch <c>t_InstanceData[InstanceIndex()]</c>
    /// (Bridge::loadSurface), so the TLAS instance order must equal the layout/instList order.
    /// The native AS reuses freed slots (LIFO), so raw slot order stops matching registration
    /// order once anything has been removed — instead every instance gets a dense
    /// SetInstanceOrderIndex in layout order and the native build emits TLAS instances sorted
    /// by it. Survivors also get SetInstanceHitGroupContribution updated, because their base
    /// offset in the flat shader table shifts when earlier geometry is added or removed.
    /// </summary>
    internal sealed class RtxptAccelRegistry : IDisposable
    {
        // Live registrations: handle → content hash of the BLAS-affecting inputs.
        private readonly Dictionary<uint, ulong> _registeredHandles = new();

        // Stable handle per (rendererId, groupIndex). Allocated from a counter so handles can
        // never collide (the previous scheme packed the renderer instance ID into 28 bits).
        private readonly Dictionary<(int rendererId, int groupIndex), uint> _handleByKey = new();
        private readonly Dictionary<uint, (int rendererId, int groupIndex)> _keyByHandle = new();
        private uint _nextHandle = 1; // 0 is never used (doubles as "no handle")

        private readonly HashSet<uint> _desiredScratch = new();
        private readonly List<uint>    _staleScratch   = new();

        // Flat per-geometry hit-group variant index (one entry per TLAS geometry, in layout
        // order), used to rebuild each ray-trace pipeline's hit-group shader table. Rebuilt only
        // on topology changes; the SBT rebuild is then issued once per pipeline.
        private NativeArray<uint> _variantIndexArray;
        private bool              _shaderTableDirty;

        /// <summary>
        /// True when the per-geometry hit-group variant layout changed since the last
        /// <see cref="MarkShaderTableClean"/>. While set, callers should rebuild the hit-group
        /// table of every ray-trace pipeline via <see cref="RebuildShaderTable"/>.
        /// </summary>
        public bool ShaderTableDirty => _shaderTableDirty;

        /// <summary>Clears the dirty flag after every pipeline's hit-group table has been rebuilt.</summary>
        public void MarkShaderTableClean() => _shaderTableDirty = false;

        /// <summary>
        /// Issues a render event to rebuild <paramref name="pipeline"/>'s hit-group table from
        /// the flat per-geometry variant array. Only does work while <see cref="ShaderTableDirty"/>
        /// is set; issue in the same CommandBuffer as the TLAS build.
        /// </summary>
        public void RebuildShaderTable(CommandBuffer cmd, RayTracePipeline pipeline)
        {
            if (!_shaderTableDirty || pipeline == null || !_variantIndexArray.IsCreated) return;
            pipeline.RebuildHitGroupTable(cmd, _variantIndexArray);
        }

        /// <summary>
        /// Diffs <paramref name="records"/> against the live TLAS registrations and brings the
        /// acceleration structure up to date. Assigns <see cref="RtxptInstanceRecord.Handle"/> on
        /// every record. Records whose native registration fails are REMOVED from the list so the
        /// GPU-buffer build sees exactly what the TLAS contains (no silent misalignment).
        /// </summary>
        public void Sync(RayTracingAccelerationStructure accel, List<RtxptInstanceRecord> records)
        {
            var desired = _desiredScratch;
            desired.Clear();

            var  variantList         = new List<uint>();
            uint runningContribution = 0; // each instance's InstanceContributionToHitGroupIndex
            uint orderIndex          = 0;

            for (int i = 0; i < records.Count; i++)
            {
                var rec = records[i];

                var key = (rec.RendererId, rec.GroupIndex);
                if (!_handleByKey.TryGetValue(key, out uint handle))
                {
                    handle              = _nextHandle++;
                    _handleByKey[key]   = handle;
                    _keyByHandle[handle] = key;
                }
                rec.Handle = handle;

                bool isRegistered = _registeredHandles.TryGetValue(handle, out ulong oldHash);
                if (isRegistered && oldHash != rec.ContentHash)
                {
                    // Sub-mesh membership or opacity flags changed → different BLAS content.
                    accel.RemoveInstance(handle);
                    _registeredHandles.Remove(handle);
                    isRegistered = false;
                }

                if (!isRegistered)
                {
                    if (accel.AddInstanceGroup(rec.Mesh, rec.Descs, handle, hitGroupContribution: runningContribution))
                    {
                        accel.SetInstanceTransform(handle, rec.MeshRenderer.transform.localToWorldMatrix);
                        _registeredHandles[handle] = rec.ContentHash;
                    }
                    else
                    {
                        Debug.LogWarning($"[RtxptAccelRegistry] AddInstanceGroup failed for '{rec.MeshRenderer.name}' group={rec.GroupIndex} — instance dropped.");
                        records.RemoveAt(i--);
                        continue;
                    }
                }
                else
                {
                    accel.SetInstanceHitGroupContribution(handle, runningContribution);
                }

                desired.Add(handle);
                accel.SetInstanceOrderIndex(handle, orderIndex++);

                for (int k = 0; k < rec.Descs.Length; k++)
                    variantList.Add(rec.HitGroupVariant);
                runningContribution += (uint)rec.Descs.Length;
            }

            // Remove registrations that are no longer desired: renderer destroyed or disabled,
            // or its group disappeared. Removal is by raw handle, so no live Unity object is
            // required.
            _staleScratch.Clear();
            foreach (var kv in _registeredHandles)
                if (!desired.Contains(kv.Key))
                    _staleScratch.Add(kv.Key);
            foreach (var h in _staleScratch)
            {
                accel.RemoveInstance(h);
                _registeredHandles.Remove(h);
                if (_keyByHandle.TryGetValue(h, out var key))
                {
                    _keyByHandle.Remove(h);
                    _handleByKey.Remove(key);
                }
            }

            // Publish the new variant array and flag pipelines for a one-time SBT rebuild.
            if (_variantIndexArray.IsCreated) _variantIndexArray.Dispose();
            _variantIndexArray = new NativeArray<uint>(variantList.Count, Allocator.Persistent);
            for (int i = 0; i < variantList.Count; i++)
                _variantIndexArray[i] = variantList[i];
            _shaderTableDirty = true;
        }

        public void Dispose()
        {
            if (_variantIndexArray.IsCreated) _variantIndexArray.Dispose();
            _registeredHandles.Clear();
            _handleByKey.Clear();
            _keyByHandle.Clear();
        }
    }
}
