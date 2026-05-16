using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Nri;
using PathTracing;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace Nrd
{
    // -----------------------------------------------------------------------
    // Abstract base – holds all common infrastructure shared by the three
    // typed denoiser subclasses (Sigma / Reblur / Relax).
    // -----------------------------------------------------------------------
    public abstract class NrdDenoiser : IDisposable
    {
        [DllImport("Denoiser")]
        private static extern int CreateDenoiserInstance(IntPtr denoisers, int count);

        [DllImport("Denoiser")]
        private static extern void DestroyDenoiserInstance(int id);

        [DllImport("Denoiser")]
        private static extern void UpdateDenoiserResources(int instanceId, IntPtr resources, int count);

        private NativeArray<NrdResourceInput> _resourceCache;

        protected readonly int    _nrdInstanceId;
        protected readonly string _cameraName;

        private       NativeArray<NrdFrameData> _buffer;
        private const int                       BufferCount = 3;

        // -----------------------------------------------------------------------
        // Fields shared by all denoiser types – filled from camera/frame state.
        // -----------------------------------------------------------------------
        public struct CommonFrameInput
        {
            public Matrix4x4 worldToView;
            public Matrix4x4 prevWorldToView;
            public Matrix4x4 viewToClip;
            public Matrix4x4 prevViewToClip;
            public float2    viewportJitter;
            public float2    prevViewportJitter;
            public float     resolutionScale;
            public float     prevResolutionScale;
            public int2      renderResolution;
            public uint      frameIndex;
            public bool      flipMotionVectors;
            public bool      enableValidation;
            public bool      isHistoryConfidenceAvailable;
            public float     splitScreen;
            public float     denoisingRange;
            public float     strandMaterialID;
        }

        // -----------------------------------------------------------------------
        protected NrdDenoiser(string camName, NrdDenoiserDesc desc)
        {
            _cameraName = camName;
            var descs = new[] { desc };
            unsafe
            {
                fixed (NrdDenoiserDesc* p = descs)
                {
                    _nrdInstanceId = CreateDenoiserInstance((IntPtr)p, 1);
                }
            }

            _buffer = new NativeArray<NrdFrameData>(BufferCount, Allocator.Persistent);
            Debug.Log($"[NRD] Created Denoiser Instance {_nrdInstanceId} | camera={_cameraName} | type={desc.denoiser} (id={desc.identifier}, enum={(uint)desc.denoiser})");
        }

        // -----------------------------------------------------------------------
        public unsafe void UpdateResources(params (ResourceType type, NriTextureResource resource)[] entries)
        {
            if (entries == null || entries.Length == 0) return;

            if (!_resourceCache.IsCreated || _resourceCache.Length < entries.Length)
            {
                if (_resourceCache.IsCreated) _resourceCache.Dispose();
                _resourceCache = new NativeArray<NrdResourceInput>(entries.Length, Allocator.Persistent);
            }

            int idx = 0;
            var ptr = (NrdResourceInput*)_resourceCache.GetUnsafePtr();

            foreach (var (type, resource) in entries)
            {
                if (resource == null) continue;
                ptr[idx++] = new NrdResourceInput { type = type, texture = resource.NriPtr, state = resource.ResourceState };
            }

            UpdateDenoiserResources(_nrdInstanceId, (IntPtr)ptr, idx);
        }

        // -----------------------------------------------------------------------
        protected unsafe void FillCommonSettings(ref NrdFrameData data, CommonFrameInput fi)
        {
            data.commonSettings.viewToClipMatrix      = fi.viewToClip;
            data.commonSettings.viewToClipMatrixPrev  = fi.prevViewToClip;
            data.commonSettings.worldToViewMatrix     = fi.worldToView;
            data.commonSettings.worldToViewMatrixPrev = fi.prevWorldToView;

            data.commonSettings.cameraJitter     = fi.viewportJitter;
            data.commonSettings.cameraJitterPrev = fi.prevViewportJitter;

            ushort rectW     = (ushort)(fi.renderResolution.x * fi.resolutionScale + 0.5f);
            ushort rectH     = (ushort)(fi.renderResolution.y * fi.resolutionScale + 0.5f);
            ushort prevRectW = (ushort)(fi.renderResolution.x * fi.prevResolutionScale + 0.5f);
            ushort prevRectH = (ushort)(fi.renderResolution.y * fi.prevResolutionScale + 0.5f);

            data.commonSettings.resourceSize[0]     = (ushort)fi.renderResolution.x;
            data.commonSettings.resourceSize[1]     = (ushort)fi.renderResolution.y;
            data.commonSettings.rectSize[0]         = rectW;
            data.commonSettings.rectSize[1]         = rectH;
            data.commonSettings.resourceSizePrev[0] = (ushort)fi.renderResolution.x;
            data.commonSettings.resourceSizePrev[1] = (ushort)fi.renderResolution.y;
            data.commonSettings.rectSizePrev[0]     = prevRectW;
            data.commonSettings.rectSizePrev[1]     = prevRectH;

            data.commonSettings.motionVectorScale          = new float3(1.0f / rectW, 1.0f / rectH, fi.flipMotionVectors ? 1.0f : -1.0f);
            data.commonSettings.isMotionVectorInWorldSpace = false;
            data.commonSettings.accumulationMode           = AccumulationMode.CONTINUE;
            data.commonSettings.frameIndex                 = fi.frameIndex;

            data.instanceId = _nrdInstanceId;
            data.width      = (ushort)fi.renderResolution.x;
            data.height     = (ushort)fi.renderResolution.y;

            data.commonSettings.denoisingRange                 = fi.denoisingRange;
            data.commonSettings.splitScreen                    = fi.splitScreen;
            data.commonSettings.strandMaterialID               = fi.strandMaterialID == 0f ? 999.0f : fi.strandMaterialID;
            data.commonSettings.enableValidation               = fi.enableValidation;
            data.commonSettings.disocclusionThresholdAlternate = 0.1f;
            data.commonSettings.isHistoryConfidenceAvailable   = fi.isHistoryConfidenceAvailable;
        }

        protected unsafe IntPtr StoreAndGetPtr(NrdFrameData data, uint frameIndex)
        {
            var index = (int)(frameIndex % BufferCount);
            _buffer[index] = data;
            return (IntPtr)_buffer.GetUnsafePtr() + index * sizeof(NrdFrameData);
        }

        public void Dispose()
        {
            if (_buffer.IsCreated) _buffer.Dispose();
            if (_resourceCache.IsCreated) _resourceCache.Dispose();
            DestroyDenoiserInstance(_nrdInstanceId);
            Debug.Log($"[NRD] Destroyed Denoiser Instance {_nrdInstanceId} for Camera {_cameraName}");
        }
    }



}