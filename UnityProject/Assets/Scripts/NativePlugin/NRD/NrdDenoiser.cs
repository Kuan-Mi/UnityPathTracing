using System;
using System.Runtime.InteropServices;
using Nri;
using PathTracing;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace Nrd
{
    public class NrdDenoiser : IDisposable
    {
        [DllImport("Denoiser")]
        private static extern int CreateDenoiserInstance(IntPtr denoisers, int count);

        [DllImport("Denoiser")]
        private static extern void DestroyDenoiserInstance(int id);

        [DllImport("Denoiser")]
        private static extern void UpdateDenoiserResources(int instanceId, IntPtr resources, int count);

        private NativeArray<NrdResourceInput> _resourceCache;

        private readonly int    _nrdInstanceId;
        private readonly string _cameraName;

        // Denoiser list captured at construction, used to populate each frame's entries[].
        private readonly NrdDenoiserDesc[] _denoisers;

        private       NativeArray<NrdFrameData> _buffer;
        private const int                       BufferCount = 3;

        // private readonly PathTracingSetting _setting;

        /// <summary>
        /// Per-frame camera data filled by PathTracingFeature from CameraFrameState.
        /// NRDDenoiser does not depend on CameraFrameState directly.
        /// </summary>
        public struct NrdFrameInput
        {
            public Matrix4x4        worldToView;
            public Matrix4x4        prevWorldToView;
            public Matrix4x4        viewToClip;
            public Matrix4x4        prevViewToClip;
            public float2           viewportJitter;
            public float2           prevViewportJitter;
            public float            resolutionScale;
            public float            prevResolutionScale;
            public int2             renderResolution;
            public uint             frameIndex;
            public float3           lightDirection;
            public CheckerboardMode checkerboardMode;

        }



        public NrdDenoiser(string camName, NrdDenoiserDesc[] denoisers)
        {
            if (denoisers == null || denoisers.Length == 0)
                throw new ArgumentException("At least one denoiser must be specified.", nameof(denoisers));
            if (denoisers.Length > NrdLayout.MaxDenoisersPerInstance)
                throw new ArgumentException(
                    $"denoisers.Length={denoisers.Length} exceeds max {NrdLayout.MaxDenoisersPerInstance}",
                    nameof(denoisers));

            // _setting    = setting;
            _cameraName = camName;
            _denoisers  = (NrdDenoiserDesc[])denoisers.Clone();

            unsafe
            {
                fixed (NrdDenoiserDesc* p = _denoisers)
                {
                    _nrdInstanceId = CreateDenoiserInstance((IntPtr)p, _denoisers.Length);
                }
            }

            _buffer = new NativeArray<NrdFrameData>(BufferCount, Allocator.Persistent);
            Debug.Log($"[NRD] Created Denoiser Instance {_nrdInstanceId} for Camera {_cameraName} with {_denoisers.Length} denoiser(s)");
        }

        /// <summary>
        /// Default factory matching legacy behaviour: SIGMA_SHADOW (id 0) + REBLUR_DIFFUSE_SPECULAR (id 1).
        /// </summary>
        public static NrdDenoiser CreateDefault(string camName)
        {
            var descs = new[]
            {
                new NrdDenoiserDesc(0, Denoiser.SIGMA_SHADOW),
                new NrdDenoiserDesc(1, Denoiser.REBLUR_DIFFUSE_SPECULAR),
            };
            return new NrdDenoiser(camName, descs);
        }

        /// <summary>
        /// Pushes NRD texture bindings to the C++ denoiser.
        /// Called by PathTracingFeature whenever textures are reallocated.
        /// Each entry is a (ResourceType, NriTextureResource) pair; null textures are skipped.
        /// </summary>
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
            Debug.Log($"[NRD] Updated Resources for Denoiser Instance {_nrdInstanceId} with {idx} resources.");
        }

        private unsafe NrdFrameData GetData(NrdFrameInput fi)
        {
            NrdFrameData data = NrdFrameData._default;

            // --- 矩阵赋值 ---
            data.commonSettings.viewToClipMatrix      = fi.viewToClip;
            data.commonSettings.viewToClipMatrixPrev  = fi.prevViewToClip;
            data.commonSettings.worldToViewMatrix     = fi.worldToView;
            data.commonSettings.worldToViewMatrixPrev = fi.prevWorldToView;

            // --- Jitter ---
            data.commonSettings.cameraJitter     =  fi.viewportJitter;
            data.commonSettings.cameraJitterPrev =  fi.prevViewportJitter;

            // --- 分辨率 ---
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

            data.commonSettings.motionVectorScale          = new float3(1.0f / rectW, 1.0f / rectH, -1.0f);
            data.commonSettings.isMotionVectorInWorldSpace = false;
            data.commonSettings.accumulationMode           = AccumulationMode.CONTINUE;
            data.commonSettings.frameIndex                 = fi.frameIndex;

            data.instanceId = _nrdInstanceId;
            data.width      = (ushort)fi.renderResolution.x;
            data.height     = (ushort)fi.renderResolution.y;

            // Common 设置
            data.commonSettings.denoisingRange   = 1000;
            data.commonSettings.splitScreen      = 0;
            data.commonSettings.enableValidation = true;

            // --- Per-denoiser settings (entries[]) ---
            data.denoiserCount = (uint)_denoisers.Length;
            for (int i = 0; i < _denoisers.Length; i++)
            {
                ref DenoiserSettingsEntry entry = ref NrdFrameData.GetEntry(ref data, i);
                entry.identifier = _denoisers[i].identifier;
                entry.denoiser   = _denoisers[i].denoiser;

                switch (_denoisers[i].denoiser)
                {
                    case Denoiser.SIGMA_SHADOW:
                    case Denoiser.SIGMA_SHADOW_TRANSLUCENCY:
                    {
                        var s = SigmaSettings._default;
                        s.lightDirection           = fi.lightDirection;
                        s.planeDistanceSensitivity = 0.02f;
                        s.maxStabilizedFrameNum    = 5;
                        entry.Write(s);
                        break;
                    }

                    case Denoiser.REBLUR_DIFFUSE:
                    case Denoiser.REBLUR_DIFFUSE_OCCLUSION:
                    case Denoiser.REBLUR_DIFFUSE_SH:
                    case Denoiser.REBLUR_SPECULAR:
                    case Denoiser.REBLUR_SPECULAR_OCCLUSION:
                    case Denoiser.REBLUR_SPECULAR_SH:
                    case Denoiser.REBLUR_DIFFUSE_SPECULAR:
                    case Denoiser.REBLUR_DIFFUSE_SPECULAR_OCCLUSION:
                    case Denoiser.REBLUR_DIFFUSE_SPECULAR_SH:
                    case Denoiser.REBLUR_DIFFUSE_DIRECTIONAL_OCCLUSION:
                    {
                        var s = ReblurSettings._default;
                        s.checkerboardMode       = fi.checkerboardMode;
                        s.minMaterialForDiffuse  = 0;
                        s.minMaterialForSpecular = 1;
                        entry.Write(s);
                        break;
                    }

                    case Denoiser.RELAX_DIFFUSE:
                    case Denoiser.RELAX_DIFFUSE_SH:
                    case Denoiser.RELAX_SPECULAR:
                    case Denoiser.RELAX_SPECULAR_SH:
                    case Denoiser.RELAX_DIFFUSE_SPECULAR:
                    case Denoiser.RELAX_DIFFUSE_SPECULAR_SH:
                    {
                        entry.Write(RelaxSettings._default);
                        break;
                    }

                    case Denoiser.REFERENCE:
                    {
                        entry.Write(ReferenceSettings._default);
                        break;
                    }

                    default:
                        // Zero-initialized blob (default defaults); still safe for NRD to consume.
                        break;
                }
            }

            return data;
        }

        public IntPtr GetInteropDataPtr(NrdFrameInput fi)
        {
            var index = (int)(fi.frameIndex % BufferCount);
            _buffer[index] = GetData(fi);
            unsafe
            {
                return (IntPtr)_buffer.GetUnsafePtr() + index * sizeof(NrdFrameData);
            }
        }

        public void Dispose()
        {
            if (_buffer.IsCreated) _buffer.Dispose();
            if (_resourceCache.IsCreated) _resourceCache.Dispose();
            DestroyDenoiserInstance(_nrdInstanceId);
            Debug.Log($"[NRD] Destroyed Denoiser Instance {_nrdInstanceId} for Camera {_cameraName} - Dispose Complete");
        }
    }
}