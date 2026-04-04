using System;
using System.Runtime.InteropServices;
using Nri;
using PathTracing;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace DLRR
{
    public class DlrrDenoiser : IDisposable
    {
        [DllImport("RenderingPlugin")]
        private static extern int CreateDLRRInstance();

        [DllImport("RenderingPlugin")]
        private static extern void DestroyDLRRInstance(int id);

        private readonly int _instanceId;
        private NativeArray<DlrrFrameData> _buffer;
        private const int BufferCount = 3;
        private readonly string _cameraName;

        /// <summary>
        /// Per-frame camera data filled by PathTracingFeature from CameraFrameState.
        /// DLRRDenoiser does not depend on CameraFrameState directly.
        /// </summary>
        public struct DlrrFrameInput
        {
            public Matrix4x4 worldToView;
            public Matrix4x4 viewToClip;
            public float2    viewportJitter;
            public int2      renderResolution;
            public uint      frameIndex;
            public ushort    outputWidth;
            public ushort    outputHeight;
        }

        /// <summary>
        /// DLSS-RR textures packed by PathTracingFeature and passed each frame.
        /// </summary>
        public struct DlrrResources
        {
            public NriTextureResource input;           // Composed
            public NriTextureResource output;          // DlssOutput
            public NriTextureResource mv;              // IN_MV
            public NriTextureResource depth;           // IN_VIEWZ
            public NriTextureResource diffAlbedo;      // RRGuide_DiffAlbedo
            public NriTextureResource specAlbedo;      // RRGuide_SpecAlbedo
            public NriTextureResource normalRoughness; // RRGuide_Normal_Roughness
            public NriTextureResource specHitDistance; // RRGuide_SpecHitDistance
        }

        public DlrrDenoiser(string camName)
        {
            _instanceId = CreateDLRRInstance();
            _cameraName = camName;
            _buffer = new NativeArray<DlrrFrameData>(BufferCount, Allocator.Persistent);
                Debug.Log($"[DLSS RR] Created Denoiser Instance {_instanceId} for Camera {_cameraName}");
        }

        private DlrrFrameData GetData(DlrrFrameInput fi, DlrrResources res, float resolutionScale, UpscalerMode upscalerMode)
        {
            ushort rectW = (ushort)(fi.renderResolution.x * resolutionScale + 0.5f);
            ushort rectH = (ushort)(fi.renderResolution.y * resolutionScale + 0.5f);

            var data = new DlrrFrameData
            {
                inputTex = res.input.NriPtr,
                outputTex = res.output.NriPtr,
                mvTex = res.mv.NriPtr,
                depthTex = res.depth.NriPtr,
                diffuseAlbedoTex = res.diffAlbedo.NriPtr,
                specularAlbedoTex = res.specAlbedo.NriPtr,
                normalRoughnessTex = res.normalRoughness.NriPtr,
                specularMvOrHitTex = res.specHitDistance.NriPtr,
                worldToViewMatrix = fi.worldToView,
                viewToClipMatrix = fi.viewToClip,
                outputWidth = fi.outputWidth,
                outputHeight = fi.outputHeight,
                currentWidth = rectW,
                currentHeight = rectH,
                upscalerMode = upscalerMode,
                cameraJitter = fi.viewportJitter,
                instanceId = _instanceId
            };

            return data;
        }

        public IntPtr GetInteropDataPtr(DlrrFrameInput fi, DlrrResources res, float resolutionScale, UpscalerMode upscalerMode)
        {
            var index = (int)(fi.frameIndex % BufferCount);
            _buffer[index] = GetData(fi, res, resolutionScale, upscalerMode);
            unsafe
            {
                return (IntPtr)_buffer.GetUnsafePtr() + index * sizeof(DlrrFrameData);
            }
        }

        public void Dispose()
        {
            if (_buffer.IsCreated)
            {
                _buffer.Dispose();
            }

            DestroyDLRRInstance(_instanceId);
            
            
            Debug.Log($"[DLSS RR] Destroyed Denoiser Instance {_instanceId} for Camera {_cameraName} - Dispose Complete");
        }
    }
}