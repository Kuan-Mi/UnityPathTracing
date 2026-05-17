using System;
using System.Runtime.InteropServices;
using Nri;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;

namespace NIS
{
    public class NisUpscaler : IDisposable
    {
        [DllImport("Denoiser")]
        private static extern int CreateNISInstance();

        [DllImport("Denoiser")]
        private static extern void DestroyNISInstance(int id);

        private readonly int _instanceId;
        private NativeArray<NisFrameData> _buffer;
        private const int BufferCount = 3;
        private readonly string _cameraName;

        /// <summary>Per-frame camera data.</summary>
        public struct NisFrameInput
        {
            public ushort outputWidth;
            public ushort outputHeight;
            public ushort currentWidth;
            public ushort currentHeight;
            public uint   frameIndex;
        }

        /// <summary>NIS textures passed each frame.</summary>
        public struct NisResources
        {
            public NriTextureResource input;   // color input (DlssOutput or TaaHistory)
            public NriTextureResource output;  // sharpened output (UAV) → PreFinal
        }

        /// <summary>Per-dispatch settings.</summary>
        public struct NisSettings
        {
            public float sharpness; // [0, 1]; default 0.5
        }

        public NisUpscaler(string camName)
        {
            _instanceId = CreateNISInstance();
            _cameraName = camName;
            _buffer     = new NativeArray<NisFrameData>(BufferCount, Allocator.Persistent);
            Debug.Log($"[NIS] Created NIS Instance {_instanceId} for Camera {_cameraName}");
        }

        private NisFrameData BuildFrameData(NisFrameInput fi, NisResources res, NisSettings settings)
        {
            return new NisFrameData
            {
                inputTex      = res.input.NriPtr,
                outputTex     = res.output.NriPtr,
                outputWidth   = fi.outputWidth,
                outputHeight  = fi.outputHeight,
                currentWidth  = fi.currentWidth,
                currentHeight = fi.currentHeight,
                sharpness     = settings.sharpness,
                instanceId    = _instanceId,
            };
        }

        public IntPtr GetInteropDataPtr(NisFrameInput fi, NisResources res, NisSettings settings)
        {
            var index = (int)(fi.frameIndex % BufferCount);
            _buffer[index] = BuildFrameData(fi, res, settings);
            unsafe
            {
                return (IntPtr)_buffer.GetUnsafePtr() + index * sizeof(NisFrameData);
            }
        }

        public void Dispose()
        {
            if (_buffer.IsCreated)
                _buffer.Dispose();

            DestroyNISInstance(_instanceId);
            Debug.Log($"[NIS] Destroyed NIS Instance {_instanceId} for Camera {_cameraName}");
        }
    }
}
