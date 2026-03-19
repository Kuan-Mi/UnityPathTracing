using System;
using System.Runtime.InteropServices;
using Nrd;
using Nri;
using PathTracing;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering.Universal;

namespace DLRR
{
    public class DLRRDenoiser : IDisposable
    {
        [DllImport("RenderingPlugin")]
        private static extern int CreateDLRRInstance();

        [DllImport("RenderingPlugin")]
        private static extern void DestroyDLRRInstance(int id);

        private readonly int instanceId;
        private NativeArray<RRFrameData> buffer;
        private const int BufferCount = 3;
        private string cameraName;

        private PathTracingSetting setting;

        /// <summary>
        /// Per-frame camera data filled by PathTracingFeature from CameraFrameState.
        /// DLRRDenoiser does not depend on CameraFrameState directly.
        /// </summary>
        public struct DlrrFrameInput
        {
            public Matrix4x4 worldToView;
            public Matrix4x4 viewToClip;
            public float2    ViewportJitter;
            public int2      renderResolution;
            public uint      FrameIndex;
        }

        /// <summary>
        /// DLSS-RR textures packed by PathTracingFeature and passed each frame.
        /// </summary>
        public struct DlrrResources
        {
            public NriTextureResource Input;           // Composed
            public NriTextureResource Output;          // DlssOutput
            public NriTextureResource Mv;              // IN_MV
            public NriTextureResource Depth;           // IN_VIEWZ
            public NriTextureResource DiffAlbedo;      // RRGuide_DiffAlbedo
            public NriTextureResource SpecAlbedo;      // RRGuide_SpecAlbedo
            public NriTextureResource NormalRoughness; // RRGuide_Normal_Roughness
            public NriTextureResource SpecHitDistance; // RRGuide_SpecHitDistance
        }

        public DLRRDenoiser(PathTracingSetting setting, string camName)
        {
            this.setting = setting;
            instanceId = CreateDLRRInstance();
            cameraName = camName;
            buffer = new NativeArray<RRFrameData>(BufferCount, Allocator.Persistent);
        }

        private RRFrameData GetData(CameraData cameraData, DlrrFrameInput fi, DlrrResources res)
        {
            RRFrameData data = new RRFrameData();

            data.inputTex  = res.Input.NriPtr;
            data.outputTex = res.Output.NriPtr;

            data.mvTex    = res.Mv.NriPtr;
            data.depthTex = res.Depth.NriPtr;

            data.diffuseAlbedoTex   = res.DiffAlbedo.NriPtr;
            data.specularAlbedoTex  = res.SpecAlbedo.NriPtr;
            data.normalRoughnessTex = res.NormalRoughness.NriPtr;
            data.specularMvOrHitTex = res.SpecHitDistance.NriPtr;

            data.worldToViewMatrix = fi.worldToView;
            data.viewToClipMatrix  = fi.viewToClip;

            var xr = cameraData.xr;
            if (xr.enabled)
            {
                var desc = xr.renderTargetDesc;
                data.outputWidth  = (ushort)desc.width;
                data.outputHeight = (ushort)desc.height;
            }
            else
            {
                data.outputWidth  = (ushort)cameraData.camera.scaledPixelWidth;
                data.outputHeight = (ushort)cameraData.camera.scaledPixelHeight;
            }

            ushort rectW = (ushort)(fi.renderResolution.x * setting.resolutionScale + 0.5f);
            ushort rectH = (ushort)(fi.renderResolution.y * setting.resolutionScale + 0.5f);

            data.currentWidth  = rectW;
            data.currentHeight = rectH;

            data.upscalerMode  = setting.upscalerMode;
            data.cameraJitter  = fi.ViewportJitter;
            data.instanceId    = instanceId;

            return data;
        }

        public IntPtr GetInteropDataPtr(RenderingData renderingData, DlrrFrameInput fi, DlrrResources res)
        {
            var index = (int)(fi.FrameIndex % BufferCount);
            buffer[index] = GetData(renderingData.cameraData, fi, res);
            unsafe
            {
                return (IntPtr)buffer.GetUnsafePtr() + index * sizeof(RRFrameData);
            }
        }

        public void Dispose()
        {
            if (buffer.IsCreated)
            {
                buffer.Dispose();
            }

            DestroyDLRRInstance(instanceId);
        }
    }
}