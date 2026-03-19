using System;
using System.Runtime.InteropServices;
using Nri;
using PathTracing;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine.Rendering.Universal;

namespace Nrd
{
    public class DLRRDenoiser : IDisposable
    {
        [DllImport("RenderingPlugin")]
        private static extern int CreateDLRRInstance();

        [DllImport("RenderingPlugin")]
        private static extern void DestroyDLRRInstance(int id);

        private readonly int instanceId;
        private NativeArray<RRFrameData> buffer;
        public uint FrameIndex;
        private const int BufferCount = 3;
        private string cameraName;

        private PathTracingSetting setting;

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


        private unsafe RRFrameData GetData(CameraData cameraData, NRDDenoiser denoiser, DlrrResources res)
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

            data.worldToViewMatrix = denoiser.worldToView;
            data.viewToClipMatrix = denoiser.viewToClip;

            var xr = cameraData.xr;
            if (xr.enabled)
            {
                var desc = xr.renderTargetDesc;
                data.outputWidth = (ushort)desc.width;
                data.outputHeight = (ushort)desc.height;
            }
            else
            {
                data.outputWidth = (ushort)cameraData.camera.scaledPixelWidth;
                data.outputHeight = (ushort)cameraData.camera.scaledPixelHeight;
            }


            ushort rectW = (ushort)(denoiser.renderResolution.x * setting.resolutionScale + 0.5f);
            ushort rectH = (ushort)(denoiser.renderResolution.y * setting.resolutionScale + 0.5f);

            data.currentWidth = rectW;
            data.currentHeight = rectH;

            data.upscalerMode = setting.upscalerMode;

            data.cameraJitter = denoiser.ViewportJitter;
            data.instanceId = instanceId;

            return data;
        }

        public IntPtr GetInteropDataPtr(RenderingData renderingData, NRDDenoiser denoiser, DlrrResources res)
        {
            var index = (int)(FrameIndex % BufferCount);
            buffer[index] = GetData(renderingData.cameraData, denoiser, res);
            FrameIndex++;
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