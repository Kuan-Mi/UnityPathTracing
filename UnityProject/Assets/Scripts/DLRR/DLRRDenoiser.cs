using System;
using System.Runtime.InteropServices;
using PathTracing;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Rendering;

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

        public DLRRDenoiser(PathTracingSetting setting, string camName)
        {
            this.setting = setting;
            instanceId = CreateDLRRInstance();
            cameraName = camName;
            buffer = new NativeArray<RRFrameData>(BufferCount, Allocator.Persistent);
        }


        private unsafe RRFrameData GetData(Camera mCamera, NRDDenoiser denoiser)
        {
            RRFrameData data = new RRFrameData();

            data.inputTex = denoiser.GetResource(ResourceType.Composed).NriPtr;
            data.outputTex = denoiser.GetResource(ResourceType.DlssOutput).NriPtr;

            data.mvTex = denoiser.GetResource(ResourceType.IN_MV).NriPtr;
            data.depthTex = denoiser.GetResource(ResourceType.IN_VIEWZ).NriPtr;

            data.diffuseAlbedoTex = denoiser.GetResource(ResourceType.RRGuide_DiffAlbedo).NriPtr;
            data.specularAlbedoTex = denoiser.GetResource(ResourceType.RRGuide_SpecAlbedo).NriPtr;
            data.normalRoughnessTex = denoiser.GetResource(ResourceType.RRGuide_Normal_Roughness).NriPtr;
            data.specularMvOrHitTex = denoiser.GetResource(ResourceType.RRGuide_SpecHitDistance).NriPtr;

            data.worldToViewMatrix = denoiser.worldToView;
            data.viewToClipMatrix = denoiser.viewToClip;
            data.outputWidth = (ushort)mCamera.pixelWidth;
            data.outputHeight = (ushort)mCamera.pixelHeight;

            ushort rectW = (ushort)(denoiser.renderResolution.x * setting.resolutionScale + 0.5f);
            ushort rectH = (ushort)(denoiser.renderResolution.y * setting.resolutionScale + 0.5f);

            data.currentWidth = rectW;
            data.currentHeight = rectH;

            data.upscalerMode = setting.upscalerMode;

            data.cameraJitter = denoiser.ViewportJitter;
            data.instanceId = instanceId;

            return data;
        }

        public IntPtr GetInteropDataPtr(Camera mCamera, NRDDenoiser denoiser)
        {
            var index = (int)(FrameIndex % BufferCount);
            buffer[index] = GetData(mCamera, denoiser);
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