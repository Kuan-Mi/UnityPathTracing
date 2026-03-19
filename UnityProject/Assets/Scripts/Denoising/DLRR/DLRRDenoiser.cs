using System;
using System.Runtime.InteropServices;
using PathTracing;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Rendering;
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

        public DLRRDenoiser(PathTracingSetting setting, string camName)
        {
            this.setting = setting;
            instanceId = CreateDLRRInstance();
            cameraName = camName;
            buffer = new NativeArray<RRFrameData>(BufferCount, Allocator.Persistent);
        }


        private unsafe RRFrameData GetData(CameraData cameraData, NRDDenoiser denoiser, PathTracingResourcePool pool)
        {
            RRFrameData data = new RRFrameData();

            data.inputTex  = pool.GetNriResource(RenderResourceType.Composed).NriPtr;
            data.outputTex = pool.GetNriResource(RenderResourceType.DlssOutput).NriPtr;

            data.mvTex    = pool.GetNrdResource(ResourceType.IN_MV).NriPtr;
            data.depthTex = pool.GetNrdResource(ResourceType.IN_VIEWZ).NriPtr;

            data.diffuseAlbedoTex   = pool.GetNriResource(RenderResourceType.RRGuide_DiffAlbedo).NriPtr;
            data.specularAlbedoTex  = pool.GetNriResource(RenderResourceType.RRGuide_SpecAlbedo).NriPtr;
            data.normalRoughnessTex = pool.GetNriResource(RenderResourceType.RRGuide_Normal_Roughness).NriPtr;
            data.specularMvOrHitTex = pool.GetNriResource(RenderResourceType.RRGuide_SpecHitDistance).NriPtr;

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

        public IntPtr GetInteropDataPtr(RenderingData renderingData, NRDDenoiser denoiser, PathTracingResourcePool pool)
        {
            var index = (int)(FrameIndex % BufferCount);
            buffer[index] = GetData(renderingData.cameraData, denoiser, pool);
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