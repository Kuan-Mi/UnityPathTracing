using System;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using Object = UnityEngine.Object;

namespace Nri
{
    public class NriTextureResource
    {
        [DllImport("Denoiser")]
        private static extern IntPtr WrapD3D12Texture(IntPtr resource, DXGI_FORMAT format);

        [DllImport("Denoiser")]
        private static extern void ReleaseTexture(IntPtr nriTex);

        /// <summary>Exposed for subclasses that need to wrap a non-2D resource.</summary>
        protected static IntPtr WrapD3D12TextureInternal(IntPtr resource, DXGI_FORMAT format)
            => WrapD3D12Texture(resource, format);

        public RTHandle Handle; // Unity RTHandle封装
        public IntPtr NativePtr; // DX12底层指针
        public IntPtr NriPtr; // NRD封装指针


        public string Name;
        public NriResourceState ResourceState;
        public GraphicsFormat GraphicsFormat;
        public bool SRGB;
        
        public bool IsCreated => Handle != null;


        public NriTextureResource(string name, GraphicsFormat graphicsFormat, NriResourceState initialState, bool srgb = false)
        {
            Name = name;
            ResourceState = initialState;
            GraphicsFormat = graphicsFormat;
            SRGB = srgb;
        }

        public void Allocate(int2 resolution)
        {
            Release(); // 确保先释放旧的
            var dxgiFormat = NriUtil.GetDXGIFormat(GraphicsFormat);

            // Debug.Log($"Allocating NRD Texture Resource: {Name}, Size: {resolution}, Format: {GraphicsFormat}");


            // 创建 RT 描述
            var desc = new RenderTextureDescriptor(resolution.x, resolution.y, GraphicsFormat, 0)
            {
                enableRandomWrite = true,
                useMipMap = false,
                msaaSamples = 1,
                sRGB = SRGB
            };

            // 创建 RT
            var rt = new RenderTexture(desc)
            {
                name = Name,
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };
            rt.Create();

            Handle = RTHandles.Alloc(rt);
            NativePtr = Handle.rt.GetNativeTexturePtr();
            NriPtr = WrapD3D12Texture(NativePtr, dxgiFormat);
        }

        public void Release()
        {
            if (NriPtr != IntPtr.Zero)
            {
                ReleaseTexture(NriPtr);
                NriPtr = IntPtr.Zero;
            }

            NativePtr = IntPtr.Zero;

            if (Handle != null)
            {
                var rt = Handle.rt;


                RTHandles.Release(Handle);
                Handle = null;
                if (rt != null)
                {
                    if (Application.isPlaying)
                        Object.Destroy(rt);
                    else
                        Object.DestroyImmediate(rt);
                }
            }
        }
    }
}