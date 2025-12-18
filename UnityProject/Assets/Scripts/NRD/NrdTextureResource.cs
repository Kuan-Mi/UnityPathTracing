using System;
using System.Runtime.InteropServices;
using Nrd;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace NRD
{
    public class NrdTextureResource
    {
        
        [DllImport("RenderingPlugin")] private static extern IntPtr WrapD3D12Texture(IntPtr resource, DXGI_FORMAT format);
        
        [DllImport("RenderingPlugin")] private static extern void ReleaseTexture(IntPtr nriTex);
        
        public RTHandle Handle;  // Unity RTHandle封装
        public IntPtr NativePtr; // DX12底层指针
        public IntPtr NriPtr;    // NRD封装指针
        public string Name;

        public bool IsCreated => Handle != null;
        
        public DXGI_FORMAT GetDXGIFormat(GraphicsFormat format)
        {
            switch (format)
            {
                case GraphicsFormat.R16G16B16A16_SFloat:
                    return DXGI_FORMAT.DXGI_FORMAT_R16G16B16A16_FLOAT;
                case GraphicsFormat.R32G32B32A32_SFloat:
                    return DXGI_FORMAT.DXGI_FORMAT_R32G32B32A32_FLOAT;
                case GraphicsFormat.R8G8B8A8_UNorm:
                    return DXGI_FORMAT.DXGI_FORMAT_R8G8B8A8_UNORM;
                default:
                    throw new ArgumentOutOfRangeException(nameof(format), format, null);
            }
        }

        public void Allocate(string name, int w, int h, GraphicsFormat graphicsFormat)
        {
            var dxgiFormat = GetDXGIFormat(graphicsFormat);
            Release(); // 确保先释放旧的
            Name = name;

            // 创建 RT 描述
            var desc = new RenderTextureDescriptor(w, h, graphicsFormat, 0)
            {
                enableRandomWrite = true,
                useMipMap = false,
                msaaSamples = 1,
                sRGB = false
            };

            // 创建 RT
            var rt = new RenderTexture(desc)
            {
                name = name,
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
                RTHandles.Release(Handle);
                Handle = null;
            }
        }
    }
}