using System;
using System.Runtime.InteropServices;
using Nrd;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using Object = UnityEngine.Object;

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
        public NriResourceState ResourceState;
        public ResourceType ResourceType;
        public GraphicsFormat GraphicsFormat;
        public bool IsCreated => Handle != null;
        
        
        public NrdTextureResource(ResourceType resourceType, GraphicsFormat graphicsFormat,NriResourceState initialState)
        {
            Name = resourceType.ToString();
            ResourceType = resourceType;
            ResourceState = initialState;
            GraphicsFormat = graphicsFormat;
        }

        public void Allocate( int w, int h)
        {
            var dxgiFormat = NRDUtil.GetDXGIFormat(GraphicsFormat);
            Release(); // 确保先释放旧的
            
            // Debug.Log($"Allocating NRD Texture Resource: {Name}, Size: {w}x{h}, Format: {GraphicsFormat}");

            // 创建 RT 描述
            var desc = new RenderTextureDescriptor(w, h, GraphicsFormat, 0)
            {
                enableRandomWrite = true,
                useMipMap = false,
                msaaSamples = 1,
                sRGB = false
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
                RTHandles.Release(Handle);
                Handle = null;
            }
        }
    }
}