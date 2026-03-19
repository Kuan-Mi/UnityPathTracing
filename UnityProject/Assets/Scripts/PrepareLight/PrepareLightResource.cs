using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using DefaultNamespace;
using Nrd;
using Nri;
using RTXDI;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Playables;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace RTXDI
{
    public class PrepareLightResource : IDisposable
    {
        [DllImport("UnityRTXDI")]
        private static extern int CreateDenoiserInstance();

        [DllImport("UnityRTXDI")]
        private static extern void DestroyDenoiserInstance(int id);


        [DllImport("UnityRTXDI")]
        private static extern IntPtr WrapD3D12Texture(IntPtr resource, DXGI_FORMAT format);
        
        [DllImport("UnityRTXDI")]
        private static extern IntPtr WrapD3D12Buffer(IntPtr resource, ushort stride);

        [DllImport("UnityRTXDI")]
        private static extern void ReleaseTexture(IntPtr nriTex);

        [DllImport("UnityRTXDI")]
        private static extern void UpdateDenoiserResources(int instanceId, IntPtr resources, int count);

        private readonly int instanceId;

        
        // private IntPtr inputNriTex;

        private NativeArray<EmissionResourceInput> m_ResourceCache;
        
        private List<Texture2D> lastSentTextures = new List<Texture2D>();

        public unsafe void SendTexture(List<Texture2D> textures)
        {

            if (lastSentTextures.SequenceEqual(textures))
            {
                return; // No change in textures, skip updating
            }
            
            if (m_ResourceCache.IsCreated) m_ResourceCache.Dispose();
            m_ResourceCache = new NativeArray<EmissionResourceInput>(textures.Count, Allocator.Persistent);

            for (int i = 0; i < textures.Count; i++)
            {
                var tex = textures[i];
                IntPtr nativePtr = tex.GetNativeTexturePtr();
                
                
                var format = tex.graphicsFormat;
                var dxgiFormat = NRIUtil.GetDXGIFormat(format);
                 
                // Debug.Log($"Sending Texture {i}: {tex.name}, Format: {format}, DXGI Format: {dxgiFormat}");
                
                IntPtr nriTex = WrapD3D12Texture(nativePtr, dxgiFormat);

                EmissionResourceInput resourceInput = new EmissionResourceInput
                {
                    texture = nriTex,
                    format =  NRIUtil.GetNriFormat(format)
                };
                m_ResourceCache[i] = resourceInput;
            }
 
            EmissionResourceInput* ptr = (EmissionResourceInput*)m_ResourceCache.GetUnsafePtr();

            UpdateDenoiserResources(instanceId, (IntPtr)ptr, m_ResourceCache.Length);
            
            lastSentTextures = new List<Texture2D>(textures);
        }
        
        
        private IntPtr nriInstanceBufferPtr;
        private IntPtr nriPrimtiveBufferPtr;
        private IntPtr nriLightInfoBufferPtr;
        private GPUScene _scene;

        public void SetBuffer(GPUScene scene)
        {
            _scene = scene;
            nriInstanceBufferPtr = WrapD3D12Buffer(scene._instanceBuffer.GetNativeBufferPtr(), (ushort)Marshal.SizeOf<InstanceData>());
            nriPrimtiveBufferPtr = WrapD3D12Buffer(scene._primitiveBuffer.GetNativeBufferPtr(), (ushort)Marshal.SizeOf<PrimitiveData>());
            nriLightInfoBufferPtr = WrapD3D12Buffer(scene._lightInfoBuffer.GetNativeBufferPtr(), (ushort)Marshal.SizeOf<RAB_LightInfo>());
        }
        
        public PrepareLightResource()
        {
            instanceId = CreateDenoiserInstance();

            buffer = new NativeArray<PrepareLightFrameData>(BufferCount, Allocator.Persistent);
            
             
        }

        public void Dispose()
        {
            DestroyDenoiserInstance(instanceId);
        }

        public uint FrameIndex;
        private NativeArray<PrepareLightFrameData> buffer;
        private const int BufferCount = 3;

        private PrepareLightFrameData GetData()
        {
            PrepareLightFrameData data = new PrepareLightFrameData
            {
                instanceBuffer = nriInstanceBufferPtr,
                primitiveBuffer = nriPrimtiveBufferPtr,
                lightDataBuffer = nriLightInfoBufferPtr,
                numPrimitives = (int)_scene.emissiveTriangleCount,
                InstanceCount = _scene._instanceBuffer.count,
                instanceId = instanceId
            };

            return data;
        }


        public IntPtr GetInteropDataPtr()
        {
            var index = (int)(FrameIndex % BufferCount);

            buffer[index] = GetData();
            FrameIndex++;
            unsafe
            {
                return (IntPtr)buffer.GetUnsafePtr() + index * sizeof(PrepareLightFrameData);
            }
        }
    }
}