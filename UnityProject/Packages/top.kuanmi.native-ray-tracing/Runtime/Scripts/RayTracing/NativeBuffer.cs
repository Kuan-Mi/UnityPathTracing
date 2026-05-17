using System;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace NativeRender
{
    public sealed class NativeBuffer : IDisposable
    {
        public  ulong Handle { get; private set; }
        private bool  _disposed;

        private const    int BufferCount  = 3; // 三缓冲
        private          int _bufferIndex = 0;
        private readonly int _singleFrameSize;

        // 1. 数据缓冲区数组
        private NativeArray<byte>[] _frameDataArray;

        // 2. 参数缓冲区数组 (用于传递给 Native)
        private NativeArray<UploadParams>[] _paramsArray;

        private readonly InternalUploadPass _internalPass;

        [StructLayout(LayoutKind.Sequential)]
        struct UploadParams
        {
            public ulong  BufferHandle;
            public IntPtr SourceDataPtr;
            public uint   Size;
        }

        public NativeBuffer(int sizeInBytes)
        {
            if (sizeInBytes <= 0) throw new ArgumentOutOfRangeException(nameof(sizeInBytes));

            _singleFrameSize = (sizeInBytes + 255) & ~255;
            Handle           = NativeRenderPlugin.NR_CreateNativeBuffer((uint)_singleFrameSize);

            _frameDataArray = new NativeArray<byte>[BufferCount];
            _paramsArray    = new NativeArray<UploadParams>[BufferCount];

            for (int i = 0; i < BufferCount; i++)
            {
                _frameDataArray[i] = new NativeArray<byte>(_singleFrameSize, Allocator.Persistent);
                _paramsArray[i]    = new NativeArray<UploadParams>(1, Allocator.Persistent);
            }

            _internalPass = new InternalUploadPass(this);
        }

        public void AdvanceFrame()
        {
            _bufferIndex = (_bufferIndex + 1) % BufferCount;
        }
        
        
        // todo 目前要手动确保每帧只调用一次
        public unsafe void Upload<T>(ScriptableRenderer renderer, T data) where T : unmanaged
        {
            if (_disposed) return;


            AdvanceFrame();
            
            // 拷贝数据到当前帧对应的缓冲区
            var   currentData = _frameDataArray[_bufferIndex];
            void* srcPtr      = UnsafeUtility.AddressOf(ref data);
            void* dstPtr      = currentData.GetUnsafePtr();
            UnsafeUtility.MemCpy(dstPtr, srcPtr, Math.Min(sizeof(T), _singleFrameSize));

            _internalPass.Setup(_bufferIndex);
            renderer.EnqueuePass(_internalPass);
        }
        
        public unsafe void UploadDirect<T>(UnsafeCommandBuffer cmd, T[] data) where T : unmanaged
        {
            if (_disposed || data == null || data.Length == 0) return;
            
            AdvanceFrame();
            
            // 固定托管数组地址进行拷贝
            fixed (void* srcPtr = data)
            {
                void* dstPtr     = _frameDataArray[_bufferIndex].GetUnsafePtr();
                long  sizeToCopy = Math.Min((long)data.Length * UnsafeUtility.SizeOf<T>(), (long)_singleFrameSize);
                UnsafeUtility.MemCpy(dstPtr, srcPtr, sizeToCopy);
            }

            Execute(cmd, _bufferIndex);
        }

        private unsafe void Execute(UnsafeCommandBuffer cmd, int index)
        {
            var currentParams = _paramsArray[index];

            currentParams[0] = new UploadParams
            {
                BufferHandle  = Handle,
                SourceDataPtr = (IntPtr)_frameDataArray[index].GetUnsafePtr(),
                Size          = (uint)_singleFrameSize
            };

            // 获取参数的固定地址传给 Native
            IntPtr pParams = (IntPtr)currentParams.GetUnsafePtr();

            // 这里的 eventId 可以自定义，data 传入参数指针
            // 注意：Native 层收到该指针后只能读取，不能 free
            cmd.IssuePluginEventAndData(NativeRenderPlugin.GetNativeBufferUploadCallbackPtr(), 0x01, pParams);
        }

        private class InternalUploadPass : ScriptableRenderPass
        {
            private readonly NativeBuffer _owner;
            private          int          _recordedIndex;

            public void Setup(int index)
            {
                _recordedIndex = index;
            }

            public InternalUploadPass(NativeBuffer owner)
            {
                _owner          = owner;
                renderPassEvent = RenderPassEvent.BeforeRendering;
            }

            public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
            {
                using var builder = renderGraph.AddUnsafePass<UploadPassData>("NativeBufferUpload", out var passData);


                passData.BufferIndex = _recordedIndex;

                builder.AllowPassCulling(false);
                builder.SetRenderFunc((UploadPassData data, UnsafeGraphContext context) => { _owner.Execute(context.cmd, data.BufferIndex); });
            }

            class UploadPassData
            {
                public int BufferIndex;
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            for (int i = 0; i < BufferCount; i++)
            {
                if (_frameDataArray[i].IsCreated) _frameDataArray[i].Dispose();
                if (_paramsArray[i].IsCreated) _paramsArray[i].Dispose();
            }

            if (Handle != 0) NativeRenderPlugin.NR_DestroyNativeBuffer(Handle);
            _disposed = true;
        }
    }
}