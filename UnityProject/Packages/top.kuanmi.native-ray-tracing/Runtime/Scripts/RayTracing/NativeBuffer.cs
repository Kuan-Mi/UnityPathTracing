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
        public ulong Handle { get; private set; }
        private bool _disposed;

        private const int BufferCount = 3; // 三缓冲
        private int _bufferIndex = 0;
        private readonly int _singleFrameSize;
        
        // 1. 数据缓冲区数组
        private NativeArray<byte>[] _frameDataArray; 
        // 2. 参数缓冲区数组 (用于传递给 Native)
        private NativeArray<UploadParams>[] _paramsArray;

        private bool _hasUpdateThisFrame;
        private int _lastUpdateFrame = -1;
        private readonly InternalUploadPass _internalPass;

        [StructLayout(LayoutKind.Sequential)]
        struct UploadParams {
            public ulong BufferHandle;
            public IntPtr SourceDataPtr;
            public uint Size;
        }

        public NativeBuffer(int sizeInBytes)
        {
            if (sizeInBytes <= 0) throw new ArgumentOutOfRangeException(nameof(sizeInBytes));

            _singleFrameSize = (sizeInBytes + 255) & ~255;
            Handle = NativeRenderPlugin.NR_CreateNativeBuffer((uint)_singleFrameSize);

            _frameDataArray = new NativeArray<byte>[BufferCount];
            _paramsArray = new NativeArray<UploadParams>[BufferCount];

            for (int i = 0; i < BufferCount; i++)
            {
                _frameDataArray[i] = new NativeArray<byte>(_singleFrameSize, Allocator.Persistent);
                _paramsArray[i] = new NativeArray<UploadParams>(1, Allocator.Persistent);
            }

            _internalPass = new InternalUploadPass(this);
        }

        public unsafe void Upload<T>(ScriptableRenderer renderer, T data) where T : unmanaged
        {
            if (_disposed) return;

            // 每一帧第一次调用时切换缓冲区索引
            if (_lastUpdateFrame != Time.frameCount)
            {
                _bufferIndex = (_bufferIndex + 1) % BufferCount;
                _lastUpdateFrame = Time.frameCount;
                _hasUpdateThisFrame = false;
            }

            // 拷贝数据到当前帧对应的缓冲区
            var currentData = _frameDataArray[_bufferIndex];
            void* srcPtr = UnsafeUtility.AddressOf(ref data);
            void* dstPtr = currentData.GetUnsafePtr();
            UnsafeUtility.MemCpy(dstPtr, srcPtr, Math.Min(sizeof(T), _singleFrameSize));

            if (!_hasUpdateThisFrame)
            {
                renderer.EnqueuePass(_internalPass);
                _hasUpdateThisFrame = true;
            }
        }

        private unsafe void Execute(UnsafeCommandBuffer cmd)
        {
            // 准备这一帧的参数
            var currentParams = _paramsArray[_bufferIndex];
            currentParams[0] = new UploadParams {
                BufferHandle = Handle,
                SourceDataPtr = (IntPtr)_frameDataArray[_bufferIndex].GetUnsafePtr(),
                Size = (uint)_singleFrameSize
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
            public InternalUploadPass(NativeBuffer owner) {
                _owner = owner;
                renderPassEvent = RenderPassEvent.BeforeRendering;
            }

            public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
            {
                using var builder = renderGraph.AddUnsafePass<EmptyPassData>("NativeBufferUpload", out _);
                builder.AllowPassCulling(false);
                builder.SetRenderFunc((EmptyPassData _, UnsafeGraphContext context) => {
                    _owner.Execute(context.cmd);
                });
            }
            class EmptyPassData { }
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