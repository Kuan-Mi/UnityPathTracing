using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Rendering;

namespace NativeRender
{
    /// <summary>
    /// A DXR pipeline instance created from a <see cref="RayTraceShader"/> asset.
    /// Owns the native D3D12 pipeline handle and all resource bindings.
    ///
    /// Multiple pipelines can be created from the same <see cref="RayTraceShader"/>,
    /// each with independent resource bindings (textures, buffers, acceleration structures, etc.).
    ///
    /// Lifetime: must be explicitly disposed via <see cref="Dispose"/>.
    /// </summary>
    public sealed class RayTracePipeline : IDisposable
    {
        private ulong _handle;
        private NativeArray<NativeRenderPlugin.RTS_RenderEventData> _eventData;
        private RayTraceShader _shader;

        /// <summary>True if the underlying D3D12 pipeline is valid and ready to dispatch.</summary>
        public bool IsValid => _handle != 0;

        // -------------------------------------------------------------------
        // Construction
        // -------------------------------------------------------------------

        /// <summary>
        /// Creates a new DXR pipeline from the given shader asset.
        /// Triggers HLSL compilation if the asset has not been compiled yet.
        /// Throws <see cref="InvalidOperationException"/> if pipeline creation fails.
        /// </summary>
        public RayTracePipeline(RayTraceShader shader)
        {
            if (shader == null)
                throw new ArgumentNullException(nameof(shader));

            _shader = shader;
            BuildNativeHandle(shader);
            _eventData = new NativeArray<NativeRenderPlugin.RTS_RenderEventData>(1, Allocator.Persistent);
            RayTraceShader.OnRecompiled += OnShaderRecompiled;
        }

        private void BuildNativeHandle(RayTraceShader shader)
        {
            byte[] dxil = shader.GetOrCompileDxil();
            if (dxil == null || dxil.Length == 0)
                throw new InvalidOperationException(
                    $"[RayTracePipeline] Shader compilation failed for: {shader.GetHlslPath()}");

            _handle = NativeRenderPlugin.NR_CreateRayTraceShaderFromBytes(dxil, (uint)dxil.Length);
            if (_handle == 0)
                throw new InvalidOperationException(
                    $"[RayTracePipeline] NR_CreateRayTraceShaderFromBytes returned 0 for: {shader.name}");
        }

        private void OnShaderRecompiled(RayTraceShader shader)
        {
            if (shader != _shader) return;

            // Destroy the old native pipeline and rebuild from the freshly compiled DXIL.
            if (_handle != 0)
            {
                GL.Flush();
                NativeRenderPlugin.NR_DestroyRayTraceShader(_handle);
                _handle = 0;
            }

            try
            {
                BuildNativeHandle(shader);
                Debug.Log($"[RayTracePipeline] Rebuilt pipeline for: {shader.name}");
            }
            catch (Exception e)
            {
                Debug.LogError(e.Message);
            }
        }

        // -------------------------------------------------------------------
        // IDisposable
        // -------------------------------------------------------------------

        public void Dispose()
        {
            RayTraceShader.OnRecompiled -= OnShaderRecompiled;

            if (_handle != 0)
            {
                GL.Flush();
                NativeRenderPlugin.NR_DestroyRayTraceShader(_handle);
                _handle = 0;
            }

            if (_eventData.IsCreated)
                _eventData.Dispose();
        }

        // -------------------------------------------------------------------
        // Resource binding
        // -------------------------------------------------------------------

        /// <summary>Binds a ComputeBuffer as a read-only structured/byte-address buffer (SRV).</summary>
        public void SetBuffer(string name, ComputeBuffer buffer)
        {
            if (!IsValid) return;
            NativeRenderPlugin.NR_RTS_SetBuffer(_handle, name, buffer.GetNativeBufferPtr());
        }

        /// <summary>Binds a GraphicsBuffer as a read-only structured/byte-address buffer (SRV).</summary>
        public void SetBuffer(string name, GraphicsBuffer buffer)
        {
            if (!IsValid) return;
            NativeRenderPlugin.NR_RTS_SetBuffer(_handle, name, buffer.GetNativeBufferPtr());
        }

        /// <summary>Binds a ComputeBuffer as an RW (read-write) buffer (UAV).</summary>
        public void SetRWBuffer(string name, ComputeBuffer buffer)
        {
            if (!IsValid) return;
            NativeRenderPlugin.NR_RTS_SetRWBuffer(_handle, name, buffer.GetNativeBufferPtr());
        }

        /// <summary>Binds a GraphicsBuffer as an RW (read-write) buffer (UAV).</summary>
        public void SetRWBuffer(string name, GraphicsBuffer buffer)
        {
            if (!IsValid) return;
            NativeRenderPlugin.NR_RTS_SetRWBuffer(_handle, name, buffer.GetNativeBufferPtr());
        }

        /// <summary>Binds a Texture2D or RenderTexture as a read-only texture (SRV).</summary>
        public void SetTexture(string name, Texture texture)
        {
            if (!IsValid) return;
            NativeRenderPlugin.NR_RTS_SetTexture(_handle, name, texture.GetNativeTexturePtr());
        }

        /// <summary>Binds a RenderTexture as a read-write texture (UAV).</summary>
        public void SetRWTexture(string name, RenderTexture texture)
        {
            if (!IsValid) return;
            NativeRenderPlugin.NR_RTS_SetRWTexture(_handle, name, texture.GetNativeTexturePtr());
        }

        /// <summary>
        /// Binds a ComputeBuffer as a constant buffer (CBV).
        /// The buffer must have been created with ComputeBufferType.Constant.
        /// </summary>
        public void SetConstantBuffer(string name, ComputeBuffer buffer)
        {
            if (!IsValid) return;
            NativeRenderPlugin.NR_RTS_SetConstantBuffer(_handle, name, buffer.GetNativeBufferPtr());
        }

        /// <summary>Binds a GraphicsBuffer as a constant buffer (CBV).</summary>
        public void SetConstantBuffer(string name, GraphicsBuffer buffer)
        {
            if (!IsValid) return;
            NativeRenderPlugin.NR_RTS_SetConstantBuffer(_handle, name, buffer.GetNativeBufferPtr());
        }

        /// <summary>
        /// Binds a GraphicsBuffer as a StructuredBuffer SRV.
        /// Passing null clears the binding, preventing stale/dangling resource pointers.
        /// </summary>
        public void SetStructuredBuffer(string name, GraphicsBuffer buffer)
        {
            if (!IsValid) return;
            IntPtr ptr    = buffer != null ? buffer.GetNativeBufferPtr() : IntPtr.Zero;
            uint   count  = buffer != null ? (uint)buffer.count  : 0;
            uint   stride = buffer != null ? (uint)buffer.stride : 0;
            NativeRenderPlugin.NR_RTS_SetStructuredBuffer(_handle, name, ptr, count, stride);
        }

        /// <summary>Binds a ComputeBuffer as a StructuredBuffer SRV with explicit element count and stride.</summary>
        public void SetStructuredBuffer(string name, ComputeBuffer buffer, int elementCount, int elementStride)
        {
            if (!IsValid) return;
            IntPtr ptr    = buffer != null ? buffer.GetNativeBufferPtr() : IntPtr.Zero;
            uint   count  = buffer != null ? (uint)elementCount  : 0;
            uint   stride = buffer != null ? (uint)elementStride : 0;
            NativeRenderPlugin.NR_RTS_SetStructuredBuffer(_handle, name, ptr, count, stride);
        }

        /// <summary>Binds the TLAS of an acceleration structure by HLSL variable name.</summary>
        public void SetAccelerationStructure(string name, RayTracingAccelerationStructure accelStructure)
        {
            if (!IsValid || accelStructure == null) return;
            NativeRenderPlugin.NR_RTS_SetAccelerationStructureHandle(_handle, name, accelStructure.Handle);
        }

        /// <summary>
        /// Binds a BindlessTexture to an unbounded Texture2D[] variable.
        /// Call again after BindlessTexture.Resize() to rebind the new descriptor range.
        /// Returns true on success.
        /// </summary>
        public bool SetBindlessTexture(string name, BindlessTexture bt)
        {
            if (!IsValid) return false;
            ulong btHandle = bt != null ? bt.Handle : 0UL;
            return NativeRenderPlugin.NR_RTS_SetBindlessTexture(_handle, name, btHandle) != 0;
        }

        /// <summary>
        /// Binds a BindlessBuffer to an unbounded ByteAddressBuffer[] variable.
        /// Call again after BindlessBuffer.Resize() to rebind the new descriptor range.
        /// Returns true on success.
        /// </summary>
        public bool SetBindlessBuffer(string name, BindlessBuffer bb)
        {
            if (!IsValid) return false;
            ulong bbHandle = bb != null ? bb.Handle : 0UL;
            return NativeRenderPlugin.NR_RTS_SetBindlessBuffer(_handle, name, bbHandle) != 0;
        }

        // -------------------------------------------------------------------
        // Dispatch
        // -------------------------------------------------------------------

        /// <summary>
        /// Enqueues a DispatchRays call into the CommandBuffer.
        /// Must be called during a URP/HDRP render pass (on the render thread).
        /// </summary>
        public void Dispatch(CommandBuffer cmd, uint width, uint height)
        {
            if (!IsValid) return;

            var ed = _eventData[0];
            ed.shaderHandle = _handle;
            ed.width        = width;
            ed.height       = height;
            _eventData[0]   = ed;

            unsafe
            {
                cmd.IssuePluginEventAndData(
                    NativeRenderPlugin.NR_RTS_GetRenderEventFunc(),
                    1,
                    (IntPtr)_eventData.GetUnsafePtr());
            }
        }
    }
}
