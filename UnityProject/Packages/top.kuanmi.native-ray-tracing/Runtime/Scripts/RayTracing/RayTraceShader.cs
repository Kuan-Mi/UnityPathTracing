using System;
using System.IO;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Rendering;

namespace NativeRender
{
    /// <summary>
    /// DXR ray tracing shader asset. Import a <c>.rayshader</c> HLSL file to create one.
    ///
    /// HLSL convention:
    ///   Resources in space0-3 are all reflected and bindable via the API.
    ///   Shader must export: RayGenShader, MissShader, AnyHitShader, ClosestHitShader.
    ///
    /// Note: Only one RayTraceShader may be dispatched per CommandBuffer recording
    ///       (D3D12 limitation: only one shader-visible CBV/SRV/UAV heap at a time).
    ///       Dispatch multiple shaders in separate CommandBuffers or sequential render passes.
    /// </summary>
    public class RayTraceShader : ScriptableObject
    {
        /// <summary>Semicolon-separated #include search directories. Written by the ScriptedImporter.</summary>
        [SerializeField, HideInInspector]
        private string[] additionalIncludePaths = Array.Empty<string>();

        /// <summary>Additional DXC compiler arguments (semicolon-separated). Written by the ScriptedImporter.</summary>
        [SerializeField, HideInInspector]
        private string _extraArgs;

        /// <summary>Pre-compiled DXIL bytecode. Populated by CompileOnly(); persisted by Unity serialization.</summary>
        [SerializeField, HideInInspector]
        private byte[] _compiledDxil;

        private ulong _handle;

        // Event data buffer reused each frame
        private NativeArray<NativeRenderPlugin.RTS_RenderEventData> _eventData;

        public bool IsValid => _handle != 0;

        /// <summary>True when compiled DXIL bytes are available (pipeline may or may not exist yet).</summary>
        public bool HasCompiledBytes => _compiledDxil is { Length: > 0 };

        /// <summary>Size in bytes of the cached DXIL bytecode, or 0 if not compiled.</summary>
        public int CompiledByteCount => _compiledDxil?.Length ?? 0;

        // -------------------------------------------------------------------
        // ScriptableObject lifecycle
        // -------------------------------------------------------------------

        private void OnEnable()
        {
            if (!string.IsNullOrEmpty(GetHlslPath()))
                CompileOnly();
        }

        private void OnDisable()
        {
            DestroyHandle();
        }

        // -------------------------------------------------------------------
        // Compilation  (ShaderCompilerPlugin — no D3D12 needed)
        // -------------------------------------------------------------------

        /// <summary>
        /// Compiles the HLSL to DXIL bytes and caches them in <see cref="_compiledDxil"/>.
        /// If bytes are already present this is a no-op; call <see cref="ForceRecompile"/>
        /// to discard the cache and recompile.
        /// Does NOT create the DXR pipeline.
        /// </summary>
        private void CompileOnly()
        {
            if (HasCompiledBytes) return;

            string hlslPath    = GetHlslPath();
            string includeDirs = BuildIncludeDirs(hlslPath);

            bool ok = NativeRenderPlugin.ShaderCompilerPlugin.NR_SC_Compile(
                hlslPath, includeDirs, string.IsNullOrEmpty(_extraArgs) ? null : _extraArgs, out IntPtr nativePtr, out uint nativeSize);

            if (!ok || nativePtr == IntPtr.Zero || nativeSize == 0)
            {
                Debug.LogError($"[RayTraceShader] Compilation failed: {hlslPath}");
                return;
            }

            _compiledDxil = new byte[nativeSize];
            Marshal.Copy(nativePtr, _compiledDxil, 0, (int)nativeSize);
            NativeRenderPlugin.ShaderCompilerPlugin.NR_SC_Free(nativePtr);

#if UNITY_EDITOR
            UnityEditor.EditorUtility.SetDirty(this);
#endif
            Debug.Log($"[RayTraceShader] Compiled {nativeSize} bytes: {hlslPath}");
        }

        /// <summary>
        /// Clears the cached DXIL bytes, destroys the existing pipeline, and recompiles.
        /// The pipeline will be re-created on the next <see cref="CreatePipeline"/> call.
        /// </summary>
        [ContextMenu("Recompile")]
        public void ForceRecompile()
        {
            DestroyHandle();
            _compiledDxil = null;
            CompileOnly();
        }

        // -------------------------------------------------------------------
        // Pipeline creation  (NativeRenderPlugin — requires D3D12 device)
        // Called by NativeRayTracingFeature, not during OnEnable.
        // -------------------------------------------------------------------

        /// <summary>
        /// Creates (or reuses) the DXR pipeline from the cached DXIL bytes.
        /// Returns true if the pipeline is ready.  Idempotent: if <see cref="IsValid"/>
        /// is already true this returns immediately without rebuilding.
        /// </summary>
        public bool CreatePipeline()
        {
            if (IsValid) return true;

            if (!HasCompiledBytes)
            {
                if (string.IsNullOrEmpty(GetHlslPath()))
                {
                    Debug.LogError("[RayTraceShader] CreatePipeline: cannot resolve hlsl path");
                    return false;
                }

                CompileOnly();
            }

            if (!HasCompiledBytes)
            {
                Debug.LogError($"[RayTraceShader] CreatePipeline: compilation failed: {GetHlslPath()}");
                return false;
            }

            if (!_eventData.IsCreated)
                _eventData = new NativeArray<NativeRenderPlugin.RTS_RenderEventData>(1, Allocator.Persistent);

            _handle = NativeRenderPlugin.NR_CreateRayTraceShaderFromBytes(_compiledDxil, (uint)_compiledDxil.Length);
            if (_handle == 0)
            {
                Debug.LogError($"[RayTraceShader] CreatePipeline failed: {GetHlslPath()}");
                return false;
            }

            return true;
        }

        private void DestroyHandle()
        {
            if (_handle != 0)
            {
                GL.Flush();
                NativeRenderPlugin.NR_DestroyRayTraceShader(_handle);
                _handle = 0;
            }

            if (_eventData.IsCreated)
                _eventData.Dispose();
        }

        private string GetHlslPath()
        {
#if UNITY_EDITOR
            string assetPath = UnityEditor.AssetDatabase.GetAssetPath(this);
            if (!string.IsNullOrEmpty(assetPath))
                return Path.GetFullPath(assetPath);
            else
            {
                Debug.LogError($"[RayTraceShader] GetHlslPath: cannot find asset path for {name}");
            }
#endif
            return string.Empty;
        }

        private string BuildIncludeDirs(string hlslPath)
        {
            string shaderDir = Path.GetDirectoryName(hlslPath);
            if (additionalIncludePaths == null || additionalIncludePaths.Length == 0)
                return shaderDir;
            return shaderDir + ";" + string.Join(";", additionalIncludePaths);
        }

        // -------------------------------------------------------------------
        // Resource binding
        // -------------------------------------------------------------------

        /// <summary>Binds a ComputeBuffer (or GraphicsBuffer) as a read-only structured/byte-address buffer (SRV).</summary>
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

        public void SetConstantBuffer(string name, GraphicsBuffer buffer)
        {
            if (!IsValid) return;
            NativeRenderPlugin.NR_RTS_SetConstantBuffer(_handle, name, buffer.GetNativeBufferPtr());
        }

        /// <summary>Binds a GraphicsBuffer as a StructuredBuffer SRV (provides stride and count automatically).</summary>
        public void SetStructuredBuffer(string name, GraphicsBuffer buffer)
        {
            if (!IsValid) return;
            // Passing null clears the binding on the C++ side (writes a null SRV),
            // which prevents stale/dangling resource pointers after buffers are disposed.
            IntPtr ptr    = buffer != null ? buffer.GetNativeBufferPtr() : IntPtr.Zero;
            uint   count  = buffer != null ? (uint)buffer.count : 0;
            uint   stride = buffer != null ? (uint)buffer.stride : 0;
            NativeRenderPlugin.NR_RTS_SetStructuredBuffer(_handle, name, ptr, count, stride);
        }

        /// <summary>Binds a ComputeBuffer as a StructuredBuffer SRV with explicit element count and stride.</summary>
        public void SetStructuredBuffer(string name, ComputeBuffer buffer, int elementCount, int elementStride)
        {
            if (!IsValid) return;
            IntPtr ptr    = buffer != null ? buffer.GetNativeBufferPtr() : IntPtr.Zero;
            uint   count  = buffer != null ? (uint)elementCount : 0;
            uint   stride = buffer != null ? (uint)elementStride : 0;
            NativeRenderPlugin.NR_RTS_SetStructuredBuffer(_handle, name, ptr, count, stride);
        }

        /// <summary>Binds the TLAS of an acceleration structure by HLSL variable name (e.g. "SceneBVH").</summary>
        public void SetAccelerationStructure(string name, RayTracingAccelerationStructure accelStructure)
        {
            if (!IsValid || accelStructure == null) return;
            // Pass the AccelerationStructure handle so the native layer reads the TLAS pointer
            // dynamically at Dispatch time, after BuildOrUpdate may have rebuilt the TLAS buffer.
            NativeRenderPlugin.NR_RTS_SetAccelerationStructureHandle(_handle, name, accelStructure.Handle);
        }

        /// <summary>
        /// Binds a BindlessTexture to an unbounded Texture2D[] variable.
        /// Call this again after BindlessTexture.Resize() to rebind the new descriptor range.
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
        /// Call this again after BindlessBuffer.Resize() to rebind the new descriptor range.
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
        /// <param name="cmd">Active CommandBuffer.</param>
        /// <param name="outputTexture">RenderTexture to write ray tracing results into.</param>
        /// <param name="camera">Used for view-projection matrix and position.</param>
        public void Dispatch(CommandBuffer cmd, uint width, uint height)
        {
            if (!IsValid)
                return;

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