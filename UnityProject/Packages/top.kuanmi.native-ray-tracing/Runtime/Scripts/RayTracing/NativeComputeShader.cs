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
    /// Native compute shader asset. Import a <c>.computeshader</c> HLSL file to create one.
    ///
    /// HLSL convention:
    ///   Resources in space0-3 are all reflected and bindable via the API.
    ///   Shader must export a single <c>[numthreads(X,Y,Z)]</c> entry point.
    /// </summary>
    public class NativeComputeShader : ScriptableObject
    {
        /// <summary>Semicolon-separated #include search directories. Written by the ScriptedImporter.</summary>
        [SerializeField, HideInInspector]
        private string[] additionalIncludePaths = Array.Empty<string>();

        /// <summary>Additional DXC compiler arguments. Written by the ScriptedImporter.</summary>
        [SerializeField, HideInInspector]
        private string[] _extraArgs = Array.Empty<string>();

        /// <summary>Preprocessor defines (e.g. "FOO=1", "BAR"). Written by the ScriptedImporter.</summary>
        [SerializeField, HideInInspector]
        private string[] _defines = Array.Empty<string>();

        /// <summary>Entry point function name (e.g. "main"). Written by the ScriptedImporter.</summary>
        [SerializeField, HideInInspector]
        private string _entryPoint = "main";

        /// <summary>DXC target profile (e.g. "cs_6_6"). Written by the ScriptedImporter.</summary>
        [SerializeField, HideInInspector]
        private string _targetProfile = "cs_6_6";

        /// <summary>Pre-compiled DXIL bytecode. Populated by CompileOnly(); persisted by Unity serialization.</summary>
        [SerializeField, HideInInspector]
        private byte[] _compiledDxil;

        private ulong _handle;

        // Event data buffer reused each frame
        private NativeArray<NativeRenderPlugin.CS_RenderEventData> _eventData;

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
        /// Does NOT create the compute pipeline.
        /// </summary>
        private void CompileOnly()
        {
            if (HasCompiledBytes) return;

            string hlslPath    = GetHlslPath();
            string includeDirs = BuildIncludeDirs(hlslPath);
            string entryPoint  = string.IsNullOrEmpty(_entryPoint) ? "main" : _entryPoint;
            string target      = string.IsNullOrEmpty(_targetProfile) ? "cs_6_6" : _targetProfile;

            string defines  = _defines  is { Length: > 0 } ? string.Join(";", _defines)  : null;
            string extraArgs = _extraArgs is { Length: > 0 } ? string.Join(";", _extraArgs) : null;

            bool ok = NativeRenderPlugin.ShaderCompilerPlugin.NR_SC_CompileCS(
                hlslPath, entryPoint, target, includeDirs, defines, extraArgs,
                out IntPtr nativePtr, out uint nativeSize);

            if (!ok || nativePtr == IntPtr.Zero || nativeSize == 0)
            {
                Debug.LogError($"[NativeComputeShader] Compilation failed: {hlslPath}");
                return;
            }

            _compiledDxil = new byte[nativeSize];
            Marshal.Copy(nativePtr, _compiledDxil, 0, (int)nativeSize);
            NativeRenderPlugin.ShaderCompilerPlugin.NR_SC_Free(nativePtr);

#if UNITY_EDITOR
            UnityEditor.EditorUtility.SetDirty(this);
#endif
            Debug.Log($"[NativeComputeShader] Compiled {nativeSize} bytes: {hlslPath}");
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
        // -------------------------------------------------------------------

        /// <summary>
        /// Creates (or reuses) the compute pipeline from the cached DXIL bytes.
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
                    Debug.LogError("[NativeComputeShader] CreatePipeline: cannot resolve hlsl path");
                    return false;
                }

                CompileOnly();
            }

            if (!HasCompiledBytes)
            {
                Debug.LogError($"[NativeComputeShader] CreatePipeline: compilation failed: {GetHlslPath()}");
                return false;
            }

            if (!_eventData.IsCreated)
                _eventData = new NativeArray<NativeRenderPlugin.CS_RenderEventData>(1, Allocator.Persistent);

            _handle = NativeRenderPlugin.NR_CreateComputeShader(_compiledDxil, (uint)_compiledDxil.Length);
            if (_handle == 0)
            {
                Debug.LogError($"[NativeComputeShader] CreatePipeline failed: {GetHlslPath()}");
                return false;
            }

            return true;
        }

        private void DestroyHandle()
        {
            if (_handle != 0)
            {
                GL.Flush();
                NativeRenderPlugin.NR_DestroyComputeShader(_handle);
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
            NativeRenderPlugin.NR_CS_SetBuffer(_handle, name, buffer.GetNativeBufferPtr());
        }

        /// <summary>Binds a GraphicsBuffer as a read-only structured/byte-address buffer (SRV).</summary>
        public void SetBuffer(string name, GraphicsBuffer buffer)
        {
            if (!IsValid) return;
            NativeRenderPlugin.NR_CS_SetBuffer(_handle, name, buffer.GetNativeBufferPtr());
        }

        /// <summary>Binds a ComputeBuffer as an RW (read-write) buffer (UAV).</summary>
        public void SetRWBuffer(string name, ComputeBuffer buffer)
        {
            if (!IsValid) return;
            NativeRenderPlugin.NR_CS_SetRWBuffer(_handle, name, buffer.GetNativeBufferPtr());
        }

        /// <summary>Binds a GraphicsBuffer as an RW (read-write) buffer (UAV).</summary>
        public void SetRWBuffer(string name, GraphicsBuffer buffer)
        {
            if (!IsValid) return;
            NativeRenderPlugin.NR_CS_SetRWBuffer(_handle, name, buffer.GetNativeBufferPtr());
        }

        /// <summary>Binds a Texture2D or RenderTexture as a read-only texture (SRV).</summary>
        public void SetTexture(string name, Texture texture)
        {
            if (!IsValid) return;
            NativeRenderPlugin.NR_CS_SetTexture(_handle, name, texture.GetNativeTexturePtr());
        }

        /// <summary>Binds a RenderTexture as a read-write texture (UAV).</summary>
        public void SetRWTexture(string name, RenderTexture texture)
        {
            if (!IsValid) return;
            NativeRenderPlugin.NR_CS_SetRWTexture(_handle, name, texture.GetNativeTexturePtr());
        }

        /// <summary>
        /// Binds a ComputeBuffer as a constant buffer (CBV).
        /// The buffer must have been created with ComputeBufferType.Constant.
        /// </summary>
        public void SetConstantBuffer(string name, ComputeBuffer buffer)
        {
            if (!IsValid) return;
            NativeRenderPlugin.NR_CS_SetConstantBuffer(_handle, name, buffer.GetNativeBufferPtr());
        }

        public void SetConstantBuffer(string name, GraphicsBuffer buffer)
        {
            if (!IsValid) return;
            NativeRenderPlugin.NR_CS_SetConstantBuffer(_handle, name, buffer.GetNativeBufferPtr());
        }

        /// <summary>Binds a GraphicsBuffer as a StructuredBuffer SRV (provides stride and count automatically).</summary>
        public void SetStructuredBuffer(string name, GraphicsBuffer buffer)
        {
            if (!IsValid) return;
            IntPtr ptr    = buffer != null ? buffer.GetNativeBufferPtr() : IntPtr.Zero;
            uint   count  = buffer != null ? (uint)buffer.count : 0;
            uint   stride = buffer != null ? (uint)buffer.stride : 0;
            NativeRenderPlugin.NR_CS_SetStructuredBuffer(_handle, name, ptr, count, stride);
        }

        /// <summary>Binds a ComputeBuffer as a StructuredBuffer SRV with explicit element count and stride.</summary>
        public void SetStructuredBuffer(string name, ComputeBuffer buffer, int elementCount, int elementStride)
        {
            if (!IsValid) return;
            IntPtr ptr    = buffer != null ? buffer.GetNativeBufferPtr() : IntPtr.Zero;
            uint   count  = buffer != null ? (uint)elementCount : 0;
            uint   stride = buffer != null ? (uint)elementStride : 0;
            NativeRenderPlugin.NR_CS_SetStructuredBuffer(_handle, name, ptr, count, stride);
        }

        /// <summary>Binds the TLAS of an acceleration structure by HLSL variable name (e.g. "SceneBVH").</summary>
        public void SetAccelerationStructure(string name, RayTracingAccelerationStructure accelStructure)
        {
            if (!IsValid || accelStructure == null) return;
            // Pass the AccelerationStructure handle so the native layer reads the TLAS pointer
            // dynamically at Dispatch time, after BuildOrUpdate may have rebuilt the TLAS buffer.
            NativeRenderPlugin.NR_CS_SetAccelerationStructureHandle(_handle, name, accelStructure.Handle);
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
            return NativeRenderPlugin.NR_CS_SetBindlessTexture(_handle, name, btHandle) != 0;
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
            return NativeRenderPlugin.NR_CS_SetBindlessBuffer(_handle, name, bbHandle) != 0;
        }

        // -------------------------------------------------------------------
        // Dispatch
        // -------------------------------------------------------------------

        /// <summary>
        /// Enqueues a Dispatch call into the CommandBuffer.
        /// Must be called during a URP/HDRP render pass (on the render thread).
        /// </summary>
        public void Dispatch(CommandBuffer cmd, uint threadGroupX, uint threadGroupY, uint threadGroupZ)
        {
            if (!IsValid)
                return;

            var ed = _eventData[0];
            ed.shaderHandle = _handle;
            ed.threadGroupX = threadGroupX;
            ed.threadGroupY = threadGroupY;
            ed.threadGroupZ = threadGroupZ;
            _eventData[0]   = ed;

            unsafe
            {
                cmd.IssuePluginEventAndData(
                    NativeRenderPlugin.NR_CS_GetRenderEventFunc(),
                    1,
                    (IntPtr)_eventData.GetUnsafePtr());
            }
        }
    }
}
