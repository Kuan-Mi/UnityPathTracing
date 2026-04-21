using System;
using System.IO;
using System.Runtime.InteropServices;
using UnityEngine;

namespace NativeRender
{
    /// <summary>
    /// DXR ray tracing shader asset. Import a <c>.rayshader</c> HLSL file to create one.
    /// This asset only manages HLSL compilation and DXIL bytecode caching.
    ///
    /// To dispatch rays, create a <see cref="RayTracePipeline"/> from this asset.
    /// Multiple pipelines can be created from the same shader, each with independent resource bindings.
    /// </summary>
    public class RayTraceShader : ScriptableObject
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

        /// <summary>Pre-compiled DXIL bytecode. Populated by EnsureCompiled(); persisted by Unity serialization.</summary>
        [SerializeField, HideInInspector]
        private byte[] _compiledDxil;

        /// <summary>True when compiled DXIL bytes are available.</summary>
        public bool HasCompiledBytes => _compiledDxil is { Length: > 0 };

        /// <summary>Size in bytes of the cached DXIL bytecode, or 0 if not compiled.</summary>
        public int CompiledByteCount => _compiledDxil?.Length ?? 0;

        // -------------------------------------------------------------------
        // ScriptableObject lifecycle
        // -------------------------------------------------------------------

        private void OnEnable()
        {
            // if (!string.IsNullOrEmpty(GetHlslPath()))
            //     EnsureCompiled();
        }

        // -------------------------------------------------------------------
        // Compilation  (ShaderCompilerPlugin — no D3D12 needed)
        // -------------------------------------------------------------------

        /// <summary>
        /// Compiles the HLSL to DXIL bytes and caches them in <see cref="_compiledDxil"/>.
        /// If bytes are already present this is a no-op; call <see cref="ForceRecompile"/>
        /// to discard the cache and recompile.
        /// </summary>
        internal void EnsureCompiled()
        {
            if (HasCompiledBytes) return;

            string hlslPath    = GetHlslPath();
            string includeDirs = BuildIncludeDirs(hlslPath);

            string defines   = _defines   is { Length: > 0 } ? string.Join(";", _defines)   : null;
            string extraArgs = _extraArgs is { Length: > 0 } ? string.Join(";", _extraArgs) : null;

            bool ok = NativeRenderPlugin.ShaderCompilerPlugin.NR_SC_Compile(
                hlslPath, includeDirs, defines, extraArgs,
                out IntPtr nativePtr, out uint nativeSize);

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
        /// Clears the cached DXIL bytes and recompiles.
        /// Any existing <see cref="RayTracePipeline"/> instances created from this asset
        /// are unaffected; new pipelines will use the freshly compiled bytes.
        /// </summary>
        [ContextMenu("Recompile")]
        public void ForceRecompile()
        {
            _compiledDxil = null;
            EnsureCompiled();
        }

        // -------------------------------------------------------------------
        // Internal access for RayTracePipeline
        // -------------------------------------------------------------------

        /// <summary>Returns the cached DXIL bytes. Triggers compilation if needed.</summary>
        internal byte[] GetOrCompileDxil()
        {
            EnsureCompiled();
            return _compiledDxil;
        }

        internal string GetHlslPath()
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
    }
}