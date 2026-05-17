using System;
using System.IO;
using System.Runtime.InteropServices;
using UnityEngine;

namespace NativeRender
{
    /// <summary>
    /// A DXIL library asset that contains only <b>ClosestHit</b> and/or <b>AnyHit</b> shaders.
    /// Used as an extra hit-group blob when building a multi-blob DXR pipeline via
    /// <see cref="RayTracePipeline(RayTraceShader, HitGroupShader[])"/>.
    ///
    /// Unlike <see cref="RayTraceShader"/>, this asset has no <c>rayGenName</c> or
    /// <c>maxPayloadSizeInBytes</c> because those are owned by the primary pipeline shader.
    ///
    /// Import a <c>.hitgroupshader</c> HLSL file to create one.
    /// </summary>
    public class HitGroupShader : ScriptableObject
    {
        /// <summary>Additional #include search directories. Written by the ScriptedImporter.</summary>
        [SerializeField, HideInInspector]
        private string[] additionalIncludePaths = Array.Empty<string>();

        /// <summary>Additional DXC compiler arguments. Written by the ScriptedImporter.</summary>
        [SerializeField, HideInInspector]
        private string[] _extraArgs = Array.Empty<string>();

        /// <summary>Preprocessor defines (e.g. "FOO=1", "BAR"). Written by the ScriptedImporter.</summary>
        [SerializeField, HideInInspector]
        private string[] _defines = Array.Empty<string>();

        /// <summary>DXC target profile (e.g. "lib_6_6", "lib_6_9"). Written by the ScriptedImporter.</summary>
        [SerializeField, HideInInspector]
        private string _targetProfile = "lib_6_6";

        /// <summary>Pre-compiled DXIL bytecode. Populated by EnsureCompiled(); persisted by Unity serialization.</summary>
        [SerializeField, HideInInspector]
        private byte[] _compiledDxil;

        /// <summary>JSON reflection data produced after each successful compilation.</summary>
        [SerializeField, HideInInspector]
        private string _reflectionJson = "";

        /// <summary>True when compiled DXIL bytes are available.</summary>
        public bool HasCompiledBytes => _compiledDxil is { Length: > 0 };

        /// <summary>DXC target profile (e.g. "lib_6_6", "lib_6_9").</summary>
        public string TargetProfile => string.IsNullOrEmpty(_targetProfile) ? "lib_6_6" : _targetProfile;

        /// <summary>JSON reflection data produced after the last successful compilation.</summary>
        public string ReflectionJson => _reflectionJson ?? "";

        /// <summary>Size in bytes of the cached DXIL bytecode, or 0 if not compiled.</summary>
        public int CompiledByteCount => _compiledDxil?.Length ?? 0;

        /// <summary>
        /// Fired after this asset has been successfully (re)compiled.
        /// Subscribe in <see cref="RayTracePipeline"/> to trigger a hot-reload.
        /// </summary>
        public static event Action<HitGroupShader> OnRecompiled;

        /// <summary>Allows the editor AssetPostprocessor to fire <see cref="OnRecompiled"/>.</summary>
        public static void InvokeOnRecompiled(HitGroupShader shader) => OnRecompiled?.Invoke(shader);

        // -------------------------------------------------------------------
        // Compilation
        // -------------------------------------------------------------------

        /// <summary>Compiles the HLSL to DXIL bytes and caches them. No-op if already compiled.</summary>
        /// <param name="hlslPath">Absolute path to the HLSL file. If null, resolved via AssetDatabase.</param>
        internal void EnsureCompiled(string hlslPath = null)
        {
            if (HasCompiledBytes) return;

            if (string.IsNullOrEmpty(hlslPath))
                hlslPath = GetHlslPath();
            string includeDirs = BuildIncludeDirs(hlslPath);
            string target      = string.IsNullOrEmpty(_targetProfile) ? "lib_6_6" : _targetProfile;
            string defines     = _defines   is { Length: > 0 } ? string.Join(";", _defines)   : null;
            string extraArgs   = _extraArgs is { Length: > 0 } ? string.Join(";", _extraArgs) : null;

            bool ok = NativeRenderPlugin.ShaderCompilerPlugin.NR_SC_Compile(
                hlslPath, target, includeDirs, defines, extraArgs,
                out IntPtr nativePtr, out uint nativeSize);

            if (!ok || nativePtr == IntPtr.Zero || nativeSize == 0)
            {
                Debug.LogError($"[HitGroupShader] Compilation failed: {hlslPath}");
                return;
            }

            _compiledDxil = new byte[nativeSize];
            Marshal.Copy(nativePtr, _compiledDxil, 0, (int)nativeSize);
            NativeRenderPlugin.ShaderCompilerPlugin.NR_SC_Free(nativePtr);

            _reflectionJson = "";
            if (NativeRenderPlugin.ShaderCompilerPlugin.NR_SC_ReflectLib(
                    _compiledDxil, (uint)_compiledDxil.Length, out IntPtr jsonPtr, out uint jsonLen)
                && jsonPtr != IntPtr.Zero && jsonLen > 0)
            {
                _reflectionJson = Marshal.PtrToStringAnsi(jsonPtr, (int)jsonLen);
                NativeRenderPlugin.ShaderCompilerPlugin.NR_SC_Free(jsonPtr);
            }

#if UNITY_EDITOR
            UnityEditor.EditorUtility.SetDirty(this);
#endif
            Debug.Log($"[HitGroupShader] Compiled {nativeSize} bytes: {hlslPath}");
            OnRecompiled?.Invoke(this);
        }

        /// <summary>Clears the cached DXIL bytes and recompiles.</summary>
        public void ForceRecompile(string hlslPath = null)
        {
            _compiledDxil = null;
            EnsureCompiled(hlslPath);
        }

        // -------------------------------------------------------------------
        // Internal access for RayTracePipeline
        // -------------------------------------------------------------------

        /// <summary>Returns the cached DXIL bytes, compiling if needed.</summary>
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
