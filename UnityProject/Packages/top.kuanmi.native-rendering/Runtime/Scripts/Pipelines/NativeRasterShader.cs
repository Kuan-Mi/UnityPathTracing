using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;

namespace NativeRender
{
    /// <summary>
    /// Native raster shader asset. Import a <c>.rastershader</c> HLSL file to create one.
    /// A single HLSL source provides BOTH a vertex entry point and a pixel entry point
    /// (compiled as <c>vs_6_x</c> / <c>ps_6_x</c>); the asset manages compilation and DXIL
    /// caching for both stages.
    ///
    /// To draw, create a <see cref="NativeRasterPipeline"/> from this asset (plus a
    /// <see cref="NativeRenderPlugin.RasterPipelineStateDesc"/> describing the render-target
    /// formats and fixed-function state). Resource bindings live in a
    /// <see cref="NativeRasterDescriptorSet"/>.
    /// </summary>
    public class NativeRasterShader : ScriptableObject
    {
        [SerializeField, HideInInspector] private string[] additionalIncludePaths = Array.Empty<string>();
        [SerializeField, HideInInspector] private string[] _extraArgs = Array.Empty<string>();
        [SerializeField, HideInInspector] private string[] _defines   = Array.Empty<string>();

        [SerializeField, HideInInspector] private string _vsEntryPoint = "VSMain";
        [SerializeField, HideInInspector] private string _psEntryPoint = "PSMain";
        [SerializeField, HideInInspector] private string _vsTargetProfile = "vs_6_6";
        [SerializeField, HideInInspector] private string _psTargetProfile = "ps_6_6";

        [SerializeField, HideInInspector] private RootConstantsHint[] _rootConstantsHints = Array.Empty<RootConstantsHint>();
        [SerializeField, HideInInspector] private string[]            _rootSRVHints       = Array.Empty<string>();
        [SerializeField, HideInInspector] private SamplerBinding[]    _samplerBindings    = Array.Empty<SamplerBinding>();

        [SerializeField, HideInInspector] private byte[] _vsDxil;
        [SerializeField, HideInInspector] private byte[] _psDxil;
        [SerializeField, HideInInspector] private string _reflectionJson = "";

        public bool HasCompiledBytes => _vsDxil is { Length: > 0 } && _psDxil is { Length: > 0 };
        public int  CompiledByteCount => (_vsDxil?.Length ?? 0) + (_psDxil?.Length ?? 0);
        public string ReflectionJson => _reflectionJson ?? "";

        /// <summary>The DXIL container shader hash (32-char hex) of the vertex-stage blob, or "".</summary>
        public string VsShaderHash => DxilContainerUtil.ExtractHashHex(_vsDxil);

        /// <summary>The DXIL container shader hash (32-char hex) of the pixel-stage blob, or "".</summary>
        public string PsShaderHash => DxilContainerUtil.ExtractHashHex(_psDxil);

        /// <summary>Combined per-stage shader hashes ("VS &lt;hash&gt;  PS &lt;hash&gt;"), or "" if not
        /// compiled. Each matches the hash PIX / RenderDoc display for that stage.</summary>
        public string ShaderHash
        {
            get
            {
                string vs = VsShaderHash, ps = PsShaderHash;
                if (vs.Length == 0 && ps.Length == 0) return "";
                return $"VS {vs}  PS {ps}";
            }
        }

        internal RootConstantsHint[] RootConstantsHints => _rootConstantsHints ?? Array.Empty<RootConstantsHint>();
        internal string[]            RootSRVHints       => _rootSRVHints       ?? Array.Empty<string>();
        internal SamplerBinding[]    SamplerBindings    => _samplerBindings    ?? Array.Empty<SamplerBinding>();
        internal SamplerHint[]       ResolveSamplerHints() => SamplerBindingResolver.Resolve(_samplerBindings);

        public static event Action<NativeRasterShader> OnRecompiled;
        public static void InvokeOnRecompiled(NativeRasterShader shader) => OnRecompiled?.Invoke(shader);

        // -------------------------------------------------------------------
        // Compilation
        // -------------------------------------------------------------------

        internal void EnsureCompiled(string hlslPath = null)
        {
            if (HasCompiledBytes) return;

            if (string.IsNullOrEmpty(hlslPath))
                hlslPath = GetHlslPath();
            string includeDirs = BuildIncludeDirs(hlslPath);
            string defines     = _defines   is { Length: > 0 } ? string.Join(";", _defines)   : null;
            string extraArgs   = _extraArgs is { Length: > 0 } ? string.Join(";", _extraArgs) : null;

            string vsEntry = string.IsNullOrEmpty(_vsEntryPoint) ? "VSMain" : _vsEntryPoint;
            string psEntry = string.IsNullOrEmpty(_psEntryPoint) ? "PSMain" : _psEntryPoint;
            string vsTgt   = string.IsNullOrEmpty(_vsTargetProfile) ? "vs_6_6" : _vsTargetProfile;
            string psTgt   = string.IsNullOrEmpty(_psTargetProfile) ? "ps_6_6" : _psTargetProfile;

            // A per-stage define lets a single source #include the stage-specific original file
            // (e.g. donut fullscreen_vs.hlsl for the VS, luminance_ps.hlsl for the PS) without the two
            // 'main' entry points colliding — the body guards each include with #ifdef NR_VERTEX_SHADER
            // / NR_PIXEL_SHADER. This is how the ToneMapper assets become thin verbatim include wrappers.
            string vsDefines = AppendDefine(defines, "NR_VERTEX_SHADER=1");
            string psDefines = AppendDefine(defines, "NR_PIXEL_SHADER=1");

            _vsDxil = CompileStage(hlslPath, vsEntry, vsTgt, includeDirs, vsDefines, extraArgs);
            _psDxil = CompileStage(hlslPath, psEntry, psTgt, includeDirs, psDefines, extraArgs);
            if (!HasCompiledBytes)
            {
                Debug.LogError($"[NativeRasterShader] Compilation failed (VS or PS): {hlslPath}", this);
                return;
            }

            // Merge reflected binding names from both stages so the slot layout the
            // pipeline builds covers VS-only and PS-only resources.
            _reflectionJson = MergeReflection(_vsDxil, _psDxil);

#if UNITY_EDITOR
            UnityEditor.EditorUtility.SetDirty(this);
#endif
            Debug.Log($"[NativeRasterShader] Compiled VS {_vsDxil.Length}B + PS {_psDxil.Length}B: {hlslPath}");
            OnRecompiled?.Invoke(this);
        }

        private static string AppendDefine(string defines, string extra)
            => string.IsNullOrEmpty(defines) ? extra : defines + ";" + extra;

        private byte[] CompileStage(string hlslPath, string entry, string target,
                                    string includeDirs, string defines, string extraArgs)
        {
            bool ok = NativeRenderPlugin.ShaderCompilerPlugin.NR_SC_CompileCS(
                hlslPath, entry, target, includeDirs, defines, extraArgs,
                NativeRenderPlugin.ShaderCompilerPlugin.PdbOutputDir,
                out IntPtr nativePtr, out uint nativeSize);
            if (!ok || nativePtr == IntPtr.Zero || nativeSize == 0)
            {
                Debug.LogError($"[NativeRasterShader] {target} '{entry}' compile failed: {hlslPath}", this);
                return null;
            }
            var bytes = new byte[nativeSize];
            Marshal.Copy(nativePtr, bytes, 0, (int)nativeSize);
            NativeRenderPlugin.ShaderCompilerPlugin.NR_SC_Free(nativePtr);
            return bytes;
        }

        private static string MergeReflection(byte[] vs, byte[] ps)
        {
            string vsJson = ReflectStage(vs);
            string psJson = ReflectStage(ps);
            // Concatenate the two "bindings" arrays into one document. Duplicate names
            // are harmless: the pipeline's slot map resolves each via NR_RAS_GetSlotIndex.
            string vsArr = ExtractBindingsArrayBody(vsJson);
            string psArr = ExtractBindingsArrayBody(psJson);
            var sb = new StringBuilder();
            sb.Append("{\"bindings\":[");
            sb.Append(vsArr);
            if (vsArr.Length > 0 && psArr.Length > 0) sb.Append(',');
            sb.Append(psArr);
            sb.Append("]}");
            return sb.ToString();
        }

        private static string ReflectStage(byte[] dxil)
        {
            if (dxil == null || dxil.Length == 0) return "";
            if (NativeRenderPlugin.ShaderCompilerPlugin.NR_SC_ReflectCS(
                    dxil, (uint)dxil.Length, out IntPtr jsonPtr, out uint jsonLen)
                && jsonPtr != IntPtr.Zero && jsonLen > 0)
            {
                string s = Marshal.PtrToStringAnsi(jsonPtr, (int)jsonLen);
                NativeRenderPlugin.ShaderCompilerPlugin.NR_SC_Free(jsonPtr);
                return s;
            }
            return "";
        }

        // Returns the inner contents of the top-level "bindings":[ ... ] array (without brackets).
        private static string ExtractBindingsArrayBody(string json)
        {
            if (string.IsNullOrEmpty(json)) return "";
            int idx = json.IndexOf("\"bindings\"", StringComparison.Ordinal);
            if (idx < 0) return "";
            int open = json.IndexOf('[', idx);
            if (open < 0) return "";
            int close = json.LastIndexOf(']');
            if (close <= open) return "";
            return json.Substring(open + 1, close - open - 1).Trim();
        }

        public void ForceRecompile(string hlslPath = null)
        {
            _vsDxil = null;
            _psDxil = null;
            EnsureCompiled(hlslPath);
        }

        // -------------------------------------------------------------------
        // Internal access for NativeRasterPipeline
        // -------------------------------------------------------------------

        internal byte[] GetOrCompileVsDxil() { EnsureCompiled(); return _vsDxil; }
        internal byte[] GetOrCompilePsDxil() { EnsureCompiled(); return _psDxil; }

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
