using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace NativeRender
{
    /// <summary>
    /// Per-draw render-target / viewport / draw-arg description for
    /// <see cref="NativeRasterPipeline.Draw"/>. Color arrays are caller-owned and may be
    /// cached/reused between frames. <c>numRenderTargets</c> entries of
    /// <c>colorResources</c> / <c>colorFormats</c> are read.
    /// </summary>
    public struct RasterDrawDesc
    {
        public uint     numRenderTargets;
        public IntPtr[] colorResources; // ID3D12Resource* per RT (length >= numRenderTargets)
        public uint[]   colorFormats; // DXGI_FORMAT per RT (must match the pipeline state)
        public IntPtr   depthResource; // ID3D12Resource* or IntPtr.Zero
        public uint     depthFormat; // DXGI_FORMAT for the depth view
        public bool     clearColor;
        public bool     clearDepth;
        public Color    clearColorValue;
        public float    clearDepthValue;
        public Rect     viewport; // in pixels (x, y, width, height)
        public uint     vertexCount; // e.g. 3 for a fullscreen triangle
        public uint     instanceCount; // 0 => 1
        public float    blendFactor; // OMSetBlendFactor constant; only used by blendMode==4 (constant-color)
    }

    /// <summary>
    /// Manages the D3D12 graphics pipeline (PSO + root signature + slot layout) created from a
    /// <see cref="NativeRasterShader"/> asset and a <see cref="NativeRenderPlugin.RasterPipelineStateDesc"/>.
    ///
    /// Resource bindings live in <see cref="NativeRasterDescriptorSet"/>; render targets are
    /// supplied per draw via <see cref="RasterDrawDesc"/>.
    /// Lifetime: must be explicitly disposed via <see cref="Dispose"/>.
    /// </summary>
    public sealed class NativeRasterPipeline : IDisposable
    {
        private ulong                                      _handle;
        private NativeRasterShader                         _shader;
        private NativeRenderPlugin.RasterPipelineStateDesc _state;
        private RootConstantsHint[]                        _rootConstantsHints;
        private string[]                                   _rootSRVHints;
        private SamplerHint[]                              _samplerHints;
        private NativeBindingLayout                        _userLayout; // hand-authored; null = auto-generate
        private NativeBindingLayout                        _layout;     // the effective (mandatory) layout

        private Dictionary<string, uint> _nameToSlot;
        private uint                     _slotCount;

        public bool IsValid => _handle != 0;

        /// <summary>The effective binding layout (hand-authored or auto-generated).</summary>
        public NativeBindingLayout Layout => _layout;

        internal IReadOnlyDictionary<string, uint> NameToSlot => _nameToSlot;
        internal uint SlotCount => _slotCount;
        internal ulong Handle => _handle;

        internal event Action<NativeRasterPipeline> OnRebuilt;

        public NativeRasterPipeline(NativeRasterShader shader,
            NativeRenderPlugin.RasterPipelineStateDesc state)
            : this(shader, state,
                shader != null ? shader.RootConstantsHints : null,
                shader != null ? shader.RootSRVHints : null)
        {
        }

        public NativeRasterPipeline(NativeRasterShader shader,
            NativeRenderPlugin.RasterPipelineStateDesc state,
            RootConstantsHint[] rootConstantsHints,
            string[] rootSRVHints)
            : this(shader, state, rootConstantsHints, rootSRVHints, null)
        {
        }

        public NativeRasterPipeline(NativeRasterShader shader,
            NativeRenderPlugin.RasterPipelineStateDesc state,
            NativeBindingLayout sharedLayout)
            : this(shader, state,
                shader != null ? shader.RootConstantsHints : null,
                shader != null ? shader.RootSRVHints : null,
                sharedLayout)
        {
        }

        public NativeRasterPipeline(NativeRasterShader shader,
            NativeRenderPlugin.RasterPipelineStateDesc state,
            RootConstantsHint[] rootConstantsHints,
            string[] rootSRVHints,
            NativeBindingLayout sharedLayout)
        {
            if (shader == null) throw new ArgumentNullException(nameof(shader));
            _shader             = shader;
            _state              = state;
            _rootConstantsHints = rootConstantsHints;
            _rootSRVHints       = rootSRVHints;
            _samplerHints       = shader.ResolveSamplerHints();
            _userLayout         = (sharedLayout != null && !sharedLayout.IsEmpty) ? sharedLayout : null;
            BuildBindingContract();
            BuildNativeHandle();
            NativeRasterShader.OnRecompiled += OnShaderRecompiled;
        }

        /// <summary>
        /// Establishes the binding contract before the native pipeline exists:
        /// picks (or auto-generates) the layout from the merged VS+PS reflection,
        /// resolves every reflected binding into it (name → layout slot), and
        /// fails loudly when a hand-authored layout doesn't cover the shader.
        /// </summary>
        private void BuildBindingContract()
        {
            byte[] vs = _shader.GetOrCompileVsDxil();
            byte[] ps = _shader.GetOrCompilePsDxil();
            if (vs == null || vs.Length == 0 || ps == null || ps.Length == 0)
                throw new InvalidOperationException(
                    $"[NativeRasterPipeline] Shader compilation failed for: {_shader.GetHlslPath()}");

            var reflected = ShaderReflectionInfo.Parse(_shader.ReflectionJson);
            _layout = _userLayout ?? NativeBindingLayout.FromReflection(
                reflected, _rootConstantsHints, _rootSRVHints, _samplerHints);

            _slotCount  = (uint)_layout.Items.Count;
            _nameToSlot = new Dictionary<string, uint>(reflected.Count);

            List<string> missing = null;
            for (int i = 0; i < reflected.Count; i++)
            {
                var b = reflected[i];
                int slot = _layout.ResolveSlot(b);
                if (slot >= 0)
                {
                    _nameToSlot[b.Name] = (uint)slot;
                }
                else if (!(b.RegClass == BindingRegClass.Sampler && _layout.HasStaticSampler(b.Reg, b.Space)))
                {
                    (missing ??= new List<string>()).Add(
                        $"'{b.Name}' ({b.RegClass} reg {b.Reg}, space {b.Space})");
                }
            }

            if (missing != null)
                throw new InvalidOperationException(
                    $"[NativeRasterPipeline] '{_shader.name}': binding layout has no item for " +
                    string.Join(", ", missing));
        }

        private void BuildNativeHandle()
        {
            byte[] vs = _shader.GetOrCompileVsDxil();
            byte[] ps = _shader.GetOrCompilePsDxil();
            if (vs == null || vs.Length == 0 || ps == null || ps.Length == 0)
                throw new InvalidOperationException(
                    $"[NativeRasterPipeline] Shader compilation failed for: {_shader.GetHlslPath()}");

            var layoutItems = _layout.BuildNativeItems();
            var staticSamplers = _layout.BuildNativeStaticSamplers();
            _handle = NativeRenderPlugin.NR_CreateRasterShaderEx(
                vs, (uint)vs.Length, ps, (uint)ps.Length, ref _state, _shader.name,
                layoutItems, (uint)layoutItems.Length,
                staticSamplers, (uint)staticSamplers.Length);

            if (_handle == 0)
                throw new InvalidOperationException(
                    $"[NativeRasterPipeline] NR_CreateRasterShaderEx returned 0 for: {_shader.name}");
        }

        private void OnShaderRecompiled(NativeRasterShader shader)
        {
            if (shader != _shader) return;
            if (_handle != 0)
            {
                using var flushCmd = new CommandBuffer { name = "NR_GpuFlush" };
                flushCmd.IssuePluginEvent(NativeRenderPlugin.NR_GetGpuFlushEventFunc(), 0);
                Graphics.ExecuteCommandBuffer(flushCmd);
                NativeRenderPlugin.NR_WaitForGpuFlush();

                NativeRenderPlugin.NR_DestroyRasterShader(_handle);
                _handle = 0;
            }

            try
            {
                BuildBindingContract();
                BuildNativeHandle();
                Debug.Log($"[NativeRasterPipeline] Rebuilt pipeline for: {shader.name}");
                OnRebuilt?.Invoke(this);
            }
            catch (Exception e)
            {
                Debug.LogError(e.Message);
            }
        }

        public void Dispose()
        {
            NativeRasterShader.OnRecompiled -= OnShaderRecompiled;
            if (_handle != 0)
            {
                GL.Flush();
                NativeRenderPlugin.NR_DestroyRasterShader(_handle);
                _handle = 0;
            }
        }

        // -------------------------------------------------------------------
        // Draw
        // -------------------------------------------------------------------

        /// <summary>
        /// Snapshots bindings from <paramref name="descriptorSet"/> and enqueues a DrawInstanced
        /// into the CommandBuffer using the render targets / viewport in <paramref name="draw"/>.
        /// </summary>
        public void Draw(CommandBuffer cmd, NativeRasterDescriptorSet descriptorSet, in RasterDrawDesc draw)
        {
            if (!IsValid || descriptorSet == null) return;

            IntPtr headerPtr = descriptorSet.SnapshotAndBuildHeader(in draw);
            if (headerPtr == IntPtr.Zero) return;

            cmd.IssuePluginEventAndData(NativeRenderPlugin.NR_RAS_GetRenderEventFunc(), 1, headerPtr);
        }
    }
}
