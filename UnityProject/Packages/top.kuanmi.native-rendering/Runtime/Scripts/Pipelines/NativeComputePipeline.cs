using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

namespace NativeRender
{
    /// <summary>
    /// Describes a CBV binding that should be promoted to root 32-bit constants
    /// (bound via SetComputeRoot32BitConstants instead of a root CBV descriptor).
    /// </summary>
    [Serializable]
    public struct RootConstantsHint
    {
        /// <summary>The HLSL variable name of the ConstantBuffer.</summary>
        public string Name;

        /// <summary>Total number of 32-bit values in the constant buffer.</summary>
        public uint Count;
    }

    /// <summary>Texture filtering mode for a <see cref="SamplerHint"/>. Order matches the native ABI.</summary>
    public enum SamplerFilter
    {
        Point       = 0,
        Linear      = 1,
        Anisotropic = 2
    }

    /// <summary>Texture address (wrap) mode for a <see cref="SamplerHint"/>. Order matches the native ABI.</summary>
    public enum SamplerAddress
    {
        Wrap       = 0,
        Clamp      = 1,
        Mirror     = 2,
        MirrorOnce = 3,
        Border     = 4
    }

    public sealed class NativeComputePipelineDesc
    {
        private readonly List<NativeBindingLayout> _bindingLayouts = new List<NativeBindingLayout>();

        public NativeShaderHandle CS;
        public NativeComputeShader ReflectionSource;
        public string DebugName;
        public IReadOnlyList<NativeBindingLayout> BindingLayouts => _bindingLayouts;

        public NativeComputePipelineDesc SetComputeShader(
            NativeShaderHandle shader,
            NativeComputeShader reflectionSource = null)
        {
            CS = shader;
            ReflectionSource = reflectionSource;
            return this;
        }

        public NativeComputePipelineDesc SetComputeShader(NativeComputeShader shader)
        {
            ReflectionSource = shader;
            DebugName = shader != null ? shader.name : DebugName;
            return this;
        }

        public NativeComputePipelineDesc SetDebugName(string debugName)
        {
            DebugName = debugName;
            return this;
        }

        public NativeComputePipelineDesc AddBindingLayout(NativeBindingLayout layout)
        {
            if (layout != null && !layout.IsEmpty)
                _bindingLayouts.Add(layout);
            return this;
        }

        public NativeComputePipelineDesc SetBindingLayouts(params NativeBindingLayout[] layouts)
        {
            _bindingLayouts.Clear();
            if (layouts == null) return this;
            for (int i = 0; i < layouts.Length; i++)
                AddBindingLayout(layouts[i]);
            return this;
        }
    }

    /// <summary>
    /// Overrides the static-sampler attributes for one HLSL sampler, replacing the
    /// name-inference convention (sampler_LinearClamp, …) used by the native plugin.
    /// Authored on the shader importer and passed to native through the explicit binding layout.
    /// </summary>
    [Serializable]
    public struct SamplerHint
    {
        /// <summary>HLSL sampler variable name to match exactly.</summary>
        public string Name;

        public SamplerFilter Filter;

        /// <summary>Address mode for the U/V/W texture axes. Usually all three are equal;
        /// they differ only for cases like an equirectangular sampler (U=Wrap, V/W=Clamp).</summary>
        public SamplerAddress AddressU;

        public SamplerAddress AddressV;
        public SamplerAddress AddressW;

        /// <summary>Sample mip levels (MaxLOD 16) when true; clamp to mip 0 otherwise.</summary>
        public bool Mips;

        /// <summary>Max anisotropy; used only when <see cref="Filter"/> is Anisotropic.</summary>
        public uint MaxAnisotropy;
    }

    /// <summary>
    /// Manages the D3D12 compute pipeline state (PSO + root signature + slot layout)
    /// created from a <see cref="NativeComputeShader"/> asset.
    ///
    /// Binding model (nvrhi-style): every pipeline has a mandatory
    /// <see cref="NativeBindingLayout"/> — hand-authored, or auto-generated here
    /// from the shader's import-time reflection JSON. The layout is the whole
    /// binding contract: it alone shapes the native root signature, defines the
    /// dispatch slot order (slot i = layout item i), and resolves both HLSL
    /// variable names (via import-time reflection, C#-side only) and
    /// register-addressed <see cref="NativeBindingSetItem"/>s. The render plugin
    /// never reflects shaders; mismatches between a hand-authored layout and the
    /// shader's registers are caught here at pipeline build, against the cached
    /// reflection JSON.
    ///
    /// Resource bindings live in <see cref="NativeComputeDescriptorSet"/>, which is
    /// created separately and passed to <see cref="Dispatch"/>.  This decouples
    /// pipeline state from per-pass resource bindings so that multiple passes can
    /// each hold their own descriptor set while sharing the same pipeline.
    ///
    /// Lifetime: must be explicitly disposed via <see cref="Dispose"/>.
    /// </summary>
    public sealed class NativeComputePipeline : IDisposable
    {
        private ulong               _handle;
        private NativeComputeShader _shader;
        private NativeShaderHandle  _shaderHandle;
        private bool                _ownsShaderHandle;
        private RootConstantsHint[] _rootConstantsHints; // may be null
        private string[]            _rootSRVHints; // may be null
        private SamplerHint[]       _samplerHints; // from shader asset; may be null
        private readonly List<NativeBindingLayout> _userLayouts = new List<NativeBindingLayout>();
        private NativeBindingLayout _layout;       // the effective (mandatory) layout
        private NativeComputePipelineDesc _pipelineDesc;

        // Slot layout: name → layout item index, resolved from the import-time
        // reflection JSON against the layout (no native round-trips).
        private Dictionary<string, uint> _nameToSlot;
        private uint                     _slotCount;

        /// <summary>True if the underlying D3D12 pipeline is valid and ready to dispatch.</summary>
        public bool IsValid => _handle != 0;

        /// <summary>The effective binding layout (hand-authored or auto-generated).</summary>
        public NativeBindingLayout Layout => _layout;

        // Internal access for NativeComputeDescriptorSet
        internal IReadOnlyDictionary<string, uint> NameToSlot => _nameToSlot;
        internal uint SlotCount => _slotCount;
        internal ulong Handle => _handle;

        /// <summary>
        /// Fired after a hot-reload successfully rebuilds the native pipeline.
        /// <see cref="NativeComputeDescriptorSet"/> subscribes to reallocate its ring buffers.
        /// </summary>
        internal event Action<NativeComputePipeline> OnRebuilt;

        // -------------------------------------------------------------------
        // Construction
        // -------------------------------------------------------------------

        /// <summary>
        /// Creates a new compute pipeline from the given shader asset.
        /// Root constants hints defined on the asset (via the importer) are applied automatically.
        /// Triggers HLSL compilation if the asset has not been compiled yet.
        /// Throws <see cref="InvalidOperationException"/> if pipeline creation fails.
        /// </summary>
        public NativeComputePipeline(NativeComputeShader shader)
            : this(shader,
                shader != null ? shader.RootConstantsHints : null,
                shader != null ? shader.RootSRVHints : null)
        {
        }

        /// <summary>
        /// Creates a new compute pipeline, promoting the specified CBV bindings to
        /// root 32-bit constants (SetComputeRoot32BitConstants).
        /// Must be called before any <see cref="NativeComputeDescriptorSet"/> is created
        /// for this pipeline.
        /// </summary>
        public NativeComputePipeline(NativeComputeShader shader, RootConstantsHint[] rootConstantsHints)
            : this(shader, rootConstantsHints, null)
        {
        }

        /// <summary>
        /// Creates a new compute pipeline with both root constants and root SRV hints.
        /// <paramref name="rootSRVHints"/> names buffer SRV / TLAS bindings to promote to
        /// inline root descriptors (SetComputeRootShaderResourceView) instead of a
        /// descriptor-table entry, reducing shader execution overhead.
        /// </summary>
        public NativeComputePipeline(NativeComputeShader shader, RootConstantsHint[] rootConstantsHints, string[] rootSRVHints)
            : this(shader, rootConstantsHints, rootSRVHints, null)
        {
        }

        public NativeComputePipeline(NativeComputeShader shader, NativeBindingLayout sharedLayout)
            : this(shader,
                shader != null ? shader.RootConstantsHints : null,
                shader != null ? shader.RootSRVHints : null,
                sharedLayout)
        {
        }

        public NativeComputePipeline(NativeComputeShader shader, BindingLayoutDesc[] bindingLayouts)
            : this(shader,
                shader != null ? shader.RootConstantsHints : null,
                shader != null ? shader.RootSRVHints : null,
                NativeBindingLayout.FromDescs(bindingLayouts))
        {
        }

        public NativeComputePipeline(NativeComputeShader shader, RootConstantsHint[] rootConstantsHints,
            string[] rootSRVHints, NativeBindingLayout sharedLayout)
            : this(BuildCompatDesc(shader, sharedLayout), rootConstantsHints, rootSRVHints)
        {
        }

        public NativeComputePipeline(NativeComputePipelineDesc pipelineDesc)
            : this(pipelineDesc,
                pipelineDesc != null && pipelineDesc.ReflectionSource != null
                    ? pipelineDesc.ReflectionSource.RootConstantsHints
                    : null,
                pipelineDesc != null && pipelineDesc.ReflectionSource != null
                    ? pipelineDesc.ReflectionSource.RootSRVHints
                    : null)
        {
        }

        private NativeComputePipeline(NativeComputePipelineDesc pipelineDesc,
            RootConstantsHint[] rootConstantsHints, string[] rootSRVHints)
        {
            if (pipelineDesc == null)
                throw new ArgumentNullException(nameof(pipelineDesc));
            if (pipelineDesc.CS == null && pipelineDesc.ReflectionSource == null)
                throw new ArgumentException("PipelineDesc must provide CS or ReflectionSource.", nameof(pipelineDesc));

            _pipelineDesc       = pipelineDesc;
            _shader             = pipelineDesc.ReflectionSource;
            _rootConstantsHints = rootConstantsHints;
            _rootSRVHints       = rootSRVHints;
            _samplerHints       = _shader != null ? _shader.ResolveSamplerHints() : null;
            for (int i = 0; i < pipelineDesc.BindingLayouts.Count; i++)
                if (pipelineDesc.BindingLayouts[i] != null && !pipelineDesc.BindingLayouts[i].IsEmpty)
                    _userLayouts.Add(pipelineDesc.BindingLayouts[i]);

            if (_shader == null)
                throw new InvalidOperationException(
                    "NativeComputePipelineDesc needs ReflectionSource for C# binding-name validation.");

            BuildBindingContract(_shader);
            BuildShaderHandle(_shader);
            BuildNativeHandle();
            NativeComputeShader.OnRecompiled += OnShaderRecompiled;
        }

        private static NativeComputePipelineDesc BuildCompatDesc(
            NativeComputeShader shader, NativeBindingLayout sharedLayout)
        {
            var desc = new NativeComputePipelineDesc()
                .SetComputeShader(shader);
            if (sharedLayout != null && !sharedLayout.IsEmpty)
                desc.AddBindingLayout(sharedLayout);
            return desc;
        }

        /// <summary>
        /// Establishes the binding contract before any native object exists:
        /// picks (or auto-generates) the layout, resolves every reflected HLSL
        /// binding into it (name → layout slot), and fails loudly — naming the
        /// HLSL variables — when a hand-authored layout doesn't cover the shader.
        /// This is the C#-side safety net replacing the plugin's old runtime
        /// reflection (nvrhi itself has no such check).
        /// </summary>
        private void BuildBindingContract(NativeComputeShader shader)
        {
            // Compile first so ReflectionJson is available for freshly imported assets.
            byte[] dxil = shader.GetOrCompileDxil();
            if (dxil == null || dxil.Length == 0)
                throw new InvalidOperationException(
                    $"[NativeComputePipeline] Shader compilation failed for: {shader.GetHlslPath()}");

            var reflected = ShaderReflectionInfo.Parse(shader.ReflectionJson);
            _layout = _userLayouts.Count != 0
                ? CombineLayouts(_userLayouts)
                : NativeBindingLayout.FromReflection(
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
                    $"[NativeComputePipeline] '{shader.name}': binding layout has no item for " +
                    string.Join(", ", missing));
        }

        private static NativeBindingLayout CombineLayouts(IReadOnlyList<NativeBindingLayout> layouts)
        {
            if (layouts.Count == 1)
                return layouts[0];

            var combined = new NativeBindingLayout();
            for (int i = 0; i < layouts.Count; i++)
            {
                foreach (var item in layouts[i].Items)
                {
                    combined.Add(item.Kind, item.Slot, item.Space, item.Count,
                        item.Num32BitValues, item.Visibility, item.BindlessLayoutIndex);
                }
                foreach (var sampler in layouts[i].StaticSamplers)
                {
                    combined.StaticSampler(sampler.Slot, sampler.Filter, sampler.AddressU,
                        sampler.AddressV, sampler.AddressW, sampler.Mips,
                        sampler.MaxAnisotropy, sampler.Space);
                }
            }
            return combined;
        }

        private void BuildShaderHandle(NativeComputeShader shader)
        {
            if (_pipelineDesc.CS != null)
            {
                _shaderHandle = _pipelineDesc.CS;
                _ownsShaderHandle = false;
                return;
            }

            _shaderHandle = NativeShaderHandle.FromComputeShader(shader);
            _ownsShaderHandle = true;
        }

        private void BuildNativeHandle()
        {
            if (_shaderHandle == null || !_shaderHandle.IsValid)
                throw new InvalidOperationException("[NativeComputePipeline] PipelineDesc.CS is invalid.");

            var layoutHandles = GetPipelineLayoutHandles();
            var pin = layoutHandles != null && layoutHandles.Length != 0
                ? GCHandle.Alloc(layoutHandles, GCHandleType.Pinned)
                : default;
            try
            {
                var desc = new NativeRenderPlugin.NR_ComputePipelineDesc
                {
                    CS = _shaderHandle.Handle,
                    bindingLayouts = pin.IsAllocated ? pin.AddrOfPinnedObject() : IntPtr.Zero,
                    bindingLayoutCount = (uint)(layoutHandles?.Length ?? 0),
                    debugName = string.IsNullOrEmpty(_pipelineDesc.DebugName)
                        ? _shader.name
                        : _pipelineDesc.DebugName
                };
                _handle = NativeRenderPlugin.NR_CreateComputePipeline(ref desc);
            }
            finally
            {
                if (pin.IsAllocated)
                    pin.Free();
            }

            if (_handle == 0)
                throw new InvalidOperationException(
                    $"[NativeComputePipeline] NR_CreateComputePipeline returned 0 for: {_shader.name}");
        }

        private ulong[] GetPipelineLayoutHandles()
        {
            if (_userLayouts.Count == 0)
                return _layout.GetNativeLayoutHandles();

            var handles = new List<ulong>(_userLayouts.Count);
            for (int i = 0; i < _userLayouts.Count; i++)
            {
                var h = _userLayouts[i].GetNativeLayoutHandles();
                if (h == null) continue;
                handles.AddRange(h);
            }
            return handles.ToArray();
        }

        private void OnShaderRecompiled(NativeComputeShader shader)
        {
            if (shader != _shader) return;

            if (_handle != 0)
            {
                // Issue a render-thread flush event so all in-flight CsDispatchCallbacks
                // finish executing before we free ring-buffer NativeArrays.
                // GL.Flush() alone only queues commands — it does NOT wait for execution,
                // causing a use-after-free when FreeRingBuffers() runs immediately after.
                using var flushCmd = new CommandBuffer { name = "NR_GpuFlush" };
                flushCmd.IssuePluginEvent(NativeRenderPlugin.NR_GetGpuFlushEventFunc(), 0);
                Graphics.ExecuteCommandBuffer(flushCmd);
                // Block main thread until render thread + GPU are fully idle.
                NativeRenderPlugin.NR_WaitForGpuFlush();

                NativeRenderPlugin.NR_DestroyComputePipeline(_handle);
                _handle = 0;
            }
            if (_ownsShaderHandle && _shaderHandle != null)
            {
                _shaderHandle.Dispose();
                _shaderHandle = null;
                _ownsShaderHandle = false;
            }

            try
            {
                // Re-derive the contract: an auto-generated layout must track the
                // recompiled shader's bindings; a hand-authored one is revalidated.
                BuildBindingContract(shader);
                BuildShaderHandle(shader);
                BuildNativeHandle();
                Debug.Log($"[NativeComputePipeline] Rebuilt pipeline for: {shader.name}");
                OnRebuilt?.Invoke(this);
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
            NativeComputeShader.OnRecompiled -= OnShaderRecompiled;

            if (_handle != 0)
            {
                GL.Flush();
                NativeRenderPlugin.NR_DestroyComputePipeline(_handle);
                _handle = 0;
            }
            if (_ownsShaderHandle && _shaderHandle != null)
            {
                _shaderHandle.Dispose();
                _shaderHandle = null;
                _ownsShaderHandle = false;
            }
        }

        // -------------------------------------------------------------------
        // Dispatch
        // -------------------------------------------------------------------

        /// <summary>
        /// Snapshots bindings from <paramref name="descriptorSet"/> into its ring buffer
        /// and enqueues a Dispatch call into the CommandBuffer.
        /// Safe to call multiple times per frame with different descriptor sets.
        /// </summary>
        public void Dispatch(CommandBuffer cmd, NativeComputeDescriptorSet descriptorSet,
            uint threadGroupX, uint threadGroupY, uint threadGroupZ)
        {
            if (!IsValid || descriptorSet == null) return;

            IntPtr headerPtr = descriptorSet.SnapshotAndBuildHeader(
                threadGroupX, threadGroupY, threadGroupZ);
            if (headerPtr == IntPtr.Zero) return;

            cmd.IssuePluginEventAndData(
                NativeRenderPlugin.NR_CS_GetRenderEventFunc(),
                1,
                headerPtr);
        }

    }
}
