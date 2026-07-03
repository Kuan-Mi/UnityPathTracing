using System;
using System.Collections.Generic;
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

    /// <summary>Texture filtering mode for a <see cref="SamplerHint"/>. Order matches the native parser.</summary>
    public enum SamplerFilter
    {
        Point       = 0,
        Linear      = 1,
        Anisotropic = 2
    }

    /// <summary>Texture address (wrap) mode for a <see cref="SamplerHint"/>. Order matches the native parser.</summary>
    public enum SamplerAddress
    {
        Wrap       = 0,
        Clamp      = 1,
        Mirror     = 2,
        MirrorOnce = 3,
        Border     = 4
    }

    /// <summary>
    /// Overrides the static-sampler attributes for one HLSL sampler, replacing the
    /// name-inference convention (sampler_LinearClamp, …) used by the native plugin.
    /// Authored on the shader importer and serialized into the pipeline hints JSON.
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
        private RootConstantsHint[] _rootConstantsHints; // may be null
        private string[]            _rootSRVHints; // may be null
        private SamplerHint[]       _samplerHints; // from shader asset; may be null
        private NativeBindingLayout _userLayout;   // hand-authored layout; null = auto-generate
        private NativeBindingLayout _layout;       // the effective (mandatory) layout

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

        public NativeComputePipeline(NativeComputeShader shader, RootConstantsHint[] rootConstantsHints,
            string[] rootSRVHints, NativeBindingLayout sharedLayout)
        {
            if (shader == null)
                throw new ArgumentNullException(nameof(shader));

            _shader             = shader;
            _rootConstantsHints = rootConstantsHints;
            _rootSRVHints       = rootSRVHints;
            _samplerHints       = shader.ResolveSamplerHints();
            _userLayout         = (sharedLayout != null && !sharedLayout.IsEmpty) ? sharedLayout : null;
            BuildBindingContract(shader);
            BuildNativeHandle(shader);
            NativeComputeShader.OnRecompiled += OnShaderRecompiled;
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
                    $"[NativeComputePipeline] '{shader.name}': binding layout has no item for " +
                    string.Join(", ", missing));
        }

        private void BuildNativeHandle(NativeComputeShader shader)
        {
            byte[] dxil = shader.GetOrCompileDxil();
            if (dxil == null || dxil.Length == 0)
                throw new InvalidOperationException(
                    $"[NativeComputePipeline] Shader compilation failed for: {shader.GetHlslPath()}");

            string layoutJson = _layout.BuildCreationJson();
            _handle = NativeRenderPlugin.NR_CreateComputeShaderEx(dxil, (uint)dxil.Length, shader.name, layoutJson);

            if (_handle == 0)
                throw new InvalidOperationException(
                    $"[NativeComputePipeline] NR_CreateComputeShaderEx returned 0 for: {shader.name}");
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

                NativeRenderPlugin.NR_DestroyComputeShader(_handle);
                _handle = 0;
            }

            try
            {
                // Re-derive the contract: an auto-generated layout must track the
                // recompiled shader's bindings; a hand-authored one is revalidated.
                BuildBindingContract(shader);
                BuildNativeHandle(shader);
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
                NativeRenderPlugin.NR_DestroyComputeShader(_handle);
                _handle = 0;
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
