using System;
using System.Collections.Generic;

namespace NativeRender
{
    public enum NativeBindingLayoutKind : uint
    {
        SRV = 0,
        UAV = 1,
        CBV = 2,
        VolatileCBV = 3,
        PushConstants = 4,
        TLAS = 5,
        BindlessSRV = 6,
        BindlessUAV = 7,
        RootSRV = 8,
        Sampler = 9
    }

    [Serializable]
    public struct NativeBindingLayoutItem
    {
        public NativeBindingLayoutKind Kind;
        public uint Slot;
        public uint Space;
        public uint Count;
        public uint Num32BitValues;

        /// <summary>Bindless items only: share the previous bindless item's root
        /// parameter (one descriptor table aliased by several unbounded ranges —
        /// nvrhi's bindless layout with multiple register spaces, e.g. donut's
        /// shared table). When false the item gets its own root parameter.</summary>
        public bool GroupWithPrevious;
    }

    /// <summary>Explicit static-sampler definition serialized with the layout.
    /// Registers and config are fully resolved on the C# side (import-time
    /// reflection + editor hints / naming convention); the render plugin just
    /// places them into the root signature.</summary>
    [Serializable]
    public struct NativeStaticSampler
    {
        public uint Slot;
        public uint Space;
        public SamplerFilter Filter;
        public SamplerAddress AddressU;
        public SamplerAddress AddressV;
        public SamplerAddress AddressW;
        public bool Mips;
        public uint MaxAnisotropy;
    }

    /// <summary>
    /// The nvrhi-style binding layout (mirrors <c>nvrhi::BindingLayoutDesc</c>):
    /// an ordered item list that IS the whole binding contract between HLSL
    /// register assignments and the native root signature. The render plugin
    /// never reflects shaders — it builds the root signature and its per-slot
    /// binding table from this declaration alone, and the per-dispatch
    /// <c>BindingSlot[]</c> payload is ordered by layout item index.
    ///
    /// A layout is MANDATORY for every pipeline. Author one by hand for
    /// RTXPT-parity passes (item order defines the descriptor-table layout) or
    /// let the pipeline auto-generate one from the import-time reflection JSON
    /// via <see cref="FromReflection"/>.
    /// </summary>
    public sealed class NativeBindingLayout
    {
        private readonly List<NativeBindingLayoutItem> _items = new List<NativeBindingLayoutItem>();
        private readonly List<NativeStaticSampler> _staticSamplers = new List<NativeStaticSampler>();

        public IReadOnlyList<NativeBindingLayoutItem> Items => _items;
        public IReadOnlyList<NativeStaticSampler> StaticSamplers => _staticSamplers;
        public bool IsEmpty => _items.Count == 0 && _staticSamplers.Count == 0;

        public NativeBindingLayout Add(NativeBindingLayoutKind kind, uint slot, uint space = 0,
            uint count = 1, uint num32BitValues = 0, bool groupWithPrevious = false)
        {
            _items.Add(new NativeBindingLayoutItem
            {
                Kind = kind,
                Slot = slot,
                Space = space,
                Count = count == 0 ? 1 : count,
                Num32BitValues = num32BitValues,
                GroupWithPrevious = groupWithPrevious
            });
            return this;
        }

        public NativeBindingLayout StructuredBufferSRV(uint slot, uint space = 0, uint count = 1)
            => Add(NativeBindingLayoutKind.SRV, slot, space, count);

        public NativeBindingLayout TextureSRV(uint slot, uint space = 0, uint count = 1)
            => Add(NativeBindingLayoutKind.SRV, slot, space, count);

        public NativeBindingLayout RayTracingAccelStruct(uint slot, uint space = 0)
            => Add(NativeBindingLayoutKind.TLAS, slot, space);

        public NativeBindingLayout StructuredBufferUAV(uint slot, uint space = 0, uint count = 1)
            => Add(NativeBindingLayoutKind.UAV, slot, space, count);

        public NativeBindingLayout TextureUAV(uint slot, uint space = 0, uint count = 1)
            => Add(NativeBindingLayoutKind.UAV, slot, space, count);

        public NativeBindingLayout ConstantBuffer(uint slot, uint space = 0)
            => Add(NativeBindingLayoutKind.CBV, slot, space);

        public NativeBindingLayout VolatileConstantBuffer(uint slot, uint space = 0)
            => Add(NativeBindingLayoutKind.VolatileCBV, slot, space);

        public NativeBindingLayout PushConstants(uint slot, uint num32BitValues, uint space = 0)
            => Add(NativeBindingLayoutKind.PushConstants, slot, space,
                count: num32BitValues * 4, num32BitValues: num32BitValues);

        public NativeBindingLayout BindlessSRV(uint space, uint firstSlot = 0, bool groupWithPrevious = false)
            => Add(NativeBindingLayoutKind.BindlessSRV, firstSlot, space, groupWithPrevious: groupWithPrevious);

        public NativeBindingLayout BindlessUAV(uint space, uint firstSlot = 0, bool groupWithPrevious = false)
            => Add(NativeBindingLayoutKind.BindlessUAV, firstSlot, space, groupWithPrevious: groupWithPrevious);

        public NativeBindingLayout RootSRV(uint slot, uint space = 0)
            => Add(NativeBindingLayoutKind.RootSRV, slot, space);

        public NativeBindingLayout Sampler(uint slot, uint space = 0, uint count = 1)
            => Add(NativeBindingLayoutKind.Sampler, slot, space, count);

        public NativeBindingLayout StaticSampler(uint slot, SamplerFilter filter,
            SamplerAddress addressU, SamplerAddress addressV, SamplerAddress addressW,
            bool mips = false, uint maxAnisotropy = 16, uint space = 0)
        {
            _staticSamplers.Add(new NativeStaticSampler
            {
                Slot = slot,
                Space = space,
                Filter = filter,
                AddressU = addressU,
                AddressV = addressV,
                AddressW = addressW,
                Mips = mips,
                MaxAnisotropy = maxAnisotropy
            });
            return this;
        }

        internal NativeRenderPlugin.NR_BindingLayoutItem[] BuildNativeItems()
        {
            if (_items.Count == 0) return Array.Empty<NativeRenderPlugin.NR_BindingLayoutItem>();
            var dst = new NativeRenderPlugin.NR_BindingLayoutItem[_items.Count];
            for (int i = 0; i < _items.Count; i++)
            {
                var item = _items[i];
                dst[i] = new NativeRenderPlugin.NR_BindingLayoutItem
                {
                    kind = (uint)item.Kind,
                    slot = item.Slot,
                    space = item.Space,
                    count = item.Count == 0 ? 1 : item.Count,
                    num32 = item.Num32BitValues,
                    groupWithPrevious = item.GroupWithPrevious ? 1u : 0u,
                };
            }
            return dst;
        }

        internal NativeRenderPlugin.NR_StaticSampler[] BuildNativeStaticSamplers()
        {
            if (_staticSamplers.Count == 0) return Array.Empty<NativeRenderPlugin.NR_StaticSampler>();
            var dst = new NativeRenderPlugin.NR_StaticSampler[_staticSamplers.Count];
            for (int i = 0; i < _staticSamplers.Count; i++)
            {
                var s = _staticSamplers[i];
                dst[i] = new NativeRenderPlugin.NR_StaticSampler
                {
                    reg = s.Slot,
                    space = s.Space,
                    filter = (uint)s.Filter,
                    addressU = (uint)s.AddressU,
                    addressV = (uint)s.AddressV,
                    addressW = (uint)s.AddressW,
                    mips = s.Mips ? 1u : 0u,
                    maxAnisotropy = s.MaxAnisotropy,
                };
            }
            return dst;
        }

        // -------------------------------------------------------------------
        // Slot resolution (the C# mirror of what the plugin derives natively)
        // -------------------------------------------------------------------

        /// <summary>
        /// Resolves a register-addressed binding to the layout item (= dispatch
        /// slot) index that covers it, or -1. Mirrors the plugin's layout walk:
        /// table kinds match by register-range containment, root-bound kinds by
        /// exact register; first matching item in declaration order wins.
        /// </summary>
        internal int ResolveSlot(BindingRegClass cls, uint reg, uint space, bool unbounded, bool isTlas)
        {
            for (int i = 0; i < _items.Count; i++)
            {
                var item = _items[i];
                bool exact     = reg == item.Slot && space == item.Space;
                bool contained = space == item.Space && reg >= item.Slot && reg < item.Slot + item.Count;

                switch (cls)
                {
                    case BindingRegClass.SRV:
                        if (unbounded)
                        {
                            if (item.Kind == NativeBindingLayoutKind.BindlessSRV && exact) return i;
                        }
                        else if (isTlas)
                        {
                            if ((item.Kind == NativeBindingLayoutKind.TLAS ||
                                 item.Kind == NativeBindingLayoutKind.SRV) && contained) return i;
                            if (item.Kind == NativeBindingLayoutKind.RootSRV && exact) return i;
                        }
                        else
                        {
                            if (item.Kind == NativeBindingLayoutKind.SRV && contained) return i;
                            if (item.Kind == NativeBindingLayoutKind.RootSRV && exact) return i;
                        }
                        break;

                    case BindingRegClass.UAV:
                        if (unbounded)
                        {
                            if (item.Kind == NativeBindingLayoutKind.BindlessUAV && exact) return i;
                        }
                        else if (item.Kind == NativeBindingLayoutKind.UAV && contained) return i;
                        break;

                    case BindingRegClass.CBV:
                        if (item.Kind == NativeBindingLayoutKind.CBV && contained) return i;
                        if ((item.Kind == NativeBindingLayoutKind.VolatileCBV ||
                             item.Kind == NativeBindingLayoutKind.PushConstants) && exact) return i;
                        break;

                    case BindingRegClass.Sampler:
                        if (item.Kind == NativeBindingLayoutKind.Sampler && contained) return i;
                        break;
                }
            }
            return -1;
        }

        internal int ResolveSlot(in ReflectedBinding b)
            => ResolveSlot(b.RegClass, b.Reg, b.Space, unbounded: b.Count == 0, isTlas: b.IsTlas);

        /// <summary>True when a static-sampler definition covers sampler register (reg, space).</summary>
        internal bool HasStaticSampler(uint reg, uint space)
        {
            for (int i = 0; i < _staticSamplers.Count; i++)
                if (_staticSamplers[i].Slot == reg && _staticSamplers[i].Space == space)
                    return true;
            return false;
        }

        // -------------------------------------------------------------------
        // Auto-generation from import-time reflection
        // -------------------------------------------------------------------

        /// <summary>
        /// Builds a layout from the shader's import-time reflection, replicating
        /// the binding policy the plugin's (removed) runtime-reflection path used:
        /// CBVs become volatile (root-descriptor) constant buffers unless promoted
        /// to push constants by a <see cref="RootConstantsHint"/>; buffer SRVs /
        /// TLAS named in <paramref name="rootSRVHints"/> become inline root SRVs;
        /// unbounded arrays become bindless items (one root parameter each);
        /// samplers become static samplers configured from
        /// <paramref name="samplerHints"/> or the Unity inline-sampler naming
        /// convention (sampler_LinearClamp, …).
        /// </summary>
        internal static NativeBindingLayout FromReflection(IReadOnlyList<ReflectedBinding> bindings,
            RootConstantsHint[] rootConstantsHints, string[] rootSRVHints, SamplerHint[] samplerHints)
        {
            var layout = new NativeBindingLayout();
            if (bindings == null) return layout;

            for (int i = 0; i < bindings.Count; i++)
            {
                var b = bindings[i];
                if (layout.HasLayoutItemFor(b))
                    continue;

                switch (b.RegClass)
                {
                    case BindingRegClass.Sampler:
                        layout.AddStaticSamplerFor(b, samplerHints);
                        break;

                    case BindingRegClass.CBV:
                    {
                        uint num32 = 0;
                        if (rootConstantsHints != null)
                        {
                            for (int h = 0; h < rootConstantsHints.Length; h++)
                            {
                                if (rootConstantsHints[h].Name == b.Name)
                                {
                                    num32 = rootConstantsHints[h].Count != 0
                                        ? rootConstantsHints[h].Count
                                        : b.SizeBytes / 4;
                                    break;
                                }
                            }
                        }
                        if (num32 != 0) layout.PushConstants(b.Reg, num32, b.Space);
                        else            layout.VolatileConstantBuffer(b.Reg, b.Space);
                        break;
                    }

                    case BindingRegClass.SRV:
                        if (b.Count == 0)
                            layout.BindlessSRV(b.Space, b.Reg);
                        else if (NameIn(rootSRVHints, b.Name))
                            layout.RootSRV(b.Reg, b.Space);
                        else if (b.IsTlas)
                            layout.RayTracingAccelStruct(b.Reg, b.Space);
                        else
                            layout.TextureSRV(b.Reg, b.Space, b.Count);
                        break;

                    case BindingRegClass.UAV:
                        if (b.Count == 0) layout.BindlessUAV(b.Space, b.Reg);
                        else              layout.TextureUAV(b.Reg, b.Space, b.Count);
                        break;
                }
            }
            return layout;
        }

        private bool HasLayoutItemFor(in ReflectedBinding b)
        {
            int slot = ResolveSlot(b);
            return slot >= 0 || (b.RegClass == BindingRegClass.Sampler && HasStaticSampler(b.Reg, b.Space));
        }

        private static bool NameIn(string[] names, string name)
        {
            if (names == null) return false;
            for (int i = 0; i < names.Length; i++)
                if (names[i] == name)
                    return true;
            return false;
        }

        private void AddStaticSamplerFor(in ReflectedBinding b, SamplerHint[] hints)
        {
            if (HasStaticSampler(b.Reg, b.Space))
                return;

            if (hints != null)
            {
                for (int i = 0; i < hints.Length; i++)
                {
                    if (hints[i].Name == b.Name)
                    {
                        var h = hints[i];
                        StaticSampler(b.Reg, h.Filter, h.AddressU, h.AddressV, h.AddressW,
                            h.Mips, h.MaxAnisotropy, b.Space);
                        return;
                    }
                }
            }

            // Unity inline-sampler naming convention (sampler_LinearClamp, …) —
            // the same inference the plugin used to run on reflected samplers.
            string lower = b.Name.ToLowerInvariant();
            var filter = SamplerFilter.Linear;
            if (lower.Contains("point") || lower.Contains("nearest")) filter = SamplerFilter.Point;
            else if (lower.Contains("aniso")) filter = SamplerFilter.Anisotropic;

            var addr = SamplerAddress.Wrap;
            if (lower.Contains("mirroronce")) addr = SamplerAddress.MirrorOnce;
            else if (lower.Contains("mirror")) addr = SamplerAddress.Mirror;
            else if (lower.Contains("clamp")) addr = SamplerAddress.Clamp;

            bool mips = lower.Contains("mipmap");
            StaticSampler(b.Reg, filter, addr, addr, addr, mips,
                filter == SamplerFilter.Anisotropic ? 16u : 0u, b.Space);
        }
    }
}
