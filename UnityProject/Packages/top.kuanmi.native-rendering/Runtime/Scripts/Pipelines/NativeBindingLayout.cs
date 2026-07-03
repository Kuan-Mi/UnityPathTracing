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

    public enum NativeResourceType : uint
    {
        None = 0,
        Texture_SRV = 1,
        Texture_UAV = 2,
        TypedBuffer_SRV = 3,
        TypedBuffer_UAV = 4,
        StructuredBuffer_SRV = 5,
        StructuredBuffer_UAV = 6,
        RawBuffer_SRV = 7,
        RawBuffer_UAV = 8,
        ConstantBuffer = 9,
        VolatileConstantBuffer = 10,
        Sampler = 11,
        RayTracingAccelStruct = 12,
        PushConstants = 13,
        Root_SRV = 14
    }

    public enum NativeShaderVisibility : uint
    {
        All = 0,
        Vertex = 1,
        Hull = 2,
        Domain = 3,
        Geometry = 4,
        Pixel = 5
    }

    [Serializable]
    public struct BindingLayoutItem
    {
        public NativeResourceType Type;
        public uint Slot;
        public uint Size;

        public static BindingLayoutItem Texture_SRV(uint slot, uint count = 1)
            => Make(NativeResourceType.Texture_SRV, slot, count);

        public static BindingLayoutItem Texture_UAV(uint slot, uint count = 1)
            => Make(NativeResourceType.Texture_UAV, slot, count);

        public static BindingLayoutItem TypedBuffer_SRV(uint slot, uint count = 1)
            => Make(NativeResourceType.TypedBuffer_SRV, slot, count);

        public static BindingLayoutItem TypedBuffer_UAV(uint slot, uint count = 1)
            => Make(NativeResourceType.TypedBuffer_UAV, slot, count);

        public static BindingLayoutItem StructuredBuffer_SRV(uint slot, uint count = 1)
            => Make(NativeResourceType.StructuredBuffer_SRV, slot, count);

        public static BindingLayoutItem StructuredBuffer_UAV(uint slot, uint count = 1)
            => Make(NativeResourceType.StructuredBuffer_UAV, slot, count);

        public static BindingLayoutItem RawBuffer_SRV(uint slot, uint count = 1)
            => Make(NativeResourceType.RawBuffer_SRV, slot, count);

        public static BindingLayoutItem RawBuffer_UAV(uint slot, uint count = 1)
            => Make(NativeResourceType.RawBuffer_UAV, slot, count);

        public static BindingLayoutItem ConstantBuffer(uint slot)
            => Make(NativeResourceType.ConstantBuffer, slot, 1);

        public static BindingLayoutItem VolatileConstantBuffer(uint slot)
            => Make(NativeResourceType.VolatileConstantBuffer, slot, 1);

        public static BindingLayoutItem PushConstants(uint slot, uint byteSize)
            => Make(NativeResourceType.PushConstants, slot, byteSize);

        public static BindingLayoutItem RayTracingAccelStruct(uint slot)
            => Make(NativeResourceType.RayTracingAccelStruct, slot, 1);

        public static BindingLayoutItem Root_SRV(uint slot)
            => Make(NativeResourceType.Root_SRV, slot, 1);

        public static BindingLayoutItem Sampler(uint slot, uint count = 1)
            => Make(NativeResourceType.Sampler, slot, count);

        private static BindingLayoutItem Make(NativeResourceType type, uint slot, uint size)
            => new BindingLayoutItem
            {
                Type = type,
                Slot = slot,
                Size = type == NativeResourceType.PushConstants ? size : size == 0 ? 1 : size
            };
    }

    public sealed class BindingLayoutDesc
    {
        private readonly List<BindingLayoutItem> _bindings =
            new List<BindingLayoutItem>();

        public NativeShaderVisibility Visibility = NativeShaderVisibility.All;
        public uint RegisterSpace;
        public IReadOnlyList<BindingLayoutItem> Bindings => _bindings;

        public BindingLayoutDesc(uint registerSpace = 0)
        {
            RegisterSpace = registerSpace;
        }

        public BindingLayoutDesc AddItem(in BindingLayoutItem item)
        {
            _bindings.Add(item);
            return this;
        }

        public BindingLayoutDesc AddBinding(in BindingLayoutItem item)
            => AddItem(item);

        public BindingLayoutDesc AddItems(params BindingLayoutItem[] items)
        {
            if (items == null) return this;
            for (int i = 0; i < items.Length; i++)
                _bindings.Add(items[i]);
            return this;
        }

        public BindingLayoutDesc AddBindings(params BindingLayoutItem[] items)
            => AddItems(items);
    }

    public sealed class BindlessLayoutDesc
    {
        private readonly List<BindingLayoutItem> _registerSpaces =
            new List<BindingLayoutItem>();

        public NativeShaderVisibility Visibility = NativeShaderVisibility.All;
        public uint FirstSlot;
        public uint MaxCapacity = 1024;
        public IReadOnlyList<BindingLayoutItem> RegisterSpaces => _registerSpaces;

        public BindlessLayoutDesc(uint firstSlot = 0, uint maxCapacity = 1024)
        {
            FirstSlot = firstSlot;
            MaxCapacity = maxCapacity == 0 ? 1 : maxCapacity;
        }

        public BindlessLayoutDesc AddRegisterSpace(in BindingLayoutItem item)
        {
            _registerSpaces.Add(item);
            return this;
        }

        public BindlessLayoutDesc AddRegisterSpaces(params BindingLayoutItem[] items)
        {
            if (items == null) return this;
            for (int i = 0; i < items.Length; i++)
                _registerSpaces.Add(items[i]);
            return this;
        }
    }

    [Serializable]
    internal struct ResolvedBindingLayoutItem
    {
        public NativeBindingLayoutKind Kind;
        public uint Slot;
        public uint Space;
        public uint Count;
        public uint Num32BitValues;
        public NativeShaderVisibility Visibility;

        /// <summary>Bindless items only: share the previous bindless item's root
        /// parameter (one descriptor table aliased by several unbounded ranges —
        /// nvrhi's bindless layout with multiple register spaces, e.g. donut's
        /// shared table). When false the item gets its own root parameter.</summary>
        public uint BindlessLayoutIndex;
    }

    internal struct NativeBindlessLayoutRange
    {
        public uint Visibility;
        public uint FirstSlot;
        public uint MaxCapacity;
        public uint FirstItem;
        public uint ItemCount;
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
    /// Runtime binding layout wrapper built from nvrhi-style
    /// <see cref="BindingLayoutDesc"/> and <see cref="BindlessLayoutDesc"/>
    /// declarations. The resolved item order is the binding contract between
    /// HLSL register assignments and the native root signature. The render plugin
    /// never reflects shaders — it builds the root signature and its per-slot
    /// binding table from this declaration alone, and the per-dispatch
    /// <c>BindingSlot[]</c> payload is ordered by layout item index.
    ///
    /// A layout is MANDATORY for every pipeline. Author one by hand for
    /// RTXPT-parity passes (item order defines the descriptor-table layout) or
    /// let the pipeline auto-generate one from the import-time reflection JSON
    /// via <see cref="FromReflection"/>.
    /// </summary>
    public sealed class NativeBindingLayout : IDisposable
    {
        private readonly List<ResolvedBindingLayoutItem> _items = new List<ResolvedBindingLayoutItem>();
        private readonly List<NativeStaticSampler> _staticSamplers = new List<NativeStaticSampler>();
        private readonly List<NativeBindlessLayoutRange> _bindlessLayouts =
            new List<NativeBindlessLayoutRange>();
        private ulong[] _nativeLayoutHandles;

        internal IReadOnlyList<ResolvedBindingLayoutItem> Items => _items;
        public IReadOnlyList<NativeStaticSampler> StaticSamplers => _staticSamplers;
        public bool IsEmpty => _items.Count == 0 && _staticSamplers.Count == 0;

        public NativeBindingLayout()
        {
        }

        public NativeBindingLayout(params BindingLayoutDesc[] descs)
        {
            AddLayouts(descs);
        }

        public static NativeBindingLayout FromDescs(params BindingLayoutDesc[] descs)
            => new NativeBindingLayout(descs);

        public NativeBindingLayout Add(NativeBindingLayoutKind kind, uint slot, uint space = 0,
            uint count = 1, uint num32BitValues = 0,
            NativeShaderVisibility visibility = NativeShaderVisibility.All,
            uint bindlessLayoutIndex = uint.MaxValue)
        {
            ReleaseNativeLayoutHandles();
            _items.Add(new ResolvedBindingLayoutItem
            {
                Kind = kind,
                Slot = slot,
                Space = space,
                Count = count == 0 ? 1 : count,
                Num32BitValues = num32BitValues,
                Visibility = visibility,
                BindlessLayoutIndex = bindlessLayoutIndex
            });
            return this;
        }

        public NativeBindingLayout AddLayouts(params BindingLayoutDesc[] descs)
        {
            if (descs == null) return this;
            for (int i = 0; i < descs.Length; i++)
                AddLayout(descs[i]);
            return this;
        }

        public NativeBindingLayout AddLayout(BindingLayoutDesc desc)
        {
            if (desc == null) return this;
            var bindings = desc.Bindings;
            for (int i = 0; i < bindings.Count; i++)
            {
                var item = bindings[i];
                var kind = ToLayoutKind(item.Type);
                if (kind == NativeBindingLayoutKind.PushConstants &&
                    (item.Size == 0 || (item.Size & 3u) != 0))
                    throw new ArgumentException(
                        $"PushConstants item at slot {item.Slot} must have a non-zero 4-byte-aligned size.");

                uint count = kind == NativeBindingLayoutKind.PushConstants
                    ? item.Size
                    : item.Size == 0 ? 1 : item.Size;
                uint num32 = kind == NativeBindingLayoutKind.PushConstants ? item.Size / 4 : 0;
                Add(kind, item.Slot, desc.RegisterSpace, count, num32, desc.Visibility);
            }
            return this;
        }

        public NativeBindingLayout AddBindlessLayout(BindlessLayoutDesc desc)
        {
            if (desc == null) return this;
            var registerSpaces = desc.RegisterSpaces;
            if (registerSpaces.Count == 0)
                throw new ArgumentException("Bindless layout must contain at least one register space.");

            uint layoutIndex = (uint)_bindlessLayouts.Count;
            uint firstItem = (uint)_items.Count;
            for (int i = 0; i < registerSpaces.Count; i++)
            {
                var item = registerSpaces[i];
                var kind = ToBindlessLayoutKind(item.Type);
                Add(kind, desc.FirstSlot, item.Slot, desc.MaxCapacity, 0,
                    desc.Visibility, layoutIndex);
            }

            _bindlessLayouts.Add(new NativeBindlessLayoutRange
            {
                Visibility = (uint)desc.Visibility,
                FirstSlot = desc.FirstSlot,
                MaxCapacity = desc.MaxCapacity == 0 ? 1 : desc.MaxCapacity,
                FirstItem = firstItem,
                ItemCount = (uint)registerSpaces.Count
            });
            return this;
        }

        private static NativeBindingLayoutKind ToLayoutKind(NativeResourceType type)
        {
            switch (type)
            {
                case NativeResourceType.Texture_SRV:
                case NativeResourceType.TypedBuffer_SRV:
                case NativeResourceType.StructuredBuffer_SRV:
                case NativeResourceType.RawBuffer_SRV:
                    return NativeBindingLayoutKind.SRV;

                case NativeResourceType.Texture_UAV:
                case NativeResourceType.TypedBuffer_UAV:
                case NativeResourceType.StructuredBuffer_UAV:
                case NativeResourceType.RawBuffer_UAV:
                    return NativeBindingLayoutKind.UAV;

                case NativeResourceType.ConstantBuffer:
                    return NativeBindingLayoutKind.CBV;
                case NativeResourceType.VolatileConstantBuffer:
                    return NativeBindingLayoutKind.VolatileCBV;
                case NativeResourceType.PushConstants:
                    return NativeBindingLayoutKind.PushConstants;
                case NativeResourceType.RayTracingAccelStruct:
                    return NativeBindingLayoutKind.TLAS;
                case NativeResourceType.Root_SRV:
                    return NativeBindingLayoutKind.RootSRV;
                case NativeResourceType.Sampler:
                    return NativeBindingLayoutKind.Sampler;
                default:
                    throw new ArgumentOutOfRangeException(nameof(type), type,
                        "Unsupported native binding resource type");
            }
        }

        private static NativeBindingLayoutKind ToBindlessLayoutKind(NativeResourceType type)
        {
            switch (type)
            {
                case NativeResourceType.Texture_SRV:
                case NativeResourceType.TypedBuffer_SRV:
                case NativeResourceType.StructuredBuffer_SRV:
                case NativeResourceType.RawBuffer_SRV:
                    return NativeBindingLayoutKind.BindlessSRV;

                case NativeResourceType.Texture_UAV:
                case NativeResourceType.TypedBuffer_UAV:
                case NativeResourceType.StructuredBuffer_UAV:
                case NativeResourceType.RawBuffer_UAV:
                    return NativeBindingLayoutKind.BindlessUAV;

                default:
                    throw new ArgumentOutOfRangeException(nameof(type), type,
                        "Unsupported bindless register-space resource type");
            }
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

        public NativeBindingLayout BindlessSRV(uint space, uint firstSlot = 0)
            => AddBindlessLayout(new BindlessLayoutDesc(firstSlot)
                .AddRegisterSpace(BindingLayoutItem.RawBuffer_SRV(space)));

        public NativeBindingLayout BindlessUAV(uint space, uint firstSlot = 0)
            => AddBindlessLayout(new BindlessLayoutDesc(firstSlot)
                .AddRegisterSpace(BindingLayoutItem.Texture_UAV(space)));

        public NativeBindingLayout RootSRV(uint slot, uint space = 0)
            => Add(NativeBindingLayoutKind.RootSRV, slot, space);

        public NativeBindingLayout Sampler(uint slot, uint space = 0, uint count = 1)
            => Add(NativeBindingLayoutKind.Sampler, slot, space, count);

        public NativeBindingLayout StaticSampler(uint slot, SamplerFilter filter,
            SamplerAddress addressU, SamplerAddress addressV, SamplerAddress addressW,
            bool mips = false, uint maxAnisotropy = 16, uint space = 0)
        {
            ReleaseNativeLayoutHandles();
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

        internal NativeRenderPlugin.NR_BindingLayoutDesc[] BuildNativeDescs()
        {
            if (_items.Count == 0) return Array.Empty<NativeRenderPlugin.NR_BindingLayoutDesc>();

            var descs = new List<NativeRenderPlugin.NR_BindingLayoutDesc>();
            int runStart = -1;
            uint runSpace = 0;
            var runVisibility = NativeShaderVisibility.All;

            for (int i = 0; i <= _items.Count; i++)
            {
                if (i < _items.Count && runStart >= 0 &&
                    !IsBindlessKind(_items[i].Kind) &&
                    _items[i].Space == runSpace &&
                    _items[i].Visibility == runVisibility)
                    continue;

                if (runStart >= 0)
                {
                    descs.Add(new NativeRenderPlugin.NR_BindingLayoutDesc
                    {
                        visibility = (uint)runVisibility,
                        registerSpace = runSpace,
                        firstItem = (uint)runStart,
                        itemCount = (uint)(i - runStart),
                    });
                    runStart = -1;
                }

                if (i < _items.Count && !IsBindlessKind(_items[i].Kind))
                {
                    runStart = i;
                    runSpace = _items[i].Space;
                    runVisibility = _items[i].Visibility;
                }
            }

            return descs.ToArray();
        }

        internal NativeRenderPlugin.NR_BindlessLayoutDesc[] BuildNativeBindlessDescs()
        {
            if (_bindlessLayouts.Count == 0) return Array.Empty<NativeRenderPlugin.NR_BindlessLayoutDesc>();
            var dst = new NativeRenderPlugin.NR_BindlessLayoutDesc[_bindlessLayouts.Count];
            for (int i = 0; i < _bindlessLayouts.Count; i++)
            {
                var desc = _bindlessLayouts[i];
                dst[i] = new NativeRenderPlugin.NR_BindlessLayoutDesc
                {
                    visibility = desc.Visibility,
                    firstSlot = desc.FirstSlot,
                    maxCapacity = desc.MaxCapacity,
                    firstItem = desc.FirstItem,
                    itemCount = desc.ItemCount
                };
            }
            return dst;
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
                    type = (uint)item.Kind,
                    slot = IsBindlessKind(item.Kind) ? item.Space : item.Slot,
                    size = item.Kind == NativeBindingLayoutKind.PushConstants
                        ? item.Num32BitValues * 4
                        : item.Count == 0 ? 1 : item.Count,
                };
            }
            return dst;
        }

        private static bool IsBindlessKind(NativeBindingLayoutKind kind)
            => kind == NativeBindingLayoutKind.BindlessSRV ||
               kind == NativeBindingLayoutKind.BindlessUAV;

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

        internal ulong[] GetNativeLayoutHandles()
        {
            if (_nativeLayoutHandles != null)
                return _nativeLayoutHandles;

            var handles = new List<ulong>(2);
            var layoutItems = BuildNativeItems();
            var layoutDescs = BuildNativeDescs();
            var staticSamplers = BuildNativeStaticSamplers();
            if (layoutDescs.Length != 0 || staticSamplers.Length != 0)
            {
                ulong handle = NativeRenderPlugin.NR_CreateBindingLayout(
                    layoutDescs, (uint)layoutDescs.Length,
                    layoutItems, (uint)layoutItems.Length,
                    staticSamplers, (uint)staticSamplers.Length);
                if (handle == 0)
                    throw new InvalidOperationException("NR_CreateBindingLayout returned 0.");
                handles.Add(handle);
            }

            var bindlessDescs = BuildNativeBindlessDescs();
            if (bindlessDescs.Length != 0)
            {
                ulong handle = NativeRenderPlugin.NR_CreateBindlessLayout(
                    bindlessDescs, (uint)bindlessDescs.Length,
                    layoutItems, (uint)layoutItems.Length);
                if (handle == 0)
                    throw new InvalidOperationException("NR_CreateBindlessLayout returned 0.");
                handles.Add(handle);
            }

            _nativeLayoutHandles = handles.Count == 0 ? Array.Empty<ulong>() : handles.ToArray();
            return _nativeLayoutHandles;
        }

        private void ReleaseNativeLayoutHandles()
        {
            if (_nativeLayoutHandles == null)
                return;

            for (int i = 0; i < _nativeLayoutHandles.Length; i++)
            {
                if (_nativeLayoutHandles[i] != 0)
                {
                    try
                    {
                        NativeRenderPlugin.NR_DestroyBindingLayout(_nativeLayoutHandles[i]);
                    }
                    catch
                    {
                    }
                }
            }
            _nativeLayoutHandles = null;
        }

        public void Dispose()
        {
            ReleaseNativeLayoutHandles();
            GC.SuppressFinalize(this);
        }

        ~NativeBindingLayout()
        {
            ReleaseNativeLayoutHandles();
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
