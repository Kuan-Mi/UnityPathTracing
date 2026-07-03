using System;
using System.Collections.Generic;
using System.Text;

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
    }

    [Serializable]
    internal struct NativeBindingLayoutSamplerDefault
    {
        public string Name;
        public SamplerFilter Filter;
        public SamplerAddress AddressU;
        public SamplerAddress AddressV;
        public SamplerAddress AddressW;
        public bool Mips;
        public uint MaxAnisotropy;
    }

    public sealed class NativeBindingLayout
    {
        private readonly List<NativeBindingLayoutItem> _items = new List<NativeBindingLayoutItem>();
        private readonly List<NativeBindingLayoutSamplerDefault> _samplerDefaults =
            new List<NativeBindingLayoutSamplerDefault>();

        public IReadOnlyList<NativeBindingLayoutItem> Items => _items;
        public bool IsEmpty => _items.Count == 0;
        internal IReadOnlyList<NativeBindingLayoutSamplerDefault> SamplerDefaults => _samplerDefaults;

        public NativeBindingLayout Add(NativeBindingLayoutKind kind, uint slot, uint space = 0,
            uint count = 1, uint num32BitValues = 0)
        {
            _items.Add(new NativeBindingLayoutItem
            {
                Kind = kind,
                Slot = slot,
                Space = space,
                Count = count == 0 ? 1 : count,
                Num32BitValues = num32BitValues
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

        public NativeBindingLayout BindlessSRV(uint space, uint firstSlot = 0)
            => Add(NativeBindingLayoutKind.BindlessSRV, firstSlot, space);

        public NativeBindingLayout BindlessUAV(uint space, uint firstSlot = 0)
            => Add(NativeBindingLayoutKind.BindlessUAV, firstSlot, space);

        public NativeBindingLayout RootSRV(uint slot, uint space = 0)
            => Add(NativeBindingLayoutKind.RootSRV, slot, space);

        public NativeBindingLayout Sampler(uint slot, uint space = 0, uint count = 1)
            => Add(NativeBindingLayoutKind.Sampler, slot, space, count);

        internal NativeBindingLayout AddSamplerDefault(string name, SamplerFilter filter,
            SamplerAddress addressU, SamplerAddress addressV, SamplerAddress addressW,
            bool mips, uint maxAnisotropy)
        {
            if (string.IsNullOrEmpty(name)) return this;
            _samplerDefaults.Add(new NativeBindingLayoutSamplerDefault
            {
                Name = name,
                Filter = filter,
                AddressU = addressU,
                AddressV = addressV,
                AddressW = addressW,
                Mips = mips,
                MaxAnisotropy = maxAnisotropy
            });
            return this;
        }

        internal void AppendJson(StringBuilder sb)
        {
            sb.Append("\"sharedLayout\":{\"items\":[");
            for (int i = 0; i < _items.Count; i++)
            {
                if (i > 0) sb.Append(',');
                var item = _items[i];
                sb.Append("{\"kind\":").Append((uint)item.Kind)
                    .Append(",\"slot\":").Append(item.Slot)
                    .Append(",\"space\":").Append(item.Space)
                    .Append(",\"count\":").Append(item.Count == 0 ? 1 : item.Count)
                    .Append(",\"num32\":").Append(item.Num32BitValues)
                    .Append('}');
            }
            sb.Append("]}");
        }
    }
}
