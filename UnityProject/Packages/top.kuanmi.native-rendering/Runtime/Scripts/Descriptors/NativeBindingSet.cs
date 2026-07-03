using System;
using System.Collections.Generic;
using UnityEngine;

namespace NativeRender
{
    /// <summary>
    /// One register-addressed resource assignment, mirroring
    /// <c>nvrhi::BindingSetItem</c>: the slot number written here is the
    /// contract with the HLSL register (and with the matching
    /// <see cref="NativeBindingLayout"/> item) — no reflection, no names.
    /// Construct via the static factories (Texture_SRV, Texture_UAV, …).
    ///
    /// Push constants are NOT binding-set items; push their data per dispatch
    /// with <see cref="NativeDescriptorSetBase.SetRootConstants{T}(int,T*,uint,uint)"/>
    /// (the analog of nvrhi's <c>commandList-&gt;setPushConstants</c>).
    /// </summary>
    public struct NativeBindingSetItem
    {
        internal BindingRegClass RegClass;
        internal bool IsTlas;
        internal bool Unbounded;
        internal uint Slot;
        internal uint Space;

        // BindingSlot payload — identical encoding to the Set* methods.
        internal ulong ObjectPtr;
        internal uint  ObjectKind;
        internal uint  Count;
        internal uint  Stride;
        internal uint  Format;

        // objectKind constants — must match C++ BindingObjectKind.
        private const uint KindNone               = 0;
        private const uint KindAccelStruct        = 1;
        private const uint KindBindlessTexture    = 2;
        private const uint KindBindlessBuffer     = 3;
        private const uint KindNativeBuffer       = 5;
        private const uint KindBindlessUAVTexture = 6;
        private const uint KindSampler            = 7;

        private static NativeBindingSetItem Make(BindingRegClass cls, uint slot, uint space,
            ulong ptr, uint objectKind, uint count = 0, uint stride = 0, uint format = 0)
        {
            return new NativeBindingSetItem
            {
                RegClass = cls,
                Slot = slot,
                Space = space,
                ObjectPtr = ptr,
                ObjectKind = objectKind,
                Count = count,
                Stride = stride,
                Format = format
            };
        }

        // --- Textures ---

        /// <summary>Texture SRV (tN). Optional single-mip-range view:
        /// firstMip = MostDetailedMip, mipCount = MipLevels (0 = all remaining).</summary>
        public static NativeBindingSetItem Texture_SRV(uint slot, IntPtr texture,
            int firstMip = 0, int mipCount = 0, uint space = 0)
            => Make(BindingRegClass.SRV, slot, space, (ulong)texture, KindNone,
                count: (uint)mipCount, stride: (uint)firstMip);

        /// <summary>Cubemap/array mip as an explicit Texture2DArray SRV (nvrhi
        /// <c>Texture_SRV(...).setDimension(Texture2DArray)</c>). Non-zero
        /// <paramref name="dxgiFormat"/> selects the array view — never pass 0.</summary>
        public static NativeBindingSetItem TextureArray_SRV(uint slot, IntPtr texture,
            int mip, uint dxgiFormat, uint space = 0)
            => Make(BindingRegClass.SRV, slot, space, (ulong)texture, KindNone,
                count: 1, stride: (uint)mip, format: dxgiFormat);

        /// <summary>Texture UAV (uN) at mip <paramref name="mipSlice"/>.</summary>
        public static NativeBindingSetItem Texture_UAV(uint slot, IntPtr texture,
            int mipSlice = 0, uint space = 0)
            => Make(BindingRegClass.UAV, slot, space, (ulong)texture, KindNone,
                stride: (uint)mipSlice);

        /// <summary>Bounded UAV array (RWTexture2D u[N]) bound to N consecutive mips
        /// of one texture starting at <paramref name="baseMip"/>.</summary>
        public static NativeBindingSetItem TextureMipArray_UAV(uint slot, IntPtr texture,
            int baseMip, uint space = 0)
            => Make(BindingRegClass.UAV, slot, space, (ulong)texture, KindNone,
                stride: (uint)baseMip);

        /// <summary>Bounded SRV array (Texture2D t[N]) bound to N consecutive
        /// single-mip views of one texture starting at <paramref name="baseMip"/>.</summary>
        public static NativeBindingSetItem TextureMipArray_SRV(uint slot, IntPtr texture,
            int baseMip, uint space = 0)
            => Make(BindingRegClass.SRV, slot, space, (ulong)texture, KindNone,
                stride: (uint)baseMip);

        // --- Buffers ---

        public static NativeBindingSetItem StructuredBuffer_SRV(uint slot, IntPtr buffer,
            int count, int stride, uint space = 0)
            => Make(BindingRegClass.SRV, slot, space, (ulong)buffer, KindNone,
                count: (uint)count, stride: (uint)stride);

        public static NativeBindingSetItem StructuredBuffer_SRV(uint slot, GraphicsBuffer buffer, uint space = 0)
            => StructuredBuffer_SRV(slot, buffer.GetNativeBufferPtr(), buffer.count, buffer.stride, space);

        public static NativeBindingSetItem StructuredBuffer_SRV(uint slot, UploadBuffer buffer, uint space = 0)
            => Make(BindingRegClass.SRV, slot, space, buffer != null ? buffer.Handle : 0, KindNativeBuffer,
                count: (uint)(buffer != null ? buffer.count : 0),
                stride: (uint)(buffer != null ? buffer.stride : 0));

        public static NativeBindingSetItem StructuredBuffer_UAV(uint slot, IntPtr buffer,
            int count, int stride, uint space = 0)
            => Make(BindingRegClass.UAV, slot, space, (ulong)buffer, KindNone,
                count: (uint)count, stride: (uint)stride);

        public static NativeBindingSetItem StructuredBuffer_UAV(uint slot, GraphicsBuffer buffer, uint space = 0)
            => StructuredBuffer_UAV(slot, buffer.GetNativeBufferPtr(), buffer.count, buffer.stride, space);

        public static NativeBindingSetItem TypedBuffer_SRV(uint slot, IntPtr buffer,
            int count, uint dxgiFormat, uint space = 0)
            => Make(BindingRegClass.SRV, slot, space, (ulong)buffer, KindNone,
                count: (uint)count, format: dxgiFormat);

        public static NativeBindingSetItem TypedBuffer_SRV(uint slot, DeviceBuffer buffer,
            int count, uint dxgiFormat, uint space = 0)
            => Make(BindingRegClass.SRV, slot, space, buffer.Handle, KindNativeBuffer,
                count: (uint)count, format: dxgiFormat);

        public static NativeBindingSetItem TypedBuffer_UAV(uint slot, IntPtr buffer,
            int count, uint dxgiFormat, uint space = 0)
            => Make(BindingRegClass.UAV, slot, space, (ulong)buffer, KindNone,
                count: (uint)count, format: dxgiFormat);

        public static NativeBindingSetItem TypedBuffer_UAV(uint slot, DeviceBuffer buffer,
            int count, uint dxgiFormat, uint space = 0)
            => Make(BindingRegClass.UAV, slot, space, buffer.Handle, KindNativeBuffer,
                count: (uint)count, format: dxgiFormat);

        public static NativeBindingSetItem RawBuffer_SRV(uint slot, IntPtr buffer, uint space = 0)
            => Make(BindingRegClass.SRV, slot, space, (ulong)buffer, KindNone);

        public static NativeBindingSetItem RawBuffer_UAV(uint slot, IntPtr buffer, uint space = 0)
            => Make(BindingRegClass.UAV, slot, space, (ulong)buffer, KindNone);

        public static NativeBindingSetItem RawBuffer_UAV(uint slot, UploadBuffer buffer, uint space = 0)
            => Make(BindingRegClass.UAV, slot, space, buffer != null ? buffer.Handle : 0, KindNativeBuffer);

        // --- Constant buffers ---

        public static NativeBindingSetItem ConstantBuffer(uint slot, IntPtr buffer, uint space = 0)
            => Make(BindingRegClass.CBV, slot, space, (ulong)buffer, KindNone);

        public static NativeBindingSetItem ConstantBuffer(uint slot, VolatileConstantBuffer buffer, uint space = 0)
            => Make(BindingRegClass.CBV, slot, space, buffer != null ? buffer.Handle : 0, KindNativeBuffer);

        // --- Acceleration structure / samplers / bindless ---

        public static NativeBindingSetItem RayTracingAccelStruct(uint slot,
            RayTracingAccelerationStructure accelStruct, uint space = 0)
        {
            var item = Make(BindingRegClass.SRV, slot, space,
                accelStruct != null ? accelStruct.Handle : 0, KindAccelStruct);
            item.IsTlas = true;
            return item;
        }

        public static NativeBindingSetItem Sampler(uint slot, SamplerFilter filter,
            SamplerAddress addressU, SamplerAddress addressV, SamplerAddress addressW,
            bool mips = true, uint maxAnisotropy = 16, uint space = 0)
        {
            uint packed = ((uint)addressU & 0xffu) |
                          (((uint)addressV & 0xffu) << 8) |
                          (((uint)addressW & 0xffu) << 16) |
                          (mips ? (1u << 24) : 0u);
            return Make(BindingRegClass.Sampler, slot, space, 0, KindSampler,
                count: (uint)filter, stride: packed, format: maxAnisotropy);
        }

        public static NativeBindingSetItem Sampler(uint slot, NativeSampler sampler, uint space = 0)
            => sampler != null
                ? Sampler(slot, sampler.Filter, sampler.AddressU, sampler.AddressV, sampler.AddressW,
                    sampler.Mips, sampler.MaxAnisotropy, space)
                : Make(BindingRegClass.Sampler, slot, space, 0, KindNone);

        public static NativeBindingSetItem BindlessTexture_SRV(uint space, BindlessTexture textures, uint firstSlot = 0)
        {
            var item = Make(BindingRegClass.SRV, firstSlot, space,
                textures != null ? textures.Handle : 0, KindBindlessTexture);
            item.Unbounded = true;
            return item;
        }

        public static NativeBindingSetItem BindlessBuffer_SRV(uint space, BindlessBuffer buffers, uint firstSlot = 0)
        {
            var item = Make(BindingRegClass.SRV, firstSlot, space,
                buffers != null ? buffers.Handle : 0, KindBindlessBuffer);
            item.Unbounded = true;
            return item;
        }

        public static NativeBindingSetItem BindlessTexture_UAV(uint space, BindlessUAVTexture textures, uint firstSlot = 0)
        {
            var item = Make(BindingRegClass.UAV, firstSlot, space,
                textures != null ? textures.Handle : 0, KindBindlessUAVTexture);
            item.Unbounded = true;
            return item;
        }
    }

    /// <summary>
    /// An ordered list of <see cref="NativeBindingSetItem"/>s — the analog of
    /// <c>nvrhi::BindingSetDesc</c>. Apply it to a descriptor set with
    /// <see cref="NativeDescriptorSetBase.Bind(NativeBindingSetDesc)"/>, or
    /// pre-resolve it against a layout as a <see cref="NativeBindingSet"/>.
    /// </summary>
    public sealed class NativeBindingSetDesc
    {
        internal readonly List<NativeBindingSetItem> Items = new List<NativeBindingSetItem>();

        public NativeBindingSetDesc Add(in NativeBindingSetItem item)
        {
            Items.Add(item);
            return this;
        }
    }

    /// <summary>
    /// A binding-set desc resolved once against a <see cref="NativeBindingLayout"/>
    /// (the analog of <c>nvrhi::IBindingSet</c>): each item is matched to its
    /// layout slot by register, purely from the declaration — no reflection.
    /// Items that match no layout item are dropped with an error at creation.
    /// Apply per frame with <see cref="NativeDescriptorSetBase.Bind(NativeBindingSet)"/>.
    /// </summary>
    public sealed class NativeBindingSet
    {
        internal struct ResolvedItem
        {
            public int Slot;
            public NativeBindingSetItem Item;
        }

        internal readonly ResolvedItem[] Resolved;

        public NativeBindingSet(NativeBindingLayout layout, NativeBindingSetDesc desc)
        {
            if (layout == null) throw new ArgumentNullException(nameof(layout));
            if (desc == null) throw new ArgumentNullException(nameof(desc));

            var resolved = new List<ResolvedItem>(desc.Items.Count);
            for (int i = 0; i < desc.Items.Count; i++)
            {
                var item = desc.Items[i];
                int slot = layout.ResolveSlot(item.RegClass, item.Slot, item.Space,
                    item.Unbounded, item.IsTlas);
                if (slot < 0)
                {
                    Debug.LogError(
                        $"[NativeBindingSet] item {i} ({item.RegClass} reg {item.Slot}, space {item.Space}) " +
                        "matches no binding-layout item — check the register against the layout declaration");
                    continue;
                }
                resolved.Add(new ResolvedItem { Slot = slot, Item = item });
            }
            Resolved = resolved.ToArray();
        }
    }
}
