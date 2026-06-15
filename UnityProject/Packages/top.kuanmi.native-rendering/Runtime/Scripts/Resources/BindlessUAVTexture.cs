using System;
using UnityEngine;

namespace NativeRender
{
    /// <summary>
    /// A GPU-visible array of per-mip-slice RWTexture2D UAV descriptors backed by a
    /// contiguous range in the global DescriptorHeapAllocator.
    ///
    /// Primary use-case: expose mip levels of a single texture as an unbounded
    /// <c>RWTexture2D&lt;T&gt; varName[]</c> array in a compute shader, enabling
    /// GenerateMips / PreprocessEnvironmentMap patterns via native CS 6.x dispatch.
    ///
    /// Usage:
    ///   int mips = tex.mipmapCount - 1;   // number of destination mip levels
    ///   var uav = new BindlessUAVTexture(mips);
    ///   for (int i = 0; i &lt; mips; i++)
    ///       uav.SetTexture(i, tex, mipSlice: i + 1, dxgiFormat: 0x29); // R32_FLOAT
    ///   ds.SetBindlessRWTexture("u_IntegratedMips", uav);
    ///   pipeline.Dispatch(cmd, ds, groupsX, groupsY, 1);
    ///   uav.Dispose();
    ///
    /// HLSL declaration:
    ///   RWTexture2D&lt;float&gt; u_IntegratedMips[] : register(u0);
    ///
    /// Notes:
    ///   • Null entries write a fallback R32_FLOAT null UAV.
    ///   • After Resize() the GPU handle changes; re-bind to all shaders that reference it.
    ///   • The BindlessUAVTexture does not own the underlying GPU resource.
    /// </summary>
    public sealed class BindlessUAVTexture : IDisposable
    {
        private ulong _handle;
        private int   _capacity;
        private bool  _disposed;

        /// <summary>Opaque native handle. Used internally by NativeComputeDescriptorSet.SetBindlessRWTexture.</summary>
        public ulong Handle => _handle;

        /// <summary>Current number of UAV descriptor slots.</summary>
        public int Capacity => _capacity;

        public bool IsValid => _handle != 0 && !_disposed;

        // -------------------------------------------------------------------
        // Construction
        // -------------------------------------------------------------------

        /// <summary>
        /// Creates a BindlessUAVTexture with <paramref name="capacity"/> slots. All slots start as null.
        /// </summary>
        public BindlessUAVTexture(int capacity)
        {
            if (capacity < 1) capacity = 1;
            _handle   = NativeRenderPlugin.NR_CreateBindlessUAVTexture((uint)capacity);
            _capacity = capacity;
            if (_handle == 0)
                Debug.LogError("[BindlessUAVTexture] NR_CreateBindlessUAVTexture failed");
        }

        // -------------------------------------------------------------------
        // Per-slot assignment
        // -------------------------------------------------------------------

        /// <summary>
        /// Sets the UAV descriptor at <paramref name="index"/> to the given texture mip slice.
        /// Pass null to write a null UAV.
        /// </summary>
        /// <param name="index">Slot index (must be &lt; Capacity).</param>
        /// <param name="tex">Texture whose mip level to expose as UAV.</param>
        /// <param name="mipSlice">Which mip level to expose (0-based).</param>
        /// <param name="dxgiFormat">DXGI_FORMAT for the UAV view; 0 = derive from resource format.</param>
        public void SetTexture(int index, Texture tex, int mipSlice = 0, uint dxgiFormat = 0)
        {
            if (index < 0 || index >= _capacity)
                throw new ArgumentOutOfRangeException(nameof(index));
            if (!IsValid) return;
            IntPtr ptr = tex != null ? tex.GetNativeTexturePtr() : IntPtr.Zero;
            NativeRenderPlugin.NR_BUAV_SetTexture(_handle, (uint)index, ptr, (uint)mipSlice, dxgiFormat);
        }

        /// <summary>Sets a slot directly from a native D3D12 resource pointer.</summary>
        public void SetNativePtr(int index, IntPtr d3d12Ptr, int mipSlice = 0, uint dxgiFormat = 0)
        {
            if (index < 0 || index >= _capacity)
                throw new ArgumentOutOfRangeException(nameof(index));
            if (!IsValid) return;
            NativeRenderPlugin.NR_BUAV_SetTexture(_handle, (uint)index, d3d12Ptr, (uint)mipSlice, dxgiFormat);
        }

        // -------------------------------------------------------------------
        // Resize
        // -------------------------------------------------------------------

        /// <summary>
        /// Resizes the UAV descriptor array to <paramref name="newCapacity"/> slots.
        /// After resize the GPU handle changes; re-bind to all shaders that reference it.
        /// </summary>
        public void Resize(int newCapacity)
        {
            if (newCapacity < 1) newCapacity = 1;
            if (!IsValid) return;
            NativeRenderPlugin.NR_BUAV_Resize(_handle, (uint)newCapacity);
            _capacity = newCapacity;
        }

        // -------------------------------------------------------------------
        // IDisposable
        // -------------------------------------------------------------------

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_handle != 0)
            {
                NativeRenderPlugin.NR_DestroyBindlessUAVTexture(_handle);
                _handle = 0;
            }
        }
    }
}
