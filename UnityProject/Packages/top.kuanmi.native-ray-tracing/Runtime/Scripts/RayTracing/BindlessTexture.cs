using System;
using UnityEngine;

namespace NativeRender
{
    /// <summary>
    /// A GPU-visible array of Texture2D SRV descriptors backed by a contiguous range in the
    /// global DescriptorHeapAllocator.  Objects of this class are completely independent of any
    /// shader type and can be bound to a RayTraceShader (or future ComputeShader, etc.) via
    /// the SetBindlessTexture() method.
    ///
    /// Usage:
    ///   var bt = new BindlessTexture(8);
    ///   bt[0] = albedo;
    ///   bt[1] = normalMap;
    ///   shader.SetBindlessTexture("MyTextures", bt);
    ///   // each frame: shader.Dispatch(...)
    ///
    ///   // To grow/shrink the array at runtime:
    ///   bt.Resize(16);
    ///   shader.SetBindlessTexture("MyTextures", bt);  // rebind after resize!
    ///
    ///   bt.Dispose();
    ///
    /// HLSL declaration:
    ///   Texture2D MyTextures[] : register(t0);
    ///   // sample: MyTextures[index].Sample(sampler, uv)
    ///
    /// Notes:
    ///   • Null entries (unassigned slots) sample as black.
    ///   • After Resize() the GPU handle changes; re-bind to all shaders that reference it.
    ///   • Textures are non-owning: the BindlessTexture does not prevent Unity from
    ///     destroying the underlying GPU resource.
    /// </summary>
    public sealed class BindlessTexture : IDisposable
    {
        private ulong    _handle;
        private Texture[] _textures;
        private bool     _disposed;

        /// <summary>Opaque native handle. Used internally by RayTraceShader.SetBindlessTexture.</summary>
        public ulong Handle => _handle;

        /// <summary>Current number of texture slots in the array.</summary>
        public int Capacity => _textures.Length;

        public bool IsValid => _handle != 0 && !_disposed;

        // -------------------------------------------------------------------
        // Construction
        // -------------------------------------------------------------------

        /// <summary>
        /// Creates a BindlessTexture with |capacity| slots. All slots start as null.
        /// </summary>
        /// <param name="capacity">Number of texture slots. Must be >= 1.</param>
        public BindlessTexture(int capacity)
        {
            if (capacity < 1) capacity = 1;
            _handle   = NativeRenderPlugin.NR_CreateBindlessTexture((uint)capacity);
            _textures = new Texture[capacity];
            if (_handle == 0)
                Debug.LogError("[BindlessTexture] NR_CreateBindlessTexture failed");
        }

        // -------------------------------------------------------------------
        // Indexer — get/set individual texture slots
        // -------------------------------------------------------------------

        /// <summary>
        /// Gets or sets the texture at |index|. Setting null clears the slot (writes a null SRV).
        /// </summary>
        public Texture this[int index]
        {
            get
            {
                if (index < 0 || index >= _textures.Length)
                    throw new ArgumentOutOfRangeException(nameof(index));
                return _textures[index];
            }
            set
            {
                if (index < 0 || index >= _textures.Length)
                    throw new ArgumentOutOfRangeException(nameof(index));
                _textures[index] = value;
                if (IsValid)
                {
                    IntPtr ptr = value != null ? value.GetNativeTexturePtr() : IntPtr.Zero;
                    NativeRenderPlugin.NR_BT_SetTexture(_handle, (uint)index, ptr);
                }
            }
        }

        /// <summary>Sets a texture slot directly from a native resource pointer (e.g. from Texture.GetNativeTexturePtr or Mesh.GetNativeIndexBufferPtr).</summary>
        public void SetNativePtr(int index, IntPtr ptr)
        {
            if (index < 0 || index >= _textures.Length)
                throw new ArgumentOutOfRangeException(nameof(index));
            _textures[index] = null; // native-only slot — no managed Texture reference
            if (IsValid)
                NativeRenderPlugin.NR_BT_SetTexture(_handle, (uint)index, ptr);
        }

        // -------------------------------------------------------------------
        // Resize
        // -------------------------------------------------------------------

        /// <summary>
        /// Resizes the texture array to |newCapacity| slots.
        /// <para>
        /// On grow: new slots are null. Existing textures at indices below the old capacity
        /// are re-registered in the new descriptor range automatically.
        /// </para>
        /// <para>
        /// On shrink: textures beyond |newCapacity| are dropped from the array.
        /// </para>
        /// <para>
        /// IMPORTANT: After resize the GPU descriptor base address changes.
        /// Call shader.SetBindlessTexture(name, this) to rebind before the next dispatch.
        /// </para>
        /// </summary>
        public void Resize(int newCapacity)
        {
            if (newCapacity < 1) newCapacity = 1;
            if (!IsValid) return;

            // Resize the native heap range (the C++ side re-writes all existing SRVs)
            NativeRenderPlugin.NR_BT_Resize(_handle, (uint)newCapacity);

            // Resize the C# tracking array
            int oldCapacity = _textures.Length;
            Array.Resize(ref _textures, newCapacity);

            // If we grew, re-upload any previously null slots that are now in range
            // (the native side already wrote null SRVs for new slots, so nothing extra needed).
            // If we shrank, entries beyond newCapacity were already dropped by Array.Resize.
            _ = oldCapacity; // suppress unused-variable warning
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
                NativeRenderPlugin.NR_DestroyBindlessTexture(_handle);
                _handle = 0;
            }
            _textures = Array.Empty<Texture>();
        }
    }
}
