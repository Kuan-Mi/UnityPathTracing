using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using Object = UnityEngine.Object;

namespace Nri
{
    /// <summary>
    /// Texture2DArray variant of NriTextureResource.
    /// The D3D12 resource carries DepthOrArraySize in its own descriptor,
    /// so WrapD3D12Texture works without any native-side changes.
    /// </summary>
    public class NriTextureArrayResource : NriTextureResource
    {
        public int ArraySize { get; }

        public NriTextureArrayResource(string name, GraphicsFormat graphicsFormat, NriResourceState initialState, int arraySize, bool srgb = false)
            : base(name, graphicsFormat, initialState, srgb)
        {
            ArraySize = arraySize;
        }

        /// <summary>Allocates (or reallocates) a Texture2DArray at the given resolution.</summary>
        public new void Allocate(int2 resolution)
        {
            // Check if already allocated at the right size
            var existing = Handle?.rt;
            if (existing != null && existing.width == resolution.x && existing.height == resolution.y)
                return;

            Release();

            var dxgiFormat = NriUtil.GetDXGIFormat(GraphicsFormat);

            var desc = new RenderTextureDescriptor(resolution.x, resolution.y, GraphicsFormat, 0)
            {
                dimension         = TextureDimension.Tex2DArray,
                volumeDepth       = ArraySize,
                enableRandomWrite = true,
                useMipMap         = false,
                msaaSamples       = 1,
                sRGB              = SRGB,
            };

            var rt = new RenderTexture(desc)
            {
                name       = Name,
                filterMode = FilterMode.Point,
                wrapMode   = TextureWrapMode.Clamp,
            };
            rt.Create();

            Handle    = RTHandles.Alloc(rt);
            NativePtr = Handle.rt.GetNativeTexturePtr();
            NriPtr    = WrapD3D12TextureInternal(NativePtr, dxgiFormat);
        }
    }
}
