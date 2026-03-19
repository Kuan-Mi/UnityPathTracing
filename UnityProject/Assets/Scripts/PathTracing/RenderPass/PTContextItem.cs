using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;

namespace PathTracing
{
    public class PTContextItem: ContextItem
    {
        
        internal TextureHandle OutputTexture;
        internal TextureHandle DirectLighting;
        internal TextureHandle DirectEmission;
        internal TextureHandle ComposedDiff;
        internal TextureHandle ComposedSpecViewZ;
        
        public override void Reset()
        {
            OutputTexture = TextureHandle.nullHandle;
            DirectLighting = TextureHandle.nullHandle;
            DirectEmission = TextureHandle.nullHandle;
            ComposedDiff = TextureHandle.nullHandle;
            ComposedSpecViewZ = TextureHandle.nullHandle;
        }
    }
}