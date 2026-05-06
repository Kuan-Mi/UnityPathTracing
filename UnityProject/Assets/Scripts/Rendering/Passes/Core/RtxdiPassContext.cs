using mini;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace PathTracing
{
    /// <summary>
    /// Shared per-frame context passed to all RTXDI render passes.
    /// Each pass reads only the fields it needs; unused fields may be null.
    /// </summary>
    public class RtxdiPassContext
    {
        // --- Constant buffers ---
        public GraphicsBuffer ConstantBuffer;
        public GraphicsBuffer ResamplingConstantBuffer;

        // --- Scene buffers ---
        public GraphicsBuffer GeometryInstanceToLight;

        // --- Current GBuffer ---
        public RTHandle ViewDepth;
        public RTHandle DiffuseAlbedo;
        public RTHandle SpecularRough;
        public RTHandle Normals;
        public RTHandle GeoNormals;

        // --- Previous GBuffer (temporal passes) ---
        public RTHandle PrevViewDepth;
        public RTHandle PrevDiffuseAlbedo;
        public RTHandle PrevSpecularRough;
        public RTHandle PrevNormals;
        public RTHandle PrevGeoNormals;

        // --- Lighting textures ---
        public RTHandle DirectLighting;
        public RTHandle Emissive;
        public RTHandle MotionVectors;
        public RTHandle LocalLightPdfTexture;

        // --- RTXDI resources ---
        public RtxdiResources RtxdiResources;

        // --- Render dimensions ---
        public int2 RenderResolution;
        public float ResolutionScale;
    }
}
