using Unity.Mathematics;
using UnityEngine;

namespace PathTracing
{
    public class RenderResource
    {
        public GraphicsBuffer ConstantBuffer;
        public GraphicsBuffer HashEntriesBuffer;
        public GraphicsBuffer AccumulationBuffer;
        public GraphicsBuffer ResolvedBuffer;

        public GraphicsBuffer SpotLightBuffer;
        public GraphicsBuffer AreaLightBuffer;
        public GraphicsBuffer PointLightBuffer;
        
        
        public int2 RenderResolution;
    }
}