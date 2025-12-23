using Unity.Mathematics;

namespace PathTracing
{
    public struct GlobalConstants
    {
        public float4x4 gViewToWorld;
        public float4x4 gWorldToView;
        public float4x4 gWorldToViewPrev;
        public float4x4 gWorldToClip;
        public float4x4 gWorldToClipPrev;

        public float4 gCameraFrustum;
        public float4 gSunBasisX;
        public float4 gSunBasisY;
        public float4 gSunDirection;

        public float2 gRectSize;
        public float2 gInvRectSize;
        public float2 gJitter;
        public float2 gRectSizePrev;


        public float2 gRenderSize;
        public float2 gInvRenderSize;

        public float gTanPixelAngularRadius;
        public float gUnproject;
        public float gTanSunAngularRadius;
        public float gNearZ;

        public float gAperture;
        public float gFocalDistance;
        public float gExposure;
        public uint gFrameIndex;


        public float gTAA;
        public uint gSampleNum;
        public uint gBounceNum;
        public float vbv;
    }
}