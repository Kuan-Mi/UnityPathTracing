using Unity.Mathematics;

namespace PathTracing
{
    public struct GlobalConstants
    {
       public float4x4 gViewToWorld;
       public float4x4 gViewToClip;
       public float4x4 gWorldToView;
       public float4x4 gWorldToViewPrev;
       public float4x4 gWorldToClip;
       public float4x4 gWorldToClipPrev;
       
       public float4 gHitDistParams;
       public float4 gCameraFrustum;
       public float4 gSunBasisX;
       public float4 gSunBasisY;
       public float4 gSunDirection;
       public float4 gCameraGlobalPos;
       public float4 gCameraGlobalPosPrev;
       public float4 gViewDirection;
       public float4 gHairBaseColor;
       
       public float2 gHairBetas;
       public float2 gOutputSize; // represents native resolution ( >= gRenderSize )
       public float2 gRenderSize; // up to native resolution ( >= gRectSize )
       public float2 gRectSize; // dynamic resolution scaling
       public float2 gInvOutputSize;
       public float2 gInvRenderSize;
       public float2 gInvRectSize;
       public float2 gRectSizePrev;
       public float2 gJitter;
       
       public float gEmissionIntensity;
       public float gNearZ;
       public float gSeparator;
       public float gRoughnessOverride;
       public float gMetalnessOverride;
       public float gUnitToMetersMultiplier;
       public float gTanSunAngularRadius;
       public float gTanPixelAngularRadius;
       public float gDebug;
       public float gPrevFrameConfidence;
       public float gUnproject;
       public float gAperture;
       public float gFocalDistance;
       public float gFocalLength;
       public float gTAA;
       public float gHdrScale;
       public float gExposure;
       public float gMipBias;
       public float gOrthoMode;
       public float gIndirectDiffuse;
       public float gIndirectSpecular;
       public float gMinProbability;
       
       public uint gSharcMaxAccumulatedFrameNum;
       public uint gDenoiserType;
       public uint gDisableShadowsAndEnableImportanceSampling; // TODO: remove - modify GetSunIntensity to return 0 if sun is below horizon
       public uint gFrameIndex;
       public uint gForcedMaterial;
       public uint gUseNormalMap;
       public uint gBounceNum;
       public uint gResolve;
       public uint gValidation;
       public uint gSR;
       public uint gRR;
       public uint gIsSrgb;
       public uint gOnScreen;
       public uint gTracingMode;
       public uint gSampleNum;
       public uint gPSR;
       public uint gSHARC;
       public uint gTrimLobe;
    }
}