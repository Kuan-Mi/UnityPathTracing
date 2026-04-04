//
// This file was automatically generated. Please don't edit by hand. Execute Editor command [ Edit > Rendering > Generate Shader Includes ] instead
//

#ifndef GLOBALCONSTANTS_CS_HLSL
#define GLOBALCONSTANTS_CS_HLSL
// Generated from PathTracing.GlobalConstants
// PackingRules = Exact
CBUFFER_START(GlobalConstants)
    float4x4 gViewToWorld;
    float4x4 gViewToWorldPrev;
    float4x4 gViewToClip;
    float4x4 gWorldToView;
    float4x4 gWorldToViewPrev;
    float4x4 gWorldToClip;
    float4x4 gWorldToClipPrev;
    float4 gHitDistParams;
    float4 gCameraFrustum;
    float4 gSunBasisX;
    float4 gSunBasisY;
    float4 gSunDirection;
    float4 gCameraGlobalPos;
    float4 gCameraGlobalPosPrev;
    float4 gViewDirection;
    float4 gHairBaseColor;
    float2 gHairBetas;
    float2 gOutputSize;
    float2 gRenderSize;
    float2 gRectSize;
    float2 gInvOutputSize;
    float2 gInvRenderSize;
    float2 gInvRectSize;
    float2 gRectSizePrev;
    float2 gJitter;
    float gEmissionIntensity;
    float gNearZ;
    float gSeparator;
    float gRoughnessOverride;
    float gMetalnessOverride;
    float gUnitToMetersMultiplier;
    float gTanSunAngularRadius;
    float gTanPixelAngularRadius;
    float gDebug;
    float gPrevFrameConfidence;
    float gUnproject;
    float gAperture;
    float gFocalDistance;
    float gFocalLength;
    float gTAA;
    float gHdrScale;
    float gExposure;
    float gMipBias;
    float gOrthoMode;
    float gIndirectDiffuse;
    float gIndirectSpecular;
    float gMinProbability;
    uint gSharcMaxAccumulatedFrameNum;
    uint gDenoiserType;
    uint gDisableShadowsAndEnableImportanceSampling;
    uint gFrameIndex;
    uint gForcedMaterial;
    uint gUseNormalMap;
    uint gBounceNum;
    uint gResolve;
    uint gValidation;
    uint gSR;
    uint gRR;
    uint gIsSrgb;
    uint gOnScreen;
    uint gTracingMode;
    uint gSampleNum;
    uint gPSR;
    uint gSHARC;
    uint gTrimLobe;
    uint gSpotLightCount;
    uint gAreaLightCount;
    uint gPointLightCount;
    float gSssMinThreshold;
    float gSssTransmissionBsdfSampleCount;
    float gSssTransmissionPerBsdfScatteringSampleCount;
    float gSssScale;
    float gSssAnisotropy;
    float gSssMaxSampleRadius;
    float gIsEditor;
    uint gShowLight;
    float gSharcDownscale;
    float gSharcSceneScale;
    uint sharcDebug;
    uint maxLightIndex;
CBUFFER_END


#endif
