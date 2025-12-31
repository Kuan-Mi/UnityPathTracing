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

        public override string ToString()
        {
            return $@"GlobalConstants {{
    gViewToWorld = {{
    {FormatFloat4x4(gViewToWorld)}
    }},
    gViewToClip = {{
    {FormatFloat4x4(gViewToClip)}
    }},
    gWorldToView = {{
    {FormatFloat4x4(gWorldToView)}
    }},
    gWorldToViewPrev = {{
    {FormatFloat4x4(gWorldToViewPrev)}
    }},
    gWorldToClip = {{
    {FormatFloat4x4(gWorldToClip)}
    }},
    gWorldToClipPrev = {{
    {FormatFloat4x4(gWorldToClipPrev)}
    }},
    gHitDistParams = {FormatFloat4(gHitDistParams)},
    gCameraFrustum = {FormatFloat4(gCameraFrustum)},
    gSunBasisX = {FormatFloat4(gSunBasisX)},
    gSunBasisY = {FormatFloat4(gSunBasisY)},
    gSunDirection = {FormatFloat4(gSunDirection)},
    gCameraGlobalPos = {FormatFloat4(gCameraGlobalPos)},
    gCameraGlobalPosPrev = {FormatFloat4(gCameraGlobalPosPrev)},
    gViewDirection = {FormatFloat4(gViewDirection)},
    gHairBaseColor = {FormatFloat4(gHairBaseColor)},
    gHairBetas = {FormatFloat2(gHairBetas)},
    gOutputSize = {FormatFloat2(gOutputSize)},
    gRenderSize = {FormatFloat2(gRenderSize)},
    gRectSize = {FormatFloat2(gRectSize)},
    gInvOutputSize = {FormatFloat2(gInvOutputSize)},
    gInvRenderSize = {FormatFloat2(gInvRenderSize)},
    gInvRectSize = {FormatFloat2(gInvRectSize)},
    gRectSizePrev = {FormatFloat2(gRectSizePrev)},
    gJitter = {FormatFloat2(gJitter)},
    gEmissionIntensity = {gEmissionIntensity:F6},
    gNearZ = {gNearZ:F6},
    gSeparator = {gSeparator:F6},
    gRoughnessOverride = {gRoughnessOverride:F6},
    gMetalnessOverride = {gMetalnessOverride:F6},
    gUnitToMetersMultiplier = {gUnitToMetersMultiplier:F6},
    gTanSunAngularRadius = {gTanSunAngularRadius:F6},
    gTanPixelAngularRadius = {gTanPixelAngularRadius:F6},
    gDebug = {gDebug:F6},
    gPrevFrameConfidence = {gPrevFrameConfidence:F6},
    gUnproject = {gUnproject:F6},
    gAperture = {gAperture:F6},
    gFocalDistance = {gFocalDistance:F6},
    gFocalLength = {gFocalLength:F6},
    gTAA = {gTAA:F6},
    gHdrScale = {gHdrScale:F6},
    gExposure = {gExposure:F6},
    gMipBias = {gMipBias:F6},
    gOrthoMode = {gOrthoMode:F6},
    gIndirectDiffuse = {gIndirectDiffuse:F6},
    gIndirectSpecular = {gIndirectSpecular:F6},
    gMinProbability = {gMinProbability:F6},
    gSharcMaxAccumulatedFrameNum = {gSharcMaxAccumulatedFrameNum},
    gDenoiserType = {gDenoiserType},
    gDisableShadowsAndEnableImportanceSampling = {gDisableShadowsAndEnableImportanceSampling},
    gFrameIndex = {gFrameIndex},
    gForcedMaterial = {gForcedMaterial},
    gUseNormalMap = {gUseNormalMap},
    gBounceNum = {gBounceNum},
    gResolve = {gResolve},
    gValidation = {gValidation},
    gSR = {gSR},
    gRR = {gRR},
    gIsSrgb = {gIsSrgb},
    gOnScreen = {gOnScreen},
    gTracingMode = {gTracingMode},
    gSampleNum = {gSampleNum},
    gPSR = {gPSR},
    gSHARC = {gSHARC},
    gTrimLobe = {gTrimLobe}
}}";
        }

        private string FormatFloat4(float4 v)
        {
            return $"{{ {v.x:F6}, {v.y:F6}, {v.z:F6}, {v.w:F6} }}";
        }

        private string FormatFloat2(float2 v)
        {
            return $"{{ {v.x:F6}, {v.y:F6} }}";
        }

        private string FormatFloat4x4(float4x4 m)
        {
            return $@"{{ {m.c0.x:F6}, {m.c0.y:F6}, {m.c0.z:F6}, {m.c0.w:F6} }},
    {{ {m.c1.x:F6}, {m.c1.y:F6}, {m.c1.z:F6}, {m.c1.w:F6} }},
    {{ {m.c2.x:F6}, {m.c2.y:F6}, {m.c2.z:F6}, {m.c2.w:F6} }},
    {{ {m.c3.x:F6}, {m.c3.y:F6}, {m.c3.z:F6}, {m.c3.w:F6} }}";
        }
    }
}