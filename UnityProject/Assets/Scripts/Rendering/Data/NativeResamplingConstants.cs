using System.Runtime.InteropServices;
using Rtxdi;
using Rtxdi.DI;
using Rtxdi.GI;
using Rtxdi.PT;
using Rtxdi.ReGIR;
using Unity.Mathematics;

namespace PathTracing
{
    // -------------------------------------------------------------------------
    // Mirrors SceneConstants from ShaderParameters.h
    // -------------------------------------------------------------------------
    [StructLayout(LayoutKind.Sequential)]
    public struct NativeSceneConstants
    {
        public uint  enableEnvironmentMap;        // Global
        public uint  environmentMapTextureIndex;  // Global
        public float environmentScale;
        public float environmentRotation;

        public uint  enableAlphaTestedGeometry;
        public uint  enableTransparentGeometry;
        public uint2 pad1;
    }

    // -------------------------------------------------------------------------
    // Mirrors ReSTIRShaderDebugParameters from
    //   ShaderDebug/ReSTIRShaderDebugParameters.h
    // -------------------------------------------------------------------------
    [StructLayout(LayoutKind.Sequential)]
    public struct NativeReSTIRShaderDebugParameters
    {
        public uint2  mouseSelectedPixel;
        public float  spatialResamplingScreenSplitRatio;
        public uint   outputDebugDirectLighting;

        public uint   outputDebugIndirectLighting;
        public uint   pad1;
        public uint   pad2;
        public uint   pad3;
    }

    // -------------------------------------------------------------------------
    // Mirrors PTNeeParameters from PTParameters.h
    // -------------------------------------------------------------------------
    [StructLayout(LayoutKind.Sequential)]
    public struct NativePTNeeParameters
    {
        public RTXDI_DIInitialSamplingParameters   initialSamplingParams;
        public RTXDI_DISpatialResamplingParameters spatialResamplingParams;
    }

    // -------------------------------------------------------------------------
    // Mirrors PTParameters from PTParameters.h
    // -------------------------------------------------------------------------
    [StructLayout(LayoutKind.Sequential)]
    public struct NativePTParameters
    {
        public uint  enableRussianRoulette;
        public float russianRouletteContinueChance;
        public uint  enableSecondaryDISpatialResampling;
        public uint  copyReSTIRDISimilarityThresholds;

        public NativePTNeeParameters nee;

        public uint  sampleEnvMapOnSecondaryMiss;
        public uint  sampleEmissivesOnSecondaryHit;
        public uint  lightSamplingMode;           // PTInitialSamplingLightSamplingMode
        public uint  extraMirrorBounceBudget;

        public float minimumPathThroughput;
        public uint  pad1;
        public uint  pad2;
        public uint  pad3;
    }

    // -------------------------------------------------------------------------
    // Mirrors ResamplingConstants from ShaderParameters.h (shader-side layout).
    // Use this instead of ResamplingConstants when passing data to native shaders
    // via NativeRtxdiGenerateInitialSamplesPass (and related passes).
    // -------------------------------------------------------------------------
    [StructLayout(LayoutKind.Sequential)]
    public struct NativeResamplingConstants
    {
        // View matrices
        public NativePlanarViewConstants view;
        public NativePlanarViewConstants prevView;
        public NativePlanarViewConstants prevPrevView;

        // Runtime
        public RTXDI_RuntimeParameters runtimeParams;

        public float4 reblurHitDistParams;

        public uint pad3;
        public uint enablePreviousTLAS;
        public uint denoiserMode;
        public uint discountNaiveSamples;

        public uint             enableBrdfIndirect;
        public uint             enableBrdfAdditiveBlend;
        public uint             enableAccumulation;   // StoreShadingOutput
        public uint             directLightingMode;   // DirectLightingMode (uint)

        public NativeSceneConstants sceneConstants;

        // Common buffer params
        public RTXDI_LightBufferParameters          lightBufferParams;
        public RTXDI_RISBufferSegmentParameters     localLightsRISBufferSegmentParams;
        public RTXDI_RISBufferSegmentParameters     environmentLightRISBufferSegmentParams;

        // Algo-specific params
        public RTXDI_Parameters    restirDI;
        public ReGIR_Parameters    regir;
        public RTXDI_GIParameters  restirGI;
        public RTXDI_PTParameters  restirPT;
        public NativePTParameters  pt;
        public BRDFPathTracing_Parameters brdfPT;

        public uint visualizeRegirCells;
        public uint enableDenoiserPSR;
        public uint usePSRMvecForResampling;
        public uint updatePSRwithResampling;

        public uint2 environmentPdfTextureSize;
        public uint2 localLightPdfTextureSize;

        public NativeReSTIRShaderDebugParameters debug;
    }
}
