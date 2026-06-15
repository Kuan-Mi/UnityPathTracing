/*
 * C# mirror of RTXPT SampleConstantBuffer.h and PathTracerShared.h.
 *
 * Layout rules:
 *   - StructLayout(Sequential, Pack = 4) — matches HLSL's 4-byte minimum alignment.
 *   - HLSL bool in cbuffer is 4 bytes (maps to uint / int in C#).
 *   - float3x4 (row_major) = 3 rows × 4 floats = 3 × Vector4 = 48 bytes.
 *   - float4x4 (row_major) = float4x4 = 64 bytes.
 *     Unity float4x4 is column-major internally; callers are responsible for
 *     transposing before upload when HLSL expects row-major (use float4x4.transpose).
 *
 * Sizes (verified against HLSL offsets):
 *   PathTracerCameraData          : 112 B
 *   PathTracerConstants           : 368 B   (144 B + 2 × 112 B cameras)
 *   SimpleViewConstants           : 368 B   (5 × mat4 + 6 × float2)
 *   EnvMapSceneParams             : 112 B
 *   EnvMapImportanceSamplingParams:  16 B
 *   DebugConstants                :  64 B
 *   SampleConstants               : 1328 B
 *   SampleMiniConstants           :  64 B
 */

using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;

namespace PathTracing
{
    // ─────────────────────────────────────────────────────────────────────────
    // PathTracerCameraData  (PathTracerShared.h)
    // ─────────────────────────────────────────────────────────────────────────
    /// <summary>
    /// Condensed version of Falcor CameraData, used by the RTXPT path tracer.
    /// 112 bytes.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct PathTracerCameraData
    {
        public Vector3 PosW;                    // +0   world-space camera position
        public float   NearZ;                   // +12
        public Vector3 DirectionW;              // +16  normalised forward direction
        public float   PixelConeSpreadAngle;    // +28  for ray-cone LOD
        public Vector3 CameraU;                 // +32  right vector × (fov-adjusted half-width)
        public float   FarZ;                    // +44
        public Vector3 CameraV;                 // +48  up vector × (fov-adjusted half-height)
        public float   FocalDistance;           // +60
        public Vector3 CameraW;                 // +64  forward vector × focal-distance
        public float   AspectRatio;             // +76  width / height
        public uint    ViewportSizeX;           // +80
        public uint    ViewportSizeY;           // +84
        public float   ApertureRadius;          // +88
        public float   _padding0;               // +92
        public float   JitterX;                 // +96
        public float   JitterY;                 // +100
        public float   _padding1;               // +104
        public float   _padding2;               // +108
        // Total: 112 B
    }

    // ─────────────────────────────────────────────────────────────────────────
    // PathTracerConstants  (PathTracerShared.h)
    // ─────────────────────────────────────────────────────────────────────────
    /// <summary>
    /// Main per-frame path-tracer constants uploaded to g_Const (b0).
    /// 368 bytes.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct PathTracerConstants
    {
        // Row 0
        public uint  imageWidth;
        public uint  imageHeight;
        public uint  sampleBaseIndex;
        public float perPixelJitterAAScale;
        // Row 1
        public uint  bounceCount;
        public uint  diffuseBounceCount;
        public float environmentMapDiffuseSampleMIPLevel;
        public float texLODBias;
        // Row 2
        public float invSubSampleCount;
        public float fireflyFilterThreshold;    // 0 = disabled
        public float preExposedGrayLuminance;
        public uint  denoisingEnabled;
        // Row 3
        public uint  frameIndex;
        public uint  useReSTIRDI;
        public uint  useReSTIRGI;
        public uint  _padding5;
        // Row 4
        public float stablePlanesSplitStopThreshold;
        public float _padding3;
        public uint  _padding4;
        public float stablePlanesSuppressPrimaryIndirectSpecularK;
        // Row 5
        public float denoiserRadianceClampK;
        public float dlssRRBrightnessClampK;
        public float stablePlanesAntiAliasingFallthrough;
        public uint  activeStablePlaneCount;
        // Row 6
        public uint  maxStablePlaneVertexDepth;
        public uint  allowPrimarySurfaceReplacement;
        public uint  genericTSLineStride;   // stride for u_SurfaceData
        public uint  genericTSPlaneStride;
        // Row 7
        public uint  neeEnabled;
        public uint  neeType;
        public uint  neeCandidateSamples;
        public uint  neeFullSamples;
        // Row 8
        public uint  _padding6;
        public uint  stfMagnificationMethod; // STF = Stochastic Texture Filtering; 0 when disabled
        public uint  stfFilterMode;
        public float stfGaussianSigma;
        // Camera data (current and previous frame)
        public PathTracerCameraData camera;      // +144
        public PathTracerCameraData prevCamera;  // +256
        // Total: 368 B
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SimpleViewConstants  (SampleConstantBuffer.h)
    // ─────────────────────────────────────────────────────────────────────────
    /// <summary>
    /// Standard view matrices plus viewport helpers.
    /// 368 bytes.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct SimpleViewConstants
    {
        public float4x4  matWorldToView;            // +0    (upload transposed!)
        public float4x4 matViewToClip;             // +64
        public float4x4 matWorldToClip;            // +128
        public float4x4 matWorldToClipNoOffset;    // +192
        public float4x4 matClipToWorldNoOffset;    // +256
        // +320
        public float2 viewportOrigin;
        public float2 viewportSize;
        // +336
        public float2 viewportSizeInv;
        public float2 pixelOffset;
        // +352
        public float2 clipToWindowScale;
        public float2 clipToWindowBias;
        // Total: 368 B
    }

    // ─────────────────────────────────────────────────────────────────────────
    // EnvMapSceneParams  (EnvMap.hlsli)
    // ─────────────────────────────────────────────────────────────────────────
    /// <summary>
    /// Environment map color/intensity/orientation.
    /// float3x4 (row_major) = 3 × Vector4 = 48 B each.
    /// 112 bytes.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct EnvMapSceneParams
    {
        // float3x4 Transform (local→world): 3 rows, each padded to float4
        public Vector4 TransformRow0;   // +0
        public Vector4 TransformRow1;   // +16
        public Vector4 TransformRow2;   // +32
        // float3x4 InvTransform (world→local)
        public Vector4 InvTransformRow0; // +48
        public Vector4 InvTransformRow1; // +64
        public Vector4 InvTransformRow2; // +80
        // float3 ColorMultiplier + float Enabled
        public Vector3 colorMultiplier; // +96
        public float   enabled;         // +108  (1.0 = on, 0.0 = off)
        // Total: 112 B
    }

    // ─────────────────────────────────────────────────────────────────────────
    // EnvMapImportanceSamplingParams  (EnvMap.hlsli)
    // ─────────────────────────────────────────────────────────────────────────
    /// <summary>16 bytes.</summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct EnvMapImportanceSamplingParams
    {
        public float importanceInvDimX;  // 1 / importance-map width
        public float importanceInvDimY;  // 1 / importance-map height
        public uint  importanceBaseMip;  // mip level for 1×1 resolution
        public uint  _padding0;
        // Total: 16 B
    }

    // ─────────────────────────────────────────────────────────────────────────
    // DebugConstants  (PathTracerDebug.hlsli)
    // ─────────────────────────────────────────────────────────────────────────
    /// <summary>
    /// Debug visualisation control block.
    /// 64 bytes.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct DebugConstants
    {
        public int   pickX;
        public int   pickY;
        public int   pick;                      // 1 = debug-pixel active
        public float debugLineScale;
        // HLSL bool is 4 B → use uint
        public uint  showWireframe;
        public int   debugViewType;             // cast from DebugViewType enum
        public int   debugViewStablePlaneIndex; // -1 = all planes
        public int   exploreDeltaTree;
        public int   imageWidth;
        public int   imageHeight;
        public int   mouseX;
        public int   mouseY;
        public Vector3 cameraPosW;
        public float   _padding0;
        // Total: 64 B
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SampleConstants  (SampleConstantBuffer.h)
    // ─────────────────────────────────────────────────────────────────────────
    /// <summary>
    /// Top-level constant buffer (g_Const, b0) for the RTXPT pipeline.
    /// 1328 bytes.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct SampleConstants
    {
        public SimpleViewConstants              view;                              // +0    368 B
        public SimpleViewConstants              previousView;                      // +368  368 B
        public EnvMapSceneParams                envMapSceneParams;                 // +736  112 B
        public EnvMapImportanceSamplingParams   envMapImportanceSamplingParams;    // +848   16 B
        public PathTracerConstants              ptConsts;                          // +864  368 B
        public DebugConstants                   debug;                             // +1232  64 B
        public Vector4                          denoisingHitParamConsts;           // +1296  16 B
        
        public uint                             materialCount;                     // +1312
        public uint                             _padding0;                         // +1316
        public uint                             _padding1;                         // +1320
        public uint                             _padding2;                         // +1324
        // Total: 1328 B
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SampleMiniConstants  (SampleConstantBuffer.h)
    // ─────────────────────────────────────────────────────────────────────────
    /// <summary>
    /// Per-pass "push constants" (g_MiniConst, b1).
    /// Used for sub-sample index and other per-pass overrides.
    /// 64 bytes.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct SampleMiniConstants
    {
        public uint params0x;
        public uint params0y;
        public uint params0z;
        public uint params0w;

        public uint params1x;
        public uint params1y;
        public uint params1z;
        public uint params1w;

        public uint params2x;
        public uint params2y;
        public uint params2z;
        public uint params2w;

        public uint params3x;
        public uint params3y;
        public uint params3z;
        public uint params3w;
        // Total: 64 B
    }
}
