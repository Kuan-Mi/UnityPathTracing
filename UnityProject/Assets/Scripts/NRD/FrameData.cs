using System;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;

namespace Nrd
{
    // ===================================================================================
    // ENUMS
    // ===================================================================================

    public enum CheckerboardMode : byte
    {
        OFF = 0,
        BLACK,
        WHITE,
        MAX_NUM
    }

    public enum AccumulationMode : byte
    {
        CONTINUE = 0,
        RESTART,
        CLEAR_AND_RESTART,
        MAX_NUM
    }

    public enum HitDistanceReconstructionMode : byte
    {
        OFF = 0,
        AREA_3X3,
        AREA_5X5,
        MAX_NUM
    }

    // ===================================================================================
    // COMMON SETTINGS
    // ===================================================================================

    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct CommonSettings
    {
        // Matrices
        public Matrix4x4 viewToClipMatrix;
        public Matrix4x4 viewToClipMatrixPrev;
        public Matrix4x4 worldToViewMatrix;

        public Matrix4x4 worldToViewMatrixPrev;

        // 可选
        public Matrix4x4 worldPrevToWorldMatrix; // Default: Identity

        // Motion Vectors
        public float3 motionVectorScale; // Default: {1.0, 1.0, 0.0}

        // Jitter
        public float2 cameraJitter;
        public float2 cameraJitterPrev;

        // Resolution
        public fixed ushort resourceSize[2];
        public fixed ushort resourceSizePrev[2];
        public fixed ushort rectSize[2];
        public fixed ushort rectSizePrev[2];

        // Scalars
        [Range(0.0f, 2.0f)] public float viewZScale; // Default: 1.0f
        public float timeDeltaBetweenFrames; // Default: 0.0f
        [Range(0.0f, 1000000.0f)] public float denoisingRange; // Default: 500000.0f
        [Range(0.01f, 0.02f)] public float disocclusionThreshold; // Default: 0.01f
        [Range(0.02f, 0.2f)] public float disocclusionThresholdAlternate; // Default: 0.05f

        // Material IDs
        public float cameraAttachedReflectionMaterialID; // Default: 999.0f
        public float strandMaterialID; // Default: 999.0f
        public float historyFixAlternatePixelStrideMaterialID; // Default: 999.0f

        public float strandThickness; // Default: 80e-6f
        [Range(0.0f, 1.0f)] public float splitScreen; // Default: 0.0f

        public fixed ushort printfAt[2]; // Default: {9999, 9999}
        public float debug; // Default: 0.0f

        public fixed uint rectOrigin[2];

        public uint frameIndex; // Default: 0

        public AccumulationMode accumulationMode; // Default: CONTINUE

        // Bools (mapped to byte for interop safety)
        private byte _isMotionVectorInWorldSpace; // Default: false
        private byte _isHistoryConfidenceAvailable; // Default: false
        private byte _isDisocclusionThresholdMixAvailable; // Default: false
        private byte _isBaseColorMetalnessAvailable; // Default: false
        private byte _enableValidation; // Default: false

        // -----------------------------------------------------------------------
        // Boolean Properties
        // -----------------------------------------------------------------------
        public bool isMotionVectorInWorldSpace
        {
            get => _isMotionVectorInWorldSpace != 0;
            set => _isMotionVectorInWorldSpace = value ? (byte)1 : (byte)0;
        }

        public bool isHistoryConfidenceAvailable
        {
            get => _isHistoryConfidenceAvailable != 0;
            set => _isHistoryConfidenceAvailable = value ? (byte)1 : (byte)0;
        }

        public bool isDisocclusionThresholdMixAvailable
        {
            get => _isDisocclusionThresholdMixAvailable != 0;
            set => _isDisocclusionThresholdMixAvailable = value ? (byte)1 : (byte)0;
        }

        public bool isBaseColorMetalnessAvailable
        {
            get => _isBaseColorMetalnessAvailable != 0;
            set => _isBaseColorMetalnessAvailable = value ? (byte)1 : (byte)0;
        }

        public bool enableValidation
        {
            get => _enableValidation != 0;
            set => _enableValidation = value ? (byte)1 : (byte)0;
        }
        
        public static readonly CommonSettings _default = CreateDefault();

        // -----------------------------------------------------------------------
        // Factory Method for C++ Defaults
        // -----------------------------------------------------------------------
        private static CommonSettings CreateDefault()
        {
            var s = new CommonSettings();

            s.worldPrevToWorldMatrix = Matrix4x4.identity;

            s.motionVectorScale[0] = 1.0f;
            s.motionVectorScale[1] = 1.0f;
            s.motionVectorScale[2] = 1.0f;
            s.isMotionVectorInWorldSpace = true;

            // Scalars
            s.viewZScale = 1.0f;
            s.timeDeltaBetweenFrames = 0.0f;
            s.denoisingRange = 500000.0f;
            s.disocclusionThreshold = 0.01f;
            s.disocclusionThresholdAlternate = 0.05f;

            // Material IDs
            s.cameraAttachedReflectionMaterialID = 999.0f;
            s.strandMaterialID = 999.0f;
            s.historyFixAlternatePixelStrideMaterialID = 999.0f;

            s.strandThickness = 0.00008f; // 80e-6f
            s.splitScreen = 0.0f;

            // printfAt = {9999, 9999}
            s.printfAt[0] = 9999;
            s.printfAt[1] = 9999;

            s.debug = 0.0f;
            s.frameIndex = 0;
            s.accumulationMode = AccumulationMode.CONTINUE;

            return s;
        }
    }

    // ===================================================================================
    // SIGMA SETTINGS
    // ===================================================================================

    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct SigmaSettings
    {
        public Vector3 lightDirection; // Default: {0.0, 0.0, 0.0}
        [Range(0.0f, 1.0f)] public float planeDistanceSensitivity; // Default: 0.02f
        [Range(0, 7)] public uint maxStabilizedFrameNum; // Default: 5
        public static SigmaSettings _default = CreateDefault();

        // -----------------------------------------------------------------------
        // Factory Method for C++ Defaults
        // -----------------------------------------------------------------------
        private static SigmaSettings CreateDefault()
        {
            var s = new SigmaSettings();

            // lightDirection defaults to 0,0,0 which is correct by default initialization

            s.planeDistanceSensitivity = 0.02f;
            s.maxStabilizedFrameNum = 5;

            return s;
        }
    }
    
    // ===================================================================================
    // REBLUR SETTINGS
    // ===================================================================================

    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct HitDistanceParameters
    {
        public float A; // Default: 3.0f
        public float B; // Default: 0.1f
        public float C; // Default: 20.0f
        public float D; // Default: -25.0f

        public static readonly HitDistanceParameters _default = CreateDefault();

        private static HitDistanceParameters CreateDefault()
        {
            return new HitDistanceParameters
            {
                A = 3.0f,
                B = 0.1f,
                C = 20.0f,
                D = -25.0f
            };
        }
    }

    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct ReblurAntilagSettings
    {
        public float luminanceSigmaScale; // Default: 4.0f
        public float luminanceSensitivity; // Default: 3.0f

        public static readonly ReblurAntilagSettings _default = CreateDefault();

        private static ReblurAntilagSettings CreateDefault()
        {
            return new ReblurAntilagSettings
            {
                luminanceSigmaScale = 4.0f,
                luminanceSensitivity = 3.0f
            };
        }
    }

    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct ResponsiveAccumulationSettings
    {
        public float roughnessThreshold; // Default: 0.0f
        public uint minAccumulatedFrameNum; // Default: 3

        public static readonly ResponsiveAccumulationSettings _default = CreateDefault();

        private static ResponsiveAccumulationSettings CreateDefault()
        {
            return new ResponsiveAccumulationSettings
            {
                roughnessThreshold = 0.0f,
                minAccumulatedFrameNum = 3
            };
        }
    }

    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct ReblurSettings
    {
        public HitDistanceParameters hitDistanceParameters;
        public ReblurAntilagSettings antilagSettings;
        public ResponsiveAccumulationSettings responsiveAccumulationSettings;

        public uint maxAccumulatedFrameNum; // Default: 30
        public uint maxFastAccumulatedFrameNum; // Default: 6
        public uint maxStabilizedFrameNum; // Default: 63 (REBLUR_MAX_HISTORY_FRAME_NUM)

        public uint historyFixFrameNum; // Default: 3
        public uint historyFixBasePixelStride; // Default: 14
        public uint historyFixAlternatePixelStride; // Default: 14

        public float fastHistoryClampingSigmaScale; // Default: 2.0f
        public float diffusePrepassBlurRadius; // Default: 30.0f
        public float specularPrepassBlurRadius; // Default: 50.0f
        public float minHitDistanceWeight; // Default: 0.1f
        public float minBlurRadius; // Default: 1.0f
        public float maxBlurRadius; // Default: 30.0f
        public float lobeAngleFraction; // Default: 0.15f
        public float roughnessFraction; // Default: 0.15f
        public float planeDistanceSensitivity; // Default: 0.02f

        public fixed float specularProbabilityThresholdsForMvModification[2]; // Default: {0.5f, 0.9f}

        public float fireflySuppressorMinRelativeScale; // Default: 2.0f
        public float minMaterialForDiffuse; // Default: 4.0f
        public float minMaterialForSpecular; // Default: 4.0f

        public CheckerboardMode checkerboardMode; // Default: OFF
        public HitDistanceReconstructionMode hitDistanceReconstructionMode; // Default: OFF

        // Bools (mapped to byte for interop safety)
        private byte _enableAntiFirefly; // Default: false
        private byte _usePrepassOnlyForSpecularMotionEstimation; // Default: false
        private byte _returnHistoryLengthInsteadOfOcclusion; // Default: false
        
        // -----------------------------------------------------------------------
        // Boolean Properties
        // -----------------------------------------------------------------------
        public bool enableAntiFirefly
        {
            get => _enableAntiFirefly != 0;
            set => _enableAntiFirefly = value ? (byte)1 : (byte)0;
        }

        public bool usePrepassOnlyForSpecularMotionEstimation
        {
            get => _usePrepassOnlyForSpecularMotionEstimation != 0;
            set => _usePrepassOnlyForSpecularMotionEstimation = value ? (byte)1 : (byte)0;
        }

        public bool returnHistoryLengthInsteadOfOcclusion
        {
            get => _returnHistoryLengthInsteadOfOcclusion != 0;
            set => _returnHistoryLengthInsteadOfOcclusion = value ? (byte)1 : (byte)0;
        }

        public static readonly ReblurSettings _default = CreateDefault();

        // -----------------------------------------------------------------------
        // Factory Method
        // -----------------------------------------------------------------------
        private static ReblurSettings CreateDefault()
        {
            var s = new ReblurSettings();
            
            s.hitDistanceParameters = HitDistanceParameters._default;
            s.antilagSettings = ReblurAntilagSettings._default;
            s.responsiveAccumulationSettings = ResponsiveAccumulationSettings._default;

            s.maxAccumulatedFrameNum = 30;
            s.maxFastAccumulatedFrameNum = 6;
            s.maxStabilizedFrameNum = 63; // REBLUR_MAX_HISTORY_FRAME_NUM

            s.historyFixFrameNum = 3;
            s.historyFixBasePixelStride = 14;
            s.historyFixAlternatePixelStride = 14;

            s.fastHistoryClampingSigmaScale = 2.0f;
            s.diffusePrepassBlurRadius = 30.0f;
            s.specularPrepassBlurRadius = 50.0f;

            s.minHitDistanceWeight = 0.1f;
            s.minBlurRadius = 1.0f;
            s.maxBlurRadius = 30.0f;

            s.lobeAngleFraction = 0.15f;
            s.roughnessFraction = 0.15f;
            s.planeDistanceSensitivity = 0.02f;

            s.specularProbabilityThresholdsForMvModification[0] = 0.5f;
            s.specularProbabilityThresholdsForMvModification[1] = 0.9f;

            s.fireflySuppressorMinRelativeScale = 2.0f;
            s.minMaterialForDiffuse = 4.0f;
            s.minMaterialForSpecular = 4.0f;

            s.checkerboardMode = CheckerboardMode.OFF;
            s.hitDistanceReconstructionMode = HitDistanceReconstructionMode.OFF;

            s.enableAntiFirefly = false;
            s.usePrepassOnlyForSpecularMotionEstimation = false;
            s.returnHistoryLengthInsteadOfOcclusion = false;

            return s;
        }
    }

    // ===================================================================================
    // FRAME DATA (Packed)
    // ===================================================================================

    [Serializable]
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct FrameData
    {
        public CommonSettings commonSettings;
        public SigmaSettings sigmaSettings;
        public ReblurSettings reblurSettings; // 新增

        public ushort width;
        public ushort height;

        // ID3D12Resource* pointers
        public IntPtr mvPointer;
        public IntPtr normalRoughnessPointer;
        public IntPtr viewZPointer;
        public IntPtr penumbraPointer;
        public IntPtr shadowTranslucencyPointer;
        public IntPtr diffRadiancePointer;      // IN_DIFF_RADIANCE_HITDIST
        public IntPtr outDiffRadiancePointer;   // OUT_DIFF_RADIANCE_HITDIST
        public IntPtr validationPointer;   

        public IntPtr nriMv;
        public IntPtr nriNormalRoughness;
        public IntPtr nriViewZ;
        public IntPtr nriPenumbra;
        public IntPtr nriShadowTranslucency;
        public IntPtr nriDiffRadiance;
        public IntPtr nriOutDiffRadiance;
        public IntPtr nriValidation;

        public int instanceId;
        
        public static FrameData _default = CreateDefault();

        // -----------------------------------------------------------------------
        // Factory Method for C++ Defaults
        // -----------------------------------------------------------------------
        private static FrameData CreateDefault()
        {
            return new FrameData
            {
                commonSettings = CommonSettings._default,
                sigmaSettings = SigmaSettings._default,
                reblurSettings = ReblurSettings._default,
                width = 0,
                height = 0,
                mvPointer = IntPtr.Zero,
                normalRoughnessPointer = IntPtr.Zero,
                viewZPointer = IntPtr.Zero,
                penumbraPointer = IntPtr.Zero,
                shadowTranslucencyPointer = IntPtr.Zero,
                diffRadiancePointer = IntPtr.Zero,
                outDiffRadiancePointer = IntPtr.Zero,
                validationPointer = IntPtr.Zero,
                instanceId = 0
            };
        }
    }
}