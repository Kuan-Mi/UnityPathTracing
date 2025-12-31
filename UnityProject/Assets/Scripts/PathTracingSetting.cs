using UnityEngine;

namespace PathTracing
{
    public enum ShowMode
    {
        None,
        BaseColor,
        Metalness,
        Normal,
        Roughness,
        Shadow,
        Diffuse,
        Specular,
        DirectLight,
        Emissive,
        Out,
        ComposedDiff,
        ComposedSpec,
        Taa,
        Final,
    }
    
    public enum DenoiserType
    {
        DENOISER_REBLUR = 0,
        DENOISER_RELAX = 1,
        DENOISER_REFERENCE = 2,
    }

    public enum RESOLUTION
    {
        RESOLUTION_FULL = 0,
        RESOLUTION_FULL_PROBABILISTIC = 1,
        RESOLUTION_HALF = 2,
    }
 

    [System.Serializable]
    public class PathTracingSetting
    {
        [Range(0.001f, 10f)]
        public float sunAngularDiameter = 0.533f;

        [Header("NRD Common Settings")]
        [Range(0.1f, 10000.0f)]
        public float denoisingRange = 5000;

        [Range(0.0f, 1.0f)]
        public float splitScreen;

        public bool isBaseColorMetalnessAvailable;

        [Header("NRD Sigma Settings")]
        [Range(0.0f, 1.0f)]
        public float planeDistanceSensitivity = 0.02f;

        [Range(0, 7)]
        public uint maxStabilizedFrameNum = 5;

        [Header("显示模式")]
        public ShowMode showMode;

        public bool showMV;
        public bool showValidation;

        [Header("景深")]
        [Range(0, 100f)]
        public float dofAperture;

        [Range(0.1f, 10f)]
        public float dofFocalDistance;

        [Header("曝光")]
        [Range(0.1f, 100f)]
        public float exposure = 1.0f;

        // [Header("TAA")]
        // [Range(0f, 1f)]
        // public float taa = 1.0f;

        [Header("采样")]
        [Range(1, 4)]
        public uint rpp = 1;

        [Range(1, 4)]
        public uint bounceNum = 1;
 
        public float mipBias;
         
        public RESOLUTION tracingMode = RESOLUTION.RESOLUTION_FULL;
        public DenoiserType denoiser = DenoiserType.DENOISER_REBLUR;
        
        public float emissionIntensity = 1.0f;
        
        public bool cameraJitter = true;
        public bool psr = false;
        public bool emission = true;
        public bool usePrevFrame = true;
        public bool TAA = true;
        public bool indirectDiffuse = true;
        public bool indirectSpecular = true;
        public bool importanceSampling = false;
        public bool SHARC = true;
        public bool specularLobeTrimming = true;
        public bool boost = false;
        public bool SR = false;
        public bool RR = false;
    }
}