using Nrd;
using Unity.Mathematics;
using UnityEngine;

namespace PathTracing
{
    [System.Serializable]
    public class PathTracingSetting
    {
        [Range(1, 10)] public int bounceCountOpaque = 5;
        [Range(1, 10)] public int bounceCountTransparent = 5;
        [Range(1, 128)] public int sampleCount = 1;

        [Range(0.001f, 10f)] public float lightOffset = 0.0001f;

        public Cubemap envTexture = null;

        public bool enableRussianRoulette = true;

        [Header("NRD Common Settings")] [Range(0.1f, 1000000.0f)]
        public float denoisingRange = 500000; // Default: 500000.0f

        [Range(0.01f, 0.02f)] public float disocclusionThreshold = 0.01f; // Default: 0.01f
        [Range(0.02f, 0.2f)] public float disocclusionThresholdAlternate = 0.05f; // Default: 0.05f
        [Range(0.0f, 1.0f)] public float splitScreen; // Default: 0.0f

        public bool isMotionVectorInWorldSpace; // Default: false
        public bool isHistoryConfidenceAvailable; // Default: false
        public bool isDisocclusionThresholdMixAvailable; // Default: false
        public bool isBaseColorMetalnessAvailable; // Default: false
        public bool enableValidation; // Default: false

        [Header("NRD Sigma Settings")] [Range(0.0f, 1.0f)]
        public float planeDistanceSensitivity = 0.02f; // Default: 0.02f

        [Range(0, 7)] public uint maxStabilizedFrameNum = 5; // Default: 5

        [Header("NRD Common Settings Override")]
        public bool useOverriddenCommonSettings;

        public Matrix4x4 viewToClipMatrix;
        public Matrix4x4 viewToClipMatrixPrev;
        public Matrix4x4 worldToViewMatrix;
        public Matrix4x4 worldToViewMatrixPrev;
        public float3 motionVectorScale = new(1.0f, 1.0f, 0.0f); // Default: {1.0, 1.0, 0.0}

        [Header("NRD Sigma Settings Override")]
        public bool useOverriddenSigmaValues;

        public Vector3 lightDir;
    }
}