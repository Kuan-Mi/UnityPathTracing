using System;
using mini;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace PathTracing
{
    /// <summary>
    /// Per-frame context passed to NativeRtxdi* compute passes.
    /// Kept separate from <see cref="RtxdiPassContext"/> so the managed RayTracingShader pipeline
    /// and the NativeComputeShader pipeline can evolve independently.
    /// </summary>
    public class NativeRtxdiPassContext
    {
        // --- Constant buffers ---
        // g_Const (b0) — GlobalConstants for prep passes, ResamplingConstants for DI/GI passes.
        // public GraphicsBuffer ConstantBuffer;
        public GraphicsBuffer ResamplingConstantBuffer;
        // g_Const (b0) for RaytracedGBuffer — holds NativeGBufferConstants (GBufferConstants layout).
        // Must NOT be confused with ConstantBuffer which holds GlobalConstants.
        public GraphicsBuffer GBufferConstantBuffer;
        // g_PerPassConstants (b1) — holds NativeRtxdiPerPassConstants.
        public GraphicsBuffer PerPassConstantBuffer;

        // --- Scene bindings ---
        // Provides TLAS, t_InstanceData / t_GeometryData / t_MaterialConstants / bindless arrays
        // in donut-compatible layout for all RTXDI-native passes.
        public NativeRtxdiGPUScene RtxdiGpuScene;

        // --- GBuffer (current frame) — bound as IntPtr SRVs via NativeComputeDescriptorSet.SetTexture(string, IntPtr) ---
        public IntPtr ViewDepthPtr;
        public IntPtr DiffuseAlbedoPtr;
        public IntPtr SpecularRoughPtr;
        public IntPtr NormalsPtr;
        public IntPtr GeoNormalsPtr;

        // --- GBuffer (previous frame, temporal passes) ---
        public IntPtr PrevViewDepthPtr;
        public IntPtr PrevDiffuseAlbedoPtr;
        public IntPtr PrevSpecularRoughPtr;
        public IntPtr PrevNormalsPtr;
        public IntPtr PrevGeoNormalsPtr;

        // --- Lighting / aux textures ---
        public IntPtr EmissivePtr;
        public IntPtr MotionVectorsPtr;             // t_MotionVectors           (t11)
        public IntPtr DenoiserNormalRoughnessPtr;   // t_DenoiserNormalRoughness (t12)
        public IntPtr LocalLightPdfTexturePtr;
        public IntPtr EnvironmentPdfTexturePtr;
        public IntPtr DeviceDepthPtr;               // u_DeviceDepth (clip-space z written by RaytracedGBuffer)

        // --- DI screen-sized UAVs (DI temporal/spatial/shade) ---
        public IntPtr DiffuseLightingPtr;           // u_DiffuseLighting          (u1)
        public IntPtr SpecularLightingPtr;          // u_SpecularLighting         (u2)
        public IntPtr TemporalSamplePositionsPtr;   // u_TemporalSamplePositions  (u3)
        public IntPtr GradientsPtr;                 // u_Gradients                (u4)
        public IntPtr RestirLuminancePtr;           // u_RestirLuminance          (u5)
        public IntPtr PrevRestirLuminancePtr;       // t_PrevRestirLuminance      (t10)
        public IntPtr DirectLightingRawPtr;         // u_DirectLightingRaw        (u17)
        public IntPtr IndirectLightingRawPtr;       // u_IndirectLightingRaw      (u18)

        // --- Confidence textures (ping-pong, written by ConfidencePass, consumed by NRD) ---
        public IntPtr DiffuseConfidencePtr;         // u_DiffuseConfidence (current frame output)
        public IntPtr PrevDiffuseConfidencePtr;     // t_PrevDiffuseConfidence (previous frame input)
        public IntPtr SpecularConfidencePtr;        // u_SpecularConfidence (current frame output)
        public IntPtr PrevSpecularConfidencePtr;    // t_PrevSpecularConfidence (previous frame input)

        // u_RayCountBuffer (u12): raw uint buffer for per-frame ray counting (debug / stats).
        // Stored as a GraphicsBuffer because it is a raw/typed buffer, not a texture.
        public GraphicsBuffer RayCountBuffer;

        // --- RTXDI resources (light data / reservoirs / RIS buffers) ---
        // FullSample-aligned bundle (no LightScene dependency); see NativeRtxdiResources.cs.
        public NativeRtxdiResources Resources;

        // Resource pool — exposed so pass authors can resolve additional IntPtr handles via Pool.GetPoint(...).
        public NativeRtxdiTextureResources Pool;

        // --- Render dimensions ---
        public int2 RenderResolution;
        public float ResolutionScale;
    }
}
