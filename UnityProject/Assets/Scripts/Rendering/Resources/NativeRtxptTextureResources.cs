using System;
using Nri;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace PathTracing
{
    /// <summary>
    /// Owns all per-camera render textures for <see cref="NativeRtxptFeature"/>.
    /// Formats mirror RenderTargets.cpp from the RTXPT sample.
    ///
    /// Denoising is performed by DLSS Ray Reconstruction (DLSS-RR) — no NRD.
    /// Stable plane count = 3 (cStablePlaneCount).
    /// </summary>
    public class NativeRtxptTextureResources : IDisposable
    {
        // ── Path tracer primary outputs ───────────────────────────────────────
        /// <summary>Main PT output color. RGBA16_FLOAT. Written by PT shader → NoDenoiserFinalMerge input.</summary>
        public NriTextureResource OutputColor;

        /// <summary>Screen-space depth. R32_FLOAT. Written by PT / ExportVisibilityBuffer.</summary>
        public NriTextureResource Depth;

        /// <summary>Screen-space motion vectors. RGBA16_FLOAT.</summary>
        public NriTextureResource ScreenMotionVectors;

        /// <summary>PSR path throughput packed as fp16×2. R32_UINT.</summary>
        public NriTextureResource Throughput;

        // ── Specular hit-distance filtering ──────────────────────────────────
        /// <summary>Specular hit distance. R32_FLOAT. Bilateral-filtered by DenoiseSpecHitT pass (×2 ping-pong).</summary>
        public NriTextureResource SpecularHitT;

        /// <summary>Scratch buffer for bilateral filter ping-pong. R32_FLOAT.</summary>
        public NriTextureResource ScratchFloat1;

        // ── Stable plane outputs ──────────────────────────────────────────────
        /// <summary>
        /// Stable plane header. R32_UINT, Texture2DArray with 4 slices.
        /// Slices 0-2 = branch IDs per plane, slice 3 = first-hit distance.
        /// </summary>
        public NriTextureResource StablePlanesHeader;

        /// <summary>Per-stable-plane stable radiance (portion not sent to denoiser). RGBA16_FLOAT.</summary>
        public NriTextureResource StableRadiance;

        // ── GBuffer (written by PT shader) ────────────────────────────────────
        /// <summary>Base color. B10G11R11_UFloatPack32.</summary>
        public NriTextureResource BaseColor;

        /// <summary>Specular normal packed. R32_UINT.</summary>
        public NriTextureResource SpecNormal;

        /// <summary>Roughness + metalness. RG16_FLOAT.</summary>
        public NriTextureResource RoughnessMetal;

        /// <summary>Material flags / info. R32_UINT.</summary>
        public NriTextureResource MaterialInfo;

        // ── DLSS-RR guide buffers (prepared by DlssBeforePass equivalent) ────
        /// <summary>Diffuse albedo guide for DLSS-RR. R11G11B10_FLOAT (sl::kBufferTypeAlbedo).</summary>
        public NriTextureResource DlssRrDiffAlbedo;

        /// <summary>Specular albedo guide for DLSS-RR. R11G11B10_FLOAT (sl::kBufferTypeSpecularAlbedo).</summary>
        public NriTextureResource DlssRrSpecAlbedo;

        /// <summary>Specular motion vectors guide for DLSS-RR. RG16_FLOAT (sl::kBufferTypeSpecularMotionVectors).</summary>
        public NriTextureResource DlssRrSpecMotionVectors;

        /// <summary>Normal + roughness guide for DLSS-RR. RGBA16_FLOAT.</summary>
        public NriTextureResource DlssRrNormalRoughness;

        // ── DLSS-RR output ────────────────────────────────────────────────────
        /// <summary>DLSS-RR denoised + upscaled output. RGBA16_FLOAT. Display resolution.</summary>
        public NriTextureResource DlssRrOutput;

        // ── Debug viz ─────────────────────────────────────────────────────────────
        /// <summary>Shader debug visualisation texture. R32_FLOAT. Bound as u_ShaderDebugVizTextureBuffer (u126).</summary>
        public NriTextureResource ShaderDebugViz;

        // ── Reference mode accumulation ───────────────────────────────────────
        /// <summary>Multi-frame accumulation buffer (reference mode only). RGBA32_FLOAT.</summary>
        public NriTextureResource AccumulatedRadiance;

        /// <summary>Post-accumulation output (reference mode). RGBA16_FLOAT. Display resolution.</summary>
        public NriTextureResource ProcessedOutputColor;

        // ── Resolved dimensions ───────────────────────────────────────────────
        public int2 renderResolution  { get; private set; }
        public int2 displayResolution { get; private set; }

        public NativeRtxptTextureResources()
        {
            var srv = new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE,         layout = Layout.SHADER_RESOURCE,         stageBits = 1 << 7  };
            var uav = new NriResourceState { accessBits = AccessBits.SHADER_RESOURCE_STORAGE, layout = Layout.SHADER_RESOURCE_STORAGE, stageBits = 1 << 10 };

            OutputColor           = new NriTextureResource("Rtxpt_OutputColor",           GraphicsFormat.R16G16B16A16_SFloat,     uav);
            Depth                 = new NriTextureResource("Rtxpt_Depth",                 GraphicsFormat.R32_SFloat,               uav);
            ScreenMotionVectors   = new NriTextureResource("Rtxpt_ScreenMotionVectors",   GraphicsFormat.R16G16B16A16_SFloat,     uav);
            Throughput            = new NriTextureResource("Rtxpt_Throughput",            GraphicsFormat.R32_UInt,                 uav);

            SpecularHitT          = new NriTextureResource("Rtxpt_SpecularHitT",          GraphicsFormat.R32_SFloat,               uav);
            ScratchFloat1         = new NriTextureResource("Rtxpt_ScratchFloat1",         GraphicsFormat.R32_SFloat,               uav);

            StablePlanesHeader    = new NriTextureResource("Rtxpt_StablePlanesHeader",    GraphicsFormat.R32_UInt,                 uav);
            StableRadiance        = new NriTextureResource("Rtxpt_StableRadiance",        GraphicsFormat.R16G16B16A16_SFloat,     uav);

            BaseColor             = new NriTextureResource("Rtxpt_BaseColor",             GraphicsFormat.B10G11R11_UFloatPack32,  uav);
            SpecNormal            = new NriTextureResource("Rtxpt_SpecNormal",            GraphicsFormat.R32_UInt,                 uav);
            RoughnessMetal        = new NriTextureResource("Rtxpt_RoughnessMetal",        GraphicsFormat.R16G16_SFloat,            uav);
            MaterialInfo          = new NriTextureResource("Rtxpt_MaterialInfo",          GraphicsFormat.R32_UInt,                 uav);

            DlssRrDiffAlbedo      = new NriTextureResource("Rtxpt_DlssRrDiffAlbedo",      GraphicsFormat.B10G11R11_UFloatPack32,  uav);
            DlssRrSpecAlbedo      = new NriTextureResource("Rtxpt_DlssRrSpecAlbedo",      GraphicsFormat.B10G11R11_UFloatPack32,  uav);
            DlssRrSpecMotionVectors = new NriTextureResource("Rtxpt_DlssRrSpecMotionVectors", GraphicsFormat.R16G16_SFloat,          uav);
            DlssRrNormalRoughness = new NriTextureResource("Rtxpt_DlssRrNormalRoughness", GraphicsFormat.R16G16B16A16_SFloat,     uav);

            DlssRrOutput          = new NriTextureResource("Rtxpt_DlssRrOutput",          GraphicsFormat.R16G16B16A16_SFloat,     uav);

            ShaderDebugViz        = new NriTextureResource("Rtxpt_ShaderDebugViz",        GraphicsFormat.R32_SFloat,               uav);
            AccumulatedRadiance   = new NriTextureResource("Rtxpt_AccumulatedRadiance",   GraphicsFormat.R32G32B32A32_SFloat,     uav);
            ProcessedOutputColor  = new NriTextureResource("Rtxpt_ProcessedOutputColor",  GraphicsFormat.R16G16B16A16_SFloat,     uav);
        }

        /// <summary>
        /// Allocates or reallocates all textures for the given render and display resolutions.
        /// Returns true if any allocation occurred (callers must re-bind DLSS-RR resources).
        /// </summary>
        public bool EnsureResources(int2 renderRes, int2 displayRes)
        {
            bool sameRender  = OutputColor.IsCreated
                               && renderResolution.x  == renderRes.x
                               && renderResolution.y  == renderRes.y;
            bool sameDisplay = DlssRrOutput.IsCreated
                               && displayResolution.x == displayRes.x
                               && displayResolution.y == displayRes.y;
            if (sameRender && sameDisplay) return false;

            renderResolution  = renderRes;
            displayResolution = displayRes;

            foreach (var tex in RenderResolutionTextures())
                tex.Allocate(renderResolution);

            // StablePlanesHeader is a Texture2DArray with 4 slices
            StablePlanesHeader.Allocate(renderResolution, slices: 4);

            // Display-resolution textures
            DlssRrOutput.Allocate(displayResolution);
            ProcessedOutputColor.Allocate(displayResolution);

            return true;
        }

        private NriTextureResource[] RenderResolutionTextures() => new[]
        {
            OutputColor, Depth, ScreenMotionVectors, Throughput,
            SpecularHitT, ScratchFloat1,
            StableRadiance,
            BaseColor, SpecNormal, RoughnessMetal, MaterialInfo,
            DlssRrDiffAlbedo, DlssRrSpecAlbedo, DlssRrSpecMotionVectors, DlssRrNormalRoughness,
            ShaderDebugViz, AccumulatedRadiance,
        };

        public void Dispose()
        {
            foreach (var tex in AllTextures())
            {
                if (!tex.IsCreated) continue;
                var h = tex.Handle;
                if (h?.rt != null)
                {
                    AsyncGPUReadback.Request(h).WaitForCompletion();
                    break;
                }
            }
            foreach (var tex in AllTextures()) tex.Release();
        }

        private NriTextureResource[] AllTextures() => new[]
        {
            OutputColor, Depth, ScreenMotionVectors, Throughput,
            SpecularHitT, ScratchFloat1,
            StablePlanesHeader, StableRadiance,
            BaseColor, SpecNormal, RoughnessMetal, MaterialInfo,
            DlssRrDiffAlbedo, DlssRrSpecAlbedo, DlssRrSpecMotionVectors, DlssRrNormalRoughness,
            ShaderDebugViz, DlssRrOutput, AccumulatedRadiance, ProcessedOutputColor,
        };
    }
}