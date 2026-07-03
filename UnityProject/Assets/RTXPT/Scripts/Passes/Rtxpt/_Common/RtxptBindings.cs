using System;
using NativeRender;

namespace PathTracing
{
    /// <summary>
    /// Shared binding helper for Rtxpt compute and RT passes.
    ///
    /// All RTXPT shaders include <c>Shaders/Bindings/ShaderResourceBindings.hlsli</c> which
    /// declares the global binding set (g_Const b0, all output UAVs, stable planes, DLSS-RR guides, …).
    ///
    /// <see cref="NativeComputeDescriptorSet"/> and <see cref="RayTracePipeline"/> both silently
    /// ignore names absent from a shader's reflection map, so this helper safely binds the full
    /// superset; each shader only consumes what it actually references.
    /// </summary>
    internal static class RtxptBindings
    {
        private static NativeBindingLayout _globalLayout;
        private static NativeBindingLayout _globalLayoutNoBindless;

        /// <summary>
        /// The RTXPT global binding layout — Sample.cpp's globalBindingLayoutDesc plus
        /// the bindless layout, declared in the exact item order the original uses so
        /// the native plugin emits a GPU-identical root signature and combined
        /// descriptor table (see ShaderBase::BuildSharedRootSignature). Matches
        /// pipelines created with { m_bindingLayout, m_bindlessLayout }: the PT RT
        /// pipelines and the ExportVisibilityBuffer compute pass.
        /// </summary>
        public static NativeBindingLayout GlobalLayout => _globalLayout ??= CreateGlobalLayout(includeBindless: true);

        /// <summary>
        /// The global layout WITHOUT the bindless tail — matches RTXPT pipelines
        /// created with { m_bindingLayout } only: the PostProcess compute family
        /// (StablePlanesDebugViz, DLSSRR-PrepareInputs, NoDenoiserFinalMerge) and
        /// DenoisingGuidesBaker.
        /// </summary>
        public static NativeBindingLayout GlobalLayoutNoBindless => _globalLayoutNoBindless ??= CreateGlobalLayout(includeBindless: false);

        private static NativeBindingLayout CreateGlobalLayout(bool includeBindless)
        {
            var l = new NativeBindingLayout()
                .VolatileConstantBuffer(0)      // g_Const
                .PushConstants(1, 16)           // g_MiniConst (SampleMiniConstants = 16 dwords)
                .RayTracingAccelStruct(0)       // SceneBVH
                .StructuredBufferSRV(1)         // t_SubInstanceData
                .StructuredBufferSRV(2)         // t_InstanceData
                .StructuredBufferSRV(3)         // t_GeometryData
                .StructuredBufferSRV(4)         // t_GeometryDebugData
                .StructuredBufferSRV(5)         // t_PTMaterialData
                .TextureSRV(6)                  // t_LdrColorScratch
                .TextureSRV(10)                 // t_EnvironmentMap
                .TextureSRV(11)                 // t_EnvironmentMapImportanceMap (unused, kept for parity)
                .StructuredBufferSRV(12)        // t_LightsCB
                .StructuredBufferSRV(13)        // t_Lights
                .StructuredBufferSRV(14)        // t_LightsEx
                .StructuredBufferSRV(15)        // t_LightProxyCounters
                .StructuredBufferSRV(16)        // t_LightProxyIndices
                .StructuredBufferSRV(17)        // t_LightLocalSamplingBuffer
                .TextureSRV(18)                 // t_EnvLookupMap
                .TextureUAV(20)                 // u_LightFeedbackTotalWeight
                .TextureUAV(21)                 // u_LightFeedbackCandidates
                .Sampler(0)                     // s_MaterialSampler
                .Sampler(1)                     // s_EnvironmentMapSampler
                .Sampler(2)                     // s_EnvironmentMapImportanceSampler
                .TextureUAV(0)                  // u_OutputColor
                .TextureUAV(1)                  // u_ProcessedOutputColor
                .TextureUAV(2)                  // u_PostTonemapOutputColor
                .TextureUAV(4)                  // u_Throughput
                .TextureUAV(5)                  // u_MotionVectors
                .TextureUAV(6)                  // u_Depth
                .TextureUAV(7)                  // u_SpecularHitT
                .TextureUAV(8)                  // u_ScratchFloat1
                .TextureUAV(31)                 // u_DenoiserViewspaceZ
                .TextureUAV(32)                 // u_DenoiserMotionVectors
                .TextureUAV(33)                 // u_DenoiserNormalRoughness
                .TextureUAV(34)                 // u_DenoiserDiffRadianceHitDist
                .TextureUAV(35)                 // u_DenoiserSpecRadianceHitDist
                .TextureUAV(36)                 // u_DenoiserDisocclusionThresholdMix
                .TextureUAV(37)                 // u_CombinedHistoryClampRelax
                .StructuredBufferUAV(51)        // u_FeedbackBuffer
                .StructuredBufferUAV(52)        // u_DebugLinesBuffer
                .StructuredBufferUAV(53)        // u_DebugDeltaPathTree
                .StructuredBufferUAV(54)        // u_DeltaPathSearchStack
                .TextureUAV(60)                 // u_SecondarySurfacePositionNormal (ReSTIR GI)
                .TextureUAV(61)                 // u_SecondarySurfaceRadiance (ReSTIR GI)
                .TextureUAV(70)                 // u_RRDiffuseAlbedo
                .TextureUAV(71)                 // u_RRSpecAlbedo
                .TextureUAV(72)                 // u_RRNormalsAndRoughness
                .TextureUAV(73)                 // u_RRSpecMotionVectors
                .TextureUAV(74)                 // u_RRTransparencyLayer
                .TextureUAV(75)                 // u_DenoisingAvgLayerRadiance
                .StructuredBufferUAV(125)       // u_ShaderDebugBuffer (SHADER_DEBUG_BUFFER_UAV_INDEX)
                .TextureUAV(126)                // u_ShaderDebugVizTextureBuffer (SHADER_DEBUG_VIZ_TEXTURE_UAV_INDEX)
                // NV HLSL extension UAV — RTXPT appends it whenever NVAPI's
                // HlslExtensionUAV feature is supported (always on the NVIDIA
                // hardware both apps are compared on).
                .StructuredBufferUAV(127)       // g_NvidiaExt (NV_SHADER_EXTN_SLOT)
                // Stable planes — appended after the main list in Sample.cpp.
                .TextureUAV(40)                 // u_StablePlanesHeader
                .StructuredBufferUAV(42)        // u_StablePlanesBuffer
                .TextureUAV(44)                 // u_StableRadiance
                .StructuredBufferUAV(45)        // u_SurfaceData
                // GBuffer
                .TextureUAV(100)                // u_BaseColor
                .TextureUAV(101)                // u_SpecNormal
                .TextureUAV(102)                // u_RoughnessMetal
                .TextureUAV(103)                // u_MaterialInfo
                .TextureUAV(10)                 // u_LocalCubemap
                // Reflection system (IntroSample), placeholders in AdvancedSample
                .TextureSRV(80)                 // t_LocalCubemapGGX
                .TextureSRV(81)                 // t_DiffuseIrradianceCube
                .TextureSRV(82)                 // t_SSRBlurChain
                .TextureSRV(83)                 // t_BRDFLUT
                .TextureSRV(84)                 // t_DepthHierarchy
                .ConstantBuffer(10)             // ReflectionConstants (static CBV, lives in the table)
                .TextureUAV(85)                 // u_SSRResult
                .TextureSRV(86)                 // t_GTAOOutput
                .TextureSRV(87);                // t_PrevDepth
            if (includeBindless)
            {
                // Bindless layout — one root param with two unbounded ranges, both
                // aliasing the same descriptor table (donut DescriptorTableManager).
                // groupWithPrevious merges the second range into the first item's
                // root parameter instead of giving it its own table.
                l.BindlessSRV(space: 1)                                  // t_BindlessBuffers[]
                 .BindlessSRV(space: 2, groupWithPrevious: true);        // t_BindlessTextures[]
            }
            return l;
        }

        /// <summary>
        /// Stage the three RTXPT global samplers into a descriptor set (Sample.cpp
        /// binding set: donut AnisotropicWrap s0, EnvMapBaker linear-wrap s1,
        /// importance-map point-clamp s2). Staged slot values persist on the set,
        /// so calling once per bind pass is enough. Names absent from a shader's
        /// reflection are silently ignored.
        /// </summary>
        public static void BindGlobalSamplers(NativeDescriptorSetBase ds)
        {
            ds.SetSampler("s_MaterialSampler", SamplerFilter.Anisotropic,
                SamplerAddress.Wrap, SamplerAddress.Wrap, SamplerAddress.Wrap, mips: true, maxAnisotropy: 16);
            ds.SetSampler("s_EnvironmentMapSampler", SamplerFilter.Linear,
                SamplerAddress.Wrap, SamplerAddress.Wrap, SamplerAddress.Wrap, mips: true);
            ds.SetSampler("s_EnvironmentMapImportanceSampler", SamplerFilter.Point,
                SamplerAddress.Clamp, SamplerAddress.Clamp, SamplerAddress.Clamp, mips: true);
        }

        /// <summary>
        /// Bind all global resources declared in ShaderResourceBindings.hlsli to a compute descriptor set.
        /// Call this first, then bind any pass-specific SRVs/UAVs not covered here.
        /// </summary>
        public static void BindCommon(NativeComputeDescriptorSet ds, RtxptPassContext ctx)
        {
            BindConstantsAndScene(ds, ctx);
            BindOutputUAVs(ds, ctx);
            BindStablePlanes(ds, ctx);
            BindDlssRrGuides(ds, ctx);
            BindGlobalSamplers(ds);
        }

        /// <summary>
        /// Bind all global resources to a DXR ray-tracing pipeline.
        /// </summary>
        public static void BindCommon(RayTracePipeline rtp, RtxptPassContext ctx)
        {
        }

        // ── Private helpers ───────────────────────────────────────────────────

        private static void BindConstantsAndScene(NativeComputeDescriptorSet ds, RtxptPassContext ctx)
        {
            // if (ctx.ConstantBuffer != null)
            //     ds.SetConstantBuffer("g_Const", ctx.ConstantBuffer);
            //
            // var tlas = ctx.NrdSampleResource?.AccelerationStructure;
            // if (tlas != null)
            //     ds.SetAccelerationStructure("SceneBVH", tlas);
            //
            // ctx.GpuScene?.BindToShader(ds);
        }

        private static void BindOutputUAVs(NativeComputeDescriptorSet ds, RtxptPassContext ctx)
        {
            if (ctx.OutputColorPtr != IntPtr.Zero) ds.SetRWTexture("u_OutputColor", ctx.OutputColorPtr);
            if (ctx.ProcessedOutputColorPtr != IntPtr.Zero) ds.SetRWTexture("u_ProcessedOutputColor", ctx.ProcessedOutputColorPtr);
            if (ctx.ThroughputPtr != IntPtr.Zero) ds.SetRWTexture("u_Throughput", ctx.ThroughputPtr);
            if (ctx.MotionVectorsPtr != IntPtr.Zero) ds.SetRWTexture("u_MotionVectors", ctx.MotionVectorsPtr);
            if (ctx.DepthPtr != IntPtr.Zero) ds.SetRWTexture("u_Depth", ctx.DepthPtr);
            if (ctx.SpecularHitTPtr != IntPtr.Zero) ds.SetRWTexture("u_SpecularHitT", ctx.SpecularHitTPtr);
            if (ctx.ScratchFloat1Ptr != IntPtr.Zero) ds.SetRWTexture("u_ScratchFloat1", ctx.ScratchFloat1Ptr);
            if (ctx.ShaderDebugVizPtr != IntPtr.Zero) ds.SetRWTexture("u_ShaderDebugVizTextureBuffer", ctx.ShaderDebugVizPtr);
        }

        private static void BindStablePlanes(NativeComputeDescriptorSet ds, RtxptPassContext ctx)
        {
            if (ctx.StablePlanesHeaderPtr != IntPtr.Zero)
                ds.SetRWTexture("u_StablePlanesHeader", ctx.StablePlanesHeaderPtr);
            if (ctx.StableRadiancePtr != IntPtr.Zero)
                ds.SetRWTexture("u_StableRadiance", ctx.StableRadiancePtr);

            var buf = ctx.Buffers;
            if (buf?.StablePlanesBufferPtr != IntPtr.Zero)
                ds.SetRWStructuredBuffer("u_StablePlanesBuffer", buf.StablePlanesBufferPtr, buf.StablePlanesBuffer.count, buf.StablePlanesBuffer.stride);

            if (buf?.SurfaceDataBufferPtr != IntPtr.Zero)
                ds.SetRWStructuredBuffer("u_SurfaceData", buf.SurfaceDataBufferPtr, buf.SurfaceDataBuffer.count, buf.SurfaceDataBuffer.stride);
        }

        private static void BindDlssRrGuides(NativeComputeDescriptorSet ds, RtxptPassContext ctx)
        {
            if (ctx.DlssRrDiffAlbedoPtr != IntPtr.Zero) ds.SetRWTexture("u_RRDiffuseAlbedo", ctx.DlssRrDiffAlbedoPtr);
            if (ctx.DlssRrSpecAlbedoPtr != IntPtr.Zero) ds.SetRWTexture("u_RRSpecAlbedo", ctx.DlssRrSpecAlbedoPtr);
            if (ctx.DlssRrNormalRoughnessPtr != IntPtr.Zero) ds.SetRWTexture("u_RRNormalsAndRoughness", ctx.DlssRrNormalRoughnessPtr);
        }
    }
}