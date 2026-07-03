using System;
using NativeRender;
using static NativeRender.BindingLayoutItem;

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
            var global = new BindingLayoutDesc(registerSpace: 0).AddItems(
                VolatileConstantBuffer(0),      // g_Const
                PushConstants(1, 16 * 4),       // g_MiniConst (SampleMiniConstants = 16 dwords)
                RayTracingAccelStruct(0),       // SceneBVH
                StructuredBuffer_SRV(1),        // t_SubInstanceData
                StructuredBuffer_SRV(2),        // t_InstanceData
                StructuredBuffer_SRV(3),        // t_GeometryData
                StructuredBuffer_SRV(4),        // t_GeometryDebugData
                StructuredBuffer_SRV(5),        // t_PTMaterialData
                Texture_SRV(6),                 // t_LdrColorScratch
                Texture_SRV(10),                // t_EnvironmentMap
                Texture_SRV(11),                // t_EnvironmentMapImportanceMap (unused, kept for parity)
                StructuredBuffer_SRV(12),       // t_LightsCB
                StructuredBuffer_SRV(13),       // t_Lights
                StructuredBuffer_SRV(14),       // t_LightsEx
                StructuredBuffer_SRV(15),       // t_LightProxyCounters
                StructuredBuffer_SRV(16),       // t_LightProxyIndices
                StructuredBuffer_SRV(17),       // t_LightLocalSamplingBuffer
                Texture_SRV(18),                // t_EnvLookupMap
                Texture_UAV(20),                // u_LightFeedbackTotalWeight
                Texture_UAV(21),                // u_LightFeedbackCandidates
                Sampler(0),                     // s_MaterialSampler
                Sampler(1),                     // s_EnvironmentMapSampler
                Sampler(2),                     // s_EnvironmentMapImportanceSampler
                Texture_UAV(0),                 // u_OutputColor
                Texture_UAV(1),                 // u_ProcessedOutputColor
                Texture_UAV(2),                 // u_PostTonemapOutputColor
                Texture_UAV(4),                 // u_Throughput
                Texture_UAV(5),                 // u_MotionVectors
                Texture_UAV(6),                 // u_Depth
                Texture_UAV(7),                 // u_SpecularHitT
                Texture_UAV(8),                 // u_ScratchFloat1
                Texture_UAV(31),                // u_DenoiserViewspaceZ
                Texture_UAV(32),                // u_DenoiserMotionVectors
                Texture_UAV(33),                // u_DenoiserNormalRoughness
                Texture_UAV(34),                // u_DenoiserDiffRadianceHitDist
                Texture_UAV(35),                // u_DenoiserSpecRadianceHitDist
                Texture_UAV(36),                // u_DenoiserDisocclusionThresholdMix
                Texture_UAV(37),                // u_CombinedHistoryClampRelax
                StructuredBuffer_UAV(51),       // u_FeedbackBuffer
                StructuredBuffer_UAV(52),       // u_DebugLinesBuffer
                StructuredBuffer_UAV(53),       // u_DebugDeltaPathTree
                StructuredBuffer_UAV(54),       // u_DeltaPathSearchStack
                Texture_UAV(60),                // u_SecondarySurfacePositionNormal (ReSTIR GI)
                Texture_UAV(61),                // u_SecondarySurfaceRadiance (ReSTIR GI)
                Texture_UAV(70),                // u_RRDiffuseAlbedo
                Texture_UAV(71),                // u_RRSpecAlbedo
                Texture_UAV(72),                // u_RRNormalsAndRoughness
                Texture_UAV(73),                // u_RRSpecMotionVectors
                Texture_UAV(74),                // u_RRTransparencyLayer
                Texture_UAV(75),                // u_DenoisingAvgLayerRadiance
                StructuredBuffer_UAV(125),      // u_ShaderDebugBuffer (SHADER_DEBUG_BUFFER_UAV_INDEX)
                Texture_UAV(126),               // u_ShaderDebugVizTextureBuffer (SHADER_DEBUG_VIZ_TEXTURE_UAV_INDEX)
                // NV HLSL extension UAV — RTXPT appends it whenever NVAPI's
                // HlslExtensionUAV feature is supported (always on the NVIDIA
                // hardware both apps are compared on).
                StructuredBuffer_UAV(127),      // g_NvidiaExt (NV_SHADER_EXTN_SLOT)
                // Stable planes — appended after the main list in Sample.cpp.
                Texture_UAV(40),                // u_StablePlanesHeader
                StructuredBuffer_UAV(42),       // u_StablePlanesBuffer
                Texture_UAV(44),                // u_StableRadiance
                StructuredBuffer_UAV(45),       // u_SurfaceData
                // GBuffer
                Texture_UAV(100),               // u_BaseColor
                Texture_UAV(101),               // u_SpecNormal
                Texture_UAV(102),               // u_RoughnessMetal
                Texture_UAV(103),               // u_MaterialInfo
                Texture_UAV(10),                // u_LocalCubemap
                // Reflection system (IntroSample), placeholders in AdvancedSample
                Texture_SRV(80),                // t_LocalCubemapGGX
                Texture_SRV(81),                // t_DiffuseIrradianceCube
                Texture_SRV(82),                // t_SSRBlurChain
                Texture_SRV(83),                // t_BRDFLUT
                Texture_SRV(84),                // t_DepthHierarchy
                ConstantBuffer(10),             // ReflectionConstants (static CBV, lives in the table)
                Texture_UAV(85),                // u_SSRResult
                Texture_SRV(86),                // t_GTAOOutput
                Texture_SRV(87));               // t_PrevDepth

            var l = new NativeBindingLayout(global);
            if (includeBindless)
            {
                // Bindless layout — one root param with two unbounded ranges, both
                // aliasing the same descriptor table (donut DescriptorTableManager).
                l.AddBindlessLayout(new BindlessLayoutDesc(firstSlot: 0, maxCapacity: 1024)
                    .AddRegisterSpaces(
                        RawBuffer_SRV(1),                                // t_BindlessBuffers[]
                        Texture_SRV(2)));                                // t_BindlessTextures[]
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
