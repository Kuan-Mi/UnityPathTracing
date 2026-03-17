using System;
using System.Runtime.InteropServices;
using DefaultNamespace;
using mini;
using Nrd;
using RTXDI;
using Rtxdi.DI;
using Unity.Mathematics;
using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;
using static PathTracing.ShaderIDs;
using static PathTracing.PathTracingUtils;

namespace PathTracing
{
    public class PathTracingPass : ScriptableRenderPass
    {
        private static readonly int GInOutMv = Shader.PropertyToID("gInOut_Mv");
        public RayTracingShader OpaqueTs;
        public RayTracingShader TransparentTs;
        public ComputeShader CompositionCs;
        public ComputeShader TaaCs;
        public ComputeShader DlssBeforeCs;
        public Material BiltMaterial;

        public ComputeShader SharcResolveCs;
        public RayTracingShader SharcUpdateTs;

        public GraphicsBuffer HashEntriesBuffer;
        public GraphicsBuffer AccumulationBuffer;
        public GraphicsBuffer ResolvedBuffer;
        // public PathTracingDataBuilder _dataBuilder;

        public RayTracingAccelerationStructure AccelerationStructure;

        public NRDDenoiser NrdDenoiser;
        public DLRRDenoiser DLRRDenoiser;

        public GraphicsBuffer ScramblingRanking;
        public GraphicsBuffer Sobol;

        // Auto-exposure
        public ComputeShader AutoExposureCs;
        public GraphicsBuffer AeHistogramBuffer;
        public GraphicsBuffer AeExposureBuffer;

        private readonly PathTracingSetting m_Settings;
        private readonly GraphicsBuffer _pathTracingSettingsBuffer;
        private readonly GraphicsBuffer _resamplingConstantsBuffer;
        private GraphicsBuffer m_SpotLightBuffer;
        private GraphicsBuffer m_AreaLightBuffer;
        private GraphicsBuffer m_PointLightBuffer;
        
        public PrepareLightResource prepareLightResource;
        public RtxdiResources  rtxdiResources;
        public ReSTIRDIContext  restirDIContext;


        [DllImport("RenderingPlugin")]
        private static extern IntPtr GetRenderEventAndDataFunc();

        class PassData
        {
            internal TextureHandle CameraTexture;

            internal GraphicsBuffer ScramblingRanking;
            internal GraphicsBuffer Sobol;

            internal TextureHandle OutputTexture;

            internal TextureHandle Mv;
            internal TextureHandle ViewZ;
            internal TextureHandle NormalRoughness;
            internal TextureHandle BaseColorMetalness;

            internal TextureHandle DirectLighting;
            internal TextureHandle DirectEmission;

            internal TextureHandle Penumbra;
            internal TextureHandle Diff;
            internal TextureHandle Spec;

            internal TextureHandle ShadowTranslucency;
            internal TextureHandle DenoisedDiff;
            internal TextureHandle DenoisedSpec;
            internal TextureHandle Validation;

            internal TextureHandle ComposedDiff;
            internal TextureHandle ComposedSpecViewZ;
            internal TextureHandle Composed;

            internal TextureHandle TaaHistory;
            internal TextureHandle TaaHistoryPrev;
            internal TextureHandle PsrThroughput;


            internal TextureHandle RRGuide_DiffAlbedo;
            internal TextureHandle RRGuide_SpecAlbedo;
            internal TextureHandle RRGuide_SpecHitDistance;
            internal TextureHandle RRGuide_Normal_Roughness;
            internal TextureHandle DlssOutput;

            // RTXDI：上一帧 GBuffer
            internal TextureHandle PrevViewZ;
            internal TextureHandle PrevNormalRoughness;
            internal TextureHandle PrevBaseColorMetalness;

            internal RayTracingShader OpaqueTs;
            internal RayTracingShader TransparentTs;
            internal ComputeShader CompositionCs;
            internal ComputeShader TaaCs;
            internal ComputeShader DlssBeforeCs;
            internal Material BlitMaterial;
            internal uint outputGridW;
            internal uint outputGridH;
            internal uint rectGridW;
            internal uint rectGridH;
            internal int2 m_RenderResolution;

            internal GlobalConstants GlobalConstants;
            internal ResamplingConstants ResamplingConstants;
            
            internal GraphicsBuffer ResamplingConstantBuffer;
            internal GraphicsBuffer ConstantBuffer;
            internal IntPtr NrdDataPtr;
            internal IntPtr RRDataPtr;
            internal PathTracingSetting Setting;
            internal float resolutionScale;


            internal ComputeShader SharcResolveCs;
            internal RayTracingShader SharcUpdateTs;

            internal GraphicsBuffer HashEntriesBuffer;
            internal GraphicsBuffer AccumulationBuffer;

            internal GraphicsBuffer ResolvedBuffer;

            internal int passIndex;
            // internal PathTracingDataBuilder _dataBuilder;

            // internal TextureHandle SpotDirect;
            internal GraphicsBuffer SpotLightBuffer;
            internal GraphicsBuffer AreaLightBuffer;
            internal GraphicsBuffer PointLightBuffer;

            // ── Auto-exposure ──
            internal ComputeShader AeCs;
            internal GraphicsBuffer AeHistogramBuffer;
            internal GraphicsBuffer AeExposureBuffer;
            internal bool AeEnabled;
            internal float AeEVMin;
            internal float AeEVMax;
            internal float AeLowPercent;
            internal float AeHighPercent;
            internal float AeSpeedUp;
            internal float AeSpeedDown;
            internal float AeDeltaTime;
            internal float AeExposureCompensation;
            internal float AeMinExposure;
            internal float AeMaxExposure;
            internal uint AeTexWidth;
            internal uint AeTexHeight;
            internal float ManualExposure;
            
            
            internal IntPtr DataPtr;
            
            
            internal RtxdiResources RtxdiResources;
            internal ReSTIRDIContext  RestirDIContext;
        }

        public PathTracingPass(PathTracingSetting setting)
        {
            m_Settings = setting;
            _pathTracingSettingsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Constant, 1, Marshal.SizeOf<GlobalConstants>());
            _resamplingConstantsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, Marshal.SizeOf<ResamplingConstants>());
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            if (data.passIndex != 0)
            {
                return;
            }

            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            natCmd.SetBufferData(data.ConstantBuffer, new[] { data.GlobalConstants });
            natCmd.SetBufferData(data.ResamplingConstantBuffer, new[] { data.ResamplingConstants });

            // Bind the exposure buffer globally so all shaders can read the current EV.
            // When auto-exposure is OFF: seed the buffer with the manual value from settings.
            // When auto-exposure is ON:  the buffer is updated later by ReduceHistogram.
            natCmd.SetGlobalBuffer("_AE_ExposureBuffer", data.AeExposureBuffer);
            if (!data.AeEnabled)
            {
                natCmd.SetBufferData(data.AeExposureBuffer, new[] { data.ManualExposure });
            }

            var prepareLightMarker = new ProfilerMarker(ProfilerCategory.Render, "PrepareLight", MarkerFlags.SampleGPU);
            var sharcUpdateMarker = new ProfilerMarker(ProfilerCategory.Render, "Sharc Update", MarkerFlags.SampleGPU);
            var sharcResolveMarker = new ProfilerMarker(ProfilerCategory.Render, "Sharc Resolve", MarkerFlags.SampleGPU);
            var opaqueTracingMarker = new ProfilerMarker(ProfilerCategory.Render, "Opaque Tracing", MarkerFlags.SampleGPU);
            var nrdDenoiseMarker = new ProfilerMarker(ProfilerCategory.Render, "NRD Denoise", MarkerFlags.SampleGPU);
            var compositionMarker = new ProfilerMarker(ProfilerCategory.Render, "Composition", MarkerFlags.SampleGPU);
            var transparentTracingMarker = new ProfilerMarker(ProfilerCategory.Render, "Transparent Tracing", MarkerFlags.SampleGPU);
            var taaMarker = new ProfilerMarker(ProfilerCategory.Render, "TAA", MarkerFlags.SampleGPU);
            var dlssBeforeMarker = new ProfilerMarker(ProfilerCategory.Render, "DLSS Before", MarkerFlags.SampleGPU);
            var dlssDenoiseMarker = new ProfilerMarker(ProfilerCategory.Render, "DLSS Denoise", MarkerFlags.SampleGPU);
            var outputBlitMarker = new ProfilerMarker(ProfilerCategory.Render, "Output Blit", MarkerFlags.SampleGPU);
            var aeMarker = new ProfilerMarker(ProfilerCategory.Render, "Auto Exposure", MarkerFlags.SampleGPU);
            var copyGBufferMarker = new ProfilerMarker(ProfilerCategory.Render, "Copy GBuffer to Prev", MarkerFlags.SampleGPU);

            
            natCmd.BeginSample(prepareLightMarker);
            natCmd.IssuePluginEventAndData(UnityRTXDI.GetRenderEventAndDataFunc(), 1, data.DataPtr);
            natCmd.EndSample(prepareLightMarker);

            
            
            
            // Sharc update
            if (data.passIndex == 0)
            {
                natCmd.BeginSample(sharcUpdateMarker);
                natCmd.SetRayTracingShaderPass(data.SharcUpdateTs, "Test2");
                natCmd.SetRayTracingConstantBufferParam(data.SharcUpdateTs, paramsID, data.ConstantBuffer, 0, data.ConstantBuffer.stride);

                natCmd.SetRayTracingBufferParam(data.SharcUpdateTs, g_HashEntriesID, data.HashEntriesBuffer);
                natCmd.SetRayTracingBufferParam(data.SharcUpdateTs, g_AccumulationBufferID, data.AccumulationBuffer);
                natCmd.SetRayTracingBufferParam(data.SharcUpdateTs, g_ResolvedBufferID, data.ResolvedBuffer);

                natCmd.SetRayTracingTextureParam(data.SharcUpdateTs, g_OutputID, data.OutputTexture);
                natCmd.SetRayTracingBufferParam(data.SharcUpdateTs, gIn_SpotLightsID, data.SpotLightBuffer);
                natCmd.SetRayTracingBufferParam(data.SharcUpdateTs, gIn_AreaLightsID, data.AreaLightBuffer);
                natCmd.SetRayTracingBufferParam(data.SharcUpdateTs, gIn_PointLightsID, data.PointLightBuffer);

                int SHARC_DOWNSCALE = 4;  

                uint w = (uint)(data.m_RenderResolution.x / SHARC_DOWNSCALE);
                uint h = (uint)(data.m_RenderResolution.y / SHARC_DOWNSCALE);

                natCmd.DispatchRays(data.SharcUpdateTs, "MainRayGenShader", w, h, 1);
                natCmd.EndSample(sharcUpdateMarker);
            }


            // Sharc resolve
            if (data.passIndex == 0)
            {
                natCmd.BeginSample(sharcResolveMarker);
                natCmd.SetComputeConstantBufferParam(data.SharcResolveCs, paramsID, data.ConstantBuffer, 0, data.ConstantBuffer.stride);
                natCmd.SetComputeBufferParam(data.SharcResolveCs, 0, g_HashEntriesID, data.HashEntriesBuffer);
                natCmd.SetComputeBufferParam(data.SharcResolveCs, 0, g_AccumulationBufferID, data.AccumulationBuffer);
                natCmd.SetComputeBufferParam(data.SharcResolveCs, 0, g_ResolvedBufferID, data.ResolvedBuffer);
 
                int LINEAR_BLOCK_SIZE = 256;
                int x = (int)((PathTracingFeature.Capacity + LINEAR_BLOCK_SIZE - 1) / LINEAR_BLOCK_SIZE);

                natCmd.DispatchCompute(data.SharcResolveCs, 0, x, 1, 1);

                natCmd.EndSample(sharcResolveMarker);
            }

            // 不透明
            {
                natCmd.BeginSample(opaqueTracingMarker);

                // natCmd.SetGlobalBuffer(gIn_InstanceDataID, data._dataBuilder._instanceBuffer);
                // natCmd.SetGlobalBuffer(gIn_PrimitiveDataID, data._dataBuilder._primitiveBuffer);


                natCmd.SetRayTracingShaderPass(data.OpaqueTs, "Test2");
                natCmd.SetRayTracingConstantBufferParam(data.OpaqueTs, paramsID, data.ConstantBuffer, 0, data.ConstantBuffer.stride);
                natCmd.SetRayTracingBufferParam(data.OpaqueTs, "ResampleConstants" , data.ResamplingConstantBuffer);
                
                natCmd.SetRayTracingBufferParam(data.OpaqueTs, t_LightDataBufferID, data.RtxdiResources.LightDataBuffer);
                natCmd.SetRayTracingBufferParam(data.OpaqueTs, t_NeighborOffsetsID, data.RtxdiResources.NeighborOffsetsBuffer);
                natCmd.SetRayTracingBufferParam(data.OpaqueTs, u_LightReservoirsID, data.RtxdiResources.LightReservoirBuffer);

                natCmd.SetRayTracingBufferParam(data.OpaqueTs, g_ScramblingRankingID, data.ScramblingRanking);
                natCmd.SetRayTracingBufferParam(data.OpaqueTs, g_SobolID, data.Sobol);

                natCmd.SetRayTracingBufferParam(data.OpaqueTs, g_HashEntriesID, data.HashEntriesBuffer);
                natCmd.SetRayTracingBufferParam(data.OpaqueTs, g_AccumulationBufferID, data.AccumulationBuffer);
                natCmd.SetRayTracingBufferParam(data.OpaqueTs, g_ResolvedBufferID, data.ResolvedBuffer);

                natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_OutputID, data.OutputTexture);

                natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_MvID, data.Mv);
                natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_ViewZID, data.ViewZ);
                natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_Normal_RoughnessID, data.NormalRoughness);
                natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_BaseColor_MetalnessID, data.BaseColorMetalness);

                natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_DirectLightingID, data.DirectLighting);
                natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_DirectEmissionID, data.DirectEmission);
                natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_PsrThroughputID, data.PsrThroughput);

                natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_ShadowDataID, data.Penumbra);
                natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_DiffID, data.Diff);
                natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_SpecID, data.Spec);

                natCmd.SetRayTracingTextureParam(data.OpaqueTs, gIn_PrevComposedDiffID, data.ComposedDiff);
                natCmd.SetRayTracingTextureParam(data.OpaqueTs, gIn_PrevComposedSpec_PrevViewZID, data.ComposedSpecViewZ);

                natCmd.SetRayTracingTextureParam(data.OpaqueTs, gIn_PrevViewZID, data.PrevViewZ);
                natCmd.SetRayTracingTextureParam(data.OpaqueTs, gIn_PrevNormalRoughnessID, data.PrevNormalRoughness);
                natCmd.SetRayTracingTextureParam(data.OpaqueTs, gIn_PrevBaseColorMetalnessID, data.PrevBaseColorMetalness);

                natCmd.SetRayTracingBufferParam(data.OpaqueTs, gIn_SpotLightsID, data.SpotLightBuffer);
                natCmd.SetRayTracingBufferParam(data.OpaqueTs, gIn_AreaLightsID, data.AreaLightBuffer);
                natCmd.SetRayTracingBufferParam(data.OpaqueTs, gIn_PointLightsID, data.PointLightBuffer);
                // natCmd.SetRayTracingTextureParam(data.OpaqueTs, gOut_SpotDirectID, data.SpotDirect);

                // Debug.Log(data.m_RenderResolution);

                uint rectWmod = (uint)(data.m_RenderResolution.x * data.resolutionScale + 0.5f);
                uint rectHmod = (uint)(data.m_RenderResolution.y * data.resolutionScale + 0.5f);

                // Debug.Log($"Dispatch Rays Size: {rectWmod} x {rectHmod}");


                natCmd.DispatchRays(data.OpaqueTs, "MainRayGenShader", rectWmod, rectHmod, 1);

                natCmd.EndSample(opaqueTracingMarker);

                // 保存当帧 GBuffer 到 prev 纹理，供下一帧 RTXDI 时间复用读取
                natCmd.BeginSample(copyGBufferMarker);
                natCmd.CopyTexture(data.ViewZ, data.PrevViewZ);
                natCmd.CopyTexture(data.NormalRoughness, data.PrevNormalRoughness);
                natCmd.CopyTexture(data.BaseColorMetalness, data.PrevBaseColorMetalness);
                natCmd.EndSample(copyGBufferMarker);
            }


            // NRD降噪
            if (!data.Setting.RR)
            {
                natCmd.BeginSample(nrdDenoiseMarker);
                natCmd.IssuePluginEventAndData(GetRenderEventAndDataFunc(), 1, data.NrdDataPtr);
                natCmd.EndSample(nrdDenoiseMarker);
            }


            // 合成
            {
                natCmd.BeginSample(compositionMarker);
                natCmd.SetComputeConstantBufferParam(data.CompositionCs, paramsID, data.ConstantBuffer, 0, data.ConstantBuffer.stride);
                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_ViewZID, data.ViewZ);
                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_Normal_RoughnessID, data.NormalRoughness);
                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_BaseColor_MetalnessID, data.BaseColorMetalness);
                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_DirectLightingID, data.DirectLighting);
                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_DirectEmissionID, data.DirectEmission);
                if (data.Setting.RR)
                {
                    natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_ShadowID, data.Penumbra);
                    natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_DiffID, data.Diff);
                    natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_SpecID, data.Spec);
                }
                else
                {
                    natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_ShadowID, data.ShadowTranslucency);
                    natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_DiffID, data.DenoisedDiff);
                    natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_SpecID, data.DenoisedSpec);
                }

                // natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_SpotDirectID, data.SpotDirect);
                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gIn_PsrThroughputID, data.PsrThroughput);
                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gOut_ComposedDiffID, data.ComposedDiff);
                natCmd.SetComputeTextureParam(data.CompositionCs, 0, gOut_ComposedSpec_ViewZID, data.ComposedSpecViewZ);

                natCmd.DispatchCompute(data.CompositionCs, 0, (int)data.rectGridW, (int)data.rectGridH, 1);

                natCmd.EndSample(compositionMarker);
            }


            // 透明
            {
                natCmd.BeginSample(transparentTracingMarker);

                natCmd.SetRayTracingShaderPass(data.TransparentTs, "Test2");
                natCmd.SetRayTracingConstantBufferParam(data.TransparentTs, paramsID, data.ConstantBuffer, 0, data.ConstantBuffer.stride);

                natCmd.SetRayTracingBufferParam(data.TransparentTs, g_HashEntriesID, data.HashEntriesBuffer);
                natCmd.SetRayTracingBufferParam(data.TransparentTs, g_AccumulationBufferID, data.AccumulationBuffer);
                natCmd.SetRayTracingBufferParam(data.TransparentTs, g_ResolvedBufferID, data.ResolvedBuffer);


                natCmd.SetRayTracingTextureParam(data.TransparentTs, gIn_ComposedDiffID, data.ComposedDiff);
                natCmd.SetRayTracingTextureParam(data.TransparentTs, gIn_ComposedSpec_ViewZID, data.ComposedSpecViewZ);
                natCmd.SetRayTracingTextureParam(data.TransparentTs, g_Normal_RoughnessID, data.NormalRoughness);
                natCmd.SetRayTracingTextureParam(data.TransparentTs, gOut_ComposedID, data.Composed);
                natCmd.SetRayTracingTextureParam(data.TransparentTs, GInOutMv, data.Mv);

                natCmd.SetRayTracingBufferParam(data.TransparentTs, gIn_SpotLightsID, data.SpotLightBuffer);
                natCmd.SetRayTracingBufferParam(data.TransparentTs, gIn_AreaLightsID, data.AreaLightBuffer);
                natCmd.SetRayTracingBufferParam(data.TransparentTs, gIn_PointLightsID, data.PointLightBuffer);

                natCmd.DispatchRays(data.TransparentTs, "MainRayGenShader", (uint)data.m_RenderResolution.x, (uint)data.m_RenderResolution.y, 1);
                natCmd.EndSample(transparentTracingMarker);
            }


            // ── Auto-exposure: histogram build + reduce (after transparent, before TAA) ──
            if (data.AeEnabled && data.AeCs != null && data.AeHistogramBuffer != null && data.AeExposureBuffer != null)
            {
                natCmd.BeginSample(aeMarker);

                int kernelClear  = data.AeCs.FindKernel("ClearHistogram");
                int kernelBuild  = data.AeCs.FindKernel("BuildHistogram");
                int kernelReduce = data.AeCs.FindKernel("ReduceHistogram");

                // -- Kernel 0: Clear --
                natCmd.SetComputeBufferParam(data.AeCs, kernelClear, "_AE_HistogramBuffer", data.AeHistogramBuffer);
                natCmd.DispatchCompute(data.AeCs, kernelClear, 1, 1, 1);

                // -- Kernel 1: Build --
                natCmd.SetComputeTextureParam(data.AeCs, kernelBuild, "_AE_ComposedTexture", data.Composed);
                natCmd.SetComputeBufferParam(data.AeCs, kernelBuild, "_AE_HistogramBuffer", data.AeHistogramBuffer);
                natCmd.SetComputeIntParam(data.AeCs, "_AE_TexWidth",  (int)data.AeTexWidth);
                natCmd.SetComputeIntParam(data.AeCs, "_AE_TexHeight", (int)data.AeTexHeight);
                natCmd.SetComputeFloatParam(data.AeCs, "_AE_EVMin", data.AeEVMin);
                natCmd.SetComputeFloatParam(data.AeCs, "_AE_EVMax", data.AeEVMax);
                uint buildX = (data.AeTexWidth  + 15u) / 16u;
                uint buildY = (data.AeTexHeight + 15u) / 16u;
                natCmd.DispatchCompute(data.AeCs, kernelBuild, (int)buildX, (int)buildY, 1);

                // -- Kernel 2: Reduce --
                natCmd.SetComputeBufferParam(data.AeCs, kernelReduce, "_AE_HistogramBuffer", data.AeHistogramBuffer);
                natCmd.SetComputeBufferParam(data.AeCs, kernelReduce, "_AE_ExposureBuffer",  data.AeExposureBuffer);
                natCmd.SetComputeFloatParam(data.AeCs, "_AE_EVMin",                data.AeEVMin);
                natCmd.SetComputeFloatParam(data.AeCs, "_AE_EVMax",                data.AeEVMax);
                natCmd.SetComputeFloatParam(data.AeCs, "_AE_LowPercent",           data.AeLowPercent);
                natCmd.SetComputeFloatParam(data.AeCs, "_AE_HighPercent",          data.AeHighPercent);
                natCmd.SetComputeFloatParam(data.AeCs, "_AE_SpeedUp",              data.AeSpeedUp);
                natCmd.SetComputeFloatParam(data.AeCs, "_AE_SpeedDown",            data.AeSpeedDown);
                natCmd.SetComputeFloatParam(data.AeCs, "_AE_DeltaTime",            data.AeDeltaTime);
                natCmd.SetComputeFloatParam(data.AeCs, "_AE_ExposureCompensation", data.AeExposureCompensation);
                natCmd.SetComputeFloatParam(data.AeCs, "_AE_MinExposure",          data.AeMinExposure);
                natCmd.SetComputeFloatParam(data.AeCs, "_AE_MaxExposure",          data.AeMaxExposure);
                natCmd.DispatchCompute(data.AeCs, kernelReduce, 1, 1, 1);

                natCmd.EndSample(aeMarker);
            }


            var isEven = (data.GlobalConstants.gFrameIndex & 1) == 0;
            var taaSrc = isEven ? data.TaaHistoryPrev : data.TaaHistory;
            var taaDst = isEven ? data.TaaHistory : data.TaaHistoryPrev;
            if (data.Setting.RR)
            {
                // dlss Before
                natCmd.BeginSample(dlssBeforeMarker);
                natCmd.SetComputeConstantBufferParam(data.DlssBeforeCs, paramsID, data.ConstantBuffer, 0, data.ConstantBuffer.stride);

                natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gIn_Normal_Roughness", data.NormalRoughness);
                natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gIn_BaseColor_Metalness", data.BaseColorMetalness);
                natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gIn_Spec", data.Spec);

                natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gInOut_ViewZ", data.ViewZ);
                natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gOut_DiffAlbedo", data.RRGuide_DiffAlbedo);
                natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gOut_SpecAlbedo", data.RRGuide_SpecAlbedo);
                natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gOut_SpecHitDistance", data.RRGuide_SpecHitDistance);
                natCmd.SetComputeTextureParam(data.DlssBeforeCs, 0, "gOut_Normal_Roughness", data.RRGuide_Normal_Roughness);


                natCmd.DispatchCompute(data.DlssBeforeCs, 0, (int)data.rectGridW, (int)data.rectGridH, 1);
                natCmd.EndSample(dlssBeforeMarker);

                // DLSS调用

                if (!data.Setting.tmpDisableRR)
                {
                    natCmd.BeginSample(dlssDenoiseMarker);
                    natCmd.IssuePluginEventAndData(GetRenderEventAndDataFunc(), 2, data.RRDataPtr);
                    natCmd.EndSample(dlssDenoiseMarker);
                }
            }
            else
            {
                // TAA
                natCmd.BeginSample(taaMarker);

                natCmd.SetComputeConstantBufferParam(data.TaaCs, paramsID, data.ConstantBuffer, 0, data.ConstantBuffer.stride);
                natCmd.SetComputeTextureParam(data.TaaCs, 0, gIn_MvID, data.Mv);
                natCmd.SetComputeTextureParam(data.TaaCs, 0, gIn_ComposedID, data.Composed);
                natCmd.SetComputeTextureParam(data.TaaCs, 0, gIn_HistoryID, taaSrc);
                natCmd.SetComputeTextureParam(data.TaaCs, 0, gOut_ResultID, taaDst);
                natCmd.SetComputeTextureParam(data.TaaCs, 0, gOut_DebugID, data.OutputTexture);
                natCmd.DispatchCompute(data.TaaCs, 0, (int)data.rectGridW, (int)data.rectGridH, 1);
                natCmd.EndSample(taaMarker);
            }


            // 显示输出
            natCmd.BeginSample(outputBlitMarker);

            natCmd.SetRenderTarget(data.CameraTexture);

            Vector4 scaleOffset = new Vector4(data.resolutionScale, data.resolutionScale, 0, 0);
            switch (data.Setting.showMode)
            {
                case ShowMode.None:
                    break;
                case ShowMode.BaseColor:
                    Blitter.BlitTexture(natCmd, data.BaseColorMetalness, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.Metalness:
                    Blitter.BlitTexture(natCmd, data.BaseColorMetalness, scaleOffset, data.BlitMaterial, (int)ShowPass.Alpha);
                    break;
                case ShowMode.Normal:
                    Blitter.BlitTexture(natCmd, data.NormalRoughness, scaleOffset, data.BlitMaterial, (int)ShowPass.Normal);
                    break;
                case ShowMode.Roughness:
                    Blitter.BlitTexture(natCmd, data.NormalRoughness, scaleOffset, data.BlitMaterial, (int)ShowPass.Roughness);
                    break;
                case ShowMode.NoiseShadow:
                    Blitter.BlitTexture(natCmd, data.Penumbra, scaleOffset, data.BlitMaterial, (int)ShowPass.NoiseShadow);
                    break;
                case ShowMode.Shadow:
                    Blitter.BlitTexture(natCmd, data.ShadowTranslucency, scaleOffset, data.BlitMaterial, (int)ShowPass.Shadow);
                    break;
                case ShowMode.Diffuse:
                    Blitter.BlitTexture(natCmd, data.Diff, scaleOffset, data.BlitMaterial, (int)ShowPass.Radiance);
                    break;
                case ShowMode.Specular:
                    Blitter.BlitTexture(natCmd, data.Spec, scaleOffset, data.BlitMaterial, (int)ShowPass.Radiance);
                    break;
                case ShowMode.DenoisedDiffuse:
                    Blitter.BlitTexture(natCmd, data.DenoisedDiff, scaleOffset, data.BlitMaterial, (int)ShowPass.Radiance);
                    break;
                case ShowMode.DenoisedSpecular:
                    Blitter.BlitTexture(natCmd, data.DenoisedSpec, scaleOffset, data.BlitMaterial, (int)ShowPass.Radiance);
                    break;
                case ShowMode.DirectLight:
                    Blitter.BlitTexture(natCmd, data.DirectLighting, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.Emissive:
                    Blitter.BlitTexture(natCmd, data.DirectEmission, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.Out:
                    Blitter.BlitTexture(natCmd, data.OutputTexture, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.ComposedDiff:
                    Blitter.BlitTexture(natCmd, data.ComposedDiff, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.ComposedSpec:
                    Blitter.BlitTexture(natCmd, data.ComposedSpecViewZ, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.Composed:
                    Blitter.BlitTexture(natCmd, data.Composed, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.Taa:
                    Blitter.BlitTexture(natCmd, taaDst, scaleOffset, data.BlitMaterial, (int)ShowPass.Alpha);
                    break;
                case ShowMode.Final:

                    if (data.Setting.RR)
                    {
                        Blitter.BlitTexture(natCmd, data.DlssOutput, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.Dlss);
                    }
                    else
                    {
                        Blitter.BlitTexture(natCmd, taaDst, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    }

                    break;
                case ShowMode.DLSS_DiffuseAlbedo:
                    Blitter.BlitTexture(natCmd, data.RRGuide_DiffAlbedo, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.DLSS_SpecularAlbedo:
                    Blitter.BlitTexture(natCmd, data.RRGuide_SpecAlbedo, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.DLSS_SpecularHitDistance:
                    Blitter.BlitTexture(natCmd, data.RRGuide_SpecHitDistance, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.DLSS_NormalRoughness:
                    Blitter.BlitTexture(natCmd, data.RRGuide_Normal_Roughness, scaleOffset, data.BlitMaterial, (int)ShowPass.Out);
                    break;
                case ShowMode.DLSS_Output:
                    Blitter.BlitTexture(natCmd, data.DlssOutput, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.Out);
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

            if (data.Setting.showMV)
            {
                Blitter.BlitTexture(natCmd, data.Mv, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.Mv);
            }

            if (data.Setting.showValidation)
            {
                Blitter.BlitTexture(natCmd, data.Validation, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.Validation);
            }

            natCmd.EndSample(outputBlitMarker);
        }

        uint GetMaxAccumulatedFrameNum(float accumulationTime, float fps)
        {
            return (uint)(accumulationTime * fps + 0.5f);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            var cameraData = frameData.Get<UniversalCameraData>();

            // 获取主光源方向
            var universalLightData = frameData.Get<UniversalLightData>();
            var lightData = universalLightData;
            var mainLight = lightData.mainLightIndex >= 0 ? lightData.visibleLights[lightData.mainLightIndex] : default;
            var mat = mainLight.localToWorldMatrix;
            Vector3 lightForward = mat.GetColumn(2);

            // // Collect visible spot lights and upload to GPU buffer
            // var spotLightList = new System.Collections.Generic.List<SpotLightData>();
            // foreach (var vl in lightData.visibleLights)
            // {
            //     if (vl.lightType != LightType.Spot) continue;
            //     var lmat       = vl.localToWorldMatrix;
            //     Vector3 pos    = lmat.GetColumn(3);
            //     Vector3 dir    = ((Vector3)lmat.GetColumn(2)).normalized;
            //     Color   fc     = vl.finalColor;
            //     float   outerHalf = vl.spotAngle * 0.5f * Mathf.Deg2Rad;
            //     float   innerHalf = vl.light != null
            //         ? vl.light.innerSpotAngle * 0.5f * Mathf.Deg2Rad
            //         : outerHalf * 0.9f;
            //     spotLightList.Add(new SpotLightData
            //     {
            //         position      = pos,
            //         range         = vl.range,
            //         direction     = dir,
            //         cosOuterAngle = Mathf.Cos(outerHalf),
            //         color         = new Vector3(fc.r, fc.g, fc.b),
            //         cosInnerAngle = Mathf.Cos(innerHalf),
            //     });
            // }


            var spotLightList = new System.Collections.Generic.List<SpotLightData>();

            // 获取场景中所有激活的 Light 组件（不受视锥体裁剪限制）
            var allLights = UnityEngine.Object.FindObjectsByType<Light>(FindObjectsSortMode.None);
            foreach (var light in allLights)
            {
                if (!light.enabled || !light.gameObject.activeInHierarchy) continue;
                if (light.type != LightType.Spot) continue;

                Vector3 pos = light.transform.position;
                Vector3 dir = light.transform.forward.normalized;
                Color fc = light.color * light.intensity;

                float outerHalf = light.spotAngle * 0.5f * Mathf.Deg2Rad;
                float innerHalf = light.innerSpotAngle * 0.5f * Mathf.Deg2Rad;

                spotLightList.Add(new SpotLightData
                {
                    position = pos,
                    range = light.range,
                    direction = dir,
                    cosOuterAngle = Mathf.Cos(outerHalf),
                    color = new Vector3(fc.r, fc.g, fc.b),
                    cosInnerAngle = Mathf.Cos(innerHalf),
                });
            }


            int spotCount = spotLightList.Count;
            int bufferCount = Mathf.Max(spotCount, 1);
            if (m_SpotLightBuffer == null || m_SpotLightBuffer.count < bufferCount)
            {
                m_SpotLightBuffer?.Release();
                m_SpotLightBuffer = new GraphicsBuffer(
                    GraphicsBuffer.Target.Structured, bufferCount,
                    Marshal.SizeOf<SpotLightData>());
            }

            if (spotCount > 0)
                m_SpotLightBuffer.SetData(spotLightList.ToArray());

            // ---------------------------------------------------------------
            // Collect area lights (LightType.Rectangle + LightType.Disc)
            // ---------------------------------------------------------------
            var areaLightList = new System.Collections.Generic.List<AreaLightData>();

            foreach (var light in allLights)
            {
                if (!light.enabled || !light.gameObject.activeInHierarchy) continue;
                if (light.type != LightType.Rectangle && light.type != LightType.Disc) continue;

                Color   fc     = light.color * light.intensity;
                Vector2 sz     = light.areaSize;
                bool    isDisc = light.type == LightType.Disc;

                areaLightList.Add(new AreaLightData
                {
                    position   = light.transform.position,
                    // Disc: areaSize.x is the radius. Rect: areaSize is full width/height.
                    halfWidth  = isDisc ? sz.x          : sz.x * 0.5f,
                    right      = light.transform.right.normalized,
                    halfHeight = isDisc ? 0f             : sz.y * 0.5f,
                    up         = light.transform.up.normalized,
                    lightType  = isDisc ? 1f : 0f,
                    color      = new Vector3(fc.r, fc.g, fc.b),
                    pad2       = 0f,
                });
            }

            int areaCount       = areaLightList.Count;
            int areaBufferCount = Mathf.Max(areaCount, 1);
            if (m_AreaLightBuffer == null || m_AreaLightBuffer.count < areaBufferCount)
            {
                m_AreaLightBuffer?.Release();
                m_AreaLightBuffer = new GraphicsBuffer(
                    GraphicsBuffer.Target.Structured, areaBufferCount,
                    Marshal.SizeOf<AreaLightData>());
            }

            if (areaCount > 0)
                m_AreaLightBuffer.SetData(areaLightList.ToArray());

            // ---------------------------------------------------------------
            // Collect point lights (LightType.Point)
            // ---------------------------------------------------------------
            var pointLightList = new System.Collections.Generic.List<PointLightData>();

            foreach (var light in allLights)
            {
                if (!light.enabled || !light.gameObject.activeInHierarchy) continue;
                if (light.type != LightType.Point) continue;

                Color fc = light.color * light.intensity;

                // Read optional sphere radius from the PointLightRadius component.
                // Falls back to 0 (hard point light) when the component is absent.
                var    plr    = light.GetComponent<PointLightRadius>();
                float  radius = plr != null ? Mathf.Max(0f, plr.radius) : 0f;

                pointLightList.Add(new PointLightData
                {
                    position = light.transform.position,
                    range    = light.range,
                    color    = new Vector3(fc.r, fc.g, fc.b),
                    radius   = radius,
                });
            }

            int pointCount       = pointLightList.Count;
            int pointBufferCount = Mathf.Max(pointCount, 1);
            if (m_PointLightBuffer == null || m_PointLightBuffer.count < pointBufferCount)
            {
                m_PointLightBuffer?.Release();
                m_PointLightBuffer = new GraphicsBuffer(
                    GraphicsBuffer.Target.Structured, pointBufferCount,
                    Marshal.SizeOf<PointLightData>());
            }

            if (pointCount > 0)
                m_PointLightBuffer.SetData(pointLightList.ToArray());

            if (cameraData.camera.cameraType != CameraType.Game && cameraData.camera.cameraType != CameraType.SceneView)
            {
                return;
            }


            // if (m_Settings.usePackedData)
            // {
            //     Shader.EnableKeyword("_USEPACK");
            // }
            // else
            {
                Shader.DisableKeyword("_USEPACK");
            }

            var resourceData = frameData.Get<UniversalResourceData>();

            int2 outputResolution = new int2((int)(cameraData.camera.pixelWidth * cameraData.renderScale), (int)(cameraData.camera.pixelHeight * cameraData.renderScale));

            // Debug.Log($"Output Resolution: {outputResolution.x} x {outputResolution.y}");
            var xrPass = cameraData.xr;
            var isXr = xrPass.enabled;
            if (xrPass.enabled)
            {
                // Debug.Log($"XR Enabled. Eye Texture Resolution: {xrPass.renderTargetDesc.width} x {xrPass.renderTargetDesc.height}");

                outputResolution = new int2(xrPass.renderTargetDesc.width, xrPass.renderTargetDesc.height);
            }

            NrdDenoiser.EnsureResources(outputResolution);

            var renderResolution = NrdDenoiser.renderResolution;
            
            Shader.SetGlobalRayTracingAccelerationStructure(g_AccelStructID, AccelerationStructure);

            using var builder = renderGraph.AddUnsafePass<PassData>("Path Tracing Pass", out var passData);

            passData.OpaqueTs = OpaqueTs;
            passData.TransparentTs = TransparentTs;
            passData.CompositionCs = CompositionCs;
            passData.TaaCs = TaaCs;
            passData.DlssBeforeCs = DlssBeforeCs;
            passData.BlitMaterial = BiltMaterial;

            passData.SharcResolveCs = SharcResolveCs;
            passData.SharcUpdateTs = SharcUpdateTs;
            passData.AccumulationBuffer = AccumulationBuffer;
            passData.HashEntriesBuffer = HashEntriesBuffer;
            passData.ResolvedBuffer = ResolvedBuffer;
            passData.passIndex = isXr ? xrPass.multipassId : 0;
            // passData._dataBuilder = _dataBuilder;
            passData.SpotLightBuffer  = m_SpotLightBuffer;
            passData.AreaLightBuffer  = m_AreaLightBuffer;
            passData.PointLightBuffer = m_PointLightBuffer;

            // Auto-exposure pass data
            passData.AeCs                  = AutoExposureCs;
            passData.AeHistogramBuffer     = AeHistogramBuffer;
            passData.AeExposureBuffer      = AeExposureBuffer;
            passData.AeEnabled             = m_Settings.enableAutoExposure;
            passData.AeEVMin               = m_Settings.aeEVMin;
            passData.AeEVMax               = m_Settings.aeEVMax;
            passData.AeLowPercent          = m_Settings.aeLowPercent;
            passData.AeHighPercent         = m_Settings.aeHighPercent;
            passData.AeSpeedUp             = m_Settings.aeAdaptationSpeedUp;
            passData.AeSpeedDown           = m_Settings.aeAdaptationSpeedDown;
            passData.AeDeltaTime           = Time.deltaTime;
            passData.AeExposureCompensation = m_Settings.aeExposureCompensation;
            passData.AeMinExposure         = m_Settings.aeMinExposure;
            passData.AeMaxExposure         = m_Settings.aeMaxExposure;
            passData.AeTexWidth            = (uint)renderResolution.x;
            passData.AeTexHeight           = (uint)renderResolution.y;
            passData.ManualExposure        = m_Settings.exposure;

            var gSunDirection = -lightForward;
            var up = new Vector3(0, 1, 0);
            var gSunBasisX = math.normalize(math.cross(new float3(up.x, up.y, up.z), new float3(gSunDirection.x, gSunDirection.y, gSunDirection.z)));
            var gSunBasisY = math.normalize(math.cross(new float3(gSunDirection.x, gSunDirection.y, gSunDirection.z), gSunBasisX));

            // var cam = cameraData.camera;


            passData.NrdDataPtr = NrdDenoiser.GetInteropDataPtr(cameraData, gSunDirection);
            passData.RRDataPtr = DLRRDenoiser.GetInteropDataPtr(cameraData, NrdDenoiser);

            passData.DataPtr = prepareLightResource.GetInteropDataPtr();
            passData.RtxdiResources = rtxdiResources;
            passData.RestirDIContext = restirDIContext;

            var proj = isXr ? xrPass.GetProjMatrix() : cameraData.camera.projectionMatrix;

            var m11 = proj.m11;


            var rectW = (uint)(renderResolution.x * NrdDenoiser.resolutionScale + 0.5f);
            var rectH = (uint)(renderResolution.y * NrdDenoiser.resolutionScale + 0.5f);

            // todo prev
            var rectWprev = (uint)(renderResolution.x * NrdDenoiser.prevResolutionScale + 0.5f);
            var rectHprev = (uint)(renderResolution.y * NrdDenoiser.prevResolutionScale + 0.5f);


            var renderSize = new float2((renderResolution.x), (renderResolution.y));
            var outputSize = new float2((outputResolution.x), (outputResolution.y));
            var rectSize = new float2(rectW, rectH);

            var rectSizePrev = new float2((rectWprev), (rectHprev));
            var jitter = (m_Settings.cameraJitter ? NrdDenoiser.ViewportJitter : 0f) / rectSize;


            float fovXRad = math.atan(1.0f / proj.m00) * 2.0f;
            float horizontalFieldOfView = fovXRad * Mathf.Rad2Deg;

            float nearZ = proj.m23 / (proj.m22 - 1.0f);

            float emissionIntensity = m_Settings.emissionIntensity * (m_Settings.emission ? 1.0f : 0.0f);

            float ACCUMULATION_TIME = 0.5f;
            int MAX_HISTORY_FRAME_NUM = 60;

            float fps = 1000.0f / Mathf.Max(Time.deltaTime * 1000.0f, 0.0001f);
            fps = math.min(fps, 121.0f);

            // Debug.Log(fps);

            float resetHistoryFactor = 1.0f;


            float otherMaxAccumulatedFrameNum = GetMaxAccumulatedFrameNum(ACCUMULATION_TIME, fps);
            otherMaxAccumulatedFrameNum = math.min(otherMaxAccumulatedFrameNum, (MAX_HISTORY_FRAME_NUM));
            otherMaxAccumulatedFrameNum *= resetHistoryFactor;


            uint sharcMaxAccumulatedFrameNum = (uint)(otherMaxAccumulatedFrameNum * (m_Settings.boost ? m_Settings.boostFactor : 1.0f) + 0.5f);
            // Debug.Log($"sharcMaxAccumulatedFrameNum: {sharcMaxAccumulatedFrameNum}");
            float taaMaxAccumulatedFrameNum = otherMaxAccumulatedFrameNum * 0.5f;
            float prevFrameMaxAccumulatedFrameNum = otherMaxAccumulatedFrameNum * 0.3f;


            float minProbability = 0.0f;
            if (m_Settings.tracingMode == RESOLUTION.RESOLUTION_FULL_PROBABILISTIC)
            {
                HitDistanceReconstructionMode mode = HitDistanceReconstructionMode.OFF;
                if (m_Settings.denoiser == DenoiserType.DENOISER_REBLUR)
                    mode = HitDistanceReconstructionMode.OFF;
                //     mode = m_ReblurSettings.hitDistanceReconstructionMode;
                // else if (m_Settings.denoiser == DenoiserType.DENOISER_RELAX)
                //     mode = m_RelaxSettings.hitDistanceReconstructionMode;

                // Min / max allowed probability to guarantee a sample in 3x3 or 5x5 area - https://godbolt.org/z/YGYo1rjnM
                if (mode == HitDistanceReconstructionMode.AREA_3X3)
                    minProbability = 1.0f / 4.0f;
                else if (mode == HitDistanceReconstructionMode.AREA_5X5)
                    minProbability = 1.0f / 16.0f;
            }


            var globalConstants = new GlobalConstants
            {
                gViewToWorld = NrdDenoiser.worldToView.inverse,
                gViewToWorldPrev =  NrdDenoiser.prevWorldToView.inverse,
                gViewToClip = NrdDenoiser.viewToClip,
                gWorldToView = NrdDenoiser.worldToView,
                gWorldToViewPrev = NrdDenoiser.prevWorldToView,
                gWorldToClip = NrdDenoiser.worldToClip,
                gWorldToClipPrev = NrdDenoiser.prevWorldToClip,

                gHitDistParams = new float4(3, 0.1f, 20, -25),
                gCameraFrustum = GetNrdFrustum(cameraData),
                gSunBasisX = new float4(gSunBasisX.x, gSunBasisX.y, gSunBasisX.z, 0),
                gSunBasisY = new float4(gSunBasisY.x, gSunBasisY.y, gSunBasisY.z, 0),
                gSunDirection = new float4(gSunDirection.x, gSunDirection.y, gSunDirection.z, 0),
                gCameraGlobalPos = new float4(NrdDenoiser.camPos, 0),
                gCameraGlobalPosPrev = new float4(NrdDenoiser.prevCamPos, 0),
                gViewDirection = new float4(cameraData.camera.transform.forward, 0),
                gHairBaseColor = new float4(0.1f, 0.1f, 0.1f, 1.0f),

                gHairBetas = new float2(0.25f, 0.3f),
                gOutputSize = outputSize,
                gRenderSize = renderSize,
                gRectSize = rectSize,
                gInvOutputSize = new float2(1.0f, 1.0f) / outputSize,
                gInvRenderSize = new float2(1.0f, 1.0f) / renderSize,
                gInvRectSize = new float2(1.0f, 1.0f) / rectSize,
                gRectSizePrev = rectSizePrev,
                gJitter = jitter,

                gEmissionIntensity = emissionIntensity,
                gNearZ = -nearZ,
                gSeparator = m_Settings.splitScreen,
                gRoughnessOverride = 0,
                gMetalnessOverride = 0,
                gUnitToMetersMultiplier = 1.0f,
                gTanSunAngularRadius = math.tan(math.radians(m_Settings.sunAngularDiameter * 0.5f)),
                gTanPixelAngularRadius = math.tan(0.5f * math.radians(horizontalFieldOfView) / rectSize.x),
                gDebug = 0,
                gPrevFrameConfidence = (m_Settings.usePrevFrame && !m_Settings.RR) ? prevFrameMaxAccumulatedFrameNum / (1.0f + prevFrameMaxAccumulatedFrameNum) : 0.0f,
                gUnproject = 1.0f / (0.5f * rectH * m11),
                gAperture = m_Settings.dofAperture * 0.01f,
                gFocalDistance = m_Settings.dofFocalDistance,
                gFocalLength = (0.5f * (35.0f * 0.001f)) / math.tan(math.radians(horizontalFieldOfView * 0.5f)),
                gTAA = (m_Settings.denoiser != DenoiserType.DENOISER_REFERENCE && m_Settings.TAA) ? 1.0f / (1.0f + taaMaxAccumulatedFrameNum) : 1.0f,
                gHdrScale = 1.0f,
                gExposure = m_Settings.exposure,
                gMipBias = m_Settings.mipBias,
                gOrthoMode = cameraData.camera.orthographic ? 1.0f : 0f,
                gIndirectDiffuse = m_Settings.indirectDiffuse ? 1.0f : 0.0f,
                gIndirectSpecular = m_Settings.indirectSpecular ? 1.0f : 0.0f,
                gMinProbability = minProbability,

                gSharcMaxAccumulatedFrameNum = sharcMaxAccumulatedFrameNum,
                gDenoiserType = (uint)m_Settings.denoiser,
                gDisableShadowsAndEnableImportanceSampling = m_Settings.importanceSampling ? 1u : 0u,
                gFrameIndex = (uint)Time.frameCount,
                gForcedMaterial = 0,
                gUseNormalMap = 1,
                gBounceNum = m_Settings.bounceNum,
                gResolve = 1,
                gValidation = 1,
                gSR = (m_Settings.SR && !m_Settings.RR) ? 1u : 0u,
                gRR = m_Settings.RR ? 1u : 0,
                gIsSrgb = 0,
                gOnScreen = 0,
                gTracingMode = m_Settings.RR ? (uint)RESOLUTION.RESOLUTION_FULL_PROBABILISTIC : (uint)m_Settings.tracingMode,
                gSampleNum = m_Settings.rpp,
                gPSR = m_Settings.psr ? (uint)1 : 0,
                gSHARC = m_Settings.SHARC ? (uint)1 : 0,
                gTrimLobe = m_Settings.specularLobeTrimming ? 1u : 0,
            };
            
            globalConstants.gSpotLightCount  = (uint)spotCount;
            globalConstants.gAreaLightCount  = (uint)areaCount;
            globalConstants.gPointLightCount = (uint)pointCount;
            globalConstants.gSssScatteringColor    = new float3(m_Settings.sssScatteringColor.r, m_Settings.sssScatteringColor.g, m_Settings.sssScatteringColor.b);
            globalConstants.gSssMinThreshold       = m_Settings.sssMinThreshold;
            globalConstants.gSssTransmissionBsdfSampleCount       = m_Settings.sssTransmissionBsdfSampleCount;
            globalConstants.gSssTransmissionPerBsdfScatteringSampleCount       = m_Settings.sssTransmissionPerBsdfScatteringSampleCount;
            
            globalConstants.gSssScale              = m_Settings.sssScale;
            globalConstants.gSssAnisotropy = m_Settings.sssAnisotropy;
            globalConstants.gSssMaxSampleRadius    = m_Settings.sssMaxSampleRadius;
            
            
            globalConstants.gIsEditor = cameraData.camera.cameraType == CameraType.SceneView ? 1u : 0u;
            
            // Debug.Log(globalConstants.ToString());

            var textureDesc = resourceData.activeColorTexture.GetDescriptor(renderGraph);
            textureDesc.enableRandomWrite = true;
            textureDesc.depthBufferBits = 0;
            textureDesc.clearBuffer = false;
            textureDesc.discardBuffer = false;
            textureDesc.width = renderResolution.x;
            textureDesc.height = renderResolution.y;

            CreateTextureHandle(renderGraph, passData, textureDesc, builder);

            passData.GlobalConstants = globalConstants;
            
            
            restirDIContext.SetFrameIndex((uint)Time.frameCount);
            
            
            var resamplingConstants = new ResamplingConstants
            {
                runtimeParams = restirDIContext.GetRuntimeParams()
            };

            resamplingConstants.lightBufferParams.localLightBufferRegion.firstLightIndex = 0;
            resamplingConstants.lightBufferParams.localLightBufferRegion.numLights = 3964;
            
            resamplingConstants.lightBufferParams.infiniteLightBufferRegion.firstLightIndex = 0;
            resamplingConstants.lightBufferParams.infiniteLightBufferRegion.numLights = 0;
            
            resamplingConstants.lightBufferParams.environmentLightParams.lightPresent = 0;
            resamplingConstants.lightBufferParams.environmentLightParams.lightIndex = (0xffffffffu);


            resamplingConstants.restirDIReservoirBufferParams = restirDIContext.GetReservoirBufferParameters();

            resamplingConstants.frameIndex = restirDIContext.GetFrameIndex();
            resamplingConstants.numInitialSamples = m_Settings.localLightSamples;
            resamplingConstants.numSpatialSamples = 0;
            resamplingConstants.useAccurateGBufferNormal = 0;
            resamplingConstants.numInitialBRDFSamples = m_Settings.brdfSamples;
            resamplingConstants.brdfCutoff = 0;
            resamplingConstants.pad2 = new uint2(0, 0);
            resamplingConstants.enableResampling = m_Settings.enableResampling ? 1u : 0u;
            resamplingConstants.unbiasedMode = 0;
            resamplingConstants.inputBufferIndex = (resamplingConstants.frameIndex & 1u) ^ 1;
            resamplingConstants.outputBufferIndex = (resamplingConstants.frameIndex & 1u);
             
            passData.ResamplingConstants = resamplingConstants;
            
            // Debug.Log($"Reservoir reservoirArrayPitch: {resamplingConstants.restirDIReservoirBufferParams.reservoirArrayPitch}, reservoirBlockRowPitch: {resamplingConstants.restirDIReservoirBufferParams.reservoirBlockRowPitch}");
            
            
            
            
            
            passData.CameraTexture = resourceData.activeColorTexture;
            passData.outputGridW = (uint)((renderResolution.x + 15) / 16);
            passData.outputGridH = (uint)((renderResolution.y + 15) / 16);
            passData.rectGridW = (uint)((rectW + 15) / 16);
            passData.rectGridH = (uint)((rectH + 15) / 16);
            passData.m_RenderResolution = renderResolution;


            passData.ConstantBuffer = _pathTracingSettingsBuffer;
            passData.ResamplingConstantBuffer = _resamplingConstantsBuffer;
            
            passData.Setting = m_Settings;
            passData.resolutionScale = NrdDenoiser.resolutionScale;
            passData.ScramblingRanking = ScramblingRanking;
            passData.Sobol = Sobol;

            builder.UseTexture(passData.CameraTexture, AccessFlags.Write);

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }

        private void CreateTextureHandle(RenderGraph renderGraph, PassData passData, TextureDesc textureDesc, IUnsafeRenderGraphBuilder builder)
        {
            passData.OutputTexture = CreateTex(textureDesc, renderGraph, "PathTracingOutput", GraphicsFormat.R16G16B16A16_SFloat);

            passData.Mv = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_MV));
            passData.ViewZ = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_VIEWZ));
            passData.NormalRoughness = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_NORMAL_ROUGHNESS));

            passData.BaseColorMetalness = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_BASECOLOR_METALNESS));
            passData.DirectLighting = CreateTex(textureDesc, renderGraph, "DirectLighting", GraphicsFormat.B10G11R11_UFloatPack32);
            passData.DirectEmission = CreateTex(textureDesc, renderGraph, "DirectEmission", GraphicsFormat.B10G11R11_UFloatPack32);
            // passData.SpotDirect = CreateTex(textureDesc, renderGraph, "SpotDirect", GraphicsFormat.B10G11R11_UFloatPack32);

            passData.Penumbra = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_PENUMBRA));
            passData.Diff = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_DIFF_RADIANCE_HITDIST));
            passData.Spec = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.IN_SPEC_RADIANCE_HITDIST));

            // 输出
            passData.ShadowTranslucency = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.OUT_SHADOW_TRANSLUCENCY));
            passData.DenoisedDiff = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.OUT_DIFF_RADIANCE_HITDIST));
            passData.DenoisedSpec = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.OUT_SPEC_RADIANCE_HITDIST));
            passData.Validation = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.OUT_VALIDATION));

            passData.ComposedDiff = CreateTex(textureDesc, renderGraph, "ComposedDiff", GraphicsFormat.R16G16B16A16_SFloat);
            passData.ComposedSpecViewZ = CreateTex(textureDesc, renderGraph, "ComposedSpec_ViewZ", GraphicsFormat.R16G16B16A16_SFloat);

            passData.Composed = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.Composed));

            passData.TaaHistory = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.TaaHistory));
            passData.TaaHistoryPrev = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.TaaHistoryPrev));
            passData.PsrThroughput = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.PsrThroughput));

            passData.RRGuide_DiffAlbedo = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.RRGuide_DiffAlbedo));
            passData.RRGuide_SpecAlbedo = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.RRGuide_SpecAlbedo));
            passData.RRGuide_SpecHitDistance = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.RRGuide_SpecHitDistance));
            passData.RRGuide_Normal_Roughness = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.RRGuide_Normal_Roughness));
            passData.DlssOutput = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.DlssOutput));

            // RTXDI：上一帧 GBuffer
            passData.PrevViewZ = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.Prev_ViewZ));
            passData.PrevNormalRoughness = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.Prev_NormalRoughness));
            passData.PrevBaseColorMetalness = renderGraph.ImportTexture(NrdDenoiser.GetRT(ResourceType.Prev_BaseColorMetalness));


            builder.UseTexture(passData.OutputTexture, AccessFlags.ReadWrite);

            builder.UseTexture(passData.Mv, AccessFlags.ReadWrite);
            builder.UseTexture(passData.ViewZ, AccessFlags.ReadWrite);
            builder.UseTexture(passData.NormalRoughness, AccessFlags.ReadWrite);
            builder.UseTexture(passData.BaseColorMetalness, AccessFlags.ReadWrite);

            builder.UseTexture(passData.DirectLighting, AccessFlags.ReadWrite);
            builder.UseTexture(passData.DirectEmission, AccessFlags.ReadWrite);
            // builder.UseTexture(passData.SpotDirect, AccessFlags.ReadWrite);

            builder.UseTexture(passData.Penumbra, AccessFlags.ReadWrite);
            builder.UseTexture(passData.Diff, AccessFlags.ReadWrite);
            builder.UseTexture(passData.Spec, AccessFlags.ReadWrite);

            // 输出
            builder.UseTexture(passData.ShadowTranslucency, AccessFlags.ReadWrite);
            builder.UseTexture(passData.DenoisedDiff, AccessFlags.ReadWrite);
            builder.UseTexture(passData.DenoisedSpec, AccessFlags.ReadWrite);
            builder.UseTexture(passData.Validation, AccessFlags.ReadWrite);

            builder.UseTexture(passData.ComposedDiff, AccessFlags.ReadWrite);
            builder.UseTexture(passData.ComposedSpecViewZ, AccessFlags.ReadWrite);
            builder.UseTexture(passData.Composed, AccessFlags.ReadWrite);

            builder.UseTexture(passData.TaaHistory, AccessFlags.ReadWrite);
            builder.UseTexture(passData.TaaHistoryPrev, AccessFlags.ReadWrite);
            builder.UseTexture(passData.PsrThroughput, AccessFlags.ReadWrite);

            builder.UseTexture(passData.RRGuide_DiffAlbedo, AccessFlags.ReadWrite);
            builder.UseTexture(passData.RRGuide_SpecAlbedo, AccessFlags.ReadWrite);
            builder.UseTexture(passData.RRGuide_SpecHitDistance, AccessFlags.ReadWrite);
            builder.UseTexture(passData.RRGuide_Normal_Roughness, AccessFlags.ReadWrite);
            builder.UseTexture(passData.DlssOutput, AccessFlags.ReadWrite);

            builder.UseTexture(passData.PrevViewZ, AccessFlags.ReadWrite);
            builder.UseTexture(passData.PrevNormalRoughness, AccessFlags.ReadWrite);
            builder.UseTexture(passData.PrevBaseColorMetalness, AccessFlags.ReadWrite);
        }

        private TextureHandle CreateTex(TextureDesc textureDesc, RenderGraph renderGraph, string name, GraphicsFormat format)
        {
            textureDesc.format = format;
            textureDesc.name = name;
            return renderGraph.CreateTexture(textureDesc);
        }

        public void Dispose()
        {
            _pathTracingSettingsBuffer?.Release();
            _resamplingConstantsBuffer?.Release();
            m_SpotLightBuffer?.Release();
            m_AreaLightBuffer?.Release();
            m_PointLightBuffer?.Release();
        }
    }
}