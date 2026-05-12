using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using DLRR;
using DLSR;
using mini;
using NativeRender;
using Nrd;
using Rtxdi;
using RTXDI;
using Rtxdi.DI;
using Rtxdi.GI;
using Rtxdi.PT;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using CheckerboardMode = Nrd.CheckerboardMode;

namespace PathTracing
{
    /// <summary>
    /// Pure NativeComputeShader RTXDI feature.
    ///
    /// Mirrors <see cref="UnityRtxdiFeature"/> but every per-pass shader is a
    /// <see cref="NativeComputeShader"/> compiled from the pre-shipped HLSL under
    /// <c>UnityProject/Assets/RTXDI/Shaders</c>. Inline RayQuery (USE_RAY_QUERY) is baked
    /// into the shipped DXIL so no <c>RayTracingShader</c> is needed.
    ///
    /// This is currently a SKELETON: pass instantiation, scene/light scene, AS build, and constant
    /// buffer initialisation match <see cref="UnityRtxdiFeature"/>, but <see cref="AddRenderPasses"/>
    /// only enqueues the reference <see cref="NativeRtxdiGenerateInitialSamplesPass"/> behind a
    /// <c>setting.useNativeRtxdiPipeline</c> guard. Remaining passes are implemented incrementally
    /// (see /memories/session/plan.md).
    /// </summary>
    public class NativeRtxdiFeature : ScriptableRendererFeature
    {
        // -------------------------------------------------------------------
        // User-visible configuration  (mirrors UnityRtxdiFeature)
        // -------------------------------------------------------------------
        public NativeRtxdiSetting setting;

        // public GlobalConstants     globalConstants;
        public NativeResamplingConstants  resamplingConstants;
        public NativeCompositingConstants compositingConstants;

        public RenderPassEvent renderPassEvent = RenderPassEvent.BeforeRenderingPostProcessing;

        public Material finalMaterial;

        // ── Native compute shaders (one per RTXDI HLSL pass) ─────────────────
        // Source: UnityProject/Assets/RTXDI/Shaders/...

        // Prepare / Environment
        public NativeComputeShader prepareLightsCs; // PrepareLights.computeshader
        public NativeComputeShader preprocessEnvironmentMapCs; // PreprocessEnvironmentMap.computeshader  (INPUT_ENVIRONMENT_MAP=1)
        public NativeComputeShader preprocessLocalLightCs; // PreprocessLocalLight.computeshader   (INPUT_ENVIRONMENT_MAP=0)

        // GBuffer
        public NativeComputeShader raytracedGBufferCs; // RaytracedGBuffer.computeshader
        public NativeComputeShader postprocessGBufferCs; // PostprocessGBuffer.computeshader

        // Presampling
        public NativeComputeShader presampleLightsCs; // LightingPasses/Presampling/PresampleLights.computeshader
        public NativeComputeShader presampleEnvironmentMapCs; // LightingPasses/Presampling/PresampleEnvironmentMap.computeshader
        public NativeComputeShader presampleReGirCs; // LightingPasses/Presampling/PresampleReGIR.computeshader

        // ReSTIR DI
        public NativeComputeShader diGenerateInitialSamplesCs; // LightingPasses/DI/GenerateInitialSamples.computeshader
        public NativeComputeShader diTemporalResamplingCs; // LightingPasses/DI/TemporalResampling.computeshader
        public NativeComputeShader diSpatialResamplingCs; // LightingPasses/DI/SpatialResampling.computeshader
        public NativeComputeShader diShadeSamplesCs; // LightingPasses/DI/ShadeSamples.computeshader

        // Indirect / ReSTIR GI
        public NativeComputeShader brdfRayTracingCs; // LightingPasses/BrdfRayTracing.computeshader
        public NativeComputeShader shadeSecondarySurfacesCs; // LightingPasses/ShadeSecondarySurfaces.computeshader
        public NativeComputeShader giTemporalResamplingCs; // LightingPasses/GI/TemporalResampling.computeshader
        public NativeComputeShader giSpatialResamplingCs; // LightingPasses/GI/SpatialResampling.computeshader
        public NativeComputeShader giFinalShadingCs; // LightingPasses/GI/FinalShading.computeshader

        // ReSTIR PT
        public NativeComputeShader ptGenerateInitialSamplesCs; // LightingPasses/PT/GenerateInitialSamples.computeshader
        public NativeComputeShader ptTemporalResamplingCs; // LightingPasses/PT/TemporalResampling.computeshader
        public NativeComputeShader ptSpatialResamplingCs; // LightingPasses/PT/SpatialResampling.computeshader
        public NativeComputeShader ptFillSampleIDCs; // LightingPasses/PT/FillSampleID.computeshader
        public NativeComputeShader ptComputeDuplicationMapCs; // LightingPasses/PT/ComputeDuplicationMap.computeshader
        public NativeComputeShader ptFinalShadingCs; // LightingPasses/PT/FinalShading.computeshader

        // Auxiliary
        public NativeComputeShader compositingPassCs; // CompositingPass.computeshader

        // Denoising passes (gradient filter + confidence)
        public NativeComputeShader filterGradientsPassCs; // DenoisingPasses/FilterGradientsPass.computeshader
        public NativeComputeShader confidencePassCs; // DenoisingPasses/ConfidencePass.computeshader

        // Tone mapping
        public NativeComputeShader toneMappingHistogramCs; // Shaders/donut/histogram.computeshader
        public NativeComputeShader toneMappingExposureCs; // Shaders/donut/exposure.computeshader
        public NativeComputeShader toneMappingCs; // Shaders/donut/tonemapping.computeshader

        // -------------------------------------------------------------------
        // Pass instances (one per implemented pass; rest filled in over time)
        // -------------------------------------------------------------------
        private NativeRtxdiPresampleLightsPass        _presampleLightsNativePass;
        private NativeRtxdiPresampleReGirPass         _presampleReGirNativePass;
        private NativeRtxdiGenerateInitialSamplesPass _diGenerateInitialSamplesPass;
        private NativeRtxdiTemporalResamplingPass     _diTemporalResamplingPass;
        private NativeRtxdiSpatialResamplingPass      _diSpatialResamplingPass;
        private NativeRtxdiShadeSamplesPass           _diShadeSamplesPass;

        private NativeRtxdiBrdfRayTracingPass         _brdfRayTracingPass;
        private NativeRtxdiShadeSecondarySurfacesPass _shadeSecondarySurfacesPass;
        private NativeRtxdiGITemporalResamplingPass   _giTemporalResamplingPass;
        private NativeRtxdiGISpatialResamplingPass    _giSpatialResamplingPass;
        private NativeRtxdiGIFinalShadingPass         _giFinalShadingPass;

        private NativeRtxdiPTGenerateInitialSamplesPass _ptGenerateInitialSamplesPass;
        private NativeRtxdiPTTemporalResamplingPass     _ptTemporalResamplingPass;
        private NativeRtxdiPTSpatialResamplingPass      _ptSpatialResamplingPass;
        private NativeRtxdiPTFillSampleIDPass           _ptFillSampleIDPass;
        private NativeRtxdiPTComputeDuplicationMapPass  _ptComputeDuplicationMapPass;
        private NativeRtxdiPTFinalShadingPass           _ptFinalShadingPass;

        private NativeFrameTick _nativeFrameTickPass;

        // Denoising: gradient filter + confidence (mirror FullSample FilterGradientsPass + ConfidencePass)
        private NativeRtxdiFilterGradientsPass _filterGradientsPass;
        private NativeRtxdiConfidencePass      _confidencePass;

        private NativeToneMappingPass _toneMappingPass;

        private          DlssSRPass                     _dlssrPass;
        private readonly Dictionary<long, DlsrUpscaler> _dlsrUpscalers = new();

        // -------------------------------------------------------------------
        // Managed scaffolding passes (still using RayTracingShader / ComputeShader).
        // Will be replaced with native equivalents once their .computeshader assets are wired in.
        // -------------------------------------------------------------------
        private NativeRtxdiPrepareLightsPass _prepareLightsPass;
        private GBufferRasterPass            _gBufferRasterPass;
        private GBufferRasterPass.Resource   _gBufferRasterResource;
        private PdfTexturePass               _pdfTexturePass;
        private GenerateMipsPass             _generateMipsPass;
        private NativeRtxdiPdfMipsPass       _pdfMipsPass;
        private PresamplePass                _presamplePass;
        private NrdPass                      _nrdDenoisePass;
        private NativeRtxdiCompositingPass   _compositingPass;
        private NativeRtxdiOutputBlitPass    _outputBlitPass;

        // Native: builds GPUScene TLAS at the head of the native pipeline.
        private NativeRtxdiBuildAccelerationStructurePass _buildAsPass;

        // Native GBuffer passes (replace managed GBufferRasterPass)
        private NativeRtxdiRaytracedGBufferPass   _raytracedGBufferPass;
        private NativeRtxdiPostprocessGBufferPass _postprocessGBufferPass;

        // -------------------------------------------------------------------
        // Shared resources
        // -------------------------------------------------------------------
        // private GraphicsBuffer _constantBuffer;
        private GraphicsBuffer _resamplingConstantBuffer;
        private GraphicsBuffer _perPassConstantBuffer;
        private GraphicsBuffer _gbufferConstantBuffer;

        private NativeRtxdiGPUScene _rtxdiGpuScene;

        private readonly NativeResamplingConstants[]   _resamplingConstantsArray = new NativeResamplingConstants[1];
        private readonly NativeRtxdiPerPassConstants[] _perPassConstantsArray    = new NativeRtxdiPerPassConstants[1];
        private readonly NativeGBufferConstants[]      _gbufferConstantsArray    = new NativeGBufferConstants[1];

        private readonly Dictionary<long, NrdDenoiser>                 _nrdDenoisers      = new();
        private readonly Dictionary<long, NativeRtxdiTextureResources> _resourcePools     = new();
        private readonly Dictionary<long, NativeRtxdiResources>        _rtxdiResources    = new();
        private readonly Dictionary<long, ImportanceSamplingContext>   _isContexts        = new();
        private readonly Dictionary<long, CameraFrameState>            _cameraFrameStates = new();

        private GraphicsBuffer _compositingConstantBuffer;

        // -------------------------------------------------------------------
        // ScriptableRendererFeature lifecycle
        // -------------------------------------------------------------------

        public override void Create()
        {
            _rtxdiGpuScene ??= new NativeRtxdiGPUScene();

            if (_gbufferConstantBuffer == null)
                InitializeBuffers();

            // Managed scaffolding passes (NOTE: PrepareLights / PdfMipmap are intentionally
            // NOT instantiated here — see TODO in AddRenderPasses about porting
            // RTXDI/Samples/FullSample/Source/RenderPasses/PrepareLightsPass.cpp).

            _prepareLightsPass      ??= new NativeRtxdiPrepareLightsPass(prepareLightsCs) { renderPassEvent = renderPassEvent };
            _gBufferRasterPass      ??= new GBufferRasterPass() { renderPassEvent                           = renderPassEvent };
            _gBufferRasterResource  ??= new GBufferRasterPass.Resource();
            _buildAsPass            ??= new NativeRtxdiBuildAccelerationStructurePass() { renderPassEvent             = renderPassEvent };
            _raytracedGBufferPass   ??= new NativeRtxdiRaytracedGBufferPass(raytracedGBufferCs) { renderPassEvent     = renderPassEvent };
            _postprocessGBufferPass ??= new NativeRtxdiPostprocessGBufferPass(postprocessGBufferCs) { renderPassEvent = renderPassEvent };
            // PresamplePass currently consumes a managed ComputeShader asset; its native port is pending.
            // PresampleReGIR is fully native via NativeRtxdiPresampleReGirPass (wired in below).
            _nrdDenoisePass  ??= new NrdPass() { renderPassEvent                                     = renderPassEvent };
            _compositingPass ??= new NativeRtxdiCompositingPass(compositingPassCs) { renderPassEvent = renderPassEvent };

            _pdfMipsPass    ??= new NativeRtxdiPdfMipsPass(preprocessLocalLightCs, preprocessEnvironmentMapCs) { renderPassEvent = renderPassEvent };
            _outputBlitPass ??= new NativeRtxdiOutputBlitPass(finalMaterial) { renderPassEvent                                   = renderPassEvent };

            _presampleLightsNativePass ??= new NativeRtxdiPresampleLightsPass(presampleLightsCs) { renderPassEvent = renderPassEvent };
            _presampleReGirNativePass  ??= new NativeRtxdiPresampleReGirPass(presampleReGirCs) { renderPassEvent   = renderPassEvent };

            _diGenerateInitialSamplesPass ??= new NativeRtxdiGenerateInitialSamplesPass(diGenerateInitialSamplesCs) { renderPassEvent = renderPassEvent };
            _diTemporalResamplingPass     ??= new NativeRtxdiTemporalResamplingPass(diTemporalResamplingCs) { renderPassEvent         = renderPassEvent };
            _diSpatialResamplingPass      ??= new NativeRtxdiSpatialResamplingPass(diSpatialResamplingCs) { renderPassEvent           = renderPassEvent };
            _diShadeSamplesPass           ??= new NativeRtxdiShadeSamplesPass(diShadeSamplesCs) { renderPassEvent                     = renderPassEvent };

            _brdfRayTracingPass         ??= new NativeRtxdiBrdfRayTracingPass(brdfRayTracingCs) { renderPassEvent                 = renderPassEvent };
            _shadeSecondarySurfacesPass ??= new NativeRtxdiShadeSecondarySurfacesPass(shadeSecondarySurfacesCs) { renderPassEvent = renderPassEvent };
            _giTemporalResamplingPass   ??= new NativeRtxdiGITemporalResamplingPass(giTemporalResamplingCs) { renderPassEvent     = renderPassEvent };
            _giSpatialResamplingPass    ??= new NativeRtxdiGISpatialResamplingPass(giSpatialResamplingCs) { renderPassEvent       = renderPassEvent };
            _giFinalShadingPass         ??= new NativeRtxdiGIFinalShadingPass(giFinalShadingCs) { renderPassEvent                 = renderPassEvent };

            _ptGenerateInitialSamplesPass ??= new NativeRtxdiPTGenerateInitialSamplesPass(ptGenerateInitialSamplesCs) { renderPassEvent = renderPassEvent };
            _ptTemporalResamplingPass     ??= new NativeRtxdiPTTemporalResamplingPass(ptTemporalResamplingCs) { renderPassEvent         = renderPassEvent };
            _ptSpatialResamplingPass      ??= new NativeRtxdiPTSpatialResamplingPass(ptSpatialResamplingCs) { renderPassEvent           = renderPassEvent };
            _ptFillSampleIDPass           ??= new NativeRtxdiPTFillSampleIDPass(ptFillSampleIDCs) { renderPassEvent                     = renderPassEvent };
            _ptComputeDuplicationMapPass  ??= new NativeRtxdiPTComputeDuplicationMapPass(ptComputeDuplicationMapCs) { renderPassEvent   = renderPassEvent };
            _ptFinalShadingPass           ??= new NativeRtxdiPTFinalShadingPass(ptFinalShadingCs) { renderPassEvent                     = renderPassEvent };

            _nativeFrameTickPass ??= new NativeFrameTick() { renderPassEvent = renderPassEvent, };

            _filterGradientsPass ??= filterGradientsPassCs != null ? new NativeRtxdiFilterGradientsPass(filterGradientsPassCs) { renderPassEvent = renderPassEvent } : null;
            _confidencePass      ??= confidencePassCs != null ? new NativeRtxdiConfidencePass(confidencePassCs) { renderPassEvent                = renderPassEvent } : null;

            _toneMappingPass ??= (toneMappingHistogramCs != null && toneMappingExposureCs != null && toneMappingCs != null)
                ? new NativeToneMappingPass(toneMappingHistogramCs, toneMappingExposureCs, toneMappingCs) { renderPassEvent = renderPassEvent }
                : null;

            _dlssrPass ??= new DlssSRPass() { renderPassEvent = renderPassEvent };
        }

        public void InitializeBuffers()
        {
            _resamplingConstantBuffer  = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, Marshal.SizeOf<NativeResamplingConstants>());
            _perPassConstantBuffer     = new GraphicsBuffer(GraphicsBuffer.Target.Constant, 1, Marshal.SizeOf<NativeRtxdiPerPassConstants>());
            _gbufferConstantBuffer     = new GraphicsBuffer(GraphicsBuffer.Target.Constant, 1, Marshal.SizeOf<NativeGBufferConstants>());
            _compositingConstantBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Constant, 1, Marshal.SizeOf<NativeCompositingConstants>());
        }

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            var cam = renderingData.cameraData.camera;
            if (cam.cameraType is CameraType.Preview or CameraType.Reflection) return;
            if (cam.cameraType != CameraType.Game && cam.cameraType != CameraType.SceneView) return;

            cam.depthTextureMode = DepthTextureMode.Depth | DepthTextureMode.MotionVectors;

            int eyeIndex = renderingData.cameraData.xr.enabled ? renderingData.cameraData.xr.multipassId : 0;
            if (eyeIndex == 1 && setting.skipRightEyeInVR) return;

            // Mandatory native pass shaders
            if (_diGenerateInitialSamplesPass == null || _diShadeSamplesPass == null)
            {
                Debug.LogWarning("[NativeRtxdiFeature] Missing required native compute shaders — skipping.");
                return;
            }

            // TODO(FullSample port): replace this with NativeRtxdiPrepareLightsPass that mirrors
            // RTXDI/Samples/FullSample/Source/RenderPasses/PrepareLightsPass.{h,cpp} — it should
            // (a) walk emissive MeshRenderers + analytic Unity Lights to build a CPU-side
            //     TaskBuffer + PrimitiveLightBuffer, and
            // (b) dispatch PrepareLights.computeshader to populate LightDataBuffer +
            //     LightIndexMappingBuffer + LocalLightPdfTexture inside NativeRtxdiResources.
            // Until then the feature renders without dynamic emissive lights.

            long uniqueKey = cam.GetInstanceID() + (eyeIndex * 100000L);
            bool isVR      = renderingData.cameraData.xrRendering;

            if (!_resourcePools.TryGetValue(uniqueKey, out var pool))
            {
                pool = new NativeRtxdiTextureResources();
                _resourcePools.Add(uniqueKey, pool);
            }

            if (!_nrdDenoisers.TryGetValue(uniqueKey, out var nrdReblur))
            {
                var camName = isVR ? $"{cam.name}_Eye{eyeIndex}" : cam.name;
                nrdReblur = new NrdDenoiser(camName + "_Rtxdi", new NrdDenoiserDesc[]
                {
                    // new(0, Denoiser.REBLUR_DIFFUSE_SPECULAR),
                    new(0, Denoiser.RELAX_DIFFUSE_SPECULAR),
                });
                _nrdDenoisers.Add(uniqueKey, nrdReblur);
            }

            if (!_cameraFrameStates.TryGetValue(uniqueKey, out var frameState))
            {
                frameState = new CameraFrameState(1);
                _cameraFrameStates.Add(uniqueKey, frameState);
            }

            int2 outputResolution = ComputeOutputResolution(renderingData.cameraData);
            bool resourcesChanged = pool.EnsureResources(outputResolution, setting.upscalerMode);
            int2 renderResolution = pool.renderResolution;

            if (!_isContexts.TryGetValue(uniqueKey, out var isContext))
            {
                var isParams = ImportanceSamplingContext_StaticParameters.Default();
                isParams.renderWidth  = (uint)renderResolution.x;
                isParams.renderHeight = (uint)renderResolution.y;
                isContext             = new ImportanceSamplingContext(isParams);
                _isContexts.Add(uniqueKey, isContext);
            }

            // NativeRtxdiGPUScene tracks scene targets / instance buffers; the native AS handles skinned
            // mesh updates internally via NativeRayTracingSkinnedTarget. The TLAS itself is
            // built later inside NativeRtxdiBuildAccelerationStructurePass (needs a CommandBuffer).
            // Must run before rtxdiResources creation so we can query actual scene counts.
            _rtxdiGpuScene.UpdateForFrame();

            if (!_rtxdiResources.TryGetValue(uniqueKey, out var rtxdiResources))
            {
                // Mirror FullSample's SceneRenderer::UpdateRtxdiResources():
                // count actual emissive meshes / triangles / geometry instances from the live scene,
                // then round up to allocation quanta so minor scene changes don't force a realloc.
                // Rebuild on overflow is deferred (TODO).
                const uint kMeshQuantum     = 128u;
                const uint kTriangleQuantum = 1024u;
                const uint kPrimQuantum     = 128u;

                var  emissiveGeos                                    = _rtxdiGpuScene.GetEmissiveGeometries();
                uint numEmissiveMeshes                               = (uint)emissiveGeos.Count;
                uint numEmissiveTriangles                            = 0u;
                foreach (var e in emissiveGeos) numEmissiveTriangles += e.TriangleCount;
                uint numPrimitiveLights                              = 0u; // analytic lights not yet ported
                uint numGeomInstances                                = (uint)_rtxdiGpuScene.TotalGeometryInstanceCount;

                // Round up to quanta (ensure at least one quantum so buffers are non-zero)
                uint allocMeshes    = Math.Max(kMeshQuantum, (numEmissiveMeshes + kMeshQuantum - 1u) & ~(kMeshQuantum - 1u));
                uint allocTriangles = Math.Max(kTriangleQuantum, (numEmissiveTriangles + kTriangleQuantum - 1u) & ~(kTriangleQuantum - 1u));
                uint allocPrims     = Math.Max(kPrimQuantum, (numPrimitiveLights + kPrimQuantum - 1u) & ~(kPrimQuantum - 1u));
                uint allocGeom      = Math.Max(1u, numGeomInstances);

                rtxdiResources = new NativeRtxdiResources(
                    isContext.GetReSTIRDIContext(),
                    isContext.GetRISBufferSegmentAllocator(),
                    allocMeshes,
                    allocTriangles,
                    allocPrims,
                    allocGeom,
                    1u, // EnvW — no environment map yet
                    1u); // EnvH
                _rtxdiResources.Add(uniqueKey, rtxdiResources);
            }

            if (resourcesChanged)
            {
                frameState.renderResolution = pool.renderResolution;
                frameState.frameIndex       = 0;
            }

            uint curFrame = frameState.frameIndex;
            frameState.Update(renderingData, false, 1);

            _gbufferConstantsArray[0] = NativeGBufferConstantsBuilder.Build(frameState, renderResolution, 1f);
            _gbufferConstantBuffer.SetData(_gbufferConstantsArray);

            bool enableDirectReStirPass = setting.directLightingMode == DirectLightingMode.ReStir;

            var localSettings = setting.lightingSettings;
            localSettings.enablePreviousTLAS        = false;
            localSettings.enableAlphaTestedGeometry = false;
            localSettings.enableTransparentGeometry = false;
            localSettings.denoiserMode              = (uint)setting.denoiserMode;
            localSettings.enableGradients           = enableDirectReStirPass && setting.denoiserMode != RtxDiDenoiserType.DENOISER_MODE_OFF && setting.enableGradients;

            bool enableBrdfAndIndirectPass = setting.directLightingMode == DirectLightingMode.Brdf || setting.indirectLightingMode != IndirectLightingMode.None;
            bool enableIndirect            = setting.indirectLightingMode != IndirectLightingMode.None;
            bool needSecondaryGBuffer      = enableIndirect || setting.directLightingMode == DirectLightingMode.Brdf;
            bool enableReSTIRGI            = setting.indirectLightingMode == IndirectLightingMode.ReStirGI;
            bool enableEmissiveSurfaces    = setting.directLightingMode == DirectLightingMode.Brdf;
            bool enableAdditiveBlend       = enableDirectReStirPass;
            bool enableAccumulation        = false;

            if (!enableDirectReStirPass)
            {
                localSettings.brdfptParams.enableSecondaryResampling = 0u;
                localSettings.enableGradients                        = false;
            }

            // ---- PrepareLight (native FullSample-style port) ----
            // Build CPU-side TaskBuffer + GeometryInstanceToLight, upload to GPU,
            // and get back the RTXDI_LightBufferParameters that encodes the current-frame
            // light region inside the double-buffered LightDataBuffer.
            RTXDI_LightBufferParameters lightBufferParams = default;
            if (_prepareLightsPass != null)
            {
                // nativeCtx is not yet built at this point; create a lightweight context with
                // only the fields needed by BuildTasksOnCpu (Resources + RtxdiGpuScene).
                var prepCtx = new NativeRtxdiPassContext
                {
                    Resources     = rtxdiResources,
                    RtxdiGpuScene = _rtxdiGpuScene,
                };
                lightBufferParams = _prepareLightsPass.BuildTasksOnCpu(
                    prepCtx,
                    setting.environmentMap,
                    setting.environmentRotation,
                    setting.environmentScale);
            }

            uint2 localLightPdfTextureSize = rtxdiResources.LocalLightPdfTextureSize;

            var baseConsts = RtxdiConstantsBuilder.Build(
                setting, localSettings, isContext, frameState, lightBufferParams, localLightPdfTextureSize,
                enableIndirect, enableAdditiveBlend, enableEmissiveSurfaces, enableAccumulation, enableReSTIRGI);

            // Patch enableDenoiserInputPacking after Build() so it isn't overwritten by setting.shadingParams
            baseConsts.restirDI.shadingParams.enableDenoiserInputPacking = !enableIndirect ? 1u : 0u;

            var viewConst = NativeGBufferConstantsBuilder.BuildViewPublic(
                frameState.worldToView, frameState.viewToClip, frameState.worldToClip,
                frameState.camPos, renderResolution, 1f, frameState.viewportJitter);
            var prevViewConst = NativeGBufferConstantsBuilder.BuildViewPublic(
                frameState.prevWorldToView, frameState.prevViewToClip, frameState.prevWorldToClip,
                frameState.prevCamPos, renderResolution, 1f, frameState.prevViewportJitter);

            resamplingConstants = new NativeResamplingConstants
            {
                view         = viewConst,
                prevView     = prevViewConst,
                prevPrevView = prevViewConst, // CameraFrameState has no prev-prev; use prev as approximation

                runtimeParams       = baseConsts.runtimeParams,
                reblurHitDistParams = new float4(3.0f, 0.1f, 20.0f, 0f),

                pad3                 = 0u,
                enablePreviousTLAS   = baseConsts.enablePreviousTLAS,
                denoiserMode         = baseConsts.denoiserMode,
                discountNaiveSamples = baseConsts.discountNaiveSamples,

                enableBrdfIndirect      = baseConsts.enableBrdfIndirect,
                enableBrdfAdditiveBlend = baseConsts.enableBrdfAdditiveBlend,
                enableAccumulation      = baseConsts.enableAccumulation,
                directLightingMode      = (uint)setting.directLightingMode,

                sceneConstants = new NativeSceneConstants
                {
                    enableEnvironmentMap       = 0u,
                    environmentMapTextureIndex = 0u,
                    environmentScale           = 1f,
                    environmentRotation        = 0f,
                    enableAlphaTestedGeometry  = localSettings.enableAlphaTestedGeometry ? 1u : 0u,
                    enableTransparentGeometry  = localSettings.enableTransparentGeometry ? 1u : 0u,
                },

                lightBufferParams                      = baseConsts.lightBufferParams,
                localLightsRISBufferSegmentParams      = baseConsts.localLightsRISBufferSegmentParams,
                environmentLightRISBufferSegmentParams = baseConsts.environmentLightRISBufferSegmentParams,

                restirDI = baseConsts.restirDI,
                regir    = baseConsts.regir,
                restirGI = baseConsts.restirGI,
                restirPT = BuildRestirPTParams(isContext),
                pt       = BuildPTParams(),
                brdfPT   = baseConsts.brdfPT,

                visualizeRegirCells     = baseConsts.visualizeRegirCells,
                enableDenoiserPSR       = 1u,
                usePSRMvecForResampling = 1u,
                updatePSRwithResampling = 1u,

                environmentPdfTextureSize = rtxdiResources.EnvironmentPdfTextureSize,
                localLightPdfTextureSize  = baseConsts.localLightPdfTextureSize,

                debug = default,
            };
            _resamplingConstantsArray[0] = resamplingConstants;
            _resamplingConstantBuffer.SetData(_resamplingConstantsArray);

            _perPassConstantsArray[0] = new NativeRtxdiPerPassConstants { rayCountBufferIndex = 0 };
            _perPassConstantBuffer.SetData(_perPassConstantsArray);

            // ---- Ping-pong GBuffer (current vs previous frame) ----
            bool isOddFrame = (curFrame % 2) == 1;

            // ---- Gradient texture (RTXDI_GRAD_FACTOR = 3) + confidence ping-pong ----
            const int GradFactor = 3; // RTXDI_GRAD_FACTOR from ShaderParameters.h
            var gradDims = new int2(
                (renderResolution.x + GradFactor - 1) / GradFactor,
                (renderResolution.y + GradFactor - 1) / GradFactor);

            bool enableGradients = enableDirectReStirPass && setting.denoiserMode != RtxDiDenoiserType.DENOISER_MODE_OFF && setting.enableGradients;
            if (enableGradients)
                pool.EnsureGradientArray(gradDims);

            // Confidence ping-pong: current frame writes, previous frame is read.
            var diffConfCurrent = isOddFrame ? pool.DiffuseConfidence : pool.PrevDiffuseConfidence;
            var diffConfPrev    = isOddFrame ? pool.PrevDiffuseConfidence : pool.DiffuseConfidence;
            var specConfCurrent = isOddFrame ? pool.SpecularConfidence : pool.PrevSpecularConfidence;
            var specConfPrev    = isOddFrame ? pool.PrevSpecularConfidence : pool.SpecularConfidence;

            RTHandle depth         = isOddFrame ? pool.Depth.Handle : pool.PrevDepth.Handle;
            RTHandle diffuseAlbedo = isOddFrame ? pool.GBufferDiffuseAlbedo.Handle : pool.PrevGBufferDiffuseAlbedo.Handle;
            RTHandle specularRough = isOddFrame ? pool.GBufferSpecularRough.Handle : pool.PrevGBufferSpecularRough.Handle;
            RTHandle normals       = isOddFrame ? pool.GBufferNormals.Handle : pool.PrevGBufferNormals.Handle;
            RTHandle geoNormals    = isOddFrame ? pool.GBufferGeoNormals.Handle : pool.PrevGBufferGeoNormals.Handle;

            // Native IntPtr context — for native compute passes
            IntPtr ToPt(RTHandle h) => h != null && h.rt != null ? h.rt.GetNativeTexturePtr() : IntPtr.Zero;

            var nativeCtx = new NativeRtxdiPassContext
            {
                ResamplingConstantBuffer = _resamplingConstantBuffer,
                PerPassConstantBuffer    = _perPassConstantBuffer,
                GBufferConstantBuffer    = _gbufferConstantBuffer,
                RtxdiGpuScene            = _rtxdiGpuScene,
                ViewDepthPtr             = isOddFrame ? pool.Depth.NativePtr : pool.PrevDepth.NativePtr,
                DiffuseAlbedoPtr         = isOddFrame ? pool.GBufferDiffuseAlbedo.NativePtr : pool.PrevGBufferDiffuseAlbedo.NativePtr,
                SpecularRoughPtr         = isOddFrame ? pool.GBufferSpecularRough.NativePtr : pool.PrevGBufferSpecularRough.NativePtr,
                NormalsPtr               = isOddFrame ? pool.GBufferNormals.NativePtr : pool.PrevGBufferNormals.NativePtr,
                GeoNormalsPtr            = isOddFrame ? pool.GBufferGeoNormals.NativePtr : pool.PrevGBufferGeoNormals.NativePtr,
                PrevViewDepthPtr         = isOddFrame ? pool.PrevDepth.NativePtr : pool.Depth.NativePtr,
                PrevDiffuseAlbedoPtr     = isOddFrame ? pool.PrevGBufferDiffuseAlbedo.NativePtr : pool.GBufferDiffuseAlbedo.NativePtr,
                PrevSpecularRoughPtr     = isOddFrame ? pool.PrevGBufferSpecularRough.NativePtr : pool.GBufferSpecularRough.NativePtr,
                PrevNormalsPtr           = isOddFrame ? pool.PrevGBufferNormals.NativePtr : pool.GBufferNormals.NativePtr,
                PrevGeoNormalsPtr        = isOddFrame ? pool.PrevGBufferGeoNormals.NativePtr : pool.GBufferGeoNormals.NativePtr,
                DirectLightingPtr        = pool.HdrColor.NativePtr,
                EmissivePtr              = pool.GBufferEmissive.NativePtr,
                MotionVectorsPtr         = pool.MotionVectors.NativePtr,
                DeviceDepthPtr           = pool.DeviceDepth.NativePtr,
                LocalLightPdfTexturePtr  = ToPt(rtxdiResources.LocalLightPdfTexture),
                // Lighting output UAVs
                DiffuseLightingPtr         = pool.DiffuseLighting.NativePtr,
                SpecularLightingPtr        = pool.SpecularLighting.NativePtr,
                TemporalSamplePositionsPtr = pool.TemporalSamplePos.NativePtr,
                RestirLuminancePtr         = pool.RestirLuminance.NativePtr,
                PrevRestirLuminancePtr     = pool.PrevRestirLuminance.NativePtr,
                DirectLightingRawPtr       = pool.DirectLightingRaw.NativePtr,
                IndirectLightingRawPtr     = pool.IndirectLightingRaw.NativePtr,
                DenoiserNormalRoughnessPtr = pool.NormalRoughness.NativePtr,
                // Gradient 2DArray (written by DI shaders when enableGradients=1, filtered by FilterGradientsPass)
                GradientsPtr = pool.GradientArrayPtr,
                // Confidence ping-pong
                DiffuseConfidencePtr      = diffConfCurrent.NativePtr,
                PrevDiffuseConfidencePtr  = diffConfPrev.NativePtr,
                SpecularConfidencePtr     = specConfCurrent.NativePtr,
                PrevSpecularConfidencePtr = specConfPrev.NativePtr,
                RayCountBuffer            = rtxdiResources.RayCountBuffer,
                Resources                 = rtxdiResources,
                Pool                      = pool,
                RenderResolution          = renderResolution,
                ResolutionScale           = 1f,
            };

            // ---- Build native TLAS (must run before RaytracedGBuffer which needs SceneBVH) ----
            _buildAsPass.Setup(_rtxdiGpuScene);
            renderer.EnqueuePass(_buildAsPass);

            // ---- PrepareLights (native CS port) ----
            // Dispatches PrepareLights.computeshader to populate LightDataBuffer,
            // LightIndexMappingBuffer, and LocalLightPdfTexture from the TaskBuffer
            // uploaded by BuildTasksOnCpu above.
            if (_prepareLightsPass != null)
            {
                _prepareLightsPass.Setup(nativeCtx);
                renderer.EnqueuePass(_prepareLightsPass);
            }

            // ---- GBuffer (native raytraced path) ----

            _raytracedGBufferPass.Setup(nativeCtx);
            renderer.EnqueuePass(_raytracedGBufferPass);


            _postprocessGBufferPass.Setup(nativeCtx);
            renderer.EnqueuePass(_postprocessGBufferPass);


            // ---- PDF mip chain (mirrors FullSample's m_localLightPdfMipmapPass / m_environmentMapPdfMipmapPass) ----
            // Must run after PrepareLights, which writes mip 0 of LocalLightPdfTexture.
            if (_pdfMipsPass != null && _prepareLightsPass != null)
            {
                _pdfMipsPass.Setup(nativeCtx);
                renderer.EnqueuePass(_pdfMipsPass);
            }

            // ---- Update ReGIR context (center = camera position, mirrors FullSample UpdateReGIRContextFromUI) ----
            {
                var regirContext = isContext.GetReGIRContext();
                setting.regirDynamicParams.center = frameState.camPos;
                regirContext.SetDynamicParameters(setting.regirDynamicParams);
            }

            // ---- Presample local lights (NATIVE) ----
            // Mirrors FullSample: runs after PDF mip chain when using non-uniform local-light sampling.
            if (setting.initialSamplingParams.localLightSamplingMode != ReSTIRDI_LocalLightSamplingMode.Uniform)
            {
                const uint presampleGroupSize = 256u; // RTXDI_PRESAMPLING_GROUP_SIZE
                var        seg                = isContext.GetLocalLightRISBufferSegmentParams();
                uint       groupsX            = (seg.tileSize + presampleGroupSize - 1u) / presampleGroupSize;
                uint       groupsY            = seg.tileCount;
                _presampleLightsNativePass.Setup(nativeCtx, groupsX, groupsY);
                renderer.EnqueuePass(_presampleLightsNativePass);
            }

            // ---- Presample ReGIR (NATIVE) ----
            // Mirrors FullSample: runs after PresampleLights when the sampling mode is ReGIR_RIS.
            if (setting.initialSamplingParams.localLightSamplingMode == ReSTIRDI_LocalLightSamplingMode.ReGIR_RIS &&
                _presampleReGirNativePass != null)
            {
                const uint reGirTileSize = 256u; // RTXDI_PRESAMPLING_GROUP_SIZE used for ReGIR
                var        regirContext  = isContext.GetReGIRContext();
                uint       groupsX       = (regirContext.GetReGIRLightSlotCount() + reGirTileSize - 1u) / reGirTileSize;
                _presampleReGirNativePass.Setup(nativeCtx, groupsX);
                renderer.EnqueuePass(_presampleReGirNativePass);
            }

            // ---- DI core (NATIVE) ----
            if (enableDirectReStirPass)
            {
                _diGenerateInitialSamplesPass.Setup(nativeCtx);
                renderer.EnqueuePass(_diGenerateInitialSamplesPass);

                if (_diTemporalResamplingPass != null &&
                    setting.diResamplingMode is ReSTIRDI_ResamplingMode.Temporal or ReSTIRDI_ResamplingMode.TemporalAndSpatial)
                {
                    _diTemporalResamplingPass.Setup(nativeCtx);
                    renderer.EnqueuePass(_diTemporalResamplingPass);
                }

                if (_diSpatialResamplingPass != null &&
                    setting.diResamplingMode is ReSTIRDI_ResamplingMode.Spatial or ReSTIRDI_ResamplingMode.TemporalAndSpatial)
                {
                    _diSpatialResamplingPass.Setup(nativeCtx);
                    renderer.EnqueuePass(_diSpatialResamplingPass);
                }

                _diShadeSamplesPass.Setup(nativeCtx);
                renderer.EnqueuePass(_diShadeSamplesPass);
            }

            // ---- BRDF + GI (NATIVE) ----
            if (enableBrdfAndIndirectPass && _brdfRayTracingPass != null)
            {
                _brdfRayTracingPass.Setup(nativeCtx);
                renderer.EnqueuePass(_brdfRayTracingPass);

                if (needSecondaryGBuffer && _shadeSecondarySurfacesPass != null)
                {
                    _shadeSecondarySurfacesPass.Setup(nativeCtx);
                    renderer.EnqueuePass(_shadeSecondarySurfacesPass);

                    if (enableReSTIRGI)
                    {
                        if (_giTemporalResamplingPass != null &&
                            setting.giResamplingMode is ReSTIRGI_ResamplingMode.Temporal or ReSTIRGI_ResamplingMode.TemporalAndSpatial)
                        {
                            _giTemporalResamplingPass.Setup(nativeCtx);
                            renderer.EnqueuePass(_giTemporalResamplingPass);
                        }

                        if (_giSpatialResamplingPass != null &&
                            setting.giResamplingMode is ReSTIRGI_ResamplingMode.Spatial or ReSTIRGI_ResamplingMode.TemporalAndSpatial)
                        {
                            _giSpatialResamplingPass.Setup(nativeCtx);
                            renderer.EnqueuePass(_giSpatialResamplingPass);
                        }

                        if (_giFinalShadingPass != null)
                        {
                            _giFinalShadingPass.Setup(nativeCtx);
                            renderer.EnqueuePass(_giFinalShadingPass);
                        }
                    }
                }
            }

            // ---- ReSTIR PT (NATIVE) ----
            if (setting.indirectLightingMode == IndirectLightingMode.ReStirPT && _ptGenerateInitialSamplesPass != null)
            {
                _ptGenerateInitialSamplesPass.Setup(nativeCtx);
                renderer.EnqueuePass(_ptGenerateInitialSamplesPass);

                if (_ptTemporalResamplingPass != null &&
                    setting.ptResamplingMode is ReSTIRPT_ResamplingMode.Temporal or ReSTIRPT_ResamplingMode.TemporalAndSpatial)
                {
                    _ptTemporalResamplingPass.Setup(nativeCtx);
                    renderer.EnqueuePass(_ptTemporalResamplingPass);
                }

                if (_ptSpatialResamplingPass != null &&
                    setting.ptResamplingMode is ReSTIRPT_ResamplingMode.Spatial or ReSTIRPT_ResamplingMode.TemporalAndSpatial)
                {
                    _ptSpatialResamplingPass.Setup(nativeCtx);
                    renderer.EnqueuePass(_ptSpatialResamplingPass);
                }

                // Optional duplication-based history reduction (FillSampleID → ComputeDuplicationMap)
                var ptTemporalParams = resamplingConstants.restirPT.temporalResampling;
                if (ptTemporalParams.duplicationBasedHistoryReduction != 0)
                {
                    if (_ptFillSampleIDPass != null)
                    {
                        _ptFillSampleIDPass.Setup(nativeCtx);
                        renderer.EnqueuePass(_ptFillSampleIDPass);
                    }

                    if (_ptComputeDuplicationMapPass != null)
                    {
                        _ptComputeDuplicationMapPass.Setup(nativeCtx);
                        renderer.EnqueuePass(_ptComputeDuplicationMapPass);
                    }
                }

                if (_ptFinalShadingPass != null)
                {
                    _ptFinalShadingPass.Setup(nativeCtx);
                    renderer.EnqueuePass(_ptFinalShadingPass);
                }
            }

            // ---- NRD denoising (REBLUR_DIFFUSE_SPECULAR) ----
            if (setting.denoiserMode != RtxDiDenoiserType.DENOISER_MODE_OFF)
            {
                // ---- Filter gradients + compute confidence (mirrors FullSample stages 2-4) ----
                if (enableGradients && _filterGradientsPass != null && _confidencePass != null)
                {
                    _filterGradientsPass.Setup(pool.GradientArrayPtr, gradDims);
                    renderer.EnqueuePass(_filterGradientsPass);

                    _confidencePass.Setup(nativeCtx, pool.GradientArrayPtr, gradDims);
                    renderer.EnqueuePass(_confidencePass);
                }

                // ---- NRD resource bindings (update every frame for ping-pong ViewDepth) ----
                var currentViewDepthNri = isOddFrame ? pool.Depth : pool.PrevDepth;

                nrdReblur.UpdateResources(
                    (ResourceType.OUT_VALIDATION, pool.NrdValidation),
                    (ResourceType.IN_MV, pool.MotionVectors),
                    (ResourceType.IN_VIEWZ, currentViewDepthNri),
                    (ResourceType.IN_NORMAL_ROUGHNESS, pool.NormalRoughness),
                    (ResourceType.IN_DIFF_RADIANCE_HITDIST, pool.DiffuseLighting),
                    (ResourceType.IN_SPEC_RADIANCE_HITDIST, pool.SpecularLighting),
                    (ResourceType.IN_DIFF_CONFIDENCE, diffConfCurrent),
                    (ResourceType.IN_SPEC_CONFIDENCE, specConfCurrent),
                    (ResourceType.OUT_DIFF_RADIANCE_HITDIST, pool.DenoisedDiffuseLighting),
                    (ResourceType.OUT_SPEC_RADIANCE_HITDIST, pool.DenoisedSpecularLighting)
                );
                var lightData = renderingData.lightData;
                var mainLight = lightData.mainLightIndex >= 0 ? lightData.visibleLights[lightData.mainLightIndex] : default;
                var lightDir  = new float3(-(Vector3)mainLight.localToWorldMatrix.GetColumn(2));

                var nrdInput = new NrdDenoiser.NrdFrameInput
                {
                    worldToView                  = frameState.worldToView,
                    prevWorldToView              = frameState.prevWorldToView,
                    viewToClip                   = frameState.viewToClip,
                    prevViewToClip               = frameState.prevViewToClip,
                    viewportJitter               = frameState.viewportJitter,
                    prevViewportJitter           = frameState.prevViewportJitter,
                    resolutionScale              = frameState.resolutionScale,
                    prevResolutionScale          = frameState.prevResolutionScale,
                    renderResolution             = frameState.renderResolution,
                    frameIndex                   = curFrame,
                    lightDirection               = lightDir,
                    denoisingRange               = 1000f,
                    isHistoryConfidenceAvailable = enableGradients,
                    enableValidation             = setting.showValidation,
                    checkerboardMode             = CheckerboardMode.OFF,
                    flipMovionVectors            = true
                };

                _nrdDenoisePass.Setup(nrdReblur.GetInteropDataPtr(nrdInput), RenderPassMarkers.NrdDenoiseRtxdi);
                renderer.EnqueuePass(_nrdDenoisePass);
            }

            // ---- Compositing (denoised diff + spec → DirectLighting) ----
            {
                compositingConstants = NativeCompositingConstantsBuilder.Build(
                    frameState, renderResolution, 1f, localSettings.denoiserMode);

                compositingConstants.enableEnvironmentMap       = setting.environmentMap != null ? 1u : 0u;
                compositingConstants.environmentMapTextureIndex = (uint)_rtxdiGpuScene.EnvironmentMapTextureIndex;
                compositingConstants.environmentRotation        = setting.environmentRotation / 360f; // HLSL subtracts this from uv.x (0..1 range)
                compositingConstants.environmentScale           = setting.environmentScale;

                var compositingConstsArray = new[] { compositingConstants };
                _compositingConstantBuffer.SetData(compositingConstsArray);

                var compositingRes = new NativeRtxdiCompositingPass.Resource
                {
                    ConstantBuffer = _compositingConstantBuffer.GetNativeBufferPtr(),
                    CurrentDepth   = ToPt(depth),
                    Pool           = pool,
                    GpuScene       = _rtxdiGpuScene,
                };
                var compositingSettings = new NativeRtxdiCompositingPass.Settings
                {
                    renderW = renderResolution.x,
                    renderH = renderResolution.y,
                };
                _compositingPass.Setup(compositingRes, compositingSettings);
                renderer.EnqueuePass(_compositingPass);
            }

            // ---- DLSS SR (upscale DirectLighting → DlssOutput) ----
            if (setting.SR && _dlssrPass != null)
            {
                if (!_dlsrUpscalers.TryGetValue(uniqueKey, out var dlsr))
                {
                    var camName = isVR ? $"{cam.name}_Eye{eyeIndex}" : cam.name;
                    dlsr = new DlsrUpscaler(camName + "_Rtxdi");
                    _dlsrUpscalers.Add(uniqueKey, dlsr);
                }

                var dlsrRes = new DlsrUpscaler.DlsrResources
                {
                    input    = pool.HdrColor,
                    output   = pool.ResolvedColor,
                    mv       = pool.MotionVectors,
                    depth    = pool.DeviceDepth,
                    exposure = default,
                    reactive = default,
                };

                var dlsrInput = new DlsrUpscaler.DlsrFrameInput
                {
                    viewportJitter   = frameState.viewportJitter,
                    renderResolution = frameState.renderResolution,
                    frameIndex       = curFrame,
                    outputWidth      = (ushort)outputResolution.x,
                    outputHeight     = (ushort)outputResolution.y,
                };

                var dlsrSettings = new DlsrUpscaler.DlsrSettings
                {
                    upscalerMode = setting.upscalerMode,
                    preset       = 0,
                    resetHistory = resourcesChanged,
                };

                var dlsrDataPtr = dlsr.GetInteropDataPtr(dlsrInput, dlsrRes, frameState.resolutionScale, dlsrSettings);
                _dlssrPass.Setup(dlsrDataPtr);
                renderer.EnqueuePass(_dlssrPass);
            }

            // ---- Tone Mapping (HDR → LDR, adaptive exposure) ----
            if (_toneMappingPass != null && setting.enableToneMapping)
            {
                var tmSource = setting.SR ? pool.ResolvedColor.NativePtr : pool.HdrColor.NativePtr;
                var tmRes    = setting.SR ? outputResolution : renderResolution;
                var tmResource = new NativeToneMappingPass.Resource
                {
                    SourceTexture = tmSource,
                    OutputTexture = pool.LdrColor.NativePtr,
                    ColorLUT      = tmSource,
                    ColorLUTSize  = 0f,
                };
                var tmSettings = new NativeToneMappingPass.Settings
                {
                    RenderResolution  = tmRes,
                    FrameTime         = Time.deltaTime,
                    ToneMappingParams = setting.toneMappingParams,
                };
                _toneMappingPass.Setup(tmResource, tmSettings);
                renderer.EnqueuePass(_toneMappingPass);
            }

            // ---- OutputBlit (shows DirectLighting = composited NRD result) ----
            if (_outputBlitPass != null)
            {
                var outputBlitResource = new NativeRtxdiOutputBlitPass.Resource
                {
                    HdrColor   = pool.HdrColor.Handle,
                    LdrColor   = pool.LdrColor.Handle,
                    DlssOutput = setting.SR ? pool.ResolvedColor.Handle : null,

                    DiffuseLighting     = pool.DiffuseLighting.Handle,
                    SpecularLighting    = pool.SpecularLighting.Handle,
                    DenoisedDiffuse     = pool.DenoisedDiffuseLighting.Handle,
                    DenoisedSpecular    = pool.DenoisedSpecularLighting.Handle,
                    DirectLightingRaw   = pool.DirectLightingRaw.Handle,
                    IndirectLightingRaw = pool.IndirectLightingRaw.Handle,

                    NrdValidation = pool.NrdValidation.Handle,
                    MotionVectors = pool.MotionVectors.Handle,

                    // GBuffer debug
                    ViewDepth     = depth,
                    DiffuseAlbedo = diffuseAlbedo,
                    SpecularRough = specularRough,
                    Normals       = normals,
                    GeoNormals    = geoNormals,

                    // PDF debug
                    LocalLightPdfTexture  = rtxdiResources.LocalLightPdfTexture,
                    EnvironmentPdfTexture = rtxdiResources.EnvironmentPdfTexture,
                };
                var outputBlitSettings = new NativeRtxdiOutputBlitPass.Settings
                {
                    showMode         = setting.showMode,
                    resolutionScale  = frameState.resolutionScale,
                    showMv           = setting.showMv,
                    showValidation   = setting.showValidation,
                    pdfMipLevel      = setting.pdfMipLevel,
                    pdfExposureStops = setting.pdfExposureStops,
                };
                _outputBlitPass.Setup(outputBlitResource, outputBlitSettings);
                renderer.EnqueuePass(_outputBlitPass);
            }


            if (renderingData.cameraData.xr.enabled)
            {
                if (setting.skipRightEyeInVR || eyeIndex == 1)
                    renderer.EnqueuePass(_nativeFrameTickPass);
            }
            else
            {
                renderer.EnqueuePass(_nativeFrameTickPass);
            }
        }

        private static RTXDI_PTParameters BuildRestirPTParams(ImportanceSamplingContext isContext)
        {
            var ctx = isContext.GetReSTIRPTContext();
            return new RTXDI_PTParameters
            {
                reservoirBuffer    = ctx.GetReservoirBufferParameters(),
                bufferIndices      = ctx.GetBufferIndices(),
                initialSampling    = ctx.GetInitialSamplingParameters(),
                reconnection       = ctx.GetReconnectionParameters(),
                temporalResampling = ctx.GetTemporalResamplingParameters(),
                hybridShift        = ctx.GetHybridShiftParameters(),
                boilingFilter      = ctx.GetBoilingFilterParameters(),
                spatialResampling  = ctx.GetSpatialResamplingParameters(),
            };
        }

        private static NativePTParameters BuildPTParams()
        {
            return new NativePTParameters
            {
                enableRussianRoulette              = 1u,
                russianRouletteContinueChance      = 0.8f,
                enableSecondaryDISpatialResampling = 0u,
                copyReSTIRDISimilarityThresholds   = 1u,
                nee = new NativePTNeeParameters
                {
                    initialSamplingParams = new RTXDI_DIInitialSamplingParameters
                    {
                        numLocalLightSamples             = 2,
                        numInfiniteLightSamples          = 1,
                        numEnvironmentSamples            = 1,
                        numBrdfSamples                   = 0,
                        brdfCutoff                       = 0f,
                        brdfRayMinT                      = 0.001f,
                        localLightSamplingMode           = (uint)ReSTIRDI_LocalLightSamplingMode.Uniform,
                        enableInitialVisibility          = 1u,
                        environmentMapImportanceSampling = 1u,
                    },
                    spatialResamplingParams = new RTXDI_DISpatialResamplingParameters
                    {
                        numSamples                   = 1,
                        numDisocclusionBoostSamples  = 0,
                        samplingRadius               = 1f,
                        biasCorrectionMode           = ReSTIRDI_SpatialBiasCorrectionMode.Basic,
                        depthThreshold               = 0.1f,
                        normalThreshold              = 0.5f,
                        targetHistoryLength          = 0,
                        enableMaterialSimilarityTest = 1u,
                        discountNaiveSamples         = 0u,
                    },
                },
                sampleEnvMapOnSecondaryMiss   = 0u,
                sampleEmissivesOnSecondaryHit = 0u,
                lightSamplingMode             = (uint)ReSTIRDI_LocalLightSamplingMode.ReGIR_RIS,
                extraMirrorBounceBudget       = 4u,
                minimumPathThroughput         = 0.05f,
            };
        }

        private static int2 ComputeOutputResolution(CameraData cameraData)
        {
            var xrPass = cameraData.xr;
            if (xrPass.enabled)
                return new int2(xrPass.renderTargetDesc.width, xrPass.renderTargetDesc.height);
            return new int2(
                (int)(cameraData.camera.pixelWidth * cameraData.renderScale),
                (int)(cameraData.camera.pixelHeight * cameraData.renderScale));
        }

        protected override void Dispose(bool disposing)
        {
            base.Dispose(disposing);

            _resamplingConstantBuffer?.Release();
            _resamplingConstantBuffer = null;
            _perPassConstantBuffer?.Release();
            _perPassConstantBuffer = null;
            _gbufferConstantBuffer?.Release();
            _gbufferConstantBuffer = null;
            _compositingConstantBuffer?.Release();
            _compositingConstantBuffer = null;

            foreach (var d in _nrdDenoisers.Values) d.Dispose();
            _nrdDenoisers.Clear();

            foreach (var p in _resourcePools.Values) p.Dispose();
            _resourcePools.Clear();

            foreach (var r in _rtxdiResources.Values) r.Dispose();
            _rtxdiResources.Clear();

            _cameraFrameStates.Clear();

            _rtxdiGpuScene?.Dispose();
            _rtxdiGpuScene = null;

            _diGenerateInitialSamplesPass?.Dispose();
            _diGenerateInitialSamplesPass = null;
            _raytracedGBufferPass?.Dispose();
            _raytracedGBufferPass = null;
            _postprocessGBufferPass?.Dispose();
            _postprocessGBufferPass = null;
            _diTemporalResamplingPass?.Dispose();
            _diTemporalResamplingPass = null;
            _diSpatialResamplingPass?.Dispose();
            _diSpatialResamplingPass = null;
            _diShadeSamplesPass?.Dispose();
            _diShadeSamplesPass = null;

            _brdfRayTracingPass?.Dispose();
            _brdfRayTracingPass = null;
            _shadeSecondarySurfacesPass?.Dispose();
            _shadeSecondarySurfacesPass = null;
            _giTemporalResamplingPass?.Dispose();
            _giTemporalResamplingPass = null;
            _giSpatialResamplingPass?.Dispose();
            _giSpatialResamplingPass = null;
            _giFinalShadingPass?.Dispose();
            _giFinalShadingPass = null;
            _ptGenerateInitialSamplesPass?.Dispose();
            _ptGenerateInitialSamplesPass = null;
            _ptTemporalResamplingPass?.Dispose();
            _ptTemporalResamplingPass = null;
            _ptSpatialResamplingPass?.Dispose();
            _ptSpatialResamplingPass = null;
            _ptFillSampleIDPass?.Dispose();
            _ptFillSampleIDPass = null;
            _ptComputeDuplicationMapPass?.Dispose();
            _ptComputeDuplicationMapPass = null;
            _ptFinalShadingPass?.Dispose();
            _ptFinalShadingPass = null;
            _presampleReGirNativePass?.Dispose();
            _presampleReGirNativePass = null;
            _filterGradientsPass?.Dispose();
            _filterGradientsPass = null;
            _confidencePass?.Dispose();
            _confidencePass = null;
            _toneMappingPass?.Dispose();
            _toneMappingPass = null;
            _compositingPass?.Dispose();
            _compositingPass = null;
            _nrdDenoisePass  = null;

            foreach (var upscaler in _dlsrUpscalers.Values) upscaler.Dispose();
            _dlsrUpscalers.Clear();
            _dlssrPass = null;
            _pdfMipsPass?.Dispose();
            _pdfMipsPass         = null;
            _nativeFrameTickPass = null;
        }

        // -------------------------------------------------------------------
        // Debug / test helpers
        // -------------------------------------------------------------------

        /// <summary>
        /// Reads back <c>LightDataBuffer</c> from the first active camera's
        /// <see cref="NativeRtxdiResources"/> and logs each non-black light entry,
        /// mirroring <c>LightScene.DebugReadback()</c> in <c>UnityRtxdiFeature</c>.
        /// Call from the Inspector button (Editor-only) after a few rendered frames.
        /// </summary>
        public void TestPrepareLight()
        {
            if (_rtxdiResources.Count == 0)
            {
                Debug.LogWarning("[NativeRtxdiFeature] No NativeRtxdiResources allocated yet — run the scene first.");
                return;
            }

            // Pick the first entry (main camera or first active eye).
            NativeRtxdiResources res = null;
            foreach (var kv in _rtxdiResources)
            {
                res = kv.Value;
                break;
            }

            if (res == null || res.LightDataBuffer == null)
            {
                Debug.LogWarning("[NativeRtxdiFeature] LightDataBuffer is null.");
                return;
            }

            // Determine the exact current-frame range from the PrepareLights pass if available,
            // otherwise fall back to scanning the whole buffer.
            int frameOffset = 0;
            int frameCount  = res.LightDataBuffer.count;
            if (_prepareLightsPass != null && _prepareLightsPass.TotalLightCount > 0)
            {
                frameOffset = (int)_prepareLightsPass.CurrentFrameOffset;
                frameCount  = _prepareLightsPass.TotalLightCount;
            }

            var debugData = new PolymorphicLightInfo[frameCount];
            res.LightDataBuffer.GetData(debugData, 0, frameOffset, frameCount);

            int validCount = 0;
            for (int i = 0; i < debugData.Length; i++)
            {
                var info = debugData[i];

                // Decode intensity from logRadiance low-16 bits.
                uint  logRad    = info.logRadiance & 0xFFFFu;
                float intensity = (logRad == 0) ? 0f : Unity.Mathematics.math.exp2(((logRad - 1) / 65534f) * 48f - 8f);

                if (intensity < 0.01f)
                    continue;

                validCount++;

                float normR = (info.colorTypeAndFlags & 0xFFu) / 255f;
                float normG = ((info.colorTypeAndFlags >> 8) & 0xFFu) / 255f;
                float normB = ((info.colorTypeAndFlags >> 16) & 0xFFu) / 255f;
                var   c     = new Color(normR * intensity, normG * intensity, normB * intensity, 1f);

                var center   = new Vector3(info.center.x, info.center.y, info.center.z);
                var edge1Dir = OctUnorm32ToDir(info.direction1);
                var edge2Dir = OctUnorm32ToDir(info.direction2);
                var normal   = Vector3.Cross(edge1Dir, edge2Dir).normalized;

                Debug.Log($"[NativePrepareLights slot {frameOffset + i}] center={center}, color={c}, normal={normal}");
                Debug.DrawLine(center, center + normal * (intensity / 10f), c, 10f);
            }

            Debug.Log($"[NativeRtxdiFeature] LightDataBuffer readback: {validCount} active lights / {frameCount} current-frame slots (offset={frameOffset}, total buffer={res.LightDataBuffer.count})");
        }

        private static Vector3 OctUnorm32ToDir(uint packed)
        {
            float px = (packed & 0xFFFFu) / (float)0xFFFEu;
            float py = (packed >> 16) / (float)0xFFFEu;
            var   p  = new Unity.Mathematics.float2(px * 2f - 1f, py * 2f - 1f);
            var   n  = new Unity.Mathematics.float3(p.x, p.y, 1f - Unity.Mathematics.math.abs(p.x) - Unity.Mathematics.math.abs(p.y));
            if (n.z < 0f)
            {
                var wrap = (1f - Unity.Mathematics.math.abs(p.yx)) * Unity.Mathematics.math.select(-1f, 1f, p.xy >= 0f);
                n.x = wrap.x;
                n.y = wrap.y;
            }

            return ((Vector3)(Unity.Mathematics.math.normalize(n)));
        }

#if UNITY_EDITOR
        private void Reset()
        {
            setting = new NativeRtxdiSetting();
            AutoFillShaders();
        }

        public void AutoFillShaders()
        {
            const string shaderRoot = "Assets/RTXDI/Shaders";

            finalMaterial = UnityEditor.AssetDatabase.LoadAssetAtPath<Material>("Assets/Shaders/Mat/KM_Final.mat");

            // Prepare / Environment
            prepareLightsCs            = LoadCS($"{shaderRoot}/PrepareLights.computeshader");
            preprocessEnvironmentMapCs = LoadCS($"{shaderRoot}/PreprocessEnvironmentMap.computeshader");
            preprocessLocalLightCs     = LoadCS($"{shaderRoot}/PreprocessLocalLight.computeshader");
            preprocessLocalLightCs     = LoadCS($"{shaderRoot}/PreprocessLocalLight.computeshader");

            // GBuffer
            raytracedGBufferCs   = LoadCS($"{shaderRoot}/RaytracedGBuffer.computeshader");
            postprocessGBufferCs = LoadCS($"{shaderRoot}/PostprocessGBuffer.computeshader");

            // Presampling
            presampleLightsCs         = LoadCS($"{shaderRoot}/LightingPasses/Presampling/PresampleLights.computeshader");
            presampleEnvironmentMapCs = LoadCS($"{shaderRoot}/LightingPasses/Presampling/PresampleEnvironmentMap.computeshader");
            presampleReGirCs          = LoadCS($"{shaderRoot}/LightingPasses/Presampling/PresampleReGIR.computeshader");

            // ReSTIR DI
            diGenerateInitialSamplesCs = LoadCS($"{shaderRoot}/LightingPasses/DI/GenerateInitialSamples.computeshader");
            diTemporalResamplingCs     = LoadCS($"{shaderRoot}/LightingPasses/DI/TemporalResampling.computeshader");
            diSpatialResamplingCs      = LoadCS($"{shaderRoot}/LightingPasses/DI/SpatialResampling.computeshader");
            diShadeSamplesCs           = LoadCS($"{shaderRoot}/LightingPasses/DI/ShadeSamples.computeshader");

            // Indirect / ReSTIR GI
            brdfRayTracingCs         = LoadCS($"{shaderRoot}/LightingPasses/BrdfRayTracing.computeshader");
            shadeSecondarySurfacesCs = LoadCS($"{shaderRoot}/LightingPasses/ShadeSecondarySurfaces.computeshader");
            giTemporalResamplingCs   = LoadCS($"{shaderRoot}/LightingPasses/GI/TemporalResampling.computeshader");
            giSpatialResamplingCs    = LoadCS($"{shaderRoot}/LightingPasses/GI/SpatialResampling.computeshader");
            giFinalShadingCs         = LoadCS($"{shaderRoot}/LightingPasses/GI/FinalShading.computeshader");

            // PT
            ptGenerateInitialSamplesCs = LoadCS($"{shaderRoot}/LightingPasses/PT/GenerateInitialSamples.computeshader");
            ptTemporalResamplingCs     = LoadCS($"{shaderRoot}/LightingPasses/PT/TemporalResampling.computeshader");
            ptSpatialResamplingCs      = LoadCS($"{shaderRoot}/LightingPasses/PT/SpatialResampling.computeshader");
            ptFillSampleIDCs           = LoadCS($"{shaderRoot}/LightingPasses/PT/FillSampleID.computeshader");
            ptComputeDuplicationMapCs  = LoadCS($"{shaderRoot}/LightingPasses/PT/ComputeDuplicationMap.computeshader");
            ptFinalShadingCs           = LoadCS($"{shaderRoot}/LightingPasses/PT/FinalShading.computeshader");

            // Managed compute helpers (kept for v1 — not yet ported to NativeComputeShader)
            compositingPassCs = LoadCS($"{shaderRoot}/CompositingPass.computeshader");

            filterGradientsPassCs = LoadCS($"{shaderRoot}/DenoisingPasses/FilterGradientsPass.computeshader");
            confidencePassCs      = LoadCS($"{shaderRoot}/DenoisingPasses/ConfidencePass.computeshader");

            toneMappingHistogramCs = LoadCS($"Assets/Shaders/donut/histogram.computeshader");
            toneMappingExposureCs  = LoadCS($"Assets/Shaders/donut/exposure.computeshader");
            toneMappingCs          = LoadCS($"Assets/Shaders/donut/tonemapping.computeshader");

            UnityEditor.EditorUtility.SetDirty(this);

            static NativeComputeShader LoadCS(string path)
            {
                var s = UnityEditor.AssetDatabase.LoadAssetAtPath<NativeComputeShader>(path);
                if (s == null)
                    Debug.LogWarning($"[NativeRtxdiFeature] Missing NativeComputeShader at: {path}");
                return s;
            }
        }
#endif
    }
}