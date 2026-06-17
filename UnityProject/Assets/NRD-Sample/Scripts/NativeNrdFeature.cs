using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using DLRR;
using SLDLRR;
using DLSR;
using NativeRender;
using NIS;
using Nrd;
using Nri;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    public class NativeNrdFeature : ScriptableRendererFeature
    {
        private const uint sharcDownscale = 5;

        public NativeNrdSampleSetting setting;
        public CommonSettings         commonSettings    = CommonSettings._default;
        public SigmaSettings          sigmaSettings     = SigmaSettings._default;
        public ReblurSettings         reblurSettings    = ReblurSettings._default;
        public RelaxSettings          relaxSettings     = RelaxSettings._default;
        public ReferenceSettings      referenceSettings = ReferenceSettings._default;

        public NRDGlobalConstants globalConstants;

        public RenderPassEvent renderPassEvent = RenderPassEvent.BeforeRenderingPostProcessing;

        public Material            finalMaterial;
        public NativeComputeShader nrdOpaqueTracingShader;
        public NativeComputeShader nrdSharcResolve;
        public NativeComputeShader nrdSharcUpdate;
        public NativeComputeShader nrdTransparentShader;
        public NativeComputeShader nrdCompositionShader;
        public NativeComputeShader nrdTaaShader;
        public NativeComputeShader nrdConfidenceBlurShader;
        public NativeComputeShader nrdFinalShader;
        public NativeComputeShader nrdDlssBeforeShader;
        public NativeComputeShader nrdDlssAfterShader;
        public ComputeShader       updateSkinnedPrimitivesCS;
        public ComputeShader       fillUintTextureCS;

        // Tone mapping
        public NativeComputeShader toneMappingHistogramCs; // Shaders/donut/histogram.computeshader
        public NativeComputeShader toneMappingExposureCs; // Shaders/donut/exposure.computeshader
        public NativeRasterShader  toneMappingCs; // Shaders/donut/tonemapping.rastershader


        public Texture2D scramblingRankingTex;
        public Texture2D sobolTex;

        private RenderTexture scramblingRankingUintTex;
        private RenderTexture sobolUintTex;

        private IntPtr scramblingRankingTexPtr;
        private IntPtr sobolTexPtr;

        private DepthBarrierFixPass     _depthBarrierFixPass;
        private NRDTlasUpdatePass       _nrdTlasUpdatePass;
        private NRDSharcPass            _nrdSharcPass;
        private NRDOpaquePass           _nrdOpaquePass;
        private NRDTransparentPass      _nrdTransparentPass;
        private NRDCompositionPass      _nrdCompositionPass;
        private NRDTaaPass              _nrdTaaPass;
        private NRDConfidenceBlurPass   _nrdConfidenceBlurPass;
        private NRDFinalPass            _nrdFinalPass;
        private NrdPass                 _nrdShadowDenoisePass;
        private NrdPass                 _nrdOpaqueDenoisePass;
        private NrdPass                 _nrdReferenceDenoisePass;
        private NRDDlssBeforePass       _nrdDlssBeforePass;
        private NRDDlssAfterPass        _nrdDlssAfterPass;
        private DlssRRPass              _dlssrrPass;
        private SLDlssrrPass            _slDlssrrPass;
        private SLDlssgInputsPass       _slDlssgInputsPass;
        private bool                    _slFgLastEnabled;
        private DlssSRPass              _dlsssrPass;
        private NisPass                 _nisPass;
        private NativeNrdOutputBlitPass _outputBlitPass;
        private NativeToneMappingPass   _toneMappingPass;
        private NrdDlssgInputsPass      _dlssgInputsPass;

        // Persistent ring backing the DLSS-G input pointers handed to IssuePluginEventAndData
        // (must outlive the render thread's consumption of the event — see DlsrUpscaler).
        // private const int DlssgInputsRingCount = 3;
        // private NativeArray<StreamlineFrameDriver.DlssgInputs> _dlssgInputsRing;

        private NRDSampleResource _nrdSampleResource;
        public NRDSampleResource NrdSampleResource => _nrdSampleResource;

        private readonly Dictionary<long, VolatileConstantBuffer> _nrdConstantBuffers = new();

        private readonly Dictionary<long, SigmaDenoiser>     _sigmaDenoisers     = new();
        private readonly Dictionary<long, ReblurDenoiser>    _reblurDenoisers    = new();
        private readonly Dictionary<long, RelaxDenoiser>     _relaxDenoisers     = new();
        private readonly Dictionary<long, ReferenceDenoiser> _referenceDenoisers = new();

        private readonly Dictionary<long, DlrrDenoiser>              _dlrrDenoisers     = new();
        private readonly Dictionary<long, SLDlssrr>                 _slDlrrDenoisers   = new();
        private readonly Dictionary<long, DlsrUpscaler>              _dlsrUpscalers     = new();
        private readonly Dictionary<long, NisUpscaler>               _nisUpscalers      = new();
        private readonly Dictionary<long, NativeNrdTextureResources> _resourcePools     = new();
        private readonly Dictionary<long, CameraFrameState>          _cameraFrameStates = new();


        public RenderTexture GetUintTexture(Texture2D sourceTex)
        {
            RenderTextureDescriptor desc    = new RenderTextureDescriptor(sourceTex.width, sourceTex.height, GraphicsFormat.R8G8B8A8_UInt, 0);
            var                     UintTex = new RenderTexture(desc) { enableRandomWrite = true };
            UintTex.hideFlags = HideFlags.DontSave;

            var scramblingData = sourceTex.GetRawTextureData<byte>();

            var buffer = new ComputeBuffer(scramblingData.Length / 4, sizeof(uint));
            buffer.SetData(scramblingData);

            var kernel = fillUintTextureCS.FindKernel("Fill");

            fillUintTextureCS.SetBuffer(kernel, "_InputBuffer", buffer);
            fillUintTextureCS.SetTexture(kernel, "_TargetTex", UintTex);
            fillUintTextureCS.SetInt("_Width", sourceTex.width);

            int threadGroupsX = Mathf.CeilToInt(sourceTex.width / 8.0f);
            int threadGroupsY = Mathf.CeilToInt(sourceTex.height / 8.0f);

            fillUintTextureCS.Dispatch(kernel, threadGroupsX, threadGroupsY, 1);
            buffer.Dispose();

            return UintTex;
        }


        public override void Create()
        {
            if (scramblingRankingTexPtr == IntPtr.Zero)
            {
                scramblingRankingUintTex = GetUintTexture(scramblingRankingTex);
                scramblingRankingTexPtr  = scramblingRankingUintTex.GetNativeTexturePtr();
            }

            if (sobolTexPtr == IntPtr.Zero)
            {
                sobolUintTex = GetUintTexture(sobolTex);
                sobolTexPtr  = sobolUintTex.GetNativeTexturePtr();
            }
        }

        private void CreatePass()
        {
            _nrdTlasUpdatePass       ??= new NRDTlasUpdatePass { updateSkinnedPrimitivesCS                                                         = updateSkinnedPrimitivesCS, renderPassEvent = renderPassEvent };
            _nrdSharcPass            ??= new NRDSharcPass(nrdSharcResolve, nrdSharcUpdate) { renderPassEvent                                       = renderPassEvent };
            _nrdOpaquePass           ??= new NRDOpaquePass(nrdOpaqueTracingShader) { renderPassEvent                                               = renderPassEvent };
            _nrdTransparentPass      ??= new NRDTransparentPass(nrdTransparentShader) { renderPassEvent                                            = renderPassEvent };
            _nrdCompositionPass      ??= new NRDCompositionPass(nrdCompositionShader) { renderPassEvent                                            = renderPassEvent };
            _nrdTaaPass              ??= new NRDTaaPass(nrdTaaShader) { renderPassEvent                                                            = renderPassEvent };
            _nrdConfidenceBlurPass   ??= new NRDConfidenceBlurPass(nrdConfidenceBlurShader) { renderPassEvent                                      = renderPassEvent };
            _nrdFinalPass            ??= new NRDFinalPass(nrdFinalShader) { renderPassEvent                                                        = renderPassEvent };
            _nrdShadowDenoisePass    ??= new NrdPass { renderPassEvent                                                                             = renderPassEvent };
            _nrdOpaqueDenoisePass    ??= new NrdPass { renderPassEvent                                                                             = renderPassEvent };
            _nrdReferenceDenoisePass ??= new NrdPass { renderPassEvent                                                                             = renderPassEvent };
            _nrdDlssBeforePass       ??= new NRDDlssBeforePass(nrdDlssBeforeShader) { renderPassEvent                                              = renderPassEvent };
            _nrdDlssAfterPass        ??= new NRDDlssAfterPass(nrdDlssAfterShader) { renderPassEvent                                                = renderPassEvent };
            _dlssrrPass              ??= new DlssRRPass { renderPassEvent                                                                          = renderPassEvent };
            _slDlssrrPass            ??= new SLDlssrrPass { renderPassEvent                                                                        = renderPassEvent };
            _slDlssgInputsPass       ??= new SLDlssgInputsPass { renderPassEvent                                                                  = renderPassEvent };
            _dlsssrPass              ??= new DlssSRPass { renderPassEvent                                                                          = renderPassEvent };
            _nisPass                 ??= new NisPass { renderPassEvent                                                                             = renderPassEvent };
            _outputBlitPass          ??= new NativeNrdOutputBlitPass(finalMaterial) { renderPassEvent                                              = renderPassEvent };
            _depthBarrierFixPass     ??= new DepthBarrierFixPass { renderPassEvent                                                                 = RenderPassEvent.AfterRendering };
            _toneMappingPass         ??= new NativeToneMappingPass(toneMappingHistogramCs, toneMappingExposureCs, toneMappingCs) { renderPassEvent = renderPassEvent };
            _dlssgInputsPass         ??= new NrdDlssgInputsPass { renderPassEvent                                                                  = renderPassEvent };
        }

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            var cam = renderingData.cameraData.camera;
            if (cam.cameraType != CameraType.Game && cam.cameraType != CameraType.SceneView)
                return;

            cam.depthTextureMode = DepthTextureMode.Depth | DepthTextureMode.MotionVectors;

            var eyeIndex = renderingData.cameraData.xr.enabled ? renderingData.cameraData.xr.multipassId : 0;

            if (eyeIndex == 1 && setting.skipRightEyeInVR)
                return;

            if (finalMaterial == null
                || nrdOpaqueTracingShader == null
                || nrdSharcResolve == null
                || nrdSharcUpdate == null
                || nrdConfidenceBlurShader == null
                || nrdFinalShader == null
                || toneMappingCs == null
                || scramblingRankingTex == null
                || sobolTex == null)
            {
#if UNITY_EDITOR
                if (!Application.isPlaying)
                {
                    AutoFillShaders();
                    return;
                }
#endif
                Debug.LogWarning("NRDFeature: Missing required assets, skipping pass.");
                return;
            }

            CreatePass();
            _nrdSampleResource ??= new NRDSampleResource();

            // MergeBlas is no longer a runtime property; merging is determined automatically
            // based on Application.isPlaying and gameObject.isStatic per-object.            if (eyeIndex == 0)
            {
                _nrdSampleResource?.UpdateForFrame();
            }

            var uniqueKey = cam.GetInstanceID() + (eyeIndex * 100000L);
            var isVR      = renderingData.cameraData.xrRendering;

            if (!_resourcePools.TryGetValue(uniqueKey, out var pool))
            {
                pool = new NativeNrdTextureResources();
                _resourcePools.Add(uniqueKey, pool);
            }

            if (!_sigmaDenoisers.TryGetValue(uniqueKey, out var nrdSigma))
            {
                var camName = isVR ? $"{cam.name}_Eye{eyeIndex}" : cam.name;
                nrdSigma = new SigmaDenoiser(camName + "_Shadow", Denoiser.SIGMA_SHADOW_TRANSLUCENCY);
                _sigmaDenoisers.Add(uniqueKey, nrdSigma);
            }

            if (!_reblurDenoisers.TryGetValue(uniqueKey, out var nrdReblur))
            {
                var camName = isVR ? $"{cam.name}_Eye{eyeIndex}" : cam.name;
                nrdReblur = new ReblurDenoiser(camName + "_Opaque", Denoiser.REBLUR_DIFFUSE_SPECULAR);
                _reblurDenoisers.Add(uniqueKey, nrdReblur);
            }

            if (!_relaxDenoisers.TryGetValue(uniqueKey, out var nrdRelax))
            {
                var camName = isVR ? $"{cam.name}_Eye{eyeIndex}" : cam.name;
                nrdRelax = new RelaxDenoiser(camName + "_Opaque", Denoiser.RELAX_DIFFUSE_SPECULAR);
                _relaxDenoisers.Add(uniqueKey, nrdRelax);
            }

            if (!_referenceDenoisers.TryGetValue(uniqueKey, out var nrdReference))
            {
                var camName = isVR ? $"{cam.name}_Eye{eyeIndex}" : cam.name;
                nrdReference = new ReferenceDenoiser(camName + "_Reference", Denoiser.REFERENCE);
                _referenceDenoisers.Add(uniqueKey, nrdReference);
            }

            if (!_dlrrDenoisers.TryGetValue(uniqueKey, out var dlrr))
            {
                var camName = isVR ? $"{cam.name}_Eye{eyeIndex}" : cam.name;
                dlrr = new DlrrDenoiser(camName);
                _dlrrDenoisers.Add(uniqueKey, dlrr);
            }

            if (!_cameraFrameStates.TryGetValue(uniqueKey, out var frameState))
            {
                frameState = new CameraFrameState(setting.resolutionScale);
                _cameraFrameStates.Add(uniqueKey, frameState);
            }

            var outputResolution = ComputeOutputResolution(renderingData.cameraData);
            var resourcesChanged = pool.EnsureResources(outputResolution, setting.upscalerMode);
            if (!pool.ViewZ.IsCreated)
            {
                Debug.LogError("NRDFeature: Resources not ready");
            }

            var renderResolution = pool.renderResolution;

            if (resourcesChanged)
            {
                frameState.renderResolution = pool.renderResolution;
                frameState.frameIndex       = 0;
            }

            var curFrame = frameState.frameIndex;
            frameState.Update(renderingData, false, setting.resolutionScale);

            // Adaptive accumulation — mirrors NRDSample.cpp PrepareFrame adaptiveAccumulation block.
            // C++: ACCUMULATION_TIME = 0.5f s, MAX_HISTORY_FRAME_NUM = min(60, REBLUR_MAX, RELAX_MAX).
            // nrd::GetMaxAccumulatedFrameNum(t, fps) = (uint)(t * fps + 0.5f).
            if (setting.adaptiveAccumulation)
            {
                const float AccumulationTime   = 0.5f; // seconds — matches C++ ACCUMULATION_TIME
                const uint  MaxHistoryFrameNum = 60u; // matches C++ MAX_HISTORY_FRAME_NUM

                float fps      = math.min(1.0f / Time.smoothDeltaTime, 121.0f);
                float accTime  = AccumulationTime * ((setting.boost && setting.SHARC) ? 0.667f : 1.0f);
                uint  maxAccum = (uint)math.max((int)(accTime * fps + 0.5f), 1);

                setting.maxAccumulatedFrameNum     = math.min(maxAccum, MaxHistoryFrameNum);
                setting.maxFastAccumulatedFrameNum = setting.maxAccumulatedFrameNum / 5u;
            }

            globalConstants = frameState.GetNrdConstants(renderingData, setting);

            if (!_nrdConstantBuffers.TryGetValue(uniqueKey, out var nrdConstantBuffer))
            {
                nrdConstantBuffer = new VolatileConstantBuffer(Marshal.SizeOf<NRDGlobalConstants>());
                _nrdConstantBuffers.Add(uniqueKey, nrdConstantBuffer);
            }

            nrdConstantBuffer.Upload(renderer, globalConstants);

            var isEven = (globalConstants.gFrameIndex & 1) == 0;

            // TLAS update
            if (eyeIndex == 0)
            {
                _nrdTlasUpdatePass.SetNRDSampleResource(_nrdSampleResource);
                renderer.EnqueuePass(_nrdTlasUpdatePass);
            }

            // SHARC
            {
                var sharcW = 16 * ((int)(renderResolution.x / sharcDownscale + 15) / 16);
                var sharcH = 16 * ((int)(renderResolution.y / sharcDownscale + 15) / 16);
                pool.EnsureSharcGradientResources(new int2(sharcW, sharcH));

                var nrdSharcResource = new NRDSharcPass.Resource
                {
                    ConstantBuffer = nrdConstantBuffer,
                    Pool           = pool
                };

                var nrdSharcSettings = new NRDSharcPass.Settings
                {
                    RenderResolution = renderResolution,
                    sharcDownscale   = sharcDownscale,
                    isEven           = isEven
                };

                _nrdSharcPass.SetNRDSampleResource(_nrdSampleResource);
                _nrdSharcPass.Setup(nrdSharcResource, nrdSharcSettings);
                renderer.EnqueuePass(_nrdSharcPass);
            }

            if (resourcesChanged)
            {
                // Shadow denoiser (SIGMA) resources
                nrdSigma.UpdateResources(
                    // Common
                    (ResourceType.IN_MV, pool.Mv),
                    (ResourceType.IN_VIEWZ, pool.ViewZ),
                    (ResourceType.IN_NORMAL_ROUGHNESS, pool.NormalRoughness),

                    // SIGMA
                    (ResourceType.IN_PENUMBRA, pool.Unfiltered_Penumbra),
                    (ResourceType.IN_TRANSLUCENCY, pool.Unfiltered_Translucency),
                    (ResourceType.OUT_SHADOW_TRANSLUCENCY, pool.Shadow)
                );

                // Opaque denoiser (REBLUR) resources
                nrdReblur.UpdateResources(
                    // Common
                    (ResourceType.IN_MV, pool.Mv),
                    (ResourceType.IN_VIEWZ, pool.ViewZ),
                    (ResourceType.IN_NORMAL_ROUGHNESS, pool.NormalRoughness),

                    // (Optional) Validation
                    (ResourceType.OUT_VALIDATION, pool.Validation),

                    // Diffuse
                    (ResourceType.IN_DIFF_RADIANCE_HITDIST, pool.Unfiltered_Diff),
                    (ResourceType.OUT_DIFF_RADIANCE_HITDIST, pool.Diff),
                    (ResourceType.IN_DIFF_CONFIDENCE, pool.Gradient_Pong),

                    // Specular
                    (ResourceType.IN_SPEC_RADIANCE_HITDIST, pool.Unfiltered_Spec),
                    (ResourceType.OUT_SPEC_RADIANCE_HITDIST, pool.Spec),
                    (ResourceType.IN_SPEC_CONFIDENCE, pool.Gradient_Pong)
                );

                // Opaque denoiser (RELAX) resources
                nrdRelax.UpdateResources(
                    // Common
                    (ResourceType.IN_MV, pool.Mv),
                    (ResourceType.IN_VIEWZ, pool.ViewZ),
                    (ResourceType.IN_NORMAL_ROUGHNESS, pool.NormalRoughness),

                    // (Optional) Validation
                    (ResourceType.OUT_VALIDATION, pool.Validation),

                    // Diffuse
                    (ResourceType.IN_DIFF_RADIANCE_HITDIST, pool.Unfiltered_Diff),
                    (ResourceType.OUT_DIFF_RADIANCE_HITDIST, pool.Diff),
                    (ResourceType.IN_DIFF_CONFIDENCE, pool.Gradient_Pong),

                    // Specular
                    (ResourceType.IN_SPEC_RADIANCE_HITDIST, pool.Unfiltered_Spec),
                    (ResourceType.OUT_SPEC_RADIANCE_HITDIST, pool.Spec),
                    (ResourceType.IN_SPEC_CONFIDENCE, pool.Gradient_Pong)
                );

                // Reference denoiser resources
                nrdReference.UpdateResources(
                    // Common (required by NRDIntegration sanity check on frame 0)
                    (ResourceType.IN_MV, pool.Mv),
                    (ResourceType.IN_VIEWZ, pool.ViewZ),
                    (ResourceType.IN_NORMAL_ROUGHNESS, pool.NormalRoughness),

                    // REFERENCE
                    (ResourceType.IN_SIGNAL, pool.Composed),
                    (ResourceType.OUT_SIGNAL, pool.Composed)
                );
            }

            // Confidence Blur (5 ping-pong iterations over SHARC gradient)
            if (!setting.RR)
            {
                var sharcW = 16 * ((int)(renderResolution.x / sharcDownscale + 15) / 16);
                var sharcH = 16 * ((int)(renderResolution.y / sharcDownscale + 15) / 16);

                var nrdConfidenceBlurResource = new NRDConfidenceBlurPass.Resource
                {
                    ConstantBuffer = nrdConstantBuffer,
                    Pool           = pool
                };

                var nrdConfidenceBlurSettings = new NRDConfidenceBlurPass.Settings
                {
                    GroupsX = (uint)((sharcW + 15) / 16),
                    GroupsY = (uint)((sharcH + 15) / 16)
                };

                _nrdConfidenceBlurPass.Setup(nrdConfidenceBlurResource, nrdConfidenceBlurSettings);
                renderer.EnqueuePass(_nrdConfidenceBlurPass);
            }

            // Opaque
            {
                var nrdOpaqueResource = new NRDOpaquePass.Resource
                {
                    ConstantBuffer    = nrdConstantBuffer,
                    ScramblingRanking = scramblingRankingTexPtr,
                    Sobol             = sobolTexPtr,
                    Pool              = pool
                };

                var nrdOpaqueSettings = new NRDOpaquePass.Settings
                {
                    m_RenderResolution = renderResolution,
                    resolutionScale    = setting.resolutionScale
                };

                _nrdOpaquePass.SetNRDSampleResource(_nrdSampleResource);
                _nrdOpaquePass.Setup(nrdOpaqueResource, nrdOpaqueSettings);
                renderer.EnqueuePass(_nrdOpaquePass);
            }

            // NRD Denoiser (skip when DLSS-RR is active — it handles denoising internally)
            if (!setting.RR)
            {
                var lightData = renderingData.lightData;
                var mainLight = lightData.mainLightIndex >= 0 ? lightData.visibleLights[lightData.mainLightIndex] : default;
                var lightDir  = new float3(-(Vector3)mainLight.localToWorldMatrix.GetColumn(2));

                var commonInput = new NrdDenoiserHelper.CommonFrameInput
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
                    enableValidation             = setting.showValidation,
                    isHistoryConfidenceAvailable = setting.confidence,
                    splitScreen                  = setting.denoiser == DenoiserType.DENOISER_REFERENCE ? 1.0f : setting.separator,
                    denoisingRange               = setting.denoisingRange,
                    strandMaterialID             = 2f
                };

                NrdDenoiserHelper.GetCommonSettings(ref commonSettings, commonInput);

                sigmaSettings.lightDirection = new float3(lightDir.x, lightDir.y, lightDir.z);

                reblurSettings.checkerboardMode              = setting.tracingMode == RESOLUTION.RESOLUTION_HALF ? CheckerboardMode.WHITE : CheckerboardMode.OFF;
                reblurSettings.hitDistanceReconstructionMode = setting.tracingMode == RESOLUTION.RESOLUTION_FULL_PROBABILISTIC ? HitDistanceReconstructionMode.AREA_3X3 : HitDistanceReconstructionMode.OFF;
                reblurSettings.maxAccumulatedFrameNum        = setting.maxAccumulatedFrameNum;
                reblurSettings.maxFastAccumulatedFrameNum    = setting.maxFastAccumulatedFrameNum;
                reblurSettings.maxStabilizedFrameNum         = setting.maxAccumulatedFrameNum;
                reblurSettings.minMaterialForDiffuse         = 0;
                reblurSettings.minMaterialForSpecular        = 1;

                relaxSettings.checkerboardMode                   = setting.tracingMode == RESOLUTION.RESOLUTION_HALF ? CheckerboardMode.WHITE : CheckerboardMode.OFF;
                relaxSettings.hitDistanceReconstructionMode      = setting.tracingMode == RESOLUTION.RESOLUTION_FULL_PROBABILISTIC ? HitDistanceReconstructionMode.AREA_3X3 : HitDistanceReconstructionMode.OFF;
                relaxSettings.diffuseMaxAccumulatedFrameNum      = setting.maxAccumulatedFrameNum;
                relaxSettings.specularMaxAccumulatedFrameNum     = setting.maxAccumulatedFrameNum;
                relaxSettings.diffuseMaxFastAccumulatedFrameNum  = setting.maxFastAccumulatedFrameNum;
                relaxSettings.specularMaxFastAccumulatedFrameNum = setting.maxFastAccumulatedFrameNum;

                // Shadow denoising (SIGMA) — matches NRDSample.cpp "Shadow denoising" block
                _nrdShadowDenoisePass.Setup(nrdSigma.GetInteropDataPtr(commonSettings, sigmaSettings), RenderPassMarkers.NrdDenoiseShadow);
                renderer.EnqueuePass(_nrdShadowDenoisePass);

                if (setting.denoiser == DenoiserType.DENOISER_RELAX)
                    _nrdOpaqueDenoisePass.Setup(nrdRelax.GetInteropDataPtr(commonSettings, relaxSettings), RenderPassMarkers.NrdDenoiseOpaque);
                else
                    _nrdOpaqueDenoisePass.Setup(nrdReblur.GetInteropDataPtr(commonSettings, reblurSettings), RenderPassMarkers.NrdDenoiseOpaque);
                renderer.EnqueuePass(_nrdOpaqueDenoisePass);
            }

            var rectGridW = (int)(renderResolution.x * setting.resolutionScale + 0.5f + 15) / 16;
            var rectGridH = (int)(renderResolution.y * setting.resolutionScale + 0.5f + 15) / 16;

            // Composition
            {
                var nrdCompositionResource = new NRDCompositionPass.Resource
                {
                    ConstantBuffer = nrdConstantBuffer,
                    Pool           = pool
                };

                _nrdCompositionPass.Setup(nrdCompositionResource, new NRDCompositionPass.Settings
                {
                    rectGridW = rectGridW,
                    rectGridH = rectGridH,
                    useRR     = setting.RR
                });
                renderer.EnqueuePass(_nrdCompositionPass);
            }

            // Transparent
            {
                var nrdTransparentResource = new NRDTransparentPass.Resource
                {
                    ConstantBuffer = nrdConstantBuffer,
                    Pool           = pool
                };

                _nrdTransparentPass.SetNRDSampleResource(_nrdSampleResource);
                _nrdTransparentPass.Setup(nrdTransparentResource, new NRDTransparentPass.Settings
                {
                    m_RenderResolution = renderResolution,
                    resolutionScale    = setting.resolutionScale
                });
                renderer.EnqueuePass(_nrdTransparentPass);
            }

            if (!setting.RR && setting.denoiser == DenoiserType.DENOISER_REFERENCE)
            {
                var refCommonSettings = commonSettings;
                refCommonSettings.splitScreen = setting.separator;
                _nrdReferenceDenoisePass.Setup(nrdReference.GetInteropDataPtr(refCommonSettings, referenceSettings), RenderPassMarkers.NrdDenoiseOpaque);
                renderer.EnqueuePass(_nrdReferenceDenoisePass);
            }

            // ---- Tone Mapping (HDR → LDR, adaptive exposure) ----
            if (setting.enableAutoExposure)
            {
                var tmSource = pool.Composed.NativePtr;
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


            if (setting.SR || setting.RR)
            {
                var nrdDlssBeforeResource = new NRDDlssBeforePass.Resource
                {
                    ConstantBuffer = nrdConstantBuffer,
                    Pool           = pool
                };

                _nrdDlssBeforePass.Setup(nrdDlssBeforeResource, new NRDDlssBeforePass.Settings
                {
                    rectGridW = rectGridW,
                    rectGridH = rectGridH
                });
                renderer.EnqueuePass(_nrdDlssBeforePass);
            }

            if (setting.RR && setting.RRViaSL)
            {
                // DLSS Ray Reconstruction through Streamline (SLDenoiser) — A/B vs the NRI path.
                if (!_slDlrrDenoisers.TryGetValue(uniqueKey, out var slDlrr))
                {
                    var camName = isVR ? $"{cam.name}_Eye{eyeIndex}" : cam.name;
                    slDlrr = new SLDlssrr(camName);
                    _slDlrrDenoisers.Add(uniqueKey, slDlrr);
                }

                var slRes = new SLDlssrr.DlssrrResources
                {
                    input              = setting.enableAutoExposure ? pool.LdrColor : pool.Composed,
                    output             = pool.DlssOutput,
                    mv                 = pool.Mv,
                    depth              = pool.ViewZ,
                    diffAlbedo         = pool.RrGuideDiffAlbedo,
                    specAlbedo         = pool.RrGuideSpecAlbedo,
                    normalRoughness    = pool.RrGuideNormalRoughness,
                    specularMvOrHitTex = pool.RrGuideSpecHitDistance
                };

                var slInput = new SLDlssrr.SLDlssrrFrameInput
                {
                    worldToView             = frameState.worldToView,
                    viewToClip              = frameState.viewToClip,
                    worldToClip             = frameState.worldToClip,
                    prevWorldToClip         = frameState.prevWorldToClip,
                    camPos                  = frameState.camPos,
                    // Gate by the cameraJitter toggle so the jitter reported to DLSS matches the
                    // ACTUAL projection offset (0 when off; see CameraFrameStateNrdExtensions.cs:64).
                    // The native side negates it to match the NRI/NGX sign convention.
                    viewportJitter          = setting.cameraJitter ? frameState.viewportJitter : new float2(0f, 0f),
                    renderResolution        = frameState.renderResolution,
                    outputResolution        = outputResolution,
                    frameIndex              = curFrame,
                    cameraNear              = cam.nearClipPlane,
                    cameraFar               = cam.farClipPlane,
                    cameraFOV               = cam.fieldOfView * Mathf.Deg2Rad,
                    cameraAspect            = cam.aspect,
                    useSpecularMotionVector = false, // pass specular hit distance (matches NRI default)
                    reset                   = false
                };

                var slDataPtr = slDlrr.GetInteropDataPtr(slInput, slRes, setting.upscalerMode);

                _slDlssrrPass.Setup(slDataPtr, new SLDlssrrPass.Settings { tmpDisableRR = false });
                renderer.EnqueuePass(_slDlssrrPass);

                EnqueueDlssAfterPass(renderer, pool, outputResolution, nrdConstantBuffer);
            }
            else if (setting.RR)
            {
                var dlrrRes = new DlrrDenoiser.DlrrResources
                {
                    input              = setting.enableAutoExposure ? pool.LdrColor : pool.Composed,
                    output             = pool.DlssOutput,
                    mv                 = pool.Mv,
                    depth              = pool.ViewZ,
                    diffAlbedo         = pool.RrGuideDiffAlbedo,
                    specAlbedo         = pool.RrGuideSpecAlbedo,
                    normalRoughness    = pool.RrGuideNormalRoughness,
                    specularMvOrHitTex = pool.RrGuideSpecHitDistance
                };

                var dlrrInput = new DlrrDenoiser.DlrrFrameInput
                {
                    worldToView      = frameState.worldToView,
                    viewToClip       = frameState.viewToClip,
                    viewportJitter   = frameState.viewportJitter,
                    renderResolution = frameState.renderResolution,
                    frameIndex       = curFrame,
                    outputWidth      = (ushort)outputResolution.x,
                    outputHeight     = (ushort)outputResolution.y
                };

                var dlssDataPtr = dlrr.GetInteropDataPtr(dlrrInput, dlrrRes, 1.0f, setting.upscalerMode);

                _dlssrrPass.Setup(dlssDataPtr, new DlssRRPass.Settings
                {
                    tmpDisableRR = false
                });
                renderer.EnqueuePass(_dlssrrPass);

                EnqueueDlssAfterPass(renderer, pool, outputResolution, nrdConstantBuffer);
            }
            else if (setting.SR)
            {
                // DLSS-SR: upscale Composed → DlssOutput (replaces TAA + Final)
                if (!_dlsrUpscalers.TryGetValue(uniqueKey, out var dlsr))
                {
                    var camName = isVR ? $"{cam.name}_Eye{eyeIndex}" : cam.name;
                    dlsr = new DlsrUpscaler(camName);
                    _dlsrUpscalers.Add(uniqueKey, dlsr);
                }

                var dlsrRes = new DlsrUpscaler.DlsrResources
                {
                    input    = setting.enableAutoExposure ? pool.LdrColor : pool.Composed,
                    output   = pool.DlssOutput,
                    mv       = pool.Mv,
                    depth    = pool.ViewZ,
                    exposure = null,
                    reactive = null
                };

                var dlsrInput = new DlsrUpscaler.DlsrFrameInput
                {
                    viewportJitter   = frameState.viewportJitter,
                    renderResolution = frameState.renderResolution,
                    frameIndex       = curFrame,
                    outputWidth      = (ushort)outputResolution.x,
                    outputHeight     = (ushort)outputResolution.y
                };

                var dlsrSettings = new DlsrUpscaler.DlsrSettings
                {
                    upscalerMode = setting.upscalerMode,
                    preset       = 0,
                    resetHistory = resourcesChanged
                };

                var dlsrDataPtr = dlsr.GetInteropDataPtr(dlsrInput, dlsrRes, setting.resolutionScale, dlsrSettings);
                _dlsssrPass.Setup(dlsrDataPtr);
                renderer.EnqueuePass(_dlsssrPass);

                EnqueueDlssAfterPass(renderer, pool, outputResolution, nrdConstantBuffer);
            }
            else
            {
                // TAA
                var nrdTaaResource = new NRDTaaPass.Resource
                {
                    ConstantBuffer = nrdConstantBuffer,
                    Pool           = pool,
                    isEven         = isEven
                };

                _nrdTaaPass.Setup(nrdTaaResource, new NRDTaaPass.Settings
                {
                    rectGridW          = rectGridW,
                    rectGridH          = rectGridH,
                    enableAutoExposure = setting.enableAutoExposure
                });
                renderer.EnqueuePass(_nrdTaaPass);
            }

            // NIS — always runs: DLSS/TAA output → PreFinal
            {
                if (!_nisUpscalers.TryGetValue(uniqueKey, out var nis))
                {
                    var camName = isVR ? $"{cam.name}_Eye{eyeIndex}" : cam.name;
                    nis = new NisUpscaler(camName);
                    _nisUpscalers.Add(uniqueKey, nis);
                }

                var                isDlss = setting.RR || setting.SR;
                ushort             currentW, currentH;
                NriTextureResource nisInputTex;
                if (isDlss)
                {
                    nisInputTex = pool.DlssOutput;
                    currentW    = (ushort)outputResolution.x;
                    currentH    = (ushort)outputResolution.y;
                }
                else
                {
                    nisInputTex = isEven ? pool.TaaHistoryPing : pool.TaaHistoryPong;
                    currentW    = (ushort)(renderResolution.x * setting.resolutionScale + 0.5f);
                    currentH    = (ushort)(renderResolution.y * setting.resolutionScale + 0.5f);
                }

                var nisInput = new NisUpscaler.NisFrameInput
                {
                    outputWidth   = (ushort)outputResolution.x,
                    outputHeight  = (ushort)outputResolution.y,
                    currentWidth  = currentW,
                    currentHeight = currentH,
                    frameIndex    = curFrame
                };

                var nisRes = new NisUpscaler.NisResources
                {
                    input  = nisInputTex,
                    output = pool.PreFinal
                };

                var nisSettings = new NisUpscaler.NisSettings { sharpness = setting.nisSharpness };

                _nisPass.Setup(nis.GetInteropDataPtr(nisInput, nisRes, nisSettings));
                renderer.EnqueuePass(_nisPass);
            }

            // DLSS-G Frame Generation via Streamline (SLDenoiser). Player-only: in the editor
            // the swapchain is never adopted so BeginFrame/inputs are no-ops. The Reflex/begin
            // tick is driven separately by SLStreamlineFG at beginContextRendering.
            if (eyeIndex == 0)
            {
                if (setting.FGViaSL != _slFgLastEnabled)
                {
                    SLDLRR.SLStreamlineFG.SetFrameGeneration(setting.FGViaSL);
                    _slFgLastEnabled = setting.FGViaSL;
                }

                if (setting.FGViaSL)
                {
                    var fgEventFunc = SLDLRR.SLStreamlineFG.GetFrameInputsEventFunc();
                    if (fgEventFunc != IntPtr.Zero)
                    {
                        var viewToWorld   = frameState.worldToView.inverse;
                        var clipToCamView = frameState.viewToClip.inverse;
                        var clipToPrev    = frameState.prevWorldToClip * frameState.worldToClip.inverse;
                        var prevToClip    = frameState.worldToClip * frameState.prevWorldToClip.inverse;
                        var camRight =  new Vector3(viewToWorld.m00, viewToWorld.m10, viewToWorld.m20);
                        var camUp    =  new Vector3(viewToWorld.m01, viewToWorld.m11, viewToWorld.m21);
                        var camFwd   = -new Vector3(viewToWorld.m02, viewToWorld.m12, viewToWorld.m22);
                        var jit      = setting.cameraJitter ? frameState.viewportJitter : float2.zero;
                        float rW     = math.max(1, frameState.renderResolution.x);
                        float rH     = math.max(1, frameState.renderResolution.y);

                        var fg = new SLDLRR.SLStreamlineFG.DlssgInputs
                        {
                            depth                = pool.ViewZ.NativePtr,
                            motionVectors        = pool.Mv.NativePtr,
                            depthState           = SLDLRR.SLStreamlineFG.D3D12_STATE_UNORDERED_ACCESS,
                            mvecState            = SLDLRR.SLStreamlineFG.D3D12_STATE_UNORDERED_ACCESS,
                            mvecDepthW           = (uint)frameState.renderResolution.x,
                            mvecDepthH           = (uint)frameState.renderResolution.y,
                            colorW               = (uint)outputResolution.x,
                            colorH               = (uint)outputResolution.y,
                            cameraViewToClip     = frameState.viewToClip,
                            clipToCameraView     = clipToCamView,
                            clipToPrevClip       = clipToPrev,
                            prevClipToClip       = prevToClip,
                            cameraPos            = new Vector3(frameState.camPos.x, frameState.camPos.y, frameState.camPos.z),
                            cameraUp             = camUp,
                            cameraRight          = camRight,
                            cameraFwd            = camFwd,
                            jitterX              = jit.x,
                            jitterY              = jit.y,
                            mvecScaleX           = 1f / rW,
                            mvecScaleY           = 1f / rH,
                            cameraNear           = cam.nearClipPlane,
                            cameraFar            = cam.farClipPlane,
                            cameraFOV            = cam.fieldOfView * Mathf.Deg2Rad,
                            cameraAspect         = cam.aspect,
                            depthInverted        = 1,
                            cameraMotionIncluded = 1,
                            motionVectors3D      = 0,
                            reset                = 0,
                        };

                        var fgPtr = SLDLRR.SLStreamlineFG.GetInteropDataPtr(fg, curFrame);
                        _slDlssgInputsPass.Setup(fgEventFunc, fgPtr);
                        renderer.EnqueuePass(_slDlssgInputsPass);
                    }
                }
            }

            // Final (tone-map + validation overlay → Final RT)
            {
                var nrdFinalResource = new NRDFinalPass.Resource
                {
                    ConstantBuffer = nrdConstantBuffer,
                    Pool           = pool
                };

                _nrdFinalPass.Setup(nrdFinalResource, new NRDFinalPass.Settings
                {
                    OutputResolution = outputResolution
                });
                renderer.EnqueuePass(_nrdFinalPass);
            }


            // Output Blit
            {
                finalMaterial.SetFloat("gTanSunAngularRadius", math.tan(math.radians(setting.sunAngularDiameter * 0.5f)));


                _outputBlitPass.Setup(pool, new NativeNrdOutputBlitPass.Settings
                {
                    ShowMode        = setting.showMode,
                    ResolutionScale = frameState.resolutionScale,
                    ShowMv          = setting.showMv,
                    ShowValidation  = setting.showValidation
                });
                renderer.EnqueuePass(_outputBlitPass);

                renderer.EnqueuePass(_depthBarrierFixPass);
            }

            // DLSS-G frame generation: feed the real path-traced color/depth/motion + camera
            // matrices to the Streamline probe (non-RR / NRD-denoising mode only; the RR path
            // is wired separately later). No-op when the probe plugin isn't present.
            // if (!setting.RR && eyeIndex == 0)
            //     PushDlssgFrameInputs(renderer, cam, frameState, pool, outputResolution, renderResolution, resourcesChanged);
        }

        // /// <summary>
        // /// Pushes this frame's real DLSS-G inputs to the Streamline probe: color = pool.Final
        // /// (output res), depth = pool.ViewZ + motion = pool.Mv (render res), and the camera
        // /// constants from <paramref name="frameState"/>. See StreamlineFrameDriver.DlssgInputs.
        // /// </summary>
        // private void PushDlssgFrameInputs(ScriptableRenderer renderer, Camera cam, CameraFrameState frameState,
        //                                   NativeNrdTextureResources pool, int2 outputResolution, int2 renderResolution,
        //                                   bool resourcesChanged)
        // {
        //     var eventFunc = StreamlineFrameDriver.GetFrameInputsEventFunc();
        //     if (eventFunc == IntPtr.Zero) return; // probe plugin absent
        //
        //     if (pool?.Final == null || pool.Mv == null || pool.ViewZ == null) return;
        //
        //     var colorPtr = pool.Final.NativePtr;
        //     var depthPtr = pool.ViewZ.NativePtr;
        //     var mvPtr    = pool.Mv.NativePtr;
        //     if (colorPtr == IntPtr.Zero || depthPtr == IntPtr.Zero || mvPtr == IntPtr.Zero) return;
        //
        //     var viewToClip      = frameState.viewToClip;
        //     var worldToClip     = frameState.worldToClip;
        //     var prevWorldToClip = frameState.prevWorldToClip;
        //     var t               = cam.transform;
        //
        //     float mvScaleX = renderResolution.x > 0 ? 1f / renderResolution.x : 1f;
        //     float mvScaleY = renderResolution.y > 0 ? 1f / renderResolution.y : 1f;
        //
        //     var inputs = new StreamlineFrameDriver.DlssgInputs
        //     {
        //         color = colorPtr, depth = depthPtr, motionVectors = mvPtr,
        //         colorW     = (uint)outputResolution.x, colorH     = (uint)outputResolution.y,
        //         mvecDepthW = (uint)renderResolution.x, mvecDepthH = (uint)renderResolution.y,
        //         // All three are last consumed as shader resources before present (Final is
        //         // sampled by the output blit; ViewZ/Mv are denoiser SRV inputs), so they sit
        //         // in a shader-resource state at present. Exact state is a verify knob.
        //         colorState = StreamlineFrameDriver.D3D12_STATE_ALL_SHADER_RESOURCE,
        //         depthState = StreamlineFrameDriver.D3D12_STATE_ALL_SHADER_RESOURCE,
        //         mvecState  = StreamlineFrameDriver.D3D12_STATE_ALL_SHADER_RESOURCE,
        //
        //         cameraViewToClip = viewToClip,
        //         clipToCameraView = viewToClip.inverse,
        //         clipToPrevClip   = prevWorldToClip * worldToClip.inverse,
        //         prevClipToClip   = worldToClip * prevWorldToClip.inverse,
        //
        //         jitterX    = frameState.viewportJitter.x, jitterY    = frameState.viewportJitter.y,
        //         mvecScaleX = mvScaleX,                     mvecScaleY = mvScaleY,
        //         cameraPosX   = frameState.camPos.x, cameraPosY   = frameState.camPos.y, cameraPosZ   = frameState.camPos.z,
        //         cameraUpX    = t.up.x,              cameraUpY    = t.up.y,              cameraUpZ    = t.up.z,
        //         cameraRightX = t.right.x,           cameraRightY = t.right.y,           cameraRightZ = t.right.z,
        //         cameraFwdX   = t.forward.x,         cameraFwdY   = t.forward.y,         cameraFwdZ   = t.forward.z,
        //         cameraNear   = cam.nearClipPlane,
        //         cameraFar    = cam.farClipPlane,
        //         cameraFOV    = cam.fieldOfView * Mathf.Deg2Rad,
        //         cameraAspect = outputResolution.y > 0 ? (float)outputResolution.x / outputResolution.y : 1f,
        //         depthInverted        = 0, // ViewZ is linear positive view distance (verify knob)
        //         cameraMotionIncluded = 1,
        //         motionVectors3D      = 0, // NRD Mv treated as 2D screen-space (verify knob)
        //         reset                = resourcesChanged ? 1 : 0,
        //     };
        //
        //     // Stash into the persistent ring and hand the stable pointer to the render thread
        //     // via IssuePluginEventAndData (the native side copies it synchronously on consume).
        //     var dataPtr = WriteDlssgInputsToRing(in inputs, frameState.frameIndex);
        //     if (dataPtr == IntPtr.Zero) return;
        //
        //     _dlssgInputsPass.Setup(eventFunc, dataPtr);
        //     renderer.EnqueuePass(_dlssgInputsPass);
        // }
        //
        // private unsafe IntPtr WriteDlssgInputsToRing(in StreamlineFrameDriver.DlssgInputs inputs, uint frameIndex)
        // {
        //     if (!_dlssgInputsRing.IsCreated)
        //         _dlssgInputsRing = new NativeArray<StreamlineFrameDriver.DlssgInputs>(DlssgInputsRingCount, Allocator.Persistent);
        //
        //     int idx = (int)(frameIndex % DlssgInputsRingCount);
        //     _dlssgInputsRing[idx] = inputs;
        //     return (IntPtr)((byte*)_dlssgInputsRing.GetUnsafePtr()
        //                     + idx * UnsafeUtility.SizeOf<StreamlineFrameDriver.DlssgInputs>());
        // }

        private void EnqueueDlssAfterPass(ScriptableRenderer renderer, NativeNrdTextureResources pool, int2 outputResolution, VolatileConstantBuffer nrdConstantBuffer)
        {
            var outputGridW = (outputResolution.x + 15) / 16;
            var outputGridH = (outputResolution.y + 15) / 16;

            _nrdDlssAfterPass.Setup(new NRDDlssAfterPass.Resource
            {
                ConstantBuffer = nrdConstantBuffer,
                Pool           = pool
            }, new NRDDlssAfterPass.Settings
            {
                outputGridW = outputGridW,
                outputGridH = outputGridH
            });
            renderer.EnqueuePass(_nrdDlssAfterPass);
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

            _nrdSampleResource?.Dispose();
            _nrdSampleResource = null;

            foreach (var cb in _nrdConstantBuffers.Values)
                cb.Dispose();
            _nrdConstantBuffers.Clear();

            foreach (var denoiser in _sigmaDenoisers.Values)
                denoiser.Dispose();
            _sigmaDenoisers.Clear();

            foreach (var denoiser in _reblurDenoisers.Values)
                denoiser.Dispose();
            _reblurDenoisers.Clear();

            foreach (var denoiser in _relaxDenoisers.Values)
                denoiser.Dispose();
            _relaxDenoisers.Clear();

            foreach (var denoiser in _referenceDenoisers.Values)
                denoiser.Dispose();
            _referenceDenoisers.Clear();

            foreach (var denoiser in _dlrrDenoisers.Values)
                denoiser.Dispose();
            _dlrrDenoisers.Clear();

            foreach (var denoiser in _slDlrrDenoisers.Values)
                denoiser.Dispose();
            _slDlrrDenoisers.Clear();

            foreach (var upscaler in _dlsrUpscalers.Values)
                upscaler.Dispose();
            _dlsrUpscalers.Clear();

            foreach (var upscaler in _nisUpscalers.Values)
                upscaler.Dispose();
            _nisUpscalers.Clear();

            _cameraFrameStates.Clear();

            foreach (var p in _resourcePools.Values)
                p.Dispose();
            _resourcePools.Clear();

            _nrdTlasUpdatePass = null;
            _nrdSharcPass?.Dispose();
            _nrdSharcPass = null;
            _nrdOpaquePass?.Dispose();
            _nrdOpaquePass = null;
            _nrdTransparentPass?.Dispose();
            _nrdTransparentPass = null;
            _nrdCompositionPass?.Dispose();
            _nrdCompositionPass = null;
            _nrdTaaPass?.Dispose();
            _nrdTaaPass = null;
            _nrdConfidenceBlurPass?.Dispose();
            _nrdConfidenceBlurPass = null;
            _nrdFinalPass?.Dispose();
            _nrdFinalPass            = null;
            _nrdShadowDenoisePass    = null;
            _nrdOpaqueDenoisePass    = null;
            _nrdReferenceDenoisePass = null;
            _nrdDlssBeforePass?.Dispose();
            _nrdDlssBeforePass = null;
            _nrdDlssAfterPass?.Dispose();
            _nrdDlssAfterPass    = null;
            _dlssrrPass          = null;
            _dlsssrPass          = null;
            _nisPass             = null;
            _outputBlitPass      = null;
            _depthBarrierFixPass = null;
            _toneMappingPass     = null;
            _dlssgInputsPass     = null;

            // if (_dlssgInputsRing.IsCreated)
            //     _dlssgInputsRing.Dispose();

            if (scramblingRankingUintTex != null)
                scramblingRankingUintTex.Release();
            if (sobolUintTex != null)
                sobolUintTex?.Release();
            scramblingRankingUintTex = null;
            sobolUintTex             = null;
            sobolTexPtr              = IntPtr.Zero;
            scramblingRankingTexPtr  = IntPtr.Zero;
        }

#if UNITY_EDITOR
        private void Reset()
        {
            setting        = new NativeNrdSampleSetting();
            sigmaSettings  = SigmaSettings._default;
            reblurSettings = ReblurSettings._default;
            relaxSettings  = RelaxSettings._default;
            AutoFillShaders();
        }

        public void AutoFillShaders()
        {
            finalMaterial = UnityEditor.AssetDatabase.LoadAssetAtPath<Material>("Assets/Shaders/Mat/KM_Final.mat");

            updateSkinnedPrimitivesCS = UnityEditor.AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/NRD-Sample/Shaders/UpdateSkinnedPrimitives.compute");
            fillUintTextureCS         = UnityEditor.AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Scripts/FillUintTexture.compute");

            nrdOpaqueTracingShader  = UnityEditor.AssetDatabase.LoadAssetAtPath<NativeComputeShader>("Assets/NRD-Sample/Shaders/TraceOpaque.computeshader");
            nrdSharcResolve         = UnityEditor.AssetDatabase.LoadAssetAtPath<NativeComputeShader>("Assets/NRD-Sample/Shaders/SharcResolve.computeshader");
            nrdSharcUpdate          = UnityEditor.AssetDatabase.LoadAssetAtPath<NativeComputeShader>("Assets/NRD-Sample/Shaders/SharcUpdate.computeshader");
            nrdTransparentShader    = UnityEditor.AssetDatabase.LoadAssetAtPath<NativeComputeShader>("Assets/NRD-Sample/Shaders/TraceTransparent.computeshader");
            nrdCompositionShader    = UnityEditor.AssetDatabase.LoadAssetAtPath<NativeComputeShader>("Assets/NRD-Sample/Shaders/Composition.computeshader");
            nrdTaaShader            = UnityEditor.AssetDatabase.LoadAssetAtPath<NativeComputeShader>("Assets/NRD-Sample/Shaders/Taa.computeshader");
            nrdConfidenceBlurShader = UnityEditor.AssetDatabase.LoadAssetAtPath<NativeComputeShader>("Assets/NRD-Sample/Shaders/ConfidenceBlur.computeshader");
            nrdFinalShader          = UnityEditor.AssetDatabase.LoadAssetAtPath<NativeComputeShader>("Assets/NRD-Sample/Shaders/Final.computeshader");
            nrdDlssBeforeShader     = UnityEditor.AssetDatabase.LoadAssetAtPath<NativeComputeShader>("Assets/NRD-Sample/Shaders/DlssBefore.computeshader");
            nrdDlssAfterShader      = UnityEditor.AssetDatabase.LoadAssetAtPath<NativeComputeShader>("Assets/NRD-Sample/Shaders/DlssAfter.computeshader");

            scramblingRankingTex = UnityEditor.AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/NRD-Sample/Assets/Textures/scrambling_ranking_128x128_2d_4spp.png");
            sobolTex             = UnityEditor.AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/NRD-Sample/Assets/Textures/sobol_256_4d.png");

            toneMappingHistogramCs = LoadCs($"Assets/Shaders/donut/histogram");
            toneMappingExposureCs  = LoadCs($"Assets/Shaders/donut/exposure");
            toneMappingCs          = UnityEditor.AssetDatabase.LoadAssetAtPath<NativeRasterShader>("Assets/Shaders/donut/tonemapping.rastershader");
            if (toneMappingCs == null)
                Debug.LogWarning("[NativeNrdFeature] Missing NativeRasterShader at: Assets/Shaders/donut/tonemapping.rastershader");


            UnityEditor.EditorUtility.SetDirty(this);
            return;

            static NativeComputeShader LoadCs(string path)
            {
                var s = UnityEditor.AssetDatabase.LoadAssetAtPath<NativeComputeShader>(path + ".computeshader");
                if (s == null)
                    Debug.LogWarning($"[NativeRtxdiFeature] Missing NativeComputeShader at: {path}");
                return s;
            }
        }
#endif
    }
}