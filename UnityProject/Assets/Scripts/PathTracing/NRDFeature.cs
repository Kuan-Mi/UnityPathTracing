using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using DLRR;
using NativeRender;
using Nrd;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering.Universal;
using UnityEngine.Serialization;

namespace PathTracing
{
    /// <summary>
    /// Standalone ScriptableRendererFeature that runs the full NRD (NativeComputeShader) path:
    /// NRDTlasUpdate → NRDSharc → NRDConfidenceBlur → NRDOpaque → NrdDenoiser →
    /// NRDComposition → NRDTransparent → NRDTaa → NRDFinal → OutputBlit.
    /// Does NOT include AccumulatePass, ReferencePtPass, DlssBeforePass, DlssRRPass, or AutoExposurePass.
    /// </summary>
    public class NRDFeature : ScriptableRendererFeature
    {
        private const uint sharcDownscale = 5;


        [FormerlySerializedAs("pathTracingSetting")]
        public NrdSampleSetting setting;

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
        public ComputeShader        dlssBeforeComputeShader;

        public Texture2D scramblingRankingTex;
        public Texture2D sobolTex;

        private IntPtr scramblingRankingTexPtr;
        private IntPtr sobolTexPtr;


        private NRDTlasUpdatePass     _nrdTlasUpdatePass;
        private NRDSharcPass          _nrdSharcPass;
        private NRDOpaquePass         _nrdOpaquePass;
        private NRDTransparentPass    _nrdTransparentPass;
        private NRDCompositionPass    _nrdCompositionPass;
        private NRDTaaPass            _nrdTaaPass;
        private NRDConfidenceBlurPass _nrdConfidenceBlurPass;
        private NRDFinalPass          _nrdFinalPass;
        private NrdPass               _nrdPass;
        private DlssBeforePass        _dlssBeforePass;
        private DlssRRPass            _dlssrrPass;
        private OutputBlitPass        _outputBlitPass;

        private NRDSampleResource _nrdSampleResource;

        private          GraphicsBuffer       _nrdConstantBuffer;
        private          IntPtr               _nrdConstantBufferPtr;
        private readonly NRDGlobalConstants[] _nrdGlobalConstantsArray = new NRDGlobalConstants[1];

        private readonly Dictionary<long, NrdDenoiser>             _nrdDenoisers      = new();
        private readonly Dictionary<long, DlrrDenoiser>             _dlrrDenoisers     = new();
        private readonly Dictionary<long, PathTracingResourcePool> _resourcePools     = new();
        private readonly Dictionary<long, CameraFrameState>        _cameraFrameStates = new();

        public override void Create()
        {
            _nrdTlasUpdatePass ??= new NRDTlasUpdatePass
            {
                renderPassEvent = renderPassEvent
            };

            _nrdSharcPass ??= new NRDSharcPass(nrdSharcResolve, nrdSharcUpdate)
            {
                renderPassEvent = renderPassEvent
            };

            _nrdOpaquePass ??= new NRDOpaquePass(nrdOpaqueTracingShader)
            {
                renderPassEvent = renderPassEvent
            };


            _nrdTransparentPass ??= new NRDTransparentPass(nrdTransparentShader)
            {
                renderPassEvent = renderPassEvent
            };


            _nrdCompositionPass ??= new NRDCompositionPass(nrdCompositionShader)
            {
                renderPassEvent = renderPassEvent
            };


            _nrdTaaPass ??= new NRDTaaPass(nrdTaaShader)
            {
                renderPassEvent = renderPassEvent
            };

            _nrdConfidenceBlurPass ??= new NRDConfidenceBlurPass(nrdConfidenceBlurShader)
            {
                renderPassEvent = renderPassEvent
            };

            _nrdFinalPass ??= new NRDFinalPass(nrdFinalShader)
            {
                renderPassEvent = renderPassEvent
            };

            _nrdPass ??= new NrdPass()
            {
                renderPassEvent = renderPassEvent
            };

            _dlssBeforePass ??= new DlssBeforePass(dlssBeforeComputeShader)
            {
                renderPassEvent = renderPassEvent
            };
            _dlssrrPass ??= new DlssRRPass()
            {
                renderPassEvent = renderPassEvent
            };

            _outputBlitPass ??= new OutputBlitPass(finalMaterial)
            {
                renderPassEvent = renderPassEvent
            };
            scramblingRankingTexPtr = scramblingRankingTex.GetNativeTexturePtr();
            sobolTexPtr             = sobolTex.GetNativeTexturePtr();
        }

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            var cam = renderingData.cameraData.camera;
            if (cam.cameraType is CameraType.Preview or CameraType.Reflection)
                return;
            if (cam.cameraType != CameraType.Game && cam.cameraType != CameraType.SceneView)
                return;

            cam.depthTextureMode = DepthTextureMode.Depth | DepthTextureMode.MotionVectors;

            var eyeIndex = renderingData.cameraData.xr.enabled ? renderingData.cameraData.xr.multipassId : 0;

            // if (eyeIndex == 1 && pathTracingSetting.skipRightEyeInVR)
            //     return;

            if (finalMaterial == null
                || nrdOpaqueTracingShader == null
                || nrdSharcResolve == null
                || nrdSharcUpdate == null
                || nrdConfidenceBlurShader == null
                || nrdFinalShader == null
                || scramblingRankingTex == null
                || sobolTex == null)
            {
                Debug.LogWarning("NRDFeature: Missing required assets, skipping pass.");
                return;
            }


            if (_nrdSampleResource == null)
                _nrdSampleResource = new NRDSampleResource(setting.mergeBlas);


            _nrdSampleResource.MergeBlas = setting.mergeBlas;
            NativeRender.NativeRenderPlugin.NR_FrameTick();
            _nrdSampleResource?.UpdateForFrame();

            var uniqueKey = cam.GetInstanceID() + (eyeIndex * 100000L);
            var isVR      = renderingData.cameraData.xrRendering;

            if (!_resourcePools.TryGetValue(uniqueKey, out var pool))
            {
                pool = new PathTracingResourcePool();
                pool.InitNrdSampleResources();
                _resourcePools.Add(uniqueKey, pool);
            }

            if (!_nrdDenoisers.TryGetValue(uniqueKey, out var nrd))
            {
                var camName = isVR ? $"{cam.name}_Eye{eyeIndex}" : cam.name;
                // nrd = NrdDenoiser.CreateDefault(pathTracingSetting, camName);
                var nrdDenoiserDescArray = new NrdDenoiserDesc[]
                {
                    new(0, Denoiser.SIGMA_SHADOW_TRANSLUCENCY),
                    new(1, Denoiser.REBLUR_DIFFUSE_SPECULAR),
                };
                nrd = new NrdDenoiser(camName, nrdDenoiserDescArray);
                _nrdDenoisers.Add(uniqueKey, nrd);
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

            var  outputResolution = ComputeOutputResolution(renderingData.cameraData);
            bool resourcesChanged = pool.EnsureResources(outputResolution, setting.upscalerMode);
            var  renderResolution = pool.renderResolution;

            if (resourcesChanged)
            {
                frameState.renderResolution = pool.renderResolution;
                frameState.frameIndex       = 0;
            }

            uint curFrame = frameState.frameIndex;
            frameState.Update(renderingData, false, setting.resolutionScale);

            globalConstants = frameState.GetNrdConstants(renderingData, setting);

            _nrdConstantBuffer ??= new GraphicsBuffer(GraphicsBuffer.Target.Constant, 1, Marshal.SizeOf<NRDGlobalConstants>());

            _nrdGlobalConstantsArray[0] = globalConstants;

            // todo : 这里可以考虑自己在C++端管理，避免GetNativeBufferPtr的开销
            _nrdConstantBuffer.SetData(_nrdGlobalConstantsArray);
            _nrdConstantBufferPtr = _nrdConstantBuffer.GetNativeBufferPtr();

            bool isEven = (globalConstants.gFrameIndex & 1) == 0;

            // TLAS update
            _nrdTlasUpdatePass.SetNRDSampleResource(_nrdSampleResource);
            renderer.EnqueuePass(_nrdTlasUpdatePass);

            // SHARC
            {
                int sharcW = 16 * ((int)(renderResolution.x / sharcDownscale + 15) / 16);
                int sharcH = 16 * ((int)(renderResolution.y / sharcDownscale + 15) / 16);
                pool.EnsureSharcGradientResources(new int2(sharcW, sharcH));

                var nrdSharcResource = new NRDSharcPass.Resource
                {
                    ConstantBuffer = _nrdConstantBufferPtr,
                    Pool           = pool,
                };

                var nrdSharcSettings = new NRDSharcPass.Settings
                {
                    RenderResolution = renderResolution,
                    sharcDownscale   = sharcDownscale,
                    isEven           = isEven,
                };

                _nrdSharcPass.SetNRDSampleResource(_nrdSampleResource);
                _nrdSharcPass.Setup(nrdSharcResource, nrdSharcSettings);
                renderer.EnqueuePass(_nrdSharcPass);
            }

            if (resourcesChanged)
            {
                nrd.UpdateResources(
                    // Common
                    (ResourceType.IN_MV, pool.GetNriResource(RenderResourceType.MV)),
                    (ResourceType.IN_VIEWZ, pool.GetNriResource(RenderResourceType.Viewz)),
                    (ResourceType.IN_NORMAL_ROUGHNESS, pool.GetNriResource(RenderResourceType.NormalRoughness)),

                    // (Optional) Validation
                    (ResourceType.OUT_VALIDATION, pool.GetNriResource(RenderResourceType.Validation)),

                    // Diffuse
                    (ResourceType.IN_DIFF_RADIANCE_HITDIST, pool.GetNriResource(RenderResourceType.Unfiltered_Diff)),
                    (ResourceType.OUT_DIFF_RADIANCE_HITDIST, pool.GetNriResource(RenderResourceType.Diff)),
                    (ResourceType.IN_DIFF_CONFIDENCE, pool.GetNriResource(RenderResourceType.Gradient_Pong)),

                    // Specular
                    (ResourceType.IN_SPEC_RADIANCE_HITDIST, pool.GetNriResource(RenderResourceType.Unfiltered_Spec)),
                    (ResourceType.OUT_SPEC_RADIANCE_HITDIST, pool.GetNriResource(RenderResourceType.Spec)),
                    (ResourceType.IN_SPEC_CONFIDENCE, pool.GetNriResource(RenderResourceType.Gradient_Pong)),

                    // SIGMA
                    (ResourceType.IN_PENUMBRA, pool.GetNriResource(RenderResourceType.Unfiltered_Penumbra)),
                    (ResourceType.IN_TRANSLUCENCY, pool.GetNriResource(RenderResourceType.Unfiltered_Translucency)),
                    (ResourceType.OUT_SHADOW_TRANSLUCENCY, pool.GetNriResource(RenderResourceType.Shadow)),

                    // REFERENCE
                    (ResourceType.IN_SIGNAL, pool.GetNriResource(RenderResourceType.Composed)),
                    (ResourceType.OUT_SIGNAL, pool.GetNriResource(RenderResourceType.Composed)
                    )
                );
            }

            // Confidence Blur (5 ping-pong iterations over SHARC gradient)
            if(!setting.RR)
            {
                int sharcW = 16 * ((int)(renderResolution.x / sharcDownscale + 15) / 16);
                int sharcH = 16 * ((int)(renderResolution.y / sharcDownscale + 15) / 16);

                var nrdConfidenceBlurResource = new NRDConfidenceBlurPass.Resource
                {
                    ConstantBuffer = _nrdConstantBufferPtr,
                    Pool           = pool,
                };

                var nrdConfidenceBlurSettings = new NRDConfidenceBlurPass.Settings
                {
                    GroupsX = (uint)((sharcW + 15) / 16),
                    GroupsY = (uint)((sharcH + 15) / 16),
                };

                _nrdConfidenceBlurPass.Setup(nrdConfidenceBlurResource, nrdConfidenceBlurSettings);
                renderer.EnqueuePass(_nrdConfidenceBlurPass);
            }

            // Opaque
            {
                var nrdOpaqueResource = new NRDOpaquePass.Resource
                {
                    ConstantBuffer    = _nrdConstantBufferPtr,
                    ScramblingRanking = scramblingRankingTexPtr,
                    Sobol             = sobolTexPtr,
                    Pool              = pool,
                };

                var nrdOpaqueSettings = new NRDOpaquePass.Settings
                {
                    m_RenderResolution = renderResolution,
                    resolutionScale    = setting.resolutionScale,
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

                var nrdInput = new NrdDenoiser.NrdFrameInput
                {
                    worldToView         = frameState.worldToView,
                    prevWorldToView     = frameState.prevWorldToView,
                    viewToClip          = frameState.viewToClip,
                    prevViewToClip      = frameState.prevViewToClip,
                    viewportJitter      = frameState.viewportJitter,
                    prevViewportJitter  = frameState.prevViewportJitter,
                    resolutionScale     = frameState.resolutionScale,
                    prevResolutionScale = frameState.prevResolutionScale,
                    renderResolution    = frameState.renderResolution,
                    frameIndex          = curFrame,
                    lightDirection      = lightDir,
                    checkerboardMode    = setting.tracingMode == RESOLUTION.RESOLUTION_HALF ? CheckerboardMode.BLACK : CheckerboardMode.OFF,
                };

                _nrdPass.Setup(nrd.GetInteropDataPtr(nrdInput));
                renderer.EnqueuePass(_nrdPass);
            }

            var rectGridW = (int)(renderResolution.x * setting.resolutionScale + 0.5f + 15) / 16;
            var rectGridH = (int)(renderResolution.y * setting.resolutionScale + 0.5f + 15) / 16;

            // Composition
            {
                var nrdCompositionResource = new NRDCompositionPass.Resource
                {
                    ConstantBuffer = _nrdConstantBufferPtr,
                    Pool           = pool,
                };

                _nrdCompositionPass.Setup(nrdCompositionResource, new NRDCompositionPass.Settings
                {
                    rectGridW = rectGridW,
                    rectGridH = rectGridH,
                    useRR = setting.RR
                });
                renderer.EnqueuePass(_nrdCompositionPass);
            }

            // Transparent
            {
                var nrdTransparentResource = new NRDTransparentPass.Resource
                {
                    ConstantBuffer = _nrdConstantBufferPtr,
                    Pool           = pool,
                };

                _nrdTransparentPass.SetNRDSampleResource(_nrdSampleResource);
                _nrdTransparentPass.Setup(nrdTransparentResource, new NRDTransparentPass.Settings
                {
                    m_RenderResolution = renderResolution,
                    resolutionScale    = setting.resolutionScale,
                });
                renderer.EnqueuePass(_nrdTransparentPass);
            }

            if (setting.RR)
            {
                // DLSS-RR: DlssBefore → DlssRR (replaces TAA + Final)
                var dlssBeforeResource = new DlssBeforePass.Resource
                {
                    ConstantBuffer = _nrdConstantBuffer,

                    NormalRoughness    = pool.GetRT(RenderResourceType.NormalRoughness),
                    BaseColorMetalness = pool.GetRT(RenderResourceType.BaseColorMetalness),
                    Spec               = pool.GetRT(RenderResourceType.Unfiltered_Spec),
                    ViewZ              = pool.GetRT(RenderResourceType.Viewz),

                    RRGuide_DiffAlbedo       = pool.GetRT(RenderResourceType.RrGuideDiffAlbedo),
                    RRGuide_SpecAlbedo       = pool.GetRT(RenderResourceType.RrGuideSpecAlbedo),
                    RRGuide_SpecHitDistance  = pool.GetRT(RenderResourceType.RrGuideSpecHitDistance),
                    RRGuide_Normal_Roughness = pool.GetRT(RenderResourceType.RrGuideNormalRoughness),
                };

                _dlssBeforePass.Setup(dlssBeforeResource, new DlssBeforePass.Settings
                {
                    rectGridW    = rectGridW,
                    rectGridH    = rectGridH,
                    tmpDisableRR = setting.tmpDisableRR,
                });
                renderer.EnqueuePass(_dlssBeforePass);

                var dlrrRes = new DlrrDenoiser.DlrrResources
                {
                    input           = pool.GetNriResource(RenderResourceType.Composed),
                    output          = pool.GetNriResource(RenderResourceType.DlssOutput),
                    mv              = pool.GetNriResource(RenderResourceType.MV),
                    depth           = pool.GetNriResource(RenderResourceType.Viewz),
                    diffAlbedo      = pool.GetNriResource(RenderResourceType.RrGuideDiffAlbedo),
                    specAlbedo      = pool.GetNriResource(RenderResourceType.RrGuideSpecAlbedo),
                    normalRoughness = pool.GetNriResource(RenderResourceType.RrGuideNormalRoughness),
                    specHitDistance = pool.GetNriResource(RenderResourceType.RrGuideSpecHitDistance),
                };

                var dlrrInput = new DlrrDenoiser.DlrrFrameInput
                {
                    worldToView      = frameState.worldToView,
                    viewToClip       = frameState.viewToClip,
                    viewportJitter   = frameState.viewportJitter,
                    renderResolution = frameState.renderResolution,
                    frameIndex       = curFrame,
                    outputWidth      = (ushort)outputResolution.x,
                    outputHeight     = (ushort)outputResolution.y,
                };

                var dlssDataPtr = dlrr.GetInteropDataPtr(dlrrInput, dlrrRes, 1.0f, setting.upscalerMode);

                _dlssrrPass.Setup(dlssDataPtr, new DlssRRPass.Settings
                {
                    tmpDisableRR = setting.tmpDisableRR,
                });
                renderer.EnqueuePass(_dlssrrPass);
            }
            else
            {
                // TAA
                {
                    var nrdTaaResource = new NRDTaaPass.Resource
                    {
                        ConstantBuffer = _nrdConstantBufferPtr,
                        Pool           = pool,
                        isEven         = isEven,
                    };

                    _nrdTaaPass.Setup(nrdTaaResource, new NRDTaaPass.Settings
                    {
                        rectGridW = rectGridW,
                        rectGridH = rectGridH,
                    });
                    renderer.EnqueuePass(_nrdTaaPass);
                }

                // Final (tone-map + validation overlay → Final RT)
                {
                    var nrdFinalResource = new NRDFinalPass.Resource
                    {
                        ConstantBuffer = _nrdConstantBufferPtr,
                        Pool           = pool,
                        IsEven         = isEven,
                    };

                    _nrdFinalPass.Setup(nrdFinalResource, new NRDFinalPass.Settings
                    {
                        OutputResolution = outputResolution,
                    });
                    renderer.EnqueuePass(_nrdFinalPass);
                }
            }

            // Output Blit
            {
                var outputBlitResource = new OutputBlitPass.Resource
                {
                    Mv                 = pool.GetRT(RenderResourceType.MV),
                    NormalRoughness    = pool.GetRT(RenderResourceType.NormalRoughness),
                    BaseColorMetalness = pool.GetRT(RenderResourceType.BaseColorMetalness),

                    Penumbra = pool.GetRT(RenderResourceType.Unfiltered_Penumbra),
                    Diff     = pool.GetRT(RenderResourceType.Unfiltered_Diff),
                    Spec     = pool.GetRT(RenderResourceType.Unfiltered_Spec),

                    ShadowTranslucency = pool.GetRT(RenderResourceType.Shadow),
                    DenoisedDiff       = pool.GetRT(RenderResourceType.Diff),
                    DenoisedSpec       = pool.GetRT(RenderResourceType.Spec),
                    Validation         = pool.GetRT(RenderResourceType.Validation),

                    Composed       = pool.GetRT(RenderResourceType.Composed),
                    DirectLighting = pool.GetRT(RenderResourceType.DirectLighting),

                    RRGuide_DiffAlbedo       = pool.GetRT(RenderResourceType.RrGuideDiffAlbedo),
                    RRGuide_SpecAlbedo       = pool.GetRT(RenderResourceType.RrGuideSpecAlbedo),
                    RRGuide_Normal_Roughness = pool.GetRT(RenderResourceType.RrGuideNormalRoughness),
                    RRGuide_SpecHitDistance  = pool.GetRT(RenderResourceType.RrGuideSpecHitDistance),
                    DlssOutput               = pool.GetRT(RenderResourceType.DlssOutput),
                    // todo
                    taaDst                   = pool.GetRT(RenderResourceType.Final),
                    ViewZ                    = pool.GetRT(RenderResourceType.Viewz),
                    Gradient                 = pool.GetRT(RenderResourceType.Gradient_Pong),

                    Output            = pool.GetRT(RenderResourceType.Final),
                    DirectEmission    = pool.GetRT(RenderResourceType.DirectEmission),
                    ComposedDiff      = pool.GetRT(RenderResourceType.ComposedDiff),
                    ComposedSpecViewZ = pool.GetRT(RenderResourceType.ComposedSpecViewZ),
                };

                _outputBlitPass.Setup(outputBlitResource, new OutputBlitPass.Settings
                {
                    showMode        = setting.showMode,
                    resolutionScale = frameState.resolutionScale,
                    enableDlssRR    = setting.RR,
                    tmpDisableRR    = setting.tmpDisableRR,
                    showMV          = false,
                    showValidation  = setting.showValidation,
                    showReference   = false,
                });
                renderer.EnqueuePass(_outputBlitPass);
            }
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

            _nrdConstantBuffer?.Release();
            _nrdConstantBuffer = null;

            foreach (var denoiser in _nrdDenoisers.Values)
                denoiser.Dispose();
            _nrdDenoisers.Clear();

            foreach (var denoiser in _dlrrDenoisers.Values)
                denoiser.Dispose();
            _dlrrDenoisers.Clear();

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
            _nrdFinalPass   = null;
            _nrdPass        = null;
            _dlssBeforePass = null;
            _dlssrrPass     = null;
            _outputBlitPass = null;
        }

#if UNITY_EDITOR
        private void Reset()
        {
            setting = new NrdSampleSetting();
            AutoFillShaders();
        }

        public void AutoFillShaders()
        {
            finalMaterial = UnityEditor.AssetDatabase.LoadAssetAtPath<Material>("Assets/Shaders/Mat/KM_Final.mat");

            nrdOpaqueTracingShader  = UnityEditor.AssetDatabase.LoadAssetAtPath<NativeComputeShader>("Assets/NRD-Sample/Shaders/TraceOpaque.computeshader");
            nrdSharcResolve         = UnityEditor.AssetDatabase.LoadAssetAtPath<NativeComputeShader>("Assets/NRD-Sample/Shaders/SharcResolve.computeshader");
            nrdSharcUpdate          = UnityEditor.AssetDatabase.LoadAssetAtPath<NativeComputeShader>("Assets/NRD-Sample/Shaders/SharcUpdate.computeshader");
            nrdTransparentShader    = UnityEditor.AssetDatabase.LoadAssetAtPath<NativeComputeShader>("Assets/NRD-Sample/Shaders/TraceTransparent.computeshader");
            nrdCompositionShader    = UnityEditor.AssetDatabase.LoadAssetAtPath<NativeComputeShader>("Assets/NRD-Sample/Shaders/Composition.computeshader");
            nrdTaaShader            = UnityEditor.AssetDatabase.LoadAssetAtPath<NativeComputeShader>("Assets/NRD-Sample/Shaders/Taa.computeshader");
            nrdConfidenceBlurShader = UnityEditor.AssetDatabase.LoadAssetAtPath<NativeComputeShader>("Assets/NRD-Sample/Shaders/ConfidenceBlur.computeshader");
            nrdFinalShader          = UnityEditor.AssetDatabase.LoadAssetAtPath<NativeComputeShader>("Assets/NRD-Sample/Shaders/Final.computeshader");
            dlssBeforeComputeShader = UnityEditor.AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/PostProcess/DlssBefore.compute");

            scramblingRankingTex = UnityEditor.AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/Textures/scrambling_ranking_128x128_2d_4spp.png");
            sobolTex             = UnityEditor.AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/Textures/sobol_256_4d.png");

            UnityEditor.EditorUtility.SetDirty(this);
        }
#endif
    }
}