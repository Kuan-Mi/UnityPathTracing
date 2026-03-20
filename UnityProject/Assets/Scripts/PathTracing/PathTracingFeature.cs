using System.Collections.Generic;
using System.Runtime.InteropServices;
using DLRR;
using mini;
using Nrd;
using RTXDI;
using Rtxdi.DI;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using static UnityEngine.Rendering.RayTracingAccelerationStructure;
using static PathTracing.ShaderIDs;

namespace PathTracing
{
    public class PathTracingFeature : ScriptableRendererFeature
    {
        public PathTracingSetting pathTracingSetting;

        public RenderPassEvent renderPassEvent = RenderPassEvent.BeforeRenderingPostProcessing;

        public Material finalMaterial;
        public RayTracingShader sharcUpdateTs;
        public RayTracingShader opaqueTracingShader;
        public RayTracingShader transparentTracingShader;
        public RayTracingShader referencePtTracingShader;

        public ComputeShader compositionComputeShader;
        public ComputeShader taaComputeShader;
        public ComputeShader dlssBeforeComputeShader;
        public ComputeShader sharcResolveCs;
        public ComputeShader autoExposureShader;

        public Texture2D scramblingRankingTex;
        public Texture2D sobolTex;

        private OutputBlitPass _outputBlitPass;
        private SharcPass _sharcPass;
        private PrepareLightPass _prepareLightPass;
        private OpaquePass _opaquePass;
        private NrdPass _nrdPass;
        private CompositionPass _compositionPass;
        private TransparentPass _transparentPass;
        private AutoExposurePass _autoExposurePass;
        private TaaPass _taaPass;
        private DlssRRPass _dlssrrPass;
        private ReferencePtPass _referencePtPass;

        private RayTracingAccelerationStructure _accelerationStructure;

        private GraphicsBuffer _constantBuffer;
        private GraphicsBuffer _resamplingConstantBuffer;

        private GraphicsBuffer _scramblingRankingUintBuffer;
        private GraphicsBuffer _sobolUintBuffer;

        private GraphicsBuffer _hashEntriesBuffer;
        private GraphicsBuffer _accumulationBuffer;
        private GraphicsBuffer _resolvedBuffer;

        private GraphicsBuffer _aeHistogramBuffer; // 256 x uint
        private GraphicsBuffer _aeExposureBuffer; // 1 x float  (current exposure multiplier)

        private GPUScene _gpuScene = new();
        private LightCollector _lightCollector = new();
        private PrepareLightResource _prepareLightResources;

        private readonly GlobalConstants[] _globalConstantsArray = new GlobalConstants[1];
        private readonly ResamplingConstants[] _resamplingConstantsArray = new ResamplingConstants[1];
        private readonly float[] _exposureArray = new float[1];

        private readonly Dictionary<long, NrdDenoiser> _nrdDenoisers = new();
        private readonly Dictionary<long, DlrrDenoiser> _dlrrDenoisers = new();
        private readonly Dictionary<long, PathTracingResourcePool> _resourcePools = new();
        private readonly Dictionary<long, ReSTIRDIContext> _restirDiContexts = new();
        private readonly Dictionary<long, RtxdiResources> _rtxdiResources = new();
        private readonly Dictionary<long, CameraFrameState> _cameraFrameStates = new();

        public override void Create()
        {
            if (_accelerationStructure == null)
            {
                var settings = new Settings
                {
                    managementMode = ManagementMode.Automatic,
                    rayTracingModeMask = RayTracingModeMask.Everything
                };
                _accelerationStructure = new RayTracingAccelerationStructure(settings);
                _accelerationStructure.Build();
                SetMask();
            }

            _gpuScene ??= new GPUScene();
            _lightCollector ??= new LightCollector();
            _prepareLightResources ??= new PrepareLightResource();

            if (!_gpuScene.isBufferInitialized)
            {
                _gpuScene.InitBuffer();
            }

            if (_scramblingRankingUintBuffer == null && scramblingRankingTex != null)
            {
                _scramblingRankingUintBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, scramblingRankingTex.width * scramblingRankingTex.height, 16);
                var scramblingRankingData = new uint4[scramblingRankingTex.width * scramblingRankingTex.height];
                var rawData = scramblingRankingTex.GetRawTextureData();
                var count = scramblingRankingData.Length;
                for (var i = 0; i < count; i++)
                {
                    scramblingRankingData[i] = new uint4(rawData[i * 4 + 0], rawData[i * 4 + 1], rawData[i * 4 + 2], rawData[i * 4 + 3]);
                }

                _scramblingRankingUintBuffer.SetData(scramblingRankingData);
            }

            if (_sobolUintBuffer == null && sobolTex != null)
            {
                _sobolUintBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, sobolTex.width * sobolTex.height, 16);
                var sobolData = new uint4[sobolTex.width * sobolTex.height];
                var rawData = sobolTex.GetRawTextureData();
                var count = sobolData.Length;
                for (var i = 0; i < count; i++)
                {
                    sobolData[i] = new uint4(rawData[i * 4 + 0], rawData[i * 4 + 1], rawData[i * 4 + 2], rawData[i * 4 + 3]);
                }

                _sobolUintBuffer.SetData(sobolData);
            }

            if (_accumulationBuffer == null)
            {
                InitializeBuffers();
            }

            _sharcPass ??= new SharcPass(sharcResolveCs, sharcUpdateTs)
            {
                renderPassEvent = renderPassEvent
            };
            _prepareLightPass ??= new PrepareLightPass()
            {
                renderPassEvent = renderPassEvent
            };
            _opaquePass ??= new OpaquePass(opaqueTracingShader)
            {
                renderPassEvent = renderPassEvent
            };
            _nrdPass ??= new NrdPass()
            {
                renderPassEvent = renderPassEvent
            };
            _compositionPass ??= new CompositionPass(compositionComputeShader)
            {
                renderPassEvent = renderPassEvent
            };
            _transparentPass ??= new TransparentPass(transparentTracingShader)
            {
                renderPassEvent = renderPassEvent
            };
            _autoExposurePass ??= new AutoExposurePass(autoExposureShader)
            {
                renderPassEvent = renderPassEvent
            };
            _taaPass ??= new TaaPass(taaComputeShader)
            {
                renderPassEvent = renderPassEvent
            };
            _dlssrrPass ??= new DlssRRPass(dlssBeforeComputeShader)
            {
                renderPassEvent = renderPassEvent
            };
            _outputBlitPass ??= new OutputBlitPass(finalMaterial)
            {
                renderPassEvent = renderPassEvent
            };
            _referencePtPass ??= new ReferencePtPass(referencePtTracingShader)
            {
                renderPassEvent = renderPassEvent
            };
        }

        public static readonly int Capacity = 1 << 23;

        public void InitializeBuffers()
        {
            if (_hashEntriesBuffer != null)
            {
                _hashEntriesBuffer.Release();
                _hashEntriesBuffer = null;
            }

            if (_accumulationBuffer != null)
            {
                _accumulationBuffer.Release();
                _accumulationBuffer = null;
            }

            if (_resolvedBuffer != null)
            {
                _resolvedBuffer.Release();
                _resolvedBuffer = null;
            }

            _constantBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Constant, 1, Marshal.SizeOf<GlobalConstants>());
            _resamplingConstantBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, Marshal.SizeOf<ResamplingConstants>());

            _hashEntriesBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Capacity, sizeof(ulong));
            var clearData = new ulong[Capacity];
            _hashEntriesBuffer.SetData(clearData);

            // 2. Accumulation Buffer: storing uint4 (16 bytes)
            // HLSL: RWStructuredBuffer<SharcAccumulationData> gInOut_SharcAccumulated;
            _accumulationBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Capacity, sizeof(uint) * 4);
            var clearAccumData = new uint4[Capacity];
            _accumulationBuffer.SetData(clearAccumData);

            // 3. Resolved Buffer: storing uint3 + uint (16 bytes)
            // HLSL: RWStructuredBuffer<SharcPackedData> gInOut_SharcResolved;
            _resolvedBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Capacity, sizeof(uint) * 4);
            var clearResolvedData = new uint4[Capacity];
            _resolvedBuffer.SetData(clearResolvedData);

            // Auto-exposure buffers
            _aeHistogramBuffer ??= new GraphicsBuffer(GraphicsBuffer.Target.Structured, 256, sizeof(uint));

            if (_aeExposureBuffer == null)
            {
                _aeExposureBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, sizeof(float));
                // Seed with the manual exposure value so first frame is not zero
                _exposureArray[0] = pathTracingSetting?.exposure ?? 1.0f;
                _aeExposureBuffer.SetData(_exposureArray);
            }
        }


        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            var cam = renderingData.cameraData.camera;
            if (cam.cameraType is CameraType.Preview or CameraType.Reflection)
                return;
            if (cam.cameraType != CameraType.Game && cam.cameraType != CameraType.SceneView)
            {
                return;
            }

            cam.depthTextureMode = DepthTextureMode.Depth | DepthTextureMode.MotionVectors;

            var eyeIndex = renderingData.cameraData.xr.enabled ? renderingData.cameraData.xr.multipassId : 0;


            if (eyeIndex == 1)
                return;


            Shader.SetGlobalRayTracingAccelerationStructure(g_AccelStructID, _accelerationStructure);
            _gpuScene.Build();


            var uniqueKey = cam.GetInstanceID() + (eyeIndex * 100000L);


            var isVR = renderingData.cameraData.xrRendering;

            if (!_resourcePools.TryGetValue(uniqueKey, out var pool))
            {
                pool = new PathTracingResourcePool(pathTracingSetting);
                _resourcePools.Add(uniqueKey, pool);
            }

            if (!_nrdDenoisers.TryGetValue(uniqueKey, out var nrd))
            {
                var camName = cam.name;
                if (isVR)
                {
                    camName = $"{cam.name}_Eye{eyeIndex}";
                }

                nrd = new NrdDenoiser(pathTracingSetting, camName);
                _nrdDenoisers.Add(uniqueKey, nrd);
            }


            if (!_dlrrDenoisers.TryGetValue(uniqueKey, out var dlrr))
            {
                var camName = cam.name;
                if (isVR)
                {
                    camName = $"{cam.name}_Eye{eyeIndex}";
                }

                dlrr = new DlrrDenoiser(pathTracingSetting, camName);
                _dlrrDenoisers.Add(uniqueKey, dlrr);
            }

            if (!_cameraFrameStates.TryGetValue(uniqueKey, out var frameState))
            {
                frameState = new CameraFrameState(pathTracingSetting.resolutionScale);
                _cameraFrameStates.Add(uniqueKey, frameState);
            }

            _prepareLightResources.SetBuffer(_gpuScene);
            _prepareLightResources.SendTexture(_gpuScene.globalTexturePool);

            if (!_restirDiContexts.TryGetValue(uniqueKey, out var restirDiContext))
            {
                var contextParams = ReSTIRDIStaticParameters.Default();
                contextParams.RenderWidth = (uint)cam.pixelWidth;
                contextParams.RenderHeight = (uint)cam.pixelHeight;

                restirDiContext = new ReSTIRDIContext(contextParams);
                _restirDiContexts.Add(uniqueKey, restirDiContext);
            }

            if (!_rtxdiResources.TryGetValue(uniqueKey, out var rtxdiResources))
            {
                rtxdiResources = new RtxdiResources(restirDiContext, _gpuScene);
                _rtxdiResources.Add(uniqueKey, rtxdiResources);
            }

            if (finalMaterial == null
                || opaqueTracingShader == null
                || transparentTracingShader == null
                || compositionComputeShader == null
                || taaComputeShader == null
                || dlssBeforeComputeShader == null
                || sharcResolveCs == null
                || sharcUpdateTs == null
                || scramblingRankingTex == null
                || sobolTex == null
               )
            {
                Debug.LogWarning("PathTracingFeature: Missing required assets, skipping path tracing pass.");
                return;
            }

            var allSkinnedMeshRenderers = FindObjectsByType<SkinnedMeshRenderer>(FindObjectsSortMode.None);
            foreach (var smr in allSkinnedMeshRenderers)
            {
                _accelerationStructure.UpdateInstanceTransform(smr);
            }

            _accelerationStructure.Build();

            _lightCollector.Collect();

            var outputResolution = ComputeOutputResolution(renderingData.cameraData);
            bool resourcesChanged = pool.EnsureResources(outputResolution);

            if (resourcesChanged)
            {
                frameState.renderResolution = pool.renderResolution;
                frameState.FrameIndex = 0;
            }

            // Update per-camera temporal state for this frame
            uint curFrame = frameState.FrameIndex;
            frameState.Update(renderingData, pathTracingSetting);

            var nrdLightData = renderingData.lightData;
            var nrdMainLight = nrdLightData.mainLightIndex >= 0 ? nrdLightData.visibleLights[nrdLightData.mainLightIndex] : default;
            var nrdLightDir = new float3(-(Vector3)nrdMainLight.localToWorldMatrix.GetColumn(2));

            var nrdInput = new NrdDenoiser.NrdFrameInput
            {
                worldToView = frameState.worldToView,
                prevWorldToView = frameState.prevWorldToView,
                viewToClip = frameState.viewToClip,
                prevViewToClip = frameState.prevViewToClip,
                viewportJitter = frameState.ViewportJitter,
                prevViewportJitter = frameState.PrevViewportJitter,
                resolutionScale = frameState.resolutionScale,
                prevResolutionScale = frameState.prevResolutionScale,
                renderResolution = frameState.renderResolution,
                frameIndex = curFrame,
                lightDirection = nrdLightDir,
            };

            var nrdDataPtr = nrd.GetInteropDataPtr(nrdInput);

            var nrdRes = new NrdDenoiser.NrdResources
            {
                inMv = pool.GetNriResource(RenderResourceType.IN_MV),
                inViewZ = pool.GetNriResource(RenderResourceType.IN_VIEWZ),
                inNormalRoughness = pool.GetNriResource(RenderResourceType.IN_NORMAL_ROUGHNESS),
                inBaseColorMetalness = pool.GetNriResource(RenderResourceType.IN_BASECOLOR_METALNESS),
                inPenumbra = pool.GetNriResource(RenderResourceType.IN_PENUMBRA),
                inDiffRadianceHitDist = pool.GetNriResource(RenderResourceType.IN_DIFF_RADIANCE_HITDIST),
                inSpecRadianceHitDist = pool.GetNriResource(RenderResourceType.IN_SPEC_RADIANCE_HITDIST),
                outShadowTranslucency = pool.GetNriResource(RenderResourceType.OUT_SHADOW_TRANSLUCENCY),
                outDiffRadianceHitDist = pool.GetNriResource(RenderResourceType.OUT_DIFF_RADIANCE_HITDIST),
                outSpecRadianceHitDist = pool.GetNriResource(RenderResourceType.OUT_SPEC_RADIANCE_HITDIST),
                outValidation = pool.GetNriResource(RenderResourceType.OUT_VALIDATION),
            };

            if (resourcesChanged)
            {
                nrd.UpdateResources(nrdRes);
            }

            var dlrrRes = new DlrrDenoiser.DlrrResources
            {
                input = pool.GetNriResource(RenderResourceType.Composed),
                output = pool.GetNriResource(RenderResourceType.DlssOutput),
                mv = pool.GetNriResource(RenderResourceType.IN_MV),
                depth = pool.GetNriResource(RenderResourceType.IN_VIEWZ),
                diffAlbedo = pool.GetNriResource(RenderResourceType.RRGuide_DiffAlbedo),
                specAlbedo = pool.GetNriResource(RenderResourceType.RRGuide_SpecAlbedo),
                normalRoughness = pool.GetNriResource(RenderResourceType.RRGuide_Normal_Roughness),
                specHitDistance = pool.GetNriResource(RenderResourceType.RRGuide_SpecHitDistance),
            };

            var dlrrInput = new DlrrDenoiser.DlrrFrameInput
            {
                worldToView = frameState.worldToView,
                viewToClip = frameState.viewToClip,
                viewportJitter = frameState.ViewportJitter,
                renderResolution = frameState.renderResolution,
                frameIndex = curFrame,
                outputWidth = (ushort)outputResolution.x,
                outputHeight = (ushort)outputResolution.y,
            };
            var dlssDataPtr = dlrr.GetInteropDataPtr(dlrrInput, dlrrRes);

            var globalConstants = frameState.GetConstants(renderingData, pathTracingSetting, _lightCollector);
            _globalConstantsArray[0] = globalConstants;
            _constantBuffer.SetData(_globalConstantsArray);

            var resamplingConstants = GetResamplingConstants(restirDiContext, rtxdiResources, frameState);
            _resamplingConstantsArray[0] = resamplingConstants;
            _resamplingConstantBuffer.SetData(_resamplingConstantsArray);

            // Sharc 
            var sharcResource = new SharcPass.Resource
            {
                ConstantBuffer = _constantBuffer,
                AccumulationBuffer = _accumulationBuffer,
                HashEntriesBuffer = _hashEntriesBuffer,
                ResolvedBuffer = _resolvedBuffer,
                PointLightBuffer = _lightCollector.PointLightBuffer,
                AreaLightBuffer = _lightCollector.AreaLightBuffer,
                SpotLightBuffer = _lightCollector.SpotLightBuffer
            };

            var sharcSettings = new SharcPass.Settings
            {
                RenderResolution = new int2(cam.pixelWidth, cam.pixelHeight)
            };

            _sharcPass.Setup(sharcResource, sharcSettings);
            renderer.EnqueuePass(_sharcPass);

            _prepareLightPass.Setup(_prepareLightResources);
            renderer.EnqueuePass(_prepareLightPass);

            var opaqueResource = new OpaquePass.Resource
            {
                ConstantBuffer = _constantBuffer,
                ResamplingConstantBuffer = _resamplingConstantBuffer,

                AccumulationBuffer = _accumulationBuffer,
                HashEntriesBuffer = _hashEntriesBuffer,
                ResolvedBuffer = _resolvedBuffer,

                PointLightBuffer = _lightCollector.PointLightBuffer,
                AreaLightBuffer = _lightCollector.AreaLightBuffer,
                SpotLightBuffer = _lightCollector.SpotLightBuffer,

                ScramblingRanking = _scramblingRankingUintBuffer,
                Sobol = _sobolUintBuffer,

                Mv = pool.GetRT(RenderResourceType.IN_MV),
                ViewZ = pool.GetRT(RenderResourceType.IN_VIEWZ),
                NormalRoughness = pool.GetRT(RenderResourceType.IN_NORMAL_ROUGHNESS),
                BaseColorMetalness = pool.GetRT(RenderResourceType.IN_BASECOLOR_METALNESS),

                Penumbra = pool.GetRT(RenderResourceType.IN_PENUMBRA),
                Diff = pool.GetRT(RenderResourceType.IN_DIFF_RADIANCE_HITDIST),
                Spec = pool.GetRT(RenderResourceType.IN_SPEC_RADIANCE_HITDIST),

                PrevViewZ = pool.GetRT(RenderResourceType.Prev_ViewZ),
                PrevNormalRoughness = pool.GetRT(RenderResourceType.Prev_NormalRoughness),
                PrevBaseColorMetalness = pool.GetRT(RenderResourceType.Prev_BaseColorMetalness),

                PsrThroughput = pool.GetRT(RenderResourceType.PsrThroughput),

                RtxdiResources = rtxdiResources
            };

            var opaqueSettings = new OpaquePass.Settings
            {
                m_RenderResolution = new int2(cam.pixelWidth, cam.pixelHeight),
                resolutionScale = pathTracingSetting.resolutionScale
            };

            _opaquePass.Setup(opaqueResource, opaqueSettings);
            renderer.EnqueuePass(_opaquePass);


            if (!pathTracingSetting.RR)
            {
                _nrdPass.Setup(nrdDataPtr);
                renderer.EnqueuePass(_nrdPass);
            }


            var compositionResource = new CompositionPass.Resource
            {
                ConstantBuffer = _constantBuffer,

                ViewZ = pool.GetRT(RenderResourceType.IN_VIEWZ),
                NormalRoughness = pool.GetRT(RenderResourceType.IN_NORMAL_ROUGHNESS),
                BaseColorMetalness = pool.GetRT(RenderResourceType.IN_BASECOLOR_METALNESS),
                PsrThroughput = pool.GetRT(RenderResourceType.PsrThroughput)
            };

            if (pathTracingSetting.RR)
            {
                compositionResource.Shadow = pool.GetRT(RenderResourceType.IN_PENUMBRA);
                compositionResource.Diff = pool.GetRT(RenderResourceType.IN_DIFF_RADIANCE_HITDIST);
                compositionResource.Spec = pool.GetRT(RenderResourceType.IN_SPEC_RADIANCE_HITDIST);
            }
            else
            {
                compositionResource.Shadow = pool.GetRT(RenderResourceType.OUT_SHADOW_TRANSLUCENCY);
                compositionResource.Diff = pool.GetRT(RenderResourceType.OUT_DIFF_RADIANCE_HITDIST);
                compositionResource.Spec = pool.GetRT(RenderResourceType.OUT_SPEC_RADIANCE_HITDIST);
            }

            var rectGridW = (int)(cam.pixelWidth * pathTracingSetting.resolutionScale + 0.5f + 15) / 16;
            var rectGridH = (int)(cam.pixelHeight * pathTracingSetting.resolutionScale + 0.5f + 15) / 16;

            var compositionSettings = new CompositionPass.Settings
            {
                rectGridW = rectGridW,
                rectGridH = rectGridH
            };

            _compositionPass.Setup(compositionResource, compositionSettings);
            renderer.EnqueuePass(_compositionPass);


            var transparentResource = new TransparentPass.Resource
            {
                ConstantBuffer = _constantBuffer,

                Mv = pool.GetRT(RenderResourceType.IN_MV),
                Composed = pool.GetRT(RenderResourceType.Composed),
                NormalRoughness = pool.GetRT(RenderResourceType.IN_NORMAL_ROUGHNESS),

                HashEntriesBuffer = _hashEntriesBuffer,
                AccumulationBuffer = _accumulationBuffer,
                ResolvedBuffer = _resolvedBuffer,

                PointLightBuffer = _lightCollector.PointLightBuffer,
                AreaLightBuffer = _lightCollector.AreaLightBuffer,
                SpotLightBuffer = _lightCollector.SpotLightBuffer,

                AeExposureBuffer = _aeExposureBuffer
            };

            var transparentSettings = new TransparentPass.Settings
            {
                m_RenderResolution = new int2(cam.pixelWidth, cam.pixelHeight)
            };

            _transparentPass.Setup(transparentResource, transparentSettings);
            renderer.EnqueuePass(_transparentPass);

            var autoExposureResource = new AutoExposurePass.Resource
            {
                AeHistogramBuffer = _aeHistogramBuffer,
                AeExposureBuffer = _aeExposureBuffer,
                Composed = pool.GetRT(RenderResourceType.Composed)
            };

            var renderResolution = pool.renderResolution;

            var aeSettings = new AutoExposurePass.Settings
            {
                AeEnabled = pathTracingSetting.enableAutoExposure,
                AeEVMin = pathTracingSetting.aeEVMin,
                AeEVMax = pathTracingSetting.aeEVMax,
                AeLowPercent = pathTracingSetting.aeLowPercent,
                AeHighPercent = pathTracingSetting.aeHighPercent,
                AeSpeedUp = pathTracingSetting.aeAdaptationSpeedUp,
                AeSpeedDown = pathTracingSetting.aeAdaptationSpeedDown,
                AeDeltaTime = Time.deltaTime,
                AeExposureCompensation = pathTracingSetting.aeExposureCompensation,
                AeMinExposure = pathTracingSetting.aeMinExposure,
                AeMaxExposure = pathTracingSetting.aeMaxExposure,
                AeTexWidth = (uint)renderResolution.x,
                AeTexHeight = (uint)renderResolution.y,
                ManualExposure = pathTracingSetting.exposure
            };

            if (!pathTracingSetting.enableAutoExposure)
            {
                _exposureArray[0] = pathTracingSetting.exposure;
                _aeExposureBuffer.SetData(_exposureArray);
            }
            else
            {
                _autoExposurePass.Setup(autoExposureResource, aeSettings);
                renderer.EnqueuePass(_autoExposurePass);
            }

            var isEven = (globalConstants.gFrameIndex & 1) == 0;

            if (pathTracingSetting.RR)
            {
                var dlssResource = new DlssRRPass.Resource
                {
                    ConstantBuffer = _constantBuffer,

                    NormalRoughness = pool.GetRT(RenderResourceType.IN_NORMAL_ROUGHNESS),
                    BaseColorMetalness = pool.GetRT(RenderResourceType.IN_BASECOLOR_METALNESS),
                    Spec = pool.GetRT(RenderResourceType.IN_SPEC_RADIANCE_HITDIST),
                    ViewZ = pool.GetRT(RenderResourceType.IN_VIEWZ),

                    RRGuide_DiffAlbedo = pool.GetRT(RenderResourceType.RRGuide_DiffAlbedo),
                    RRGuide_SpecAlbedo = pool.GetRT(RenderResourceType.RRGuide_SpecAlbedo),
                    RRGuide_SpecHitDistance = pool.GetRT(RenderResourceType.RRGuide_SpecHitDistance),
                    RRGuide_Normal_Roughness = pool.GetRT(RenderResourceType.RRGuide_Normal_Roughness),
                };

                var dlssSettings = new DlssRRPass.Settings
                {
                    rectGridW = rectGridW,
                    rectGridH = rectGridH,
                    tmpDisableRR = pathTracingSetting.tmpDisableRR
                };

                _dlssrrPass.Setup(dlssDataPtr, dlssResource, dlssSettings);
                renderer.EnqueuePass(_dlssrrPass);
            }
            else
            {
                var taaResource = new TaaPass.Resource
                {
                    ConstantBuffer = _constantBuffer,

                    Mv = pool.GetRT(RenderResourceType.IN_MV),
                    Composed = pool.GetRT(RenderResourceType.Composed),
                    taaSrc = pool.GetRT(isEven ? RenderResourceType.TaaHistoryPrev : RenderResourceType.TaaHistory),
                    taaDst = pool.GetRT(isEven ? RenderResourceType.TaaHistory : RenderResourceType.TaaHistoryPrev)
                };

                var taaSettings = new TaaPass.Settings
                {
                    rectGridW = rectGridW,
                    rectGridH = rectGridH
                };

                _taaPass.Setup(taaResource, taaSettings);
                renderer.EnqueuePass(_taaPass);
            }

            if (pathTracingSetting.useReferencePathTracing)
            {
                var referencePtResource = new ReferencePtPass.Resource
                {
                    ConstantBuffer = _constantBuffer,

                    PointLightBuffer = _lightCollector.PointLightBuffer,
                    AreaLightBuffer = _lightCollector.AreaLightBuffer,
                    SpotLightBuffer = _lightCollector.SpotLightBuffer,
                    AeExposureBuffer = _aeExposureBuffer
                };

                var referencePtSettings = new ReferencePtPass.Settings
                {
                    m_RenderResolution = new int2(cam.pixelWidth, cam.pixelHeight),
                    resolutionScale = pathTracingSetting.resolutionScale,
                    referenceBounceNum = pathTracingSetting.referenceBounceNum,
                    convergenceStep = frameState.convergenceStep,
                    split = pathTracingSetting.split
                };

                _referencePtPass.Setup(referencePtResource, referencePtSettings);
                renderer.EnqueuePass(_referencePtPass);
            }

            var outputBlitResource = new OutputBlitPass.Resource
            {
                Mv = pool.GetRT(RenderResourceType.IN_MV),
                NormalRoughness = pool.GetRT(RenderResourceType.IN_NORMAL_ROUGHNESS),
                BaseColorMetalness = pool.GetRT(RenderResourceType.IN_BASECOLOR_METALNESS),


                Penumbra = pool.GetRT(RenderResourceType.IN_PENUMBRA),
                Diff = pool.GetRT(RenderResourceType.IN_DIFF_RADIANCE_HITDIST),
                Spec = pool.GetRT(RenderResourceType.IN_SPEC_RADIANCE_HITDIST),

                ShadowTranslucency = pool.GetRT(RenderResourceType.OUT_SHADOW_TRANSLUCENCY),
                DenoisedDiff = pool.GetRT(RenderResourceType.OUT_DIFF_RADIANCE_HITDIST),
                DenoisedSpec = pool.GetRT(RenderResourceType.OUT_SPEC_RADIANCE_HITDIST),
                Validation = pool.GetRT(RenderResourceType.OUT_VALIDATION),

                Composed = pool.GetRT(RenderResourceType.Composed),

                RRGuide_DiffAlbedo = pool.GetRT(RenderResourceType.RRGuide_DiffAlbedo),
                RRGuide_SpecAlbedo = pool.GetRT(RenderResourceType.RRGuide_SpecAlbedo),
                RRGuide_Normal_Roughness = pool.GetRT(RenderResourceType.RRGuide_Normal_Roughness),
                RRGuide_SpecHitDistance = pool.GetRT(RenderResourceType.RRGuide_SpecHitDistance),
                DlssOutput = pool.GetRT(RenderResourceType.DlssOutput),
                taaDst = pool.GetRT(isEven ? RenderResourceType.TaaHistory : RenderResourceType.TaaHistoryPrev),
            };

            var outputBlitSettings = new OutputBlitPass.Settings
            {
                showMode = pathTracingSetting.showMode,
                resolutionScale = frameState.resolutionScale,
                enableDlssRR = pathTracingSetting.RR,
                showMV = pathTracingSetting.showMV,
                showValidation = pathTracingSetting.showValidation,
                showReference = pathTracingSetting.useReferencePathTracing,
            };

            _outputBlitPass.Setup(outputBlitResource, outputBlitSettings);
            renderer.EnqueuePass(_outputBlitPass);
        }

        private ResamplingConstants GetResamplingConstants(ReSTIRDIContext restirDiContext, RtxdiResources rtxdiResources, CameraFrameState frameState)
        {
            restirDiContext.SetFrameIndex(frameState.FrameIndex);

            var resamplingConstants = new ResamplingConstants
            {
                runtimeParams = restirDiContext.GetRuntimeParams()
            };

            resamplingConstants.lightBufferParams.localLightBufferRegion.firstLightIndex = 0;
            resamplingConstants.lightBufferParams.localLightBufferRegion.numLights = rtxdiResources.Scene.emissiveTriangleCount;

            resamplingConstants.lightBufferParams.infiniteLightBufferRegion.firstLightIndex = 0;
            resamplingConstants.lightBufferParams.infiniteLightBufferRegion.numLights = 0;

            resamplingConstants.lightBufferParams.environmentLightParams.lightPresent = 0;
            resamplingConstants.lightBufferParams.environmentLightParams.lightIndex = (0xffffffffu);

            resamplingConstants.restirDIReservoirBufferParams = restirDiContext.GetReservoirBufferParameters();

            resamplingConstants.frameIndex = restirDiContext.GetFrameIndex();
            resamplingConstants.numInitialSamples = pathTracingSetting.localLightSamples;
            resamplingConstants.numSpatialSamples = pathTracingSetting.spatialSamples;
            resamplingConstants.useAccurateGBufferNormal = 0;
            resamplingConstants.numInitialBRDFSamples = pathTracingSetting.brdfSamples;
            resamplingConstants.brdfCutoff = 0;
            resamplingConstants.pad2 = new uint2(0, 0);
            resamplingConstants.enableResampling = pathTracingSetting.enableResampling ? 1u : 0u;
            resamplingConstants.unbiasedMode = 1;
            resamplingConstants.inputBufferIndex = (resamplingConstants.frameIndex & 1u) ^ 1;
            resamplingConstants.outputBufferIndex = (resamplingConstants.frameIndex & 1u);
            return resamplingConstants;
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
            Debug.LogWarning("PathTracingFeature Dispose");
            base.Dispose(disposing);

            _accelerationStructure?.Dispose();
            _accelerationStructure = null;

            _constantBuffer?.Release();
            _constantBuffer = null;

            _resamplingConstantBuffer?.Release();
            _resamplingConstantBuffer = null;

            foreach (var denoiser in _nrdDenoisers.Values)
            {
                denoiser.Dispose();
            }

            _nrdDenoisers.Clear();

            foreach (var denoiser in _dlrrDenoisers.Values)
            {
                denoiser.Dispose();
            }

            _dlrrDenoisers.Clear();

            _cameraFrameStates.Clear();

            foreach (var pool in _resourcePools.Values)
            {
                pool.Dispose();
            }

            _resourcePools.Clear();

            foreach (var res in _rtxdiResources.Values)
            {
                res.Dispose();
            }

            _rtxdiResources.Clear();

            _restirDiContexts.Clear();

            _lightCollector?.Dispose();
            _lightCollector = null;

            _gpuScene?.Dispose();
            _gpuScene = null;

            _prepareLightResources.Dispose();
            _prepareLightResources = null;

            _scramblingRankingUintBuffer?.Release();
            _scramblingRankingUintBuffer = null;

            _sobolUintBuffer?.Release();
            _sobolUintBuffer = null;

            _accumulationBuffer?.Release();
            _accumulationBuffer = null;

            _hashEntriesBuffer?.Release();
            _hashEntriesBuffer = null;

            _resolvedBuffer?.Release();
            _resolvedBuffer = null;

            _aeHistogramBuffer?.Release();
            _aeHistogramBuffer = null;

            _aeExposureBuffer?.Release();
            _aeExposureBuffer = null;

            _sharcPass = null;
            _prepareLightPass = null;
            _opaquePass = null;
            _transparentPass = null;
            _compositionPass = null;
            _nrdPass = null;
            _taaPass = null;
            _dlssrrPass = null;
            _autoExposurePass = null;
            _outputBlitPass = null;
        }

        public void Test()
        {
            _gpuScene.DebugReadback();
        }

        // #define FLAG_NON_TRANSPARENT                0x01 // geometry flag: non-transparent
        // #define FLAG_TRANSPARENT                    0x02 // geometry flag: transparent
        // #define FLAG_FORCED_EMISSION                0x04 // animated emissive cube
        // #define FLAG_STATIC                         0x08 // no velocity
        // #define FLAG_HAIR                           0x10 // hair
        // #define FLAG_LEAF                           0x20 // leaf
        // #define FLAG_SKIN                           0x40 // skin
        // #define FLAG_MORPH                          0x80 // morph

        public void SetMask()
        {
            var allRenderers = FindObjectsByType<Renderer>(FindObjectsSortMode.None);
            foreach (var r in allRenderers)
            {
                var materials = r.sharedMaterials;
                var hasTransparent = false;
                var hasOpaque = false;
                // bool isSSS = false;
                foreach (var mat in materials)
                {
                    if (mat != null)
                    {
                        if (mat.renderQueue >= 3000)
                        {
                            hasTransparent = true;
                        }
                        else
                        {
                            hasOpaque = true;
                        }

                        // if (mat.IsKeywordEnabled("_SSS"))
                        // {
                        //     isSSS = true;
                        //     Debug.Log($"Renderer {r.name} marked as SSS.");
                        // }
                    }
                }

                uint mask = 0;

                if (hasOpaque)
                    mask |= 0x01; // FLAG_NON_TRANSPARENT
                if (hasTransparent)
                    mask |= 0x02; // FLAG_TRANSPARENT

                // if (isSSS)
                //     mask |= 0x40; // FLAG_SKIN

                // Debug.Log($"Renderer {r.name} Mask: {mask}");

                _accelerationStructure.UpdateInstanceMask(r, mask); // 1 表示包含在内
            }
        }
    }
}