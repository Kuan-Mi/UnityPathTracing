using System.Collections.Generic;
using System.Runtime.InteropServices;
using DLRR;
using mini;
using Rtxdi;
using RTXDI;
using Rtxdi.DI;
using Rtxdi.GI;
using Rtxdi.ReGIR;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using static UnityEngine.Rendering.RayTracingAccelerationStructure;
using static PathTracing.ShaderIDs;

namespace PathTracing
{
    public class RtxdiFeature : ScriptableRendererFeature
    {
        public RtxdiSetting setting;

        public GlobalConstants     globalConstants;
        public ResamplingConstants resamplingConstants;

        public RenderPassEvent renderPassEvent = RenderPassEvent.BeforeRenderingPostProcessing;

        public Material finalMaterial;

        public RayTracingShader gBufferTracingShader;
        public RayTracingShader brdfRayTracingShader;
        public RayTracingShader shadeSecondarySurfacesShader;
        public ComputeShader    shadeSecondarySurfacesComputeCs;
        public RayTracingShader generateInitialShader;
        public ComputeShader    generateInitialComputeCs;

        // ReSTIR GI resampling & shading shaders
        public RayTracingShader giTemporalResamplingShader;
        public ComputeShader    giTemporalResamplingComputeCs;
        public RayTracingShader giSpatialResamplingShader;
        public ComputeShader    giSpatialResamplingComputeCs;
        public RayTracingShader giFinalShadingShader;
        public ComputeShader    giFinalShadingComputeCs;


        public RayTracingShader temporalResamplingShader;
        public ComputeShader    temporalResamplingComputeCs;
        public RayTracingShader spatialResamplingShader;
        public ComputeShader    spatialResamplingComputeCs;
        public RayTracingShader shadeSamplesShader;
        public ComputeShader    shadeSamplesComputeCs;

        public ComputeShader dlssBeforeCs;
        public ComputeShader pdfTextureCs;
        public ComputeShader presampleCs;
        public ComputeShader presampleReGirCs;
        public ComputeShader genMipsCs;

        private OutputBlitPass             _outputBlitPass;
        private PrepareLightPass           _prepareLightPass;
        private GBufferPass                _gBufferPass;
        private GBufferRasterPass          _gBufferRasterPass;
        private GBufferRasterPass.Resource _gBufferRasterResource;
        private GenerateInitialSamplesPass _generateInitialSamplesPass;
        private TemporalResamplingPass     _temporalResamplingPass;
        private SpatialResamplingPass      _spatialResamplingPass;
        private ShadeSamplesPass           _shadeSamplesPass;
        private PdfTexturePass             _pdfTexturePass;
        private PresamplePass              _presamplePass;
        private PresampleReGirLightsPass   _presampleReGirLightsPass;
        private GenerateMipsPass           _generateMipsPass;
        private BrdfRayTracingPass         _brdfRayTracingPass;
        private ShadeSecondarySurfacesPass _shadeSecondarySurfacesPass;

        // ReSTIR GI passes
        private GITemporalResamplingPass _giTemporalResamplingPass;
        private GISpatialResamplingPass  _giSpatialResamplingPass;
        private GIFinalShadingPass       _giFinalShadingPass;


        private DlssRRPass          _dlssrrPass;
        private RxtdiDlssBeforePass _rtxdiDlssBeforePass;


        private RayTracingAccelerationStructure _accelerationStructure;

        private GraphicsBuffer _constantBuffer;
        private GraphicsBuffer _resamplingConstantBuffer;

        private GPUScene             _gpuScene       = new();
        private PrepareLightResource _prepareLightResources;

        private readonly GlobalConstants[]     _globalConstantsArray     = new GlobalConstants[1];
        private readonly ResamplingConstants[] _resamplingConstantsArray = new ResamplingConstants[1];

        private readonly Dictionary<long, DlrrDenoiser>              _dlrrDenoisers     = new();
        private readonly Dictionary<long, PathTracingResourcePool>   _resourcePools     = new();
        private readonly Dictionary<long, RtxdiResources>            _rtxdiResources    = new();
        private readonly Dictionary<long, ImportanceSamplingContext> _isContexts        = new();
        private readonly Dictionary<long, CameraFrameState>          _cameraFrameStates = new();

        public override void Create()
        {
            if (_accelerationStructure == null)
            {
                var settings = new Settings
                {
                    managementMode     = ManagementMode.Automatic,
                    rayTracingModeMask = RayTracingModeMask.Everything
                };
                _accelerationStructure = new RayTracingAccelerationStructure(settings);
                _accelerationStructure.Build();
                SetMask();
            }

            _gpuScene              ??= new GPUScene();
            _prepareLightResources ??= new PrepareLightResource();

            if (!_gpuScene.isBufferInitialized)
            {
                _gpuScene.InitBuffer();
            }


            if (_constantBuffer == null)
            {
                InitializeBuffers();
            }


            _prepareLightPass ??= new PrepareLightPass()
            {
                renderPassEvent = renderPassEvent
            };
            _gBufferPass ??= new GBufferPass(gBufferTracingShader)
            {
                renderPassEvent = renderPassEvent
            };
            _gBufferRasterPass ??= new GBufferRasterPass
            {
                renderPassEvent = renderPassEvent
            };
            _gBufferRasterResource ??= new GBufferRasterPass.Resource();
            _generateInitialSamplesPass ??= new GenerateInitialSamplesPass(generateInitialShader, generateInitialComputeCs)
            {
                renderPassEvent = renderPassEvent
            };

            _temporalResamplingPass ??= new TemporalResamplingPass(temporalResamplingShader, temporalResamplingComputeCs)
            {
                renderPassEvent = renderPassEvent
            };

            _spatialResamplingPass ??= new SpatialResamplingPass(spatialResamplingShader, spatialResamplingComputeCs)
            {
                renderPassEvent = renderPassEvent
            };

            _shadeSamplesPass ??= new ShadeSamplesPass(shadeSamplesShader, shadeSamplesComputeCs)
            {
                renderPassEvent = renderPassEvent
            };

            _generateMipsPass ??= new GenerateMipsPass(genMipsCs)
            {
                renderPassEvent = renderPassEvent
            };

            _brdfRayTracingPass ??= new BrdfRayTracingPass(brdfRayTracingShader)
            {
                renderPassEvent = renderPassEvent
            };

            _shadeSecondarySurfacesPass ??= new ShadeSecondarySurfacesPass(shadeSecondarySurfacesShader, shadeSecondarySurfacesComputeCs)
            {
                renderPassEvent = renderPassEvent
            };

            _giTemporalResamplingPass ??= new GITemporalResamplingPass(giTemporalResamplingShader, giTemporalResamplingComputeCs)
            {
                renderPassEvent = renderPassEvent
            };

            _giSpatialResamplingPass ??= new GISpatialResamplingPass(giSpatialResamplingShader, giSpatialResamplingComputeCs)
            {
                renderPassEvent = renderPassEvent
            };

            _giFinalShadingPass ??= new GIFinalShadingPass(giFinalShadingShader, giFinalShadingComputeCs)
            {
                renderPassEvent = renderPassEvent
            };

            _pdfTexturePass ??= new PdfTexturePass(pdfTextureCs)
            {
                renderPassEvent = renderPassEvent
            };

            _presamplePass ??= new PresamplePass(presampleCs)
            {
                renderPassEvent = renderPassEvent
            };

            _presampleReGirLightsPass ??= new PresampleReGirLightsPass(presampleReGirCs)
            {
                renderPassEvent = renderPassEvent
            };


            _rtxdiDlssBeforePass ??= new RxtdiDlssBeforePass(dlssBeforeCs)
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
        }

        public static readonly int Capacity = 1 << 23;

        public void InitializeBuffers()
        {
            _constantBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Constant, 1, Marshal.SizeOf<GlobalConstants>());

            // var size = Marshal.SizeOf<ResamplingConstants>();
            // Debug.Log($"ResamplingConstants size: {size} bytes");
            _resamplingConstantBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, Marshal.SizeOf<ResamplingConstants>());
        }


        [System.Serializable]
        public struct RenderSettings
        {
            public uint denoiserMode;

            public bool enablePreviousTLAS;
            public bool enableAlphaTestedGeometry;
            public bool enableTransparentGeometry;
            public bool enableRayCounts;
            public bool visualizeRegirCells;

            public bool  enableGradients;
            public float gradientLogDarknessBias;
            public float gradientSensitivity;
            public float confidenceHistoryLength;

            public BRDFPathTracing_Parameters brdfptParams;

            public static RenderSettings Default()
            {
                return new RenderSettings
                {
                    denoiserMode = 0,

                    enablePreviousTLAS        = true,
                    enableAlphaTestedGeometry = true,
                    enableTransparentGeometry = true,
                    enableRayCounts           = true,
                    visualizeRegirCells       = false,

                    enableGradients         = true,
                    gradientLogDarknessBias = -12.0f,
                    gradientSensitivity     = 8.0f,
                    confidenceHistoryLength = 0.75f,

                    brdfptParams = BRDFPathTracing_Parameters.Default()
                };
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

            _gpuScene.Build(_accelerationStructure, setting.enableEnv);
            // _gpuScene.UpdateInstanceID(_accelerationStructure);

            Shader.SetGlobalRayTracingAccelerationStructure(g_AccelStructID, _accelerationStructure);

            var uniqueKey = cam.GetInstanceID() + (eyeIndex * 100000L);


            var isVR = renderingData.cameraData.xrRendering;

            if (!_resourcePools.TryGetValue(uniqueKey, out var pool))
            {
                pool = new PathTracingResourcePool();
                pool.InitRtxdiResources();
                _resourcePools.Add(uniqueKey, pool);
            }

            if (!_dlrrDenoisers.TryGetValue(uniqueKey, out var dlrr))
            {
                var camName = cam.name;
                if (isVR)
                {
                    camName = $"{cam.name}_Eye{eyeIndex}";
                }

                dlrr = new DlrrDenoiser(camName);
                _dlrrDenoisers.Add(uniqueKey, dlrr);
            }

            if (!_cameraFrameStates.TryGetValue(uniqueKey, out var frameState))
            {
                frameState = new CameraFrameState(1);
                _cameraFrameStates.Add(uniqueKey, frameState);
            }

            if (!_isContexts.TryGetValue(uniqueKey, out var isContext))
            {
                var isParams = ImportanceSamplingContext_StaticParameters.Default();
                isParams.renderWidth  = (uint)cam.pixelWidth;
                isParams.renderHeight = (uint)cam.pixelHeight;
                isContext             = new ImportanceSamplingContext(isParams);
                _isContexts.Add(uniqueKey, isContext);
            }

            if (!_rtxdiResources.TryGetValue(uniqueKey, out var rtxdiResources))
            {
                rtxdiResources = new RtxdiResources(isContext.GetReSTIRDIContext(), isContext.GetRISBufferSegmentAllocator(), _gpuScene);
                _rtxdiResources.Add(uniqueKey, rtxdiResources);
            }


            if (finalMaterial == null
                || gBufferTracingShader == null
                || dlssBeforeCs == null
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

            var  outputResolution = ComputeOutputResolution(renderingData.cameraData);
            bool resourcesChanged = pool.EnsureResources(outputResolution, setting.upscalerMode);

            if (resourcesChanged)
            {
                frameState.renderResolution = pool.renderResolution;
                frameState.frameIndex       = 0;
            }

            // Update per-camera temporal state for this frame
            uint curFrame = frameState.frameIndex;
            frameState.Update(renderingData, false, 1);

            globalConstants          = frameState.GetConstants(renderingData, setting);
            _globalConstantsArray[0] = globalConstants;
            _constantBuffer.SetData(_globalConstantsArray);


            var lightingSettings = setting.lightingSettings;
            lightingSettings.enablePreviousTLAS        = false;
            lightingSettings.enableAlphaTestedGeometry = false;
            lightingSettings.enableTransparentGeometry = false;

            lightingSettings.denoiserMode    = 0;
            lightingSettings.enableGradients = false;

            bool checkerboard = false;

            bool enableDirectReStirPass    = setting.directLightingMode == DirectLightingMode.ReStir;
            bool enableBrdfAndIndirectPass = setting.directLightingMode == DirectLightingMode.Brdf || setting.indirectLightingMode != IndirectLightingMode.None;
            bool enableIndirect            = setting.indirectLightingMode != IndirectLightingMode.None;


            RTXDI_ShadingParameters restirDIShadingParams = isContext.GetReSTIRDIContext().GetShadingParameters();
            restirDIShadingParams.enableDenoiserInputPacking = !enableIndirect ? 1u : 0u;
            isContext.GetReSTIRDIContext().SetShadingParameters(restirDIShadingParams);

            if (!enableDirectReStirPass)
            {
                lightingSettings.brdfptParams.enableSecondaryResampling = 0u;
                lightingSettings.enableGradients                        = false;
            }


            #region prepareLight

            if (enableDirectReStirPass || enableIndirect)
            {
                _prepareLightResources.SetBuffer(_gpuScene);
                _prepareLightResources.SendTexture(_gpuScene.globalTexturePool);
                _prepareLightPass.Setup(_prepareLightResources);
                renderer.EnqueuePass(_prepareLightPass);
            }


            bool enableReSTIRGI = setting.indirectLightingMode == IndirectLightingMode.ReStirGI;

            bool enableAdditiveBlend    = enableDirectReStirPass;
            bool enableEmissiveSurfaces = setting.directLightingMode == DirectLightingMode.Brdf;
            bool enableAccumulation     = false;

            resamplingConstants = GetResamplingConstants(isContext, lightingSettings, frameState, enableIndirect, enableAdditiveBlend, enableEmissiveSurfaces, enableAccumulation, enableReSTIRGI);

            _resamplingConstantsArray[0] = resamplingConstants;
            _resamplingConstantBuffer.SetData(_resamplingConstantsArray);

            #endregion

            var isOddFrame        = (curFrame % 2) == 1;
            var viewDepth         = isOddFrame ? pool.GetRT(RenderResourceType.RtxdiViewDepth) : pool.GetRT(RenderResourceType.RtxdiPrevViewDepth);
            var diffuseAlbedo     = isOddFrame ? pool.GetRT(RenderResourceType.RtxdiDiffuseAlbedo) : pool.GetRT(RenderResourceType.RtxdiPrevDiffuseAlbedo);
            var specularRough     = isOddFrame ? pool.GetRT(RenderResourceType.RtxdiSpecularRough) : pool.GetRT(RenderResourceType.RtxdiPrevSpecularRough);
            var normals           = isOddFrame ? pool.GetRT(RenderResourceType.RtxdiNormals) : pool.GetRT(RenderResourceType.RtxdiPrevNormals);
            var geoNormals        = isOddFrame ? pool.GetRT(RenderResourceType.RtxdiGeoNormals) : pool.GetRT(RenderResourceType.RtxdiPrevGeoNormals);
            var prevViewDepth     = isOddFrame ? pool.GetRT(RenderResourceType.RtxdiPrevViewDepth) : pool.GetRT(RenderResourceType.RtxdiViewDepth);
            var prevDiffuseAlbedo = isOddFrame ? pool.GetRT(RenderResourceType.RtxdiPrevDiffuseAlbedo) : pool.GetRT(RenderResourceType.RtxdiDiffuseAlbedo);
            var prevSpecularRough = isOddFrame ? pool.GetRT(RenderResourceType.RtxdiPrevSpecularRough) : pool.GetRT(RenderResourceType.RtxdiSpecularRough);
            var prevNormals       = isOddFrame ? pool.GetRT(RenderResourceType.RtxdiPrevNormals) : pool.GetRT(RenderResourceType.RtxdiNormals);
            var prevGeoNormals    = isOddFrame ? pool.GetRT(RenderResourceType.RtxdiPrevGeoNormals) : pool.GetRT(RenderResourceType.RtxdiGeoNormals);

            var rtxdiCtx = new RtxdiPassContext
            {
                ConstantBuffer           = _constantBuffer,
                ResamplingConstantBuffer = _resamplingConstantBuffer,
                GeometryInstanceToLight  = _gpuScene._geometryInstanceToLight,
                ViewDepth                = viewDepth,
                DiffuseAlbedo            = diffuseAlbedo,
                SpecularRough            = specularRough,
                Normals                  = normals,
                GeoNormals               = geoNormals,
                PrevViewDepth            = prevViewDepth,
                PrevDiffuseAlbedo        = prevDiffuseAlbedo,
                PrevSpecularRough        = prevSpecularRough,
                PrevNormals              = prevNormals,
                PrevGeoNormals           = prevGeoNormals,
                DirectLighting           = pool.GetRT(RenderResourceType.DirectLighting),
                Emissive                 = pool.GetRT(RenderResourceType.RtxdiEmissive),
                MotionVectors            = pool.GetRT(RenderResourceType.RtxdiMotionVectors),
                LocalLightPdfTexture     = _gpuScene.localLightPdfTexture,
                RtxdiResources           = rtxdiResources,
                RenderResolution         = new int2(cam.pixelWidth, cam.pixelHeight),
                ResolutionScale          = 1,
            };

            #region GBuffer

            if (setting.useRasterGBuffer)
            {
                _gBufferRasterResource.EnsureResources(pool.renderResolution);
                _gBufferRasterPass.Setup(rtxdiCtx, _gBufferRasterResource);
                renderer.EnqueuePass(_gBufferRasterPass);
            }
            else
            {
                _gBufferPass.Setup(rtxdiCtx);
                renderer.EnqueuePass(_gBufferPass);
            }

            #endregion

            #region PreSample

            _pdfTexturePass.Setup(rtxdiCtx);
            renderer.EnqueuePass(_pdfTexturePass);

            _generateMipsPass.Setup(rtxdiCtx);
            renderer.EnqueuePass(_generateMipsPass);

            if (setting.initialSamplingParams.localLightSamplingMode != ReSTIRDI_LocalLightSamplingMode.Uniform)
            {
                var RTXDI_PRESAMPLING_GROUP_SIZE = 256;

                var x = (isContext.GetLocalLightRISBufferSegmentParams().tileSize + RTXDI_PRESAMPLING_GROUP_SIZE - 1) / RTXDI_PRESAMPLING_GROUP_SIZE;
                var y = isContext.GetLocalLightRISBufferSegmentParams().tileCount;

                _presamplePass.Setup(rtxdiCtx, (int)x, (int)y);
                renderer.EnqueuePass(_presamplePass);
            }

            if (setting.initialSamplingParams.localLightSamplingMode == ReSTIRDI_LocalLightSamplingMode.ReGIR_RIS)
            {
                var ReGIR_TILE_SIZE = 256;

                var regirContext = isContext.GetReGIRContext();
                var x            = (regirContext.GetReGIRLightSlotCount() + ReGIR_TILE_SIZE - 1) / ReGIR_TILE_SIZE;

                _presampleReGirLightsPass.Setup(rtxdiCtx, (int)x, 1);
                renderer.EnqueuePass(_presampleReGirLightsPass);
            }

            #endregion

            #region DI

            if (enableDirectReStirPass)
            {
                // GenerateInitialSamplesPass
                _generateInitialSamplesPass.Setup(rtxdiCtx, setting.useComputeForGis);
                renderer.EnqueuePass(_generateInitialSamplesPass);

                // TemporalResamplingPass
                if (setting.diResamplingMode is ReSTIRDI_ResamplingMode.Temporal or ReSTIRDI_ResamplingMode.TemporalAndSpatial)
                {
                    _temporalResamplingPass.Setup(rtxdiCtx, setting.useComputeForTemporalResampling);
                    renderer.EnqueuePass(_temporalResamplingPass);
                }

                // SpatialResamplingPass
                if (setting.diResamplingMode is ReSTIRDI_ResamplingMode.Spatial or ReSTIRDI_ResamplingMode.TemporalAndSpatial)
                {
                    _spatialResamplingPass.Setup(rtxdiCtx, setting.useComputeForSpatialResampling);
                    renderer.EnqueuePass(_spatialResamplingPass);
                }

                // ShadeSamplesPass
                _shadeSamplesPass.Setup(rtxdiCtx, setting.enableDIFinalShading, setting.useComputeForShadeSamples);
                renderer.EnqueuePass(_shadeSamplesPass);
            }

            #endregion

            #region GI

            if (enableBrdfAndIndirectPass)
            {
                _brdfRayTracingPass.Setup(rtxdiCtx);
                renderer.EnqueuePass(_brdfRayTracingPass);

                if (enableIndirect)
                {
                    _shadeSecondarySurfacesPass.Setup(rtxdiCtx, setting.useComputeForShadeSecondarySurfaces);
                    renderer.EnqueuePass(_shadeSecondarySurfacesPass);

                    if (enableReSTIRGI)
                    {
                        // ReSTIR GI – Temporal Resampling
                        if (setting.giResamplingMode is ReSTIRGI_ResamplingMode.Temporal or ReSTIRGI_ResamplingMode.TemporalAndSpatial)
                        {
                            _giTemporalResamplingPass.Setup(rtxdiCtx, setting.useComputeForGITemporalResampling);
                            renderer.EnqueuePass(_giTemporalResamplingPass);
                        }

                        // ReSTIR GI – Spatial Resampling
                        if (setting.giResamplingMode is ReSTIRGI_ResamplingMode.Spatial or ReSTIRGI_ResamplingMode.TemporalAndSpatial)
                        {
                            _giSpatialResamplingPass.Setup(rtxdiCtx, setting.useComputeForGISpatialResampling);
                            renderer.EnqueuePass(_giSpatialResamplingPass);
                        }

                        // ReSTIR GI – Final Shading
                        _giFinalShadingPass.Setup(rtxdiCtx, setting.enableGIFinalShading, setting.useComputeForGIFinalShading);
                        renderer.EnqueuePass(_giFinalShadingPass);
                    }
                }
            }

            #endregion

            #region DLSS

            var dlrrRes = new DlrrDenoiser.DlrrResources
            {
                input           = pool.GetNriResource(RenderResourceType.DirectLighting),
                output          = pool.GetNriResource(RenderResourceType.DlssOutput),
                mv              = pool.GetNriResource(RenderResourceType.RtxdiMotionVectors),
                depth           = pool.GetNriResource(isOddFrame ? RenderResourceType.RtxdiViewDepth : RenderResourceType.RtxdiPrevViewDepth),
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
            var dlssDataPtr = dlrr.GetInteropDataPtr(dlrrInput, dlrrRes, 1, setting.upscalerMode);

            var rectGridW = (int)(cam.pixelWidth * 1 + 0.5f + 15) / 16;
            var rectGridH = (int)(cam.pixelHeight * 1 + 0.5f + 15) / 16;

            _rtxdiDlssBeforePass.Setup(rtxdiCtx,
                pool.GetRT(RenderResourceType.RrGuideDiffAlbedo),
                pool.GetRT(RenderResourceType.RrGuideSpecAlbedo),
                pool.GetRT(RenderResourceType.RrGuideSpecHitDistance),
                pool.GetRT(RenderResourceType.RrGuideNormalRoughness),
                rectGridW, rectGridH);
            renderer.EnqueuePass(_rtxdiDlssBeforePass);

            var dlssSettings = new DlssRRPass.Settings
            {
                tmpDisableRR = setting.tmpDisableRR
            };

            _dlssrrPass.Setup(dlssDataPtr, dlssSettings);
            renderer.EnqueuePass(_dlssrrPass);

            #endregion

            var outputBlitResource = new OutputBlitPass.Resource
            {
                Mv                       = pool.GetRT(RenderResourceType.RtxdiMotionVectors),
                DirectLighting           = pool.GetRT(RenderResourceType.DirectLighting),
                RRGuide_DiffAlbedo       = pool.GetRT(RenderResourceType.RrGuideDiffAlbedo),
                RRGuide_SpecAlbedo       = pool.GetRT(RenderResourceType.RrGuideSpecAlbedo),
                RRGuide_Normal_Roughness = pool.GetRT(RenderResourceType.RrGuideNormalRoughness),
                RRGuide_SpecHitDistance  = pool.GetRT(RenderResourceType.RrGuideSpecHitDistance),
                DlssOutput               = pool.GetRT(RenderResourceType.DlssOutput),
            };

            var outputBlitSettings = new OutputBlitPass.Settings
            {
                showMode        = setting.showMode,
                resolutionScale = frameState.resolutionScale,
                enableDlssRR    = true,
                tmpDisableRR    = setting.tmpDisableRR,
                showMV          = setting.showMv,
                showValidation  = false,
                showReference   = false,
            };

            _outputBlitPass.Setup(outputBlitResource, outputBlitSettings);
            renderer.EnqueuePass(_outputBlitPass);
        }


        void FillReSTIRDIConstants(ref RTXDI_Parameters rparams, ReSTIRDIContext restirDIContext, RTXDI_LightBufferParameters lightBufferParameters)
        {
            rparams.reservoirBufferParams = restirDIContext.GetReservoirBufferParameters();
            rparams.bufferIndices         = restirDIContext.GetBufferIndices();
            rparams.initialSamplingParams = restirDIContext.GetInitialSamplingParameters();

            rparams.initialSamplingParams.environmentMapImportanceSampling = lightBufferParameters.environmentLightParams.lightPresent;
            if (rparams.initialSamplingParams.environmentMapImportanceSampling == 0)
                rparams.initialSamplingParams.numEnvironmentSamples = 0;
            rparams.temporalResamplingParams = restirDIContext.GetTemporalResamplingParameters();
            rparams.spatialResamplingParams  = restirDIContext.GetSpatialResamplingParameters();
            rparams.shadingParams            = restirDIContext.GetShadingParameters();
        }

        void FillReSTIRGIConstants(ref RTXDI_GIParameters constants, ReSTIRGIContext restirGIContext)
        {
            constants.reservoirBufferParams    = restirGIContext.GetReservoirBufferParameters();
            constants.bufferIndices            = restirGIContext.GetBufferIndices();
            constants.temporalResamplingParams = restirGIContext.GetTemporalResamplingParameters();
            constants.spatialResamplingParams  = restirGIContext.GetSpatialResamplingParameters();
            constants.finalShadingParams       = restirGIContext.GetFinalShadingParameters();
        }

        void FillReGIRConstants(ref ReGIR_Parameters ReGIRParams, ReGIRContext regirContext)
        {
            var staticParams  = regirContext.GetReGIRStaticParameters();
            var dynamicParams = regirContext.GetReGIRDynamicParameters();
            var gridParams    = regirContext.GetReGIRGridCalculatedParameters();
            var onionParams   = regirContext.GetReGIROnionCalculatedParameters();

            ReGIRParams.gridParams.cellsX = staticParams.gridParameters.GridSize.x;
            ReGIRParams.gridParams.cellsY = staticParams.gridParameters.GridSize.y;
            ReGIRParams.gridParams.cellsZ = staticParams.gridParameters.GridSize.z;

            ReGIRParams.commonParams.numRegirBuildSamples = dynamicParams.regirNumBuildSamples;
            ReGIRParams.commonParams.risBufferOffset      = regirContext.GetReGIRCellOffset();
            ReGIRParams.commonParams.lightsPerCell        = staticParams.LightsPerCell;
            ReGIRParams.commonParams.centerX              = dynamicParams.center.x;
            ReGIRParams.commonParams.centerY              = dynamicParams.center.y;
            ReGIRParams.commonParams.centerZ              = dynamicParams.center.z;
            ReGIRParams.commonParams.cellSize = (staticParams.Mode == ReGIRMode.Onion)
                ? dynamicParams.regirCellSize * 0.5f // Onion operates with radii, while "size" feels more like diameter
                : dynamicParams.regirCellSize;
            ReGIRParams.commonParams.localLightSamplingFallbackMode = (uint)(dynamicParams.fallbackSamplingMode);
            ReGIRParams.commonParams.localLightPresamplingMode      = (uint)(dynamicParams.presamplingMode);
            ReGIRParams.commonParams.samplingJitter                 = math.max(0.0f, dynamicParams.regirSamplingJitter * 2.0f);
            ReGIRParams.onionParams.cubicRootFactor                 = onionParams.regirOnionCubicRootFactor;
            ReGIRParams.onionParams.linearFactor                    = onionParams.regirOnionLinearFactor;
            ReGIRParams.onionParams.numLayerGroups                  = (uint)(onionParams.regirOnionLayers.Count);

            // assert(onionParams.regirOnionLayers.size() <= RTXDI_ONION_MAX_LAYER_GROUPS);
            for (int group = 0; group < onionParams.regirOnionLayers.Count; group++)
            {
                var layer = onionParams.regirOnionLayers[group];
                layer.innerRadius *= ReGIRParams.commonParams.cellSize;
                layer.outerRadius *= ReGIRParams.commonParams.cellSize;


                ReGIRParams.onionParams.SetLayer(group, layer);
            }

            // assert(onionParams.regirOnionRings.size() <= RTXDI_ONION_MAX_RINGS);
            for (int n = 0; n < (onionParams.regirOnionRings.Count); n++)
            {
                // ReGIRParams.onionParams.rings[n] = onionParams.regirOnionRings[n];
                ReGIRParams.onionParams.SetRing(n, onionParams.regirOnionRings[n]);
            }

            ReGIRParams.onionParams.cubicRootFactor = regirContext.GetReGIROnionCalculatedParameters().regirOnionCubicRootFactor;
        }

        private void FillResamplingConstants(ref ResamplingConstants constants, RenderSettings localSettings, ImportanceSamplingContext isContext)
        {
            // RTXDI_LightBufferParameters lightBufferParameters = isContext.GetLightBufferParameters();

            constants.enablePreviousTLAS = localSettings.denoiserMode;
            constants.denoiserMode       = localSettings.denoiserMode;
            // constants.sceneConstants.enableAlphaTestedGeometry = lightingSettings.enableAlphaTestedGeometry;
            // constants.sceneConstants.enableTransparentGeometry = lightingSettings.enableTransparentGeometry;
            constants.visualizeRegirCells = localSettings.visualizeRegirCells ? 1u : 0u;

            constants.lightBufferParams                      = isContext.GetLightBufferParameters();
            constants.localLightsRISBufferSegmentParams      = isContext.GetLocalLightRISBufferSegmentParams();
            constants.environmentLightRISBufferSegmentParams = isContext.GetEnvironmentLightRISBufferSegmentParams();
            constants.runtimeParams                          = isContext.GetReSTIRDIContext().GetRuntimeParams();

            FillReSTIRDIConstants(ref constants.restirDI, isContext.GetReSTIRDIContext(), isContext.GetLightBufferParameters());
            FillReGIRConstants(ref constants.regir, isContext.GetReGIRContext());
            FillReSTIRGIConstants(ref constants.restirGI, isContext.GetReSTIRGIContext());


            constants.localLightPdfTextureSize = _gpuScene.localLightPdfTextureSize;

            // if (lightBufferParameters.environmentLightParams.lightPresent)
            // {
            //     constants.environmentPdfTextureSize = m_environmentPdfTextureSize;
            // }

            // m_currentFrameOutputReservoir = isContext.GetReSTIRDIContext().GetBufferIndices().shadingInputBufferIndex;
        }

        private void FillBRDFPTConstants(ref BRDFPathTracing_Parameters constants, RenderSettings localSettings, RTXDI_LightBufferParameters getLightBufferParameters)
        {
            constants = setting.brdfptParams;

            constants.materialOverrideParams.minSecondaryRoughness = 0;
            constants.materialOverrideParams.roughnessOverride     = -1.0f;
            constants.materialOverrideParams.metalnessOverride     = -1.0f;

            constants.secondarySurfaceReSTIRDIParams.initialSamplingParams.environmentMapImportanceSampling = 0;
            if (constants.secondarySurfaceReSTIRDIParams.initialSamplingParams.environmentMapImportanceSampling == 0)
                constants.secondarySurfaceReSTIRDIParams.initialSamplingParams.numEnvironmentSamples = 0;
        }


        private ResamplingConstants GetResamplingConstants(
            ImportanceSamplingContext isContext,
            RenderSettings localSettings,
            CameraFrameState frameState,
            bool enableIndirect,
            bool enableAdditiveBlend,
            bool enableEmissiveSurfaces,
            bool enableAccumulation,
            bool enableReSTIRGI)
        {
            var restirDIContext = isContext.GetReSTIRDIContext();
            var restirGIContext = isContext.GetReSTIRGIContext();


            RTXDI_LightBufferParameters lightBufferParams = _gpuScene.GetLightBufferParameters();
            isContext.SetLightBufferParams(lightBufferParams);
            restirDIContext.SetFrameIndex(frameState.frameIndex);
            restirDIContext.SetResamplingMode(setting.diResamplingMode);
            restirDIContext.SetInitialSamplingParameters(setting.initialSamplingParams);
            restirDIContext.SetTemporalResamplingParameters(setting.temporalResamplingParams);
            restirDIContext.SetSpatialResamplingParameters(setting.spatialResamplingParams);
            restirDIContext.SetShadingParameters(setting.shadingParams);


            restirGIContext.SetFrameIndex(frameState.frameIndex);
            restirGIContext.SetResamplingMode(setting.giResamplingMode);
            restirGIContext.SetTemporalResamplingParameters(setting.giTemporalResamplingParams);
            restirGIContext.SetSpatialResamplingParameters(setting.giSpatialResamplingParams);
            restirGIContext.SetFinalShadingParameters(setting.giFinalShadingParams);


            var regirContext = isContext.GetReGIRContext();
            setting.regirDynamicParams.center = frameState.camPos;
            regirContext.SetDynamicParameters(setting.regirDynamicParams);


            var constants = new ResamplingConstants();

            constants.frameIndex   = restirDIContext.GetFrameIndex();
            constants.denoiserMode = localSettings.denoiserMode;

            constants.enableBrdfIndirect      = enableIndirect ? 1u : 0u;
            constants.enableBrdfAdditiveBlend = enableAdditiveBlend ? 1u : 0u;
            constants.enableAccumulation      = enableAccumulation ? 1u : 0u;

            // constants.sceneConstants.enableEnvironmentMap = (environmentLight.textureIndex >= 0);
            // constants.sceneConstants.environmentMapTextureIndex = (environmentLight.textureIndex >= 0) ? environmentLight.textureIndex : 0;
            // constants.sceneConstants.environmentScale = environmentLight.radianceScale.x;
            // constants.sceneConstants.environmentRotation = environmentLight.rotation;


            FillResamplingConstants(ref constants, localSettings, isContext);
            FillBRDFPTConstants(ref constants.brdfPT, localSettings, isContext.GetLightBufferParameters());

            constants.brdfPT = setting.brdfptParams;

            constants.brdfPT.enableIndirectEmissiveSurfaces = enableEmissiveSurfaces ? 1u : 0u;
            constants.brdfPT.enableReSTIRGI                 = enableReSTIRGI ? 1u : 0u;

            // ReSTIRGI_BufferIndices restirGIBufferIndices = restirGIContext.GetBufferIndices();
            // m_currentFrameGIOutputReservoir = restirGIBufferIndices.finalShadingInputBufferIndex;

            return constants;
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
            Debug.Log("PathTracingFeature Dispose");
            base.Dispose(disposing);

            _accelerationStructure?.Dispose();
            _accelerationStructure = null;

            _constantBuffer?.Release();
            _constantBuffer = null;

            _resamplingConstantBuffer?.Release();
            _resamplingConstantBuffer = null;

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

            _gpuScene?.Dispose();
            _gpuScene = null;

            _prepareLightResources?.Dispose();
            _prepareLightResources = null;

            _prepareLightPass = null;
            _gBufferPass      = null;
            _dlssrrPass       = null;
            _outputBlitPass   = null;
        }

        public void Test()
        {
            _gpuScene.DebugReadback();
        }

#if UNITY_EDITOR
        private void Reset()
        {
            setting = new RtxdiSetting();
            AutoFillShaders();
        }

        public void AutoFillShaders()
        {
            finalMaterial = UnityEditor.AssetDatabase.LoadAssetAtPath<Material>("Assets/Shaders/Mat/KM_Final.mat");

            gBufferTracingShader         = UnityEditor.AssetDatabase.LoadAssetAtPath<RayTracingShader>("Assets/Shaders/RayTracing/GBuffer.raytrace");
            brdfRayTracingShader         = UnityEditor.AssetDatabase.LoadAssetAtPath<RayTracingShader>("Assets/Shaders/RayTracing/GI/BrdfRayTracing.raytrace");
            shadeSecondarySurfacesShader = UnityEditor.AssetDatabase.LoadAssetAtPath<RayTracingShader>("Assets/Shaders/RayTracing/GI/ShadeSecondarySurfaces.raytrace");
            generateInitialShader        = UnityEditor.AssetDatabase.LoadAssetAtPath<RayTracingShader>("Assets/Shaders/RayTracing/DI/GenerateInitialSamples.raytrace");
            giTemporalResamplingShader   = UnityEditor.AssetDatabase.LoadAssetAtPath<RayTracingShader>("Assets/Shaders/RayTracing/GI/TemporalResampling.raytrace");
            giSpatialResamplingShader    = UnityEditor.AssetDatabase.LoadAssetAtPath<RayTracingShader>("Assets/Shaders/RayTracing/GI/SpatialResampling.raytrace");
            giFinalShadingShader         = UnityEditor.AssetDatabase.LoadAssetAtPath<RayTracingShader>("Assets/Shaders/RayTracing/GI/FinalShading.raytrace");
            temporalResamplingShader     = UnityEditor.AssetDatabase.LoadAssetAtPath<RayTracingShader>("Assets/Shaders/RayTracing/DI/TemporalResampling.raytrace");
            spatialResamplingShader      = UnityEditor.AssetDatabase.LoadAssetAtPath<RayTracingShader>("Assets/Shaders/RayTracing/DI/SpatialResampling.raytrace");
            shadeSamplesShader           = UnityEditor.AssetDatabase.LoadAssetAtPath<RayTracingShader>("Assets/Shaders/RayTracing/DI/ShadeSamples.raytrace");

            shadeSecondarySurfacesComputeCs = UnityEditor.AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/RayTracing/GI/ShadeSecondarySurfaces.compute");
            generateInitialComputeCs        = UnityEditor.AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/RayTracing/DI/GenerateInitialSamples.compute");
            giTemporalResamplingComputeCs   = UnityEditor.AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/RayTracing/GI/TemporalResampling.compute");
            giSpatialResamplingComputeCs    = UnityEditor.AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/RayTracing/GI/SpatialResampling.compute");
            giFinalShadingComputeCs         = UnityEditor.AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/RayTracing/GI/FinalShading.compute");
            temporalResamplingComputeCs     = UnityEditor.AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/RayTracing/DI/TemporalResampling.compute");
            spatialResamplingComputeCs      = UnityEditor.AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/RayTracing/DI/SpatialResampling.compute");
            shadeSamplesComputeCs           = UnityEditor.AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/RayTracing/DI/ShadeSamples.compute");
            dlssBeforeCs                    = UnityEditor.AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/PostProcess/RtxdiDlssBefore.compute");
            pdfTextureCs                    = UnityEditor.AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/RayTracing/Presampling/PdfTexture.compute");
            presampleCs                     = UnityEditor.AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/RayTracing/Presampling/Presample.compute");
            presampleReGirCs                = UnityEditor.AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/RayTracing/Presampling/PresampleReGIR.compute");
            genMipsCs                       = UnityEditor.AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shaders/RayTracing/DI/GenerateMips.compute");

            UnityEditor.EditorUtility.SetDirty(this);
        }
#endif

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
                var materials      = r.sharedMaterials;
                var hasTransparent = false;
                var hasOpaque      = false;
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