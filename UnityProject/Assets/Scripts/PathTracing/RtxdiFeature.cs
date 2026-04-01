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
using UnityEngine.Serialization;
using static UnityEngine.Rendering.RayTracingAccelerationStructure;
using static PathTracing.ShaderIDs;

namespace PathTracing
{
    public class RtxdiFeature : ScriptableRendererFeature
    {
        public PathTracingSetting pathTracingSetting;

        public GlobalConstants globalConstants;
        public ResamplingConstants resamplingConstants;

        public RenderPassEvent renderPassEvent = RenderPassEvent.BeforeRenderingPostProcessing;

        public Material finalMaterial;

        public RayTracingShader gBufferTracingShader;
        public RayTracingShader brdfRayTracingShader;
        public RayTracingShader shadeSecondarySurfacesShader;
        public RayTracingShader generateInitialShader;
        public ComputeShader generateInitialComputeCs;

        // ReSTIR GI resampling & shading shaders
        public RayTracingShader giTemporalResamplingShader;
        public ComputeShader giTemporalResamplingComputeCs;
        public RayTracingShader giSpatialResamplingShader;
        public ComputeShader giSpatialResamplingComputeCs;
        public RayTracingShader giFinalShadingShader;
        public ComputeShader giFinalShadingComputeCs;


        public RayTracingShader temporalResamplingShader;
        public ComputeShader temporalResamplingComputeCs;
        public RayTracingShader spatialResamplingShader;
        public ComputeShader spatialResamplingComputeCs;
        public RayTracingShader shadeSamplesShader;
        public ComputeShader shadeSamplesComputeCs;

        public ComputeShader dlssBeforeCs;
        public ComputeShader pdfTextureCs;
        public ComputeShader presampleCs;
        public ComputeShader presampleReGirCs;
        public ComputeShader genMipsCs;

        private OutputBlitPass _outputBlitPass;

        private PrepareLightPass _prepareLightPass;
        private GBufferPass _gBufferPass;
        private GBufferRasterPass _gBufferRasterPass;
        private GBufferRasterPass.Resource _gBufferRasterResource;
        private GenerateInitialSamplesPass _generateInitialSamplesPass;
        private GenerateInitialSamplesComputePass _generateInitialSamplesComputePass;
        private TemporalResamplingPass _temporalResamplingPass;
        private TemporalResamplingComputePass _temporalResamplingComputePass;
        private SpatialResamplingPass _spatialResamplingPass;
        private SpatialResamplingComputePass _spatialResamplingComputePass;
        private ShadeSamplesPass _shadeSamplesPass;
        private ShadeSamplesComputePass _shadeSamplesComputePass;
        private PdfTexturePass _pdfTexturePass;
        private PresamplePass _presamplePass;
        private PresampleReGirLightsPass _presampleReGirLightsPass;
        private GenerateMipsPass _generateMipsPass;

        private BrdfRayTracingPass _brdfRayTracingPass;
        private ShadeSecondarySurfacesPass _shadeSecondarySurfacesPass;

        // ReSTIR GI passes
        private GITemporalResamplingPass _giTemporalResamplingPass;
        private GITemporalResamplingComputePass _giTemporalResamplingComputePass;
        private GISpatialResamplingPass _giSpatialResamplingPass;
        private GISpatialResamplingComputePass _giSpatialResamplingComputePass;
        private GIFinalShadingPass _giFinalShadingPass;
        private GIFinalShadingComputePass _giFinalShadingComputePass;


        private DlssRRPass _dlssrrPass;
        private RxtdiDlssBeforePass _rtxdiDlssBeforePass;


        private RayTracingAccelerationStructure _accelerationStructure;

        private GraphicsBuffer _constantBuffer;
        private GraphicsBuffer _resamplingConstantBuffer;

        private GPUScene _gpuScene = new();
        private LightCollector _lightCollector = new();
        private PrepareLightResource _prepareLightResources;

        private readonly GlobalConstants[] _globalConstantsArray = new GlobalConstants[1];
        private readonly ResamplingConstants[] _resamplingConstantsArray = new ResamplingConstants[1];

        private readonly Dictionary<long, DlrrDenoiser> _dlrrDenoisers = new();

        private readonly Dictionary<long, PathTracingResourcePool> _resourcePools = new();

        private readonly Dictionary<long, RtxdiResources> _rtxdiResources = new();
        private readonly Dictionary<long, ImportanceSamplingContext> _isContexts = new();
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
            _generateInitialSamplesPass ??= new GenerateInitialSamplesPass(generateInitialShader)
            {
                renderPassEvent = renderPassEvent
            };
            _generateInitialSamplesComputePass ??= new GenerateInitialSamplesComputePass(generateInitialComputeCs)
            {
                renderPassEvent = renderPassEvent
            };

            _temporalResamplingPass ??= new TemporalResamplingPass(temporalResamplingShader)
            {
                renderPassEvent = renderPassEvent
            };
            _temporalResamplingComputePass ??= new TemporalResamplingComputePass(temporalResamplingComputeCs)
            {
                renderPassEvent = renderPassEvent
            };

            _spatialResamplingPass ??= new SpatialResamplingPass(spatialResamplingShader)
            {
                renderPassEvent = renderPassEvent
            };
            _spatialResamplingComputePass ??= new SpatialResamplingComputePass(spatialResamplingComputeCs)
            {
                renderPassEvent = renderPassEvent
            };

            _shadeSamplesPass ??= new ShadeSamplesPass(shadeSamplesShader)
            {
                renderPassEvent = renderPassEvent
            };
            _shadeSamplesComputePass ??= new ShadeSamplesComputePass(shadeSamplesComputeCs)
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

            _shadeSecondarySurfacesPass ??= new ShadeSecondarySurfacesPass(shadeSecondarySurfacesShader)
            {
                renderPassEvent = renderPassEvent
            };

            _giTemporalResamplingPass ??= new GITemporalResamplingPass(giTemporalResamplingShader)
            {
                renderPassEvent = renderPassEvent
            };
            _giTemporalResamplingComputePass ??= new GITemporalResamplingComputePass(giTemporalResamplingComputeCs)
            {
                renderPassEvent = renderPassEvent
            };

            _giSpatialResamplingPass ??= new GISpatialResamplingPass(giSpatialResamplingShader)
            {
                renderPassEvent = renderPassEvent
            };
            _giSpatialResamplingComputePass ??= new GISpatialResamplingComputePass(giSpatialResamplingComputeCs)
            {
                renderPassEvent = renderPassEvent
            };

            _giFinalShadingPass ??= new GIFinalShadingPass(giFinalShadingShader)
            {
                renderPassEvent = renderPassEvent
            };
            _giFinalShadingComputePass ??= new GIFinalShadingComputePass(giFinalShadingComputeCs)
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

            _gpuScene.Build(_accelerationStructure, pathTracingSetting.enableEnv);
            // _gpuScene.UpdateInstanceID(_accelerationStructure);

            Shader.SetGlobalRayTracingAccelerationStructure(g_AccelStructID, _accelerationStructure);

            var uniqueKey = cam.GetInstanceID() + (eyeIndex * 100000L);


            var isVR = renderingData.cameraData.xrRendering;

            if (!_resourcePools.TryGetValue(uniqueKey, out var pool))
            {
                pool = new PathTracingResourcePool(pathTracingSetting);
                _resourcePools.Add(uniqueKey, pool);
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

            if (!_isContexts.TryGetValue(uniqueKey, out var isContext))
            {
                var isParams = ImportanceSamplingContext_StaticParameters.Default();
                isParams.renderWidth = (uint)cam.pixelWidth;
                isParams.renderHeight = (uint)cam.pixelHeight;
                isContext = new ImportanceSamplingContext(isParams);
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

            globalConstants = frameState.GetConstants(renderingData, pathTracingSetting, _lightCollector);
            _globalConstantsArray[0] = globalConstants;
            _constantBuffer.SetData(_globalConstantsArray);

            resamplingConstants = GetResamplingConstants(isContext, rtxdiResources, frameState);
            _resamplingConstantsArray[0] = resamplingConstants;
            _resamplingConstantBuffer.SetData(_resamplingConstantsArray);


            #region prepareLight

            if (pathTracingSetting.prepareLight)
            {
                _prepareLightResources.SetBuffer(_gpuScene);
                _prepareLightResources.SendTexture(_gpuScene.globalTexturePool);
                _prepareLightPass.Setup(_prepareLightResources);
                renderer.EnqueuePass(_prepareLightPass);
            }

            #endregion

            bool isOddFrame = (curFrame % 2) == 1;
            var viewDepth = isOddFrame ? pool.GetRT(RenderResourceType.RtxdiViewDepth) : pool.GetRT(RenderResourceType.RtxdiPrevViewDepth);
            var diffuseAlbedo = isOddFrame ? pool.GetRT(RenderResourceType.RtxdiDiffuseAlbedo) : pool.GetRT(RenderResourceType.RtxdiPrevDiffuseAlbedo);
            var specularRough = isOddFrame ? pool.GetRT(RenderResourceType.RtxdiSpecularRough) : pool.GetRT(RenderResourceType.RtxdiPrevSpecularRough);
            var normals = isOddFrame ? pool.GetRT(RenderResourceType.RtxdiNormals) : pool.GetRT(RenderResourceType.RtxdiPrevNormals);
            var geoNormals = isOddFrame ? pool.GetRT(RenderResourceType.RtxdiGeoNormals) : pool.GetRT(RenderResourceType.RtxdiPrevGeoNormals);

            var prevViewDepth = isOddFrame ? pool.GetRT(RenderResourceType.RtxdiPrevViewDepth) : pool.GetRT(RenderResourceType.RtxdiViewDepth);
            var prevDiffuseAlbedo = isOddFrame ? pool.GetRT(RenderResourceType.RtxdiPrevDiffuseAlbedo) : pool.GetRT(RenderResourceType.RtxdiDiffuseAlbedo);
            var prevSpecularRough = isOddFrame ? pool.GetRT(RenderResourceType.RtxdiPrevSpecularRough) : pool.GetRT(RenderResourceType.RtxdiSpecularRough);
            var prevNormals = isOddFrame ? pool.GetRT(RenderResourceType.RtxdiPrevNormals) : pool.GetRT(RenderResourceType.RtxdiNormals);
            var prevGeoNormals = isOddFrame ? pool.GetRT(RenderResourceType.RtxdiPrevGeoNormals) : pool.GetRT(RenderResourceType.RtxdiGeoNormals);


            #region gBuffer Pass

            var gBufferResource = new GBufferPass.Resource
            {
                ConstantBuffer = _constantBuffer,

                ViewDepth = viewDepth,
                DiffuseAlbedo = diffuseAlbedo,
                SpecularRough = specularRough,
                Normals = normals,
                GeoNormals = geoNormals,
                Emissive = pool.GetRT(RenderResourceType.RtxdiEmissive),
                MotionVectors = pool.GetRT(RenderResourceType.RtxdiMotionVectors),
            };

            var gBufferSettings = new GBufferPass.Settings
            {
                m_RenderResolution = new int2(cam.pixelWidth, cam.pixelHeight),
                resolutionScale = pathTracingSetting.resolutionScale
            };

            if (pathTracingSetting.useRasterGBuffer)
            {
                _gBufferRasterResource.EnsureResources(pool.renderResolution);
                _gBufferRasterPass.Setup(gBufferResource, _gBufferRasterResource, gBufferSettings);
                renderer.EnqueuePass(_gBufferRasterPass);
            }
            else
            {
                _gBufferPass.Setup(gBufferResource, gBufferSettings);
                renderer.EnqueuePass(_gBufferPass);
            }

            #endregion


            var pdfResource = new PdfTexturePass.Resource
            {
                ResamplingConstantBuffer = _resamplingConstantBuffer,
                u_LocalLightPdfTexture = _gpuScene.localLightPdfTexture,
                RtxdiResources = rtxdiResources
            };

            var pdfSettings = new PdfTexturePass.Settings
            {
            };


            _pdfTexturePass.Setup(pdfResource, pdfSettings);
            renderer.EnqueuePass(_pdfTexturePass);


            var genMipsResource = new GenerateMipsPass.Resource
            {
                u_LocalLightPdfTexture = _gpuScene.localLightPdfTexture,
            };

            var genMipsSettings = new GenerateMipsPass.Settings
            {
                width = _gpuScene.localLightPdfTexture.rt.width,
                height = _gpuScene.localLightPdfTexture.rt.height,
                mipCount = _gpuScene.localLightPdfTexture.rt.mipmapCount
            };

            _generateMipsPass.Setup(genMipsResource, genMipsSettings);
            renderer.EnqueuePass(_generateMipsPass);

            if (pathTracingSetting.initialSamplingParams.localLightSamplingMode != ReSTIRDI_LocalLightSamplingMode.Uniform)
            {
                var preResource = new PresamplePass.Resource
                {
                    ConstantBuffer = _constantBuffer,
                    ResamplingConstantBuffer = _resamplingConstantBuffer,
                    u_LocalLightPdfTexture = _gpuScene.localLightPdfTexture,

                    RtxdiResources = rtxdiResources
                };
                var RTXDI_PRESAMPLING_GROUP_SIZE = 256;
                var x = (isContext.GetLocalLightRISBufferSegmentParams().tileSize + RTXDI_PRESAMPLING_GROUP_SIZE - 1) / RTXDI_PRESAMPLING_GROUP_SIZE;
                var y = isContext.GetLocalLightRISBufferSegmentParams().tileCount;

                // dm::int2 presampleDispatchSize = {
                //     dm::div_ceil(isContext.GetLocalLightRISBufferSegmentParams().tileSize, RTXDI_PRESAMPLING_GROUP_SIZE),
                //     int(isContext.GetLocalLightRISBufferSegmentParams().tileCount)
                // };


                var preSettings = new PresamplePass.Settings
                {
                    x = (int)x,
                    y = (int)y,
                    z = 1
                };

                _presamplePass.Setup(preResource, preSettings);
                renderer.EnqueuePass(_presamplePass);
            }

            if (pathTracingSetting.initialSamplingParams.localLightSamplingMode == ReSTIRDI_LocalLightSamplingMode.ReGIR_RIS)
            {
                var preReGirResource = new PresampleReGirLightsPass.Resource
                {
                    ConstantBuffer = _constantBuffer,
                    ResamplingConstantBuffer = _resamplingConstantBuffer,
                    u_LocalLightPdfTexture = _gpuScene.localLightPdfTexture,

                    RtxdiResources = rtxdiResources
                };
                var ReGIR_TILE_SIZE = 256;

                var regirContext = isContext.GetReGIRContext();
                var x = (regirContext.GetReGIRLightSlotCount() + ReGIR_TILE_SIZE - 1) / ReGIR_TILE_SIZE;


                var preReGirSettings = new PresampleReGirLightsPass.Settings
                {
                    x = (int)x,
                    y = 1,
                    z = 1
                };

                _presampleReGirLightsPass.Setup(preReGirResource, preReGirSettings);
                renderer.EnqueuePass(_presampleReGirLightsPass);
            }

            // GenerateInitialSamplesPass
            var gisResource = new GenerateInitialSamplesPass.Resource
            {
                ConstantBuffer = _constantBuffer,
                ResamplingConstantBuffer = _resamplingConstantBuffer,
                t_GeometryInstanceToLight = _gpuScene._geometryInstanceToLight,

                ViewDepth = viewDepth,
                DiffuseAlbedo = diffuseAlbedo,
                SpecularRough = specularRough,
                Normals = normals,
                GeoNormals = geoNormals,


                u_LocalLightPdfTexture = _gpuScene.localLightPdfTexture,

                RtxdiResources = rtxdiResources,
            };

            var gisSettings = new GenerateInitialSamplesPass.Settings
            {
                m_RenderResolution = new int2(cam.pixelWidth, cam.pixelHeight),
                resolutionScale = pathTracingSetting.resolutionScale,
            };

            if (pathTracingSetting.useComputeForGIS)
            {
                var gisComputeResource = new GenerateInitialSamplesComputePass.Resource
                {
                    ConstantBuffer = gisResource.ConstantBuffer,
                    ResamplingConstantBuffer = gisResource.ResamplingConstantBuffer,
                    t_GeometryInstanceToLight = gisResource.t_GeometryInstanceToLight,
                    ViewDepth = gisResource.ViewDepth,
                    DiffuseAlbedo = gisResource.DiffuseAlbedo,
                    SpecularRough = gisResource.SpecularRough,
                    Normals = gisResource.Normals,
                    GeoNormals = gisResource.GeoNormals,
                    DirectLighting = pool.GetRT(RenderResourceType.DirectLighting),
                    u_LocalLightPdfTexture = gisResource.u_LocalLightPdfTexture,
                    RtxdiResources = gisResource.RtxdiResources,
                };
                var gisComputeSettings = new GenerateInitialSamplesComputePass.Settings
                {
                    m_RenderResolution = gisSettings.m_RenderResolution,
                    resolutionScale = gisSettings.resolutionScale,
                };
                _generateInitialSamplesComputePass.Setup(gisComputeResource, gisComputeSettings);
                renderer.EnqueuePass(_generateInitialSamplesComputePass);
            }
            else
            {
                _generateInitialSamplesPass.Setup(gisResource, gisSettings);
                renderer.EnqueuePass(_generateInitialSamplesPass);
            }


            // TemporalResamplingPass
            if (pathTracingSetting.enableTemporalResampling)
            {
                var temResource = new TemporalResamplingPass.Resource
                {
                    ConstantBuffer = _constantBuffer,
                    ResamplingConstantBuffer = _resamplingConstantBuffer,

                    Mv = pool.GetRT(RenderResourceType.RtxdiMotionVectors),
                    DirectLighting = pool.GetRT(RenderResourceType.DirectLighting),


                    ViewDepth = viewDepth,
                    DiffuseAlbedo = diffuseAlbedo,
                    SpecularRough = specularRough,
                    Normals = normals,
                    GeoNormals = geoNormals,

                    PrevViewDepth = prevViewDepth,
                    PrevDiffuseAlbedo = prevDiffuseAlbedo,
                    PrevSpecularRough = prevSpecularRough,
                    PrevNormals = prevNormals,
                    PrevGeoNormals = prevGeoNormals,

                    RtxdiResources = rtxdiResources,
                };

                var temSettings = new TemporalResamplingPass.Settings
                {
                    m_RenderResolution = new int2(cam.pixelWidth, cam.pixelHeight),
                    resolutionScale = pathTracingSetting.resolutionScale,
                };

                if (pathTracingSetting.useComputeForTemporalResampling)
                {
                    var temComputeResource = new TemporalResamplingComputePass.Resource
                    {
                        ConstantBuffer = temResource.ConstantBuffer,
                        ResamplingConstantBuffer = temResource.ResamplingConstantBuffer,
                        Mv = temResource.Mv,
                        DirectLighting = temResource.DirectLighting,
                        ViewDepth = temResource.ViewDepth,
                        DiffuseAlbedo = temResource.DiffuseAlbedo,
                        SpecularRough = temResource.SpecularRough,
                        Normals = temResource.Normals,
                        GeoNormals = temResource.GeoNormals,
                        PrevViewDepth = temResource.PrevViewDepth,
                        PrevDiffuseAlbedo = temResource.PrevDiffuseAlbedo,
                        PrevSpecularRough = temResource.PrevSpecularRough,
                        PrevNormals = temResource.PrevNormals,
                        PrevGeoNormals = temResource.PrevGeoNormals,
                        RtxdiResources = temResource.RtxdiResources,
                    };
                    var temComputeSettings = new TemporalResamplingComputePass.Settings
                    {
                        m_RenderResolution = temSettings.m_RenderResolution,
                        resolutionScale = temSettings.resolutionScale,
                    };
                    _temporalResamplingComputePass.Setup(temComputeResource, temComputeSettings);
                    renderer.EnqueuePass(_temporalResamplingComputePass);
                }
                else
                {
                    _temporalResamplingPass.Setup(temResource, temSettings);
                    renderer.EnqueuePass(_temporalResamplingPass);
                }
            }


            // SpatialResamplingPass
            if (pathTracingSetting.enableSpatialResampling)
            {
                var SpResource = new SpatialResamplingPass.Resource
                {
                    ConstantBuffer = _constantBuffer,
                    ResamplingConstantBuffer = _resamplingConstantBuffer,


                    ViewDepth = viewDepth,
                    DiffuseAlbedo = diffuseAlbedo,
                    SpecularRough = specularRough,
                    Normals = normals,
                    GeoNormals = geoNormals,


                    RtxdiResources = rtxdiResources,
                };

                var SpSettings = new SpatialResamplingPass.Settings
                {
                    m_RenderResolution = new int2(cam.pixelWidth, cam.pixelHeight),
                    resolutionScale = pathTracingSetting.resolutionScale,
                };

                if (pathTracingSetting.useComputeForSpatialResampling)
                {
                    var spComputeResource = new SpatialResamplingComputePass.Resource
                    {
                        ConstantBuffer = SpResource.ConstantBuffer,
                        ResamplingConstantBuffer = SpResource.ResamplingConstantBuffer,
                        ViewDepth = SpResource.ViewDepth,
                        DiffuseAlbedo = SpResource.DiffuseAlbedo,
                        SpecularRough = SpResource.SpecularRough,
                        Normals = SpResource.Normals,
                        GeoNormals = SpResource.GeoNormals,
                        RtxdiResources = SpResource.RtxdiResources,
                    };
                    var spComputeSettings = new SpatialResamplingComputePass.Settings
                    {
                        m_RenderResolution = SpSettings.m_RenderResolution,
                        resolutionScale = SpSettings.resolutionScale,
                    };
                    _spatialResamplingComputePass.Setup(spComputeResource, spComputeSettings);
                    renderer.EnqueuePass(_spatialResamplingComputePass);
                }
                else
                {
                    _spatialResamplingPass.Setup(SpResource, SpSettings);
                    renderer.EnqueuePass(_spatialResamplingPass);
                }
            }


            // ShadeSamplesPass
            var shaResource = new ShadeSamplesPass.Resource
            {
                ConstantBuffer = _constantBuffer,
                ResamplingConstantBuffer = _resamplingConstantBuffer,
                t_GeometryInstanceToLight = _gpuScene._geometryInstanceToLight,

                ViewDepth = viewDepth,
                DiffuseAlbedo = diffuseAlbedo,
                SpecularRough = specularRough,
                Normals = normals,
                GeoNormals = geoNormals,

                DirectLighting = pool.GetRT(RenderResourceType.DirectLighting),

                Emissive = pool.GetRT(RenderResourceType.RtxdiEmissive),


                RtxdiResources = rtxdiResources,
            };

            var shaSettings = new ShadeSamplesPass.Settings
            {
                m_RenderResolution = new int2(cam.pixelWidth, cam.pixelHeight),
                resolutionScale = pathTracingSetting.resolutionScale,
                shading = pathTracingSetting.enableFinalShading
            };

            if (pathTracingSetting.useComputeForShadeSamples)
            {
                var shaComputeResource = new ShadeSamplesComputePass.Resource
                {
                    ConstantBuffer = shaResource.ConstantBuffer,
                    ResamplingConstantBuffer = shaResource.ResamplingConstantBuffer,
                    t_GeometryInstanceToLight = shaResource.t_GeometryInstanceToLight,
                    ViewDepth = shaResource.ViewDepth,
                    DiffuseAlbedo = shaResource.DiffuseAlbedo,
                    SpecularRough = shaResource.SpecularRough,
                    Normals = shaResource.Normals,
                    GeoNormals = shaResource.GeoNormals,
                    DirectLighting = shaResource.DirectLighting,
                    RtxdiResources = shaResource.RtxdiResources,
                    Emissive = shaResource.Emissive,
                };
                var shaComputeSettings = new ShadeSamplesComputePass.Settings
                {
                    m_RenderResolution = shaSettings.m_RenderResolution,
                    resolutionScale = shaSettings.resolutionScale,
                    shading = shaSettings.shading,
                };
                _shadeSamplesComputePass.Setup(shaComputeResource, shaComputeSettings);
                renderer.EnqueuePass(_shadeSamplesComputePass);
            }
            else
            {
                _shadeSamplesPass.Setup(shaResource, shaSettings);
                renderer.EnqueuePass(_shadeSamplesPass);
            }


            // GI

            var brdfResource = new BrdfRayTracingPass.Resource
            {
                ConstantBuffer = _constantBuffer,
                ViewDepth = shaResource.ViewDepth,
                DiffuseAlbedo = shaResource.DiffuseAlbedo,
                SpecularRough = shaResource.SpecularRough,
                Normals = shaResource.Normals,
                GeoNormals = shaResource.GeoNormals,
                DirectLighting = shaResource.DirectLighting,

                ResamplingConstantBuffer = _resamplingConstantBuffer,
                RtxdiResources = shaResource.RtxdiResources,
            };

            var brdfSettings = new BrdfRayTracingPass.Settings
            {
                m_RenderResolution = new int2(cam.pixelWidth, cam.pixelHeight),
                resolutionScale = pathTracingSetting.resolutionScale,
            };

            _brdfRayTracingPass.Setup(brdfResource, brdfSettings);
            renderer.EnqueuePass(_brdfRayTracingPass);

            var shadeSecondaryResource = new ShadeSecondarySurfacesPass.Resource
            {
                ConstantBuffer = _constantBuffer,
                ViewDepth = shaResource.ViewDepth,
                DiffuseAlbedo = shaResource.DiffuseAlbedo,
                SpecularRough = shaResource.SpecularRough,
                Normals = shaResource.Normals,
                GeoNormals = shaResource.GeoNormals,
                DirectLighting = shaResource.DirectLighting,

                ResamplingConstantBuffer = _resamplingConstantBuffer,
                RtxdiResources = shaResource.RtxdiResources,
            };

            var shadeSecondarySettings = new ShadeSecondarySurfacesPass.Settings
            {
                m_RenderResolution = new int2(cam.pixelWidth, cam.pixelHeight),
                resolutionScale = pathTracingSetting.resolutionScale,
            };

            _shadeSecondarySurfacesPass.Setup(shadeSecondaryResource, shadeSecondarySettings);
            renderer.EnqueuePass(_shadeSecondarySurfacesPass);


            // ReSTIR GI – Temporal Resampling
            if (pathTracingSetting.enableGITemporalResampling)
            {
                var giTemResource = new GITemporalResamplingPass.Resource
                {
                    ConstantBuffer = _constantBuffer,
                    ResamplingConstantBuffer = _resamplingConstantBuffer,

                    Mv = pool.GetRT(RenderResourceType.RtxdiMotionVectors),

                    ViewDepth = viewDepth,
                    DiffuseAlbedo = diffuseAlbedo,
                    SpecularRough = specularRough,
                    Normals = normals,
                    GeoNormals = geoNormals,

                    PrevViewDepth = prevViewDepth,
                    PrevDiffuseAlbedo = prevDiffuseAlbedo,
                    PrevSpecularRough = prevSpecularRough,
                    PrevNormals = prevNormals,
                    PrevGeoNormals = prevGeoNormals,

                    RtxdiResources = rtxdiResources,
                };

                var giTemSettings = new GITemporalResamplingPass.Settings
                {
                    m_RenderResolution = new int2(cam.pixelWidth, cam.pixelHeight),
                    resolutionScale = pathTracingSetting.resolutionScale,
                };

                if (pathTracingSetting.useComputeForGITemporalResampling)
                {
                    var giTemComputeResource = new GITemporalResamplingComputePass.Resource
                    {
                        ConstantBuffer = giTemResource.ConstantBuffer,
                        ResamplingConstantBuffer = giTemResource.ResamplingConstantBuffer,
                        Mv = giTemResource.Mv,
                        ViewDepth = giTemResource.ViewDepth,
                        DiffuseAlbedo = giTemResource.DiffuseAlbedo,
                        SpecularRough = giTemResource.SpecularRough,
                        Normals = giTemResource.Normals,
                        GeoNormals = giTemResource.GeoNormals,
                        PrevViewDepth = giTemResource.PrevViewDepth,
                        PrevDiffuseAlbedo = giTemResource.PrevDiffuseAlbedo,
                        PrevSpecularRough = giTemResource.PrevSpecularRough,
                        PrevNormals = giTemResource.PrevNormals,
                        PrevGeoNormals = giTemResource.PrevGeoNormals,
                        RtxdiResources = giTemResource.RtxdiResources,
                    };
                    var giTemComputeSettings = new GITemporalResamplingComputePass.Settings
                    {
                        m_RenderResolution = giTemSettings.m_RenderResolution,
                        resolutionScale = giTemSettings.resolutionScale,
                    };
                    _giTemporalResamplingComputePass.Setup(giTemComputeResource, giTemComputeSettings);
                    renderer.EnqueuePass(_giTemporalResamplingComputePass);
                }
                else
                {
                    _giTemporalResamplingPass.Setup(giTemResource, giTemSettings);
                    renderer.EnqueuePass(_giTemporalResamplingPass);
                }
            }

            // ReSTIR GI – Spatial Resampling
            if (pathTracingSetting.enableGISpatialResampling)
            {
                var giSpResource = new GISpatialResamplingPass.Resource
                {
                    ConstantBuffer = _constantBuffer,
                    ResamplingConstantBuffer = _resamplingConstantBuffer,

                    ViewDepth = viewDepth,
                    DiffuseAlbedo = diffuseAlbedo,
                    SpecularRough = specularRough,
                    Normals = normals,
                    GeoNormals = geoNormals,

                    RtxdiResources = rtxdiResources,
                };

                var giSpSettings = new GISpatialResamplingPass.Settings
                {
                    m_RenderResolution = new int2(cam.pixelWidth, cam.pixelHeight),
                    resolutionScale = pathTracingSetting.resolutionScale,
                };

                if (pathTracingSetting.useComputeForGISpatialResampling)
                {
                    var giSpComputeResource = new GISpatialResamplingComputePass.Resource
                    {
                        ConstantBuffer = giSpResource.ConstantBuffer,
                        ResamplingConstantBuffer = giSpResource.ResamplingConstantBuffer,
                        ViewDepth = giSpResource.ViewDepth,
                        DiffuseAlbedo = giSpResource.DiffuseAlbedo,
                        SpecularRough = giSpResource.SpecularRough,
                        Normals = giSpResource.Normals,
                        GeoNormals = giSpResource.GeoNormals,
                        RtxdiResources = giSpResource.RtxdiResources,
                    };
                    var giSpComputeSettings = new GISpatialResamplingComputePass.Settings
                    {
                        m_RenderResolution = giSpSettings.m_RenderResolution,
                        resolutionScale = giSpSettings.resolutionScale,
                    };
                    _giSpatialResamplingComputePass.Setup(giSpComputeResource, giSpComputeSettings);
                    renderer.EnqueuePass(_giSpatialResamplingComputePass);
                }
                else
                {
                    _giSpatialResamplingPass.Setup(giSpResource, giSpSettings);
                    renderer.EnqueuePass(_giSpatialResamplingPass);
                }
            }

            // ReSTIR GI – Final Shading
            {
                var giFsResource = new GIFinalShadingPass.Resource
                {
                    ConstantBuffer = _constantBuffer,
                    ResamplingConstantBuffer = _resamplingConstantBuffer,

                    ViewDepth = viewDepth,
                    DiffuseAlbedo = diffuseAlbedo,
                    SpecularRough = specularRough,
                    Normals = normals,
                    GeoNormals = geoNormals,

                    DirectLighting = pool.GetRT(RenderResourceType.DirectLighting),

                    RtxdiResources = rtxdiResources,
                };

                var giFsSettings = new GIFinalShadingPass.Settings
                {
                    m_RenderResolution = new int2(cam.pixelWidth, cam.pixelHeight),
                    resolutionScale = pathTracingSetting.resolutionScale,
                    shading = pathTracingSetting.enableGIFinalShading,
                };

                if (pathTracingSetting.useComputeForGIFinalShading)
                {
                    var giFsComputeResource = new GIFinalShadingComputePass.Resource
                    {
                        ConstantBuffer = giFsResource.ConstantBuffer,
                        ResamplingConstantBuffer = giFsResource.ResamplingConstantBuffer,
                        ViewDepth = giFsResource.ViewDepth,
                        DiffuseAlbedo = giFsResource.DiffuseAlbedo,
                        SpecularRough = giFsResource.SpecularRough,
                        Normals = giFsResource.Normals,
                        GeoNormals = giFsResource.GeoNormals,
                        DirectLighting = giFsResource.DirectLighting,
                        RtxdiResources = giFsResource.RtxdiResources,
                    };
                    var giFsComputeSettings = new GIFinalShadingComputePass.Settings
                    {
                        m_RenderResolution = giFsSettings.m_RenderResolution,
                        resolutionScale = giFsSettings.resolutionScale,
                        shading = giFsSettings.shading,
                    };
                    _giFinalShadingComputePass.Setup(giFsComputeResource, giFsComputeSettings);
                    renderer.EnqueuePass(_giFinalShadingComputePass);
                }
                else
                {
                    _giFinalShadingPass.Setup(giFsResource, giFsSettings);
                    renderer.EnqueuePass(_giFinalShadingPass);
                }
            }


            var dlrrRes = new DlrrDenoiser.DlrrResources
            {
                input = pool.GetNriResource(pathTracingSetting.debugRtxdi ? RenderResourceType.DirectLighting : RenderResourceType.Composed),
                output = pool.GetNriResource(RenderResourceType.DlssOutput),
                mv = pool.GetNriResource(RenderResourceType.RtxdiMotionVectors),
                depth = pool.GetNriResource(isOddFrame ? RenderResourceType.RtxdiViewDepth : RenderResourceType.RtxdiPrevViewDepth),

                diffAlbedo = pool.GetNriResource(RenderResourceType.RrGuideDiffAlbedo),
                specAlbedo = pool.GetNriResource(RenderResourceType.RrGuideSpecAlbedo),
                normalRoughness = pool.GetNriResource(RenderResourceType.RrGuideNormalRoughness),
                specHitDistance = pool.GetNriResource(RenderResourceType.RrGuideSpecHitDistance),
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


            var rtxdiDlssBeforeResource = new RxtdiDlssBeforePass.Resource
            {
                ConstantBuffer = _constantBuffer,
                ViewDepth = viewDepth,
                DiffuseAlbedo = diffuseAlbedo,
                SpecularRough = specularRough,
                Normals = normals,

                RRGuide_DiffAlbedo = pool.GetRT(RenderResourceType.RrGuideDiffAlbedo),
                RRGuide_SpecAlbedo = pool.GetRT(RenderResourceType.RrGuideSpecAlbedo),
                RRGuide_SpecHitDistance = pool.GetRT(RenderResourceType.RrGuideSpecHitDistance),
                RRGuide_Normal_Roughness = pool.GetRT(RenderResourceType.RrGuideNormalRoughness),
            };


            var rectGridW = (int)(cam.pixelWidth * pathTracingSetting.resolutionScale + 0.5f + 15) / 16;
            var rectGridH = (int)(cam.pixelHeight * pathTracingSetting.resolutionScale + 0.5f + 15) / 16;

            var rtxdiDlssBeforeSettings = new RxtdiDlssBeforePass.Settings
            {
                rectGridW = rectGridW,
                rectGridH = rectGridH,
            };

            _rtxdiDlssBeforePass.Setup(rtxdiDlssBeforeResource, rtxdiDlssBeforeSettings);
            renderer.EnqueuePass(_rtxdiDlssBeforePass);

            var dlssSettings = new DlssRRPass.Settings
            {
                tmpDisableRR = pathTracingSetting.tmpDisableRR
            };

            _dlssrrPass.Setup(dlssDataPtr, dlssSettings);
            renderer.EnqueuePass(_dlssrrPass);


            var outputBlitResource = new OutputBlitPass.Resource
            {
                Mv = pool.GetRT(RenderResourceType.RtxdiMotionVectors),
                NormalRoughness = pool.GetRT(RenderResourceType.NormalRoughness),
                BaseColorMetalness = pool.GetRT(RenderResourceType.BasecolorMetalness),


                Penumbra = pool.GetRT(RenderResourceType.Penumbra),
                Diff = pool.GetRT(RenderResourceType.DiffRadianceHitdist),
                Spec = pool.GetRT(RenderResourceType.SpecRadianceHitdist),

                ShadowTranslucency = pool.GetRT(RenderResourceType.OutShadowTranslucency),
                DenoisedDiff = pool.GetRT(RenderResourceType.OutDiffRadianceHitdist),
                DenoisedSpec = pool.GetRT(RenderResourceType.OutSpecRadianceHitdist),
                Validation = pool.GetRT(RenderResourceType.Validation),

                Composed = pool.GetRT(RenderResourceType.Composed),
                DirectLighting = pool.GetRT(RenderResourceType.DirectLighting),

                RRGuide_DiffAlbedo = pool.GetRT(RenderResourceType.RrGuideDiffAlbedo),
                RRGuide_SpecAlbedo = pool.GetRT(RenderResourceType.RrGuideSpecAlbedo),
                RRGuide_Normal_Roughness = pool.GetRT(RenderResourceType.RrGuideNormalRoughness),
                RRGuide_SpecHitDistance = pool.GetRT(RenderResourceType.RrGuideSpecHitDistance),
                DlssOutput = pool.GetRT(RenderResourceType.DlssOutput),
            };

            var outputBlitSettings = new OutputBlitPass.Settings
            {
                showMode = pathTracingSetting.showMode,
                resolutionScale = frameState.resolutionScale,
                enableDlssRR = pathTracingSetting.RR,
                showMV = pathTracingSetting.showMv,
                showValidation = pathTracingSetting.showValidation,
                showReference = pathTracingSetting.useReferencePathTracing,
            };

            _outputBlitPass.Setup(outputBlitResource, outputBlitSettings);
            renderer.EnqueuePass(_outputBlitPass);
        }


        void FillReSTIRDIConstants(ref ReSTIRDI_Parameters rparams, ReSTIRDIContext restirDIContext, RTXDI_LightBufferParameters lightBufferParameters)
        {
            rparams.reservoirBufferParams = restirDIContext.GetReservoirBufferParameters();
            rparams.bufferIndices = restirDIContext.GetBufferIndices();
            rparams.initialSamplingParams = restirDIContext.GetInitialSamplingParameters();
            rparams.initialSamplingParams.environmentMapImportanceSampling = lightBufferParameters.environmentLightParams.lightPresent;
            if (rparams.initialSamplingParams.environmentMapImportanceSampling == 0)
                rparams.initialSamplingParams.numPrimaryEnvironmentSamples = 0;
            rparams.temporalResamplingParams = restirDIContext.GetTemporalResamplingParameters();
            rparams.spatialResamplingParams = restirDIContext.GetSpatialResamplingParameters();
            rparams.shadingParams = restirDIContext.GetShadingParameters();
        }

        void FillReSTIRGIConstants(ref ReSTIRGI_Parameters constants, ReSTIRGIContext restirGIContext)
        {
            constants.reservoirBufferParams = restirGIContext.GetReservoirBufferParameters();
            constants.bufferIndices = restirGIContext.GetBufferIndices();
            constants.temporalResamplingParams = restirGIContext.GetTemporalResamplingParameters();
            constants.spatialResamplingParams = restirGIContext.GetSpatialResamplingParameters();
            constants.finalShadingParams = restirGIContext.GetFinalShadingParameters();
        }

        void FillReGIRConstants(ref ReGIR_Parameters ReGIRParams, ReGIRContext regirContext)
        {
            var staticParams = regirContext.GetReGIRStaticParameters();
            var dynamicParams = regirContext.GetReGIRDynamicParameters();
            var gridParams = regirContext.GetReGIRGridCalculatedParameters();
            var onionParams = regirContext.GetReGIROnionCalculatedParameters();

            ReGIRParams.gridParams.cellsX = staticParams.gridParameters.GridSize.x;
            ReGIRParams.gridParams.cellsY = staticParams.gridParameters.GridSize.y;
            ReGIRParams.gridParams.cellsZ = staticParams.gridParameters.GridSize.z;

            ReGIRParams.commonParams.numRegirBuildSamples = dynamicParams.regirNumBuildSamples;
            ReGIRParams.commonParams.risBufferOffset = regirContext.GetReGIRCellOffset();
            ReGIRParams.commonParams.lightsPerCell = staticParams.LightsPerCell;
            ReGIRParams.commonParams.centerX = dynamicParams.center.x;
            ReGIRParams.commonParams.centerY = dynamicParams.center.y;
            ReGIRParams.commonParams.centerZ = dynamicParams.center.z;
            ReGIRParams.commonParams.cellSize = (staticParams.Mode == ReGIRMode.Onion)
                ? dynamicParams.regirCellSize * 0.5f // Onion operates with radii, while "size" feels more like diameter
                : dynamicParams.regirCellSize;
            ReGIRParams.commonParams.localLightSamplingFallbackMode = (uint)(dynamicParams.fallbackSamplingMode);
            ReGIRParams.commonParams.localLightPresamplingMode = (uint)(dynamicParams.presamplingMode);
            ReGIRParams.commonParams.samplingJitter = math.max(0.0f, dynamicParams.regirSamplingJitter * 2.0f);
            ReGIRParams.onionParams.cubicRootFactor = onionParams.regirOnionCubicRootFactor;
            ReGIRParams.onionParams.linearFactor = onionParams.regirOnionLinearFactor;
            ReGIRParams.onionParams.numLayerGroups = (uint)(onionParams.regirOnionLayers.Count);

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


        private void FillResamplingConstants(ref ResamplingConstants constants, ImportanceSamplingContext isContext)
        {
            // RTXDI_LightBufferParameters lightBufferParameters = isContext.GetLightBufferParameters();

            constants.enablePreviousTLAS = pathTracingSetting.enablePreviousTLAS ? 1u : 0u;
            constants.denoiserMode = pathTracingSetting.denoiserMode;
            // constants.sceneConstants.enableAlphaTestedGeometry = lightingSettings.enableAlphaTestedGeometry;
            // constants.sceneConstants.enableTransparentGeometry = lightingSettings.enableTransparentGeometry;
            constants.visualizeRegirCells = pathTracingSetting.visualizeRegirCells ? 1u : 0u;

            constants.lightBufferParams = isContext.GetLightBufferParameters();
            constants.localLightsRISBufferSegmentParams = isContext.GetLocalLightRISBufferSegmentParams();
            constants.environmentLightRISBufferSegmentParams = isContext.GetEnvironmentLightRISBufferSegmentParams();
            constants.runtimeParams = isContext.GetReSTIRDIContext().GetRuntimeParams();

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


        private void FillBRDFPTConstants(ref BRDFPathTracing_Parameters constants, RTXDI_LightBufferParameters getLightBufferParameters)
        {
            constants = pathTracingSetting.brdfptParams;
            constants.materialOverrideParams.minSecondaryRoughness = 0;
            constants.materialOverrideParams.roughnessOverride = -1.0f;
            constants.materialOverrideParams.metalnessOverride = -1.0f;
            constants.secondarySurfaceReSTIRDIParams.initialSamplingParams.environmentMapImportanceSampling = 0;
            if (constants.secondarySurfaceReSTIRDIParams.initialSamplingParams.environmentMapImportanceSampling == 0)
                constants.secondarySurfaceReSTIRDIParams.initialSamplingParams.numPrimaryEnvironmentSamples = 0;
        }


        private ResamplingConstants GetResamplingConstants(
            ImportanceSamplingContext isContext
            , RtxdiResources rtxdiResources
            , CameraFrameState frameState)
        {
            var restirDIContext = isContext.GetReSTIRDIContext();
            var restirGIContext = isContext.GetReSTIRGIContext();


            RTXDI_LightBufferParameters lightBufferParams = _gpuScene.GetLightBufferParameters();
            isContext.SetLightBufferParams(lightBufferParams);
            restirDIContext.SetFrameIndex(frameState.FrameIndex);
            restirDIContext.SetResamplingMode(pathTracingSetting.resamplingMode);
            restirDIContext.SetInitialSamplingParameters(pathTracingSetting.initialSamplingParams);
            restirDIContext.SetTemporalResamplingParameters(pathTracingSetting.temporalResamplingParams);
            restirDIContext.SetSpatialResamplingParameters(pathTracingSetting.spatialResamplingParams);
            restirDIContext.SetShadingParameters(pathTracingSetting.shadingParams);

            var regirContext = isContext.GetReGIRContext();
            pathTracingSetting.regirDynamicParams.center = frameState.camPos;
            regirContext.SetDynamicParameters(pathTracingSetting.regirDynamicParams);


            var constants = new ResamplingConstants();

            constants.frameIndex = restirDIContext.GetFrameIndex();
            constants.denoiserMode = pathTracingSetting.denoiserMode;

            constants.enableBrdfIndirect = pathTracingSetting.enableBrdfIndirect ? 1u : 0u;
            constants.enableBrdfAdditiveBlend = pathTracingSetting.enableBrdfAdditiveBlend ? 1u : 0u;
            constants.enableAccumulation = 0;

            // constants.sceneConstants.enableEnvironmentMap = (environmentLight.textureIndex >= 0);
            // constants.sceneConstants.environmentMapTextureIndex = (environmentLight.textureIndex >= 0) ? environmentLight.textureIndex : 0;
            // constants.sceneConstants.environmentScale = environmentLight.radianceScale.x;
            // constants.sceneConstants.environmentRotation = environmentLight.rotation;


            FillResamplingConstants(ref constants, isContext);
            FillBRDFPTConstants(ref constants.brdfPT, isContext.GetLightBufferParameters());

            constants.brdfPT = pathTracingSetting.brdfptParams;


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

            // _restirDiContexts.Clear();

            _lightCollector?.Dispose();
            _lightCollector = null;

            _gpuScene?.Dispose();
            _gpuScene = null;

            _prepareLightResources.Dispose();
            _prepareLightResources = null;
            ;
            _prepareLightPass = null;
            _gBufferPass = null;
            _dlssrrPass = null;
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