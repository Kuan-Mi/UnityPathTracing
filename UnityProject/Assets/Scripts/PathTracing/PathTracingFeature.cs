using System.Collections.Generic;
using System.Runtime.InteropServices;
using DefaultNamespace;
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
using static PathTracing.PathTracingUtils;

namespace PathTracing
{
    public class PathTracingFeature : ScriptableRendererFeature
    {
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
                bool hasTransparent = false;
                bool hasOpaque = false;
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

                accelerationStructure.UpdateInstanceMask(r, mask); // 1 表示包含在内
            }
        }

        public PathTracingSetting pathTracingSetting;

        public Material finalMaterial;
        public RayTracingShader opaqueTracingShader;
        public RayTracingShader transparentTracingShader;
        public ComputeShader compositionComputeShader;
        public ComputeShader taaComputeShader;
        public ComputeShader dlssBeforeComputeShader;
        public ComputeShader sharcResolveCs;
        public RayTracingShader sharcUpdateTs;
        public ComputeShader autoExposureShader;

        public Texture2D scramblingRankingTex;
        public Texture2D sobolTex;


        private PathTracingPass _pathTracingPass;
        private SharcPass _sharcPass;
        private PrepareLightPass _prepareLightPass;
        private OpaquePass _opaquePass;

        private RayTracingAccelerationStructure accelerationStructure;
        private Settings settings;

        private GraphicsBuffer scramblingRankingUintBuffer;
        private GraphicsBuffer sobolUintBuffer;

        private GraphicsBuffer ConstantBuffer;
        private GraphicsBuffer ResamplingConstantBuffer;
        
        private GraphicsBuffer _hashEntriesBuffer;
        private GraphicsBuffer _accumulationBuffer;
        private GraphicsBuffer _resolvedBuffer;
        
        
        private GraphicsBuffer m_SpotLightBuffer;
        private GraphicsBuffer m_AreaLightBuffer;
        private GraphicsBuffer m_PointLightBuffer;

        private int spotCount;
        private int areaCount;
        private int pointCount;

        // Auto-exposure buffers (persistent across frames)
        private GraphicsBuffer _aeHistogramBuffer; // 256 x uint
        private GraphicsBuffer _aeExposureBuffer; // 1 x float  (current exposure multiplier)

        private Dictionary<long, NRDDenoiser> _nrdDenoisers = new();
        private Dictionary<long, DLRRDenoiser> _dlrrDenoisers = new();

        private Dictionary<long, ReSTIRDIContext> _restirDIContexts = new();
        private Dictionary<long, RtxdiResources> _rtxdiResources = new();


        private Dictionary<long, PrepareLightResource> _prepareLightResources = new Dictionary<long, PrepareLightResource>();

        public GPUScene gpuScene = new GPUScene();


        // public PathTracingDataBuilder _dataBuilder = new PathTracingDataBuilder();

        // [ContextMenu("ReBuild AccelerationStructure")]
        // public void ReBuild()
        // {
        //     _dataBuilder.Build(accelerationStructure);
        // }

        public override void Create()
        {
            if (accelerationStructure == null)
            {
                settings = new Settings
                {
                    managementMode = ManagementMode.Automatic,
                    rayTracingModeMask = RayTracingModeMask.Everything
                };
                accelerationStructure = new RayTracingAccelerationStructure(settings);

                accelerationStructure.Build();

                SetMask();
            }

            if (!gpuScene.isBufferInitialized)
            {
                gpuScene.InitBuffer();
            }

            // if (_dataBuilder.IsEmpty())
            // {
            //     _dataBuilder.Build(accelerationStructure);
            // }

            // ReBuild();

            if (scramblingRankingUintBuffer == null && scramblingRankingTex != null)
            {
                scramblingRankingUintBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, scramblingRankingTex.width * scramblingRankingTex.height, 16);
                var scramblingRankingData = new uint4[scramblingRankingTex.width * scramblingRankingTex.height];
                var rawData = scramblingRankingTex.GetRawTextureData();
                var count = scramblingRankingData.Length;
                for (var i = 0; i < count; i++)
                {
                    scramblingRankingData[i] = new uint4(rawData[i * 4 + 0], rawData[i * 4 + 1], rawData[i * 4 + 2], rawData[i * 4 + 3]);
                }

                scramblingRankingUintBuffer.SetData(scramblingRankingData);
            }

            if (sobolUintBuffer == null && sobolTex != null)
            {
                sobolUintBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, sobolTex.width * sobolTex.height, 16);
                var sobolData = new uint4[sobolTex.width * sobolTex.height];
                var rawData = sobolTex.GetRawTextureData();
                var count = sobolData.Length;
                for (var i = 0; i < count; i++)
                {
                    sobolData[i] = new uint4(rawData[i * 4 + 0], rawData[i * 4 + 1], rawData[i * 4 + 2], rawData[i * 4 + 3]);
                }

                sobolUintBuffer.SetData(sobolData);
            }

            if (_accumulationBuffer == null)
            {
                InitializeBuffers();
            }

            // Auto-exposure buffers
            if (_aeHistogramBuffer == null)
            {
                _aeHistogramBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 256, sizeof(uint));
            }

            if (_aeExposureBuffer == null)
            {
                _aeExposureBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, sizeof(float));
                // Seed with the manual exposure value so first frame is not zero
                _aeExposureBuffer.SetData(new float[] { pathTracingSetting != null ? pathTracingSetting.exposure : 1.0f });
            }

            _pathTracingPass = new PathTracingPass(pathTracingSetting)
            {
                renderPassEvent = RenderPassEvent.BeforeRenderingTransparents,
                OpaqueTs = opaqueTracingShader,
                TransparentTs = transparentTracingShader,
                CompositionCs = compositionComputeShader,
                TaaCs = taaComputeShader,
                DlssBeforeCs = dlssBeforeComputeShader,
                AccelerationStructure = accelerationStructure,
                ScramblingRanking = scramblingRankingUintBuffer,
                Sobol = sobolUintBuffer,
                BiltMaterial = finalMaterial,
                SharcResolveCs = sharcResolveCs,
                SharcUpdateTs = sharcUpdateTs,
                HashEntriesBuffer = _hashEntriesBuffer,
                AccumulationBuffer = _accumulationBuffer,
                ResolvedBuffer = _resolvedBuffer,
                // _dataBuilder = _dataBuilder,
                AutoExposureCs = autoExposureShader,
                AeHistogramBuffer = _aeHistogramBuffer,
                AeExposureBuffer = _aeExposureBuffer
            };

            _sharcPass = new SharcPass(sharcResolveCs, sharcUpdateTs);
            _prepareLightPass = new PrepareLightPass();
            _opaquePass = new OpaquePass(opaqueTracingShader);
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

            ConstantBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Constant, 1, Marshal.SizeOf<GlobalConstants>());
            ResamplingConstantBuffer =  new GraphicsBuffer(GraphicsBuffer.Target.Structured, 1, Marshal.SizeOf<ResamplingConstants>());

            _hashEntriesBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Capacity, sizeof(ulong));
            ulong[] clearData = new ulong[Capacity];
            _hashEntriesBuffer.SetData(clearData);

            // 2. Accumulation Buffer: storing uint4 (16 bytes)
            // HLSL: RWStructuredBuffer<SharcAccumulationData> gInOut_SharcAccumulated;
            _accumulationBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Capacity, sizeof(uint) * 4);
            uint4[] clearAccumData = new uint4[Capacity];
            _accumulationBuffer.SetData(clearAccumData);

            // 3. Resolved Buffer: storing uint3 + uint (16 bytes)
            // HLSL: RWStructuredBuffer<SharcPackedData> gInOut_SharcResolved;
            _resolvedBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Capacity, sizeof(uint) * 4);
            uint4[] clearResolvedData = new uint4[Capacity];
            _resolvedBuffer.SetData(clearResolvedData);
        }

        
        uint GetMaxAccumulatedFrameNum(float accumulationTime, float fps)
        {
            return (uint)(accumulationTime * fps + 0.5f);
        }
        
        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            Camera cam = renderingData.cameraData.camera;
            if (cam.cameraType is CameraType.Preview or CameraType.Reflection)
                return;
            if (cam.cameraType != CameraType.Game && cam.cameraType != CameraType.SceneView)
            {
                return;
            }

            cam.depthTextureMode = DepthTextureMode.Depth | DepthTextureMode.MotionVectors;

            int eyeIndex = renderingData.cameraData.xr.enabled ? renderingData.cameraData.xr.multipassId : 0;


            if (eyeIndex == 1)
                return;


            gpuScene.Build();


            long uniqueKey = cam.GetInstanceID() + (eyeIndex * 100000L);


            var isVR = renderingData.cameraData.xrRendering;

            if (!_nrdDenoisers.TryGetValue(uniqueKey, out var nrd))
            {
                var camName = cam.name;
                if (isVR)
                {
                    camName = $"{cam.name}_Eye{eyeIndex}";
                }

                nrd = new NRDDenoiser(pathTracingSetting, camName);
                _nrdDenoisers.Add(uniqueKey, nrd);
            }


            if (!_dlrrDenoisers.TryGetValue(uniqueKey, out var dlrr))
            {
                var camName = cam.name;
                if (isVR)
                {
                    camName = $"{cam.name}_Eye{eyeIndex}";
                }

                dlrr = new DLRRDenoiser(pathTracingSetting, camName);
                _dlrrDenoisers.Add(uniqueKey, dlrr);
            }


            if (!_prepareLightResources.TryGetValue(uniqueKey, out var prepareLightResource))
            {
                prepareLightResource = new PrepareLightResource();
                _prepareLightResources.Add(uniqueKey, prepareLightResource);
                prepareLightResource.SetBuffer(gpuScene);
            }
                prepareLightResource.SendTexture(gpuScene.globalTexturePool);

            if (!_restirDIContexts.TryGetValue(uniqueKey, out var restirDIContext))
            {
                ReSTIRDIStaticParameters contextParams = ReSTIRDIStaticParameters.Default();
                contextParams.RenderWidth = (uint)cam.pixelWidth;
                contextParams.RenderHeight = (uint)cam.pixelHeight;

                restirDIContext = new ReSTIRDIContext(contextParams);
                _restirDIContexts.Add(uniqueKey, restirDIContext);
            }

            if (!_rtxdiResources.TryGetValue(uniqueKey, out var rtxdiResources))
            {
                rtxdiResources = new RtxdiResources(restirDIContext, gpuScene);
                _rtxdiResources.Add(uniqueKey, rtxdiResources);
            }

            _pathTracingPass.NrdDenoiser = nrd;
            _pathTracingPass.DLRRDenoiser = dlrr;

            _pathTracingPass.AccumulationBuffer = _accumulationBuffer;
            _pathTracingPass.HashEntriesBuffer = _hashEntriesBuffer;
            _pathTracingPass.ResolvedBuffer = _resolvedBuffer;
            _pathTracingPass.AeHistogramBuffer = _aeHistogramBuffer;
            _pathTracingPass.AeExposureBuffer = _aeExposureBuffer;

            // _pathTracingPass.prepareLightResource = prepareLightResource;
            _pathTracingPass.rtxdiResources = rtxdiResources;
            _pathTracingPass.restirDIContext = restirDIContext;

            if (finalMaterial == null
                || opaqueTracingShader == null
                || transparentTracingShader == null
                || compositionComputeShader == null
                || taaComputeShader == null
                || dlssBeforeComputeShader == null
                || sharcResolveCs == null
                || sharcUpdateTs == null
                || finalMaterial == null
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
                accelerationStructure.UpdateInstanceTransform(smr);
            }

            accelerationStructure.Build();

            
            
            Light();

            // GlobalConstants globalConstants =   GetConstants();

            var globalConstants = GetConstants(renderingData, nrd);
            ConstantBuffer.SetData(new[] { globalConstants });

            var resamplingConstants = GetResamplingConstants(restirDIContext, rtxdiResources);
            ResamplingConstantBuffer.SetData(new[] { resamplingConstants });
            
            // Sharc 
            var sharcResource = new SharcPass.Resource
            {
                ConstantBuffer =  ConstantBuffer,
                AccumulationBuffer = _accumulationBuffer,
                HashEntriesBuffer = _hashEntriesBuffer,
                ResolvedBuffer = _resolvedBuffer,
                PointLightBuffer = m_PointLightBuffer,
                AreaLightBuffer = m_AreaLightBuffer,
                SpotLightBuffer = m_SpotLightBuffer
            };
            
            var sharcSettings = new SharcPass.Settings
            {
                RenderResolution = new int2(cam.pixelWidth, cam.pixelHeight)
            };

            _sharcPass.Setup(sharcResource,sharcSettings);
            renderer.EnqueuePass(_sharcPass);
            
            _prepareLightPass.Setup(prepareLightResource);
            renderer.EnqueuePass(_prepareLightPass);
            
            var opaqueResource = new OpaquePass.Resource
            {
                ConstantBuffer =  ConstantBuffer,
                ResamplingConstantBuffer = ResamplingConstantBuffer,
                
                AccumulationBuffer = _accumulationBuffer,
                HashEntriesBuffer = _hashEntriesBuffer,
                ResolvedBuffer = _resolvedBuffer,
                
                PointLightBuffer = m_PointLightBuffer,
                AreaLightBuffer = m_AreaLightBuffer,
                SpotLightBuffer = m_SpotLightBuffer,
                
                ScramblingRanking = scramblingRankingUintBuffer,
                Sobol = sobolUintBuffer,
                
                Mv =  nrd.GetRT(ResourceType.IN_MV),
                ViewZ = nrd.GetRT(ResourceType.IN_VIEWZ),
                NormalRoughness = nrd.GetRT(ResourceType.IN_NORMAL_ROUGHNESS),
                BaseColorMetalness = nrd.GetRT(ResourceType.IN_BASECOLOR_METALNESS),
                
                Penumbra = nrd.GetRT(ResourceType.IN_PENUMBRA),
                Diff = nrd.GetRT(ResourceType.IN_DIFF_RADIANCE_HITDIST),
                Spec = nrd.GetRT(ResourceType.IN_SPEC_RADIANCE_HITDIST),
                
                PrevViewZ = nrd.GetRT(ResourceType.Prev_ViewZ),
                PrevNormalRoughness = nrd.GetRT(ResourceType.Prev_NormalRoughness),
                PrevBaseColorMetalness = nrd.GetRT(ResourceType.Prev_BaseColorMetalness),
                
                PsrThroughput = nrd.GetRT(ResourceType.PsrThroughput),
                
                RtxdiResources = rtxdiResources
            };
            
            var opaqueSettings = new OpaquePass.Settings
            {
                m_RenderResolution = new int2(cam.pixelWidth, cam.pixelHeight),
                resolutionScale = pathTracingSetting.resolutionScale
            };
            
            _opaquePass.Setup(opaqueResource, opaqueSettings);
            renderer.EnqueuePass(_opaquePass);
            
            


            _pathTracingPass.m_SpotLightBuffer = m_SpotLightBuffer;
            _pathTracingPass.m_AreaLightBuffer = m_AreaLightBuffer;
            _pathTracingPass.m_PointLightBuffer = m_PointLightBuffer;
            _pathTracingPass._pathTracingSettingsBuffer =  ConstantBuffer;
            
            _pathTracingPass.Setup();
            renderer.EnqueuePass(_pathTracingPass);
            
        }

        private ResamplingConstants GetResamplingConstants(ReSTIRDIContext restirDIContext, RtxdiResources rtxdiResources)
        {
            restirDIContext.SetFrameIndex((uint)Time.frameCount);


            var resamplingConstants = new ResamplingConstants
            {
                runtimeParams = restirDIContext.GetRuntimeParams()
            };

            
            resamplingConstants.lightBufferParams.localLightBufferRegion.firstLightIndex = 0;
            resamplingConstants.lightBufferParams.localLightBufferRegion.numLights = rtxdiResources.Scene.emissiveTriangleCount;
            
            resamplingConstants.lightBufferParams.infiniteLightBufferRegion.firstLightIndex = 0;
            resamplingConstants.lightBufferParams.infiniteLightBufferRegion.numLights = 0;
            
            resamplingConstants.lightBufferParams.environmentLightParams.lightPresent = 0;
            resamplingConstants.lightBufferParams.environmentLightParams.lightIndex = (0xffffffffu);


            resamplingConstants.restirDIReservoirBufferParams = restirDIContext.GetReservoirBufferParameters();

            resamplingConstants.frameIndex = restirDIContext.GetFrameIndex();
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

        private GlobalConstants GetConstants(RenderingData renderingData, NRDDenoiser nrd)
        {
            var NrdDenoiser =  nrd;
            var cameraData = renderingData.cameraData;
            var m_Settings = pathTracingSetting;

            var lightData = renderingData.lightData;
            var mainLight = lightData.mainLightIndex >= 0 ? lightData.visibleLights[lightData.mainLightIndex] : default;
            var mat = mainLight.localToWorldMatrix;
            Vector3 lightForward = mat.GetColumn(2);
            
            var gSunDirection = -lightForward;
            var up = new Vector3(0, 1, 0);
            var gSunBasisX = math.normalize(math.cross(new float3(up.x, up.y, up.z), new float3(gSunDirection.x, gSunDirection.y, gSunDirection.z)));
            var gSunBasisY = math.normalize(math.cross(new float3(gSunDirection.x, gSunDirection.y, gSunDirection.z), gSunBasisX));
            
            int2 outputResolution = new int2((int)(cameraData.camera.pixelWidth * cameraData.renderScale), (int)(cameraData.camera.pixelHeight * cameraData.renderScale));
            
            
            var xrPass = cameraData.xr;
            var isXr = xrPass.enabled;
            if (xrPass.enabled)
            {
                // Debug.Log($"XR Enabled. Eye Texture Resolution: {xrPass.renderTargetDesc.width} x {xrPass.renderTargetDesc.height}");

                outputResolution = new int2(xrPass.renderTargetDesc.width, xrPass.renderTargetDesc.height);
            }
            
            NrdDenoiser.EnsureResources(outputResolution);
            
            var renderResolution = NrdDenoiser.renderResolution;
            
            Shader.SetGlobalRayTracingAccelerationStructure(g_AccelStructID, accelerationStructure);
            
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
            globalConstants.gShowLight = m_Settings.gShowLight?  1u : 0u;

            return globalConstants;
        }


        private void Light()
        {
            var spotLightList = new List<SpotLightData>();

            // 获取场景中所有激活的 Light 组件（不受视锥体裁剪限制）
            var allLights = FindObjectsByType<Light>(FindObjectsSortMode.None);
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


            spotCount = spotLightList.Count;
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
            var areaLightList = new List<AreaLightData>();

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

            areaCount       = areaLightList.Count;
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
            var pointLightList = new List<PointLightData>();

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

            pointCount       = pointLightList.Count;
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
        }

        protected override void Dispose(bool disposing)
        {
            Debug.Log("PathTracingFeature Dispose");
            base.Dispose(disposing);
            // accelerationStructure.Dispose();
            // accelerationStructure.Release();
            // accelerationStructure = null;
            _pathTracingPass.Dispose();
            _pathTracingPass = null;

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


            foreach (var resource in _prepareLightResources.Values)
            {
                resource.Dispose();
            }

            _prepareLightResources.Clear();

            scramblingRankingUintBuffer?.Release();
            scramblingRankingUintBuffer = null;

            sobolUintBuffer?.Release();
            sobolUintBuffer = null;

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
        }

        public void Test()
        {
            gpuScene.DebugReadback();
        }
    }
}