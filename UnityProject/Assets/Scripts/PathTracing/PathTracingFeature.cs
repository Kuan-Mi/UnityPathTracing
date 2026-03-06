using System.Collections.Generic;
using DefaultNamespace;
using Nrd;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using static UnityEngine.Rendering.RayTracingAccelerationStructure;

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

        void SetMask()
        {
            var allRenderers = GameObject.FindObjectsByType<Renderer>(FindObjectsSortMode.None);
            foreach (var r in allRenderers)
            {
                var materials = r.sharedMaterials;
                bool hasTransparent = false;
                bool hasOpaque = false;
                foreach (var mat in materials)
                {
                    if (mat != null && mat.renderQueue >= 3000)
                    {
                        hasTransparent = true;
                    }
                    else
                    {
                        hasOpaque = true;
                    }
                }

                uint mask = 0;

                if (hasOpaque)
                    mask |= 0x01; // FLAG_NON_TRANSPARENT
                if (hasTransparent)
                    mask |= 0x02; // FLAG_TRANSPARENT

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

        private RayTracingAccelerationStructure accelerationStructure;
        private Settings settings;

        private GraphicsBuffer scramblingRankingUintBuffer;
        private GraphicsBuffer sobolUintBuffer;

        private GraphicsBuffer _hashEntriesBuffer;
        private GraphicsBuffer _accumulationBuffer;
        private GraphicsBuffer _resolvedBuffer;

        // Auto-exposure buffers (persistent across frames)
        private GraphicsBuffer _aeHistogramBuffer; // 256 x uint
        private GraphicsBuffer _aeExposureBuffer;  // 1 x float  (current exposure multiplier)

        private Dictionary<long, NRDDenoiser> _nrdDenoisers = new();
        private Dictionary<long, DLRRDenoiser> _dlrrDenoisers = new();

        public PathTracingDataBuilder _dataBuilder = new PathTracingDataBuilder();

        [ContextMenu("ReBuild AccelerationStructure")]
        public void ReBuild()
        {
            _dataBuilder.Build(accelerationStructure);
        }

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

                // SetMask();
            }

            if (_dataBuilder.IsEmpty())
            {
                _dataBuilder.Build(accelerationStructure);
            }

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
                _dataBuilder = _dataBuilder,
                AutoExposureCs = autoExposureShader,
                AeHistogramBuffer = _aeHistogramBuffer,
                AeExposureBuffer = _aeExposureBuffer
            };
        }

        static readonly int Capacity = 1 << 22;

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

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            Camera cam = renderingData.cameraData.camera;
            if (cam.cameraType is CameraType.Preview or CameraType.Reflection)
                return;
            
            
            cam.depthTextureMode = DepthTextureMode.Depth | DepthTextureMode.MotionVectors;
            
            

            int eyeIndex = renderingData.cameraData.xr.enabled ? renderingData.cameraData.xr.multipassId : 0;


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

            _pathTracingPass.NrdDenoiser = nrd;
            _pathTracingPass.DLRRDenoiser = dlrr;

            _pathTracingPass.AccumulationBuffer = _accumulationBuffer;
            _pathTracingPass.HashEntriesBuffer = _hashEntriesBuffer;
            _pathTracingPass.ResolvedBuffer = _resolvedBuffer;
            _pathTracingPass.AeHistogramBuffer = _aeHistogramBuffer;
            _pathTracingPass.AeExposureBuffer = _aeExposureBuffer;

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

            var allSkinnedMeshRenderers = GameObject.FindObjectsByType<SkinnedMeshRenderer>(FindObjectsSortMode.None);
            foreach (var smr in allSkinnedMeshRenderers)
            {
                accelerationStructure.UpdateInstanceTransform(smr);
            }

            accelerationStructure.Build();
            if (pathTracingSetting.usePackedData)
            {
                if (!_dataBuilder.IsEmpty())
                {
                    renderer.EnqueuePass(_pathTracingPass);
                }
            }
            else
            {
                renderer.EnqueuePass(_pathTracingPass);
            }
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
    }
}