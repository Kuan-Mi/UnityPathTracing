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

        public Material finalMaterial;
        public RayTracingShader opaqueTracingShader;
        public RayTracingShader transparentTracingShader;

        public ComputeShader compositionComputeShader;
        public ComputeShader taaComputeShader;
        public ComputeShader dlssBeforeComputeShader;

        public ComputeShader sharcResolveCs;
        public RayTracingShader sharcUpdateTs;

        public PathTracingSetting pathTracingSetting;

        private PathTracingPassSingle _pathTracingPass;

        public RayTracingAccelerationStructure accelerationStructure;
        public Settings settings;

        public Texture2D gIn_ScramblingRanking;
        public Texture2D gIn_Sobol;
        public GraphicsBuffer gIn_ScramblingRankingUint;
        public GraphicsBuffer gIn_SobolUint;


        private GraphicsBuffer _hashEntriesBuffer;
        private GraphicsBuffer _accumulationBuffer;
        private GraphicsBuffer _resolvedBuffer;


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

            if (gIn_ScramblingRankingUint == null)
            {
                gIn_ScramblingRankingUint = new GraphicsBuffer(GraphicsBuffer.Target.Structured, gIn_ScramblingRanking.width * gIn_ScramblingRanking.height, 16);
                var scramblingRankingData = new uint4[gIn_ScramblingRanking.width * gIn_ScramblingRanking.height];
                var rawData = gIn_ScramblingRanking.GetRawTextureData();
                var count = scramblingRankingData.Length;
                for (var i = 0; i < count; i++)
                {
                    scramblingRankingData[i] = new uint4(rawData[i * 4 + 0], rawData[i * 4 + 1], rawData[i * 4 + 2], rawData[i * 4 + 3]);
                }

                gIn_ScramblingRankingUint.SetData(scramblingRankingData);

                gIn_SobolUint = new GraphicsBuffer(GraphicsBuffer.Target.Structured, gIn_Sobol.width * gIn_Sobol.height, 16);
                var sobolData = new uint4[gIn_Sobol.width * gIn_Sobol.height];
                rawData = gIn_Sobol.GetRawTextureData();
                count = sobolData.Length;
                for (var i = 0; i < count; i++)
                {
                    sobolData[i] = new uint4(rawData[i * 4 + 0], rawData[i * 4 + 1], rawData[i * 4 + 2], rawData[i * 4 + 3]);
                }

                gIn_SobolUint.SetData(sobolData);
            }

            if (_accumulationBuffer == null)
            {
                InitializeBuffers();
            }

            _pathTracingPass = new PathTracingPassSingle(pathTracingSetting)
            {
                renderPassEvent = RenderPassEvent.BeforeRenderingTransparents,
                OpaqueTs = opaqueTracingShader,
                TransparentTs = transparentTracingShader,
                CompositionCs = compositionComputeShader,
                TaaCs = taaComputeShader,
                DlssBeforeCs = dlssBeforeComputeShader,
                AccelerationStructure = accelerationStructure,
                ScramblingRanking = gIn_ScramblingRankingUint,
                Sobol = gIn_SobolUint,
                BiltMaterial = finalMaterial,
                SharcResolveCs = sharcResolveCs,
                SharcUpdateTs = sharcUpdateTs,
                HashEntriesBuffer = _hashEntriesBuffer,
                AccumulationBuffer = _accumulationBuffer,
                ResolvedBuffer = _resolvedBuffer,
                _dataBuilder = _dataBuilder
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

            if (compositionComputeShader == null
                || taaComputeShader == null
                || finalMaterial == null
                || opaqueTracingShader == null
                || transparentTracingShader == null
                || sharcResolveCs == null
                || sharcUpdateTs == null
               )
                return;

            
            var allSkinnedMeshRenderers = GameObject.FindObjectsByType<SkinnedMeshRenderer>(FindObjectsSortMode.None);
            foreach (var smr in allSkinnedMeshRenderers)            {
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

            gIn_ScramblingRankingUint?.Release();
            gIn_ScramblingRankingUint = null;

            gIn_SobolUint?.Release();
            gIn_SobolUint = null;

            _accumulationBuffer?.Release();
            _accumulationBuffer = null;
            _hashEntriesBuffer?.Release();
            _hashEntriesBuffer = null;
            _resolvedBuffer?.Release();
            _resolvedBuffer = null;
        }
    }
}