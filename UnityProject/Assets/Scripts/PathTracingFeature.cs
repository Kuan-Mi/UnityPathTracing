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
    public struct InstanceData
    {
        public Matrix4x4 prevObjectToWorld;
        // 你也可以在这里存其他东西，比如材质 ID
    }


    public class PathTracingFeature : ScriptableRendererFeature
    {
        // private List<Renderer> m_ActiveRenderers = new List<Renderer>();
        // private Dictionary<Renderer, Matrix4x4> m_PrevMatrices = new Dictionary<Renderer, Matrix4x4>();
        // private ComputeBuffer m_InstanceDataBuffer;

        //
        // void UpdateAccelerationStructure()
        // {
        //     // 1. 准备数据数组
        //     int count = m_ActiveRenderers.Count;
        //     InstanceData[] data = new InstanceData[count];
        //
        //     // 2. 清除并重新填充 AS
        //     accelerationStructure.ClearInstances();
        //
        //     for (int i = 0; i < count; i++)
        //     {
        //         Renderer r = m_ActiveRenderers[i];
        //
        //         // 获取上一帧矩阵（如果找不到则用当前矩阵兜底）
        //         Matrix4x4 prevM = m_PrevMatrices.ContainsKey(r) ? m_PrevMatrices[r] : r.localToWorldMatrix;
        //
        //         data[i] = new InstanceData { prevObjectToWorld = prevM };
        //
        //         // 构造配置
        //         var config = new RayTracingMeshInstanceConfig(mesh, subMeshIndex, material);
        //
        //         // 关键：将当前索引 i 作为 id 传入
        //         // 这样在 Shader 中调用 InstanceID() 就会返回 i
        //         accelerationStructure.AddInstance(config, r.localToWorldMatrix, prevM, (uint)i);
        //
        //         // 记录本帧矩阵供下一帧使用
        //         m_PrevMatrices[r] = r.localToWorldMatrix;
        //     }
        //
        //     // 3. 更新并上传 GPU Buffer
        //     if (m_InstanceDataBuffer == null || m_InstanceDataBuffer.count != count)
        //     {
        //         m_InstanceDataBuffer?.Release();
        //         m_InstanceDataBuffer = new ComputeBuffer(count, Marshal.SizeOf<InstanceData>());
        //     }
        //
        //     m_InstanceDataBuffer.SetData(data);
        //
        //     // 4. 构建 TLAS
        //     accelerationStructure.Build();
        // }

        private List<Renderer> m_TrackedRenderers = new List<Renderer>();
        private Matrix4x4[] m_PrevMatrices;
        private GraphicsBuffer m_PrevMatrixBuffer;


        // void InitializeTracking()
        // {
        //     // 1. 找到所有符合条件的 Renderer (需要与你的 AS Settings 掩码一致)
        //     // 注意：这里建议根据 Layer 或特定 Component 筛选
        //     Renderer[] allRenderers = GameObject.FindObjectsByType<Renderer>(FindObjectsSortMode.None);
        //
        //     m_TrackedRenderers.Clear();
        //     foreach (var r in allRenderers)
        //     {
        //         // 这里的判断逻辑要和你的 RayTracingModeMask 匹配
        //         // 如果物体设置了 RayTracingMode.Off，Unity AS 就不会包含它
        //         // if (r.rayTracingObjectsConfig != null) // 举例：如果是 HDRP/URP 可能会有特定配置
        //         {
        //             m_TrackedRenderers.Add(r);
        //         }
        //     }
        //
        //     int count = m_TrackedRenderers.Count;
        //     m_PrevMatrices = new Matrix4x4[count];
        //
        //     // 2. 关键：强制设置 InstanceID，使其与数组索引 [i] 一致
        //     for (int i = 0; i < count; i++)
        //     {
        //         // 这一步非常关键：它把 GPU 端的 InstanceID() 和我们的数组索引绑定了
        //         accelerationStructure.UpdateInstanceID(m_TrackedRenderers[i], (uint)i);
        //
        //         // 记录初始矩阵
        //         m_PrevMatrices[i] = m_TrackedRenderers[i].localToWorldMatrix;
        //     }
        //
        //     // 3. 创建 GPU Buffer
        //     m_PrevMatrixBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, count, 64);
        // }

        void Update()
        {
            // 1. 在 Unity 自动 Build 之前或之后（只要在渲染前）同步数据
            // 注意：Automatic 模式下 Unity 会自动处理 Build，但你可以在渲染前更新 Buffer

            int count = m_TrackedRenderers.Count;
            for (int i = 0; i < count; i++)
            {
                // 注意：这里逻辑要清晰
                // 我们在这一帧提交的是“上一帧”的记录
                // 所以先上传 m_PrevMatrices 到 GPU
            }

            m_PrevMatrixBuffer.SetData(m_PrevMatrices);

            // 2. 将 Buffer 传给 Ray Tracing Shader
            // cmd.SetRayTracingBufferParam(rayTracingShader, "MyHitGroup", "_PrevObjectToWorldMatrices", m_PrevMatrixBuffer);

            // 3. 最后，更新 CPU 端的“上一帧”记录，供下帧使用
            for (int i = 0; i < count; i++)
            {
                m_PrevMatrices[i] = m_TrackedRenderers[i].localToWorldMatrix;
            }
        }


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
                SetMask();
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
                renderPassEvent = RenderPassEvent.BeforeRenderingPostProcessing,
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