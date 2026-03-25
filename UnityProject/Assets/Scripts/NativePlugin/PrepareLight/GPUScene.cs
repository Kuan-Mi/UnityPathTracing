using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Rtxdi;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using Object = UnityEngine.Object;

namespace RTXDI
{
    [StructLayout(LayoutKind.Sequential)]
    public struct PrimitiveData
    {
        public float2 uv0;
        public float2 uv1;
        public float2 uv2;

        public float3 pos0;
        public float3 pos1;
        public float3 pos2;

        public uint instanceId;
    }

    // 对应不是每个 MeshRenderer，而是每个发光 SubMesh 实例（一个 Renderer 多个发光 SubMesh 会产生多条记录）
    // 是transform + 材质信息
    [StructLayout(LayoutKind.Sequential)]
    public struct InstanceData
    {
        public float4x4 transform;

        public float3 emissiveColor;
        public uint emissiveTextureIndex;
    }


    public enum PolymorphicLightType
    {
        kSphere = 0,
        kCylinder,
        kDisk,
        kRect,
        kTriangle,
        kDirectional,
        kEnvironment,
        kPoint
    };


    [StructLayout(LayoutKind.Sequential)]
    public struct PolymorphicLightInfo
    {
        // uint4[0]
        public float3 center;
        public uint colorTypeAndFlags; // RGB8 + uint8 (see the kPolymorphicLight... constants above)

        // uint4[1]
        public uint direction1; // oct-encoded
        public uint direction2; // oct-encoded
        public uint scalars; // 2x float16
        public uint logRadiance; // uint16

        // uint4[2] -- optional, contains only shaping data
        public uint iesProfileIndex;
        public uint primaryAxis; // oct-encoded
        public uint cosConeAngleAndSoftness; // 2x float16
        public uint padding;

        public void SetColorAndType(Color color, PolymorphicLightType typeCode)
        {
            const float kMinLog2Radiance = -8f;
            const float kMaxLog2Radiance = 40f;
            const uint kTypeShift = 24u;

            float intensity = math.max(color.r, math.max(color.g, color.b));
            if (intensity > 0f)
            {
                // log-encode radiance (mirrors packLightColor in HLSL)
                float normalizedLog = math.saturate(
                    (math.log2(intensity) - kMinLog2Radiance) / (kMaxLog2Radiance - kMinLog2Radiance));
                uint packedRadiance = (uint)math.min((uint)math.ceil(normalizedLog * 65534f) + 1u, 0xFFFFu);
                logRadiance = packedRadiance; // stored in low 16 bits

                // decode back to find actual encoded intensity, then normalize color to [0,1]
                float unpackedIntensity = math.exp2(
                    ((packedRadiance - 1f) / 65534f) * (kMaxLog2Radiance - kMinLog2Radiance) + kMinLog2Radiance);
                float3 normalizedColor = math.saturate(new float3(color.r, color.g, color.b) / unpackedIntensity);

                // Pack_R8G8B8_UFLOAT: R in bits[0:7], G in bits[8:15], B in bits[16:23]
                uint r = (uint)math.round(normalizedColor.x * 255f) & 0xFFu;
                uint g = (uint)math.round(normalizedColor.y * 255f) & 0xFFu;
                uint b = (uint)math.round(normalizedColor.z * 255f) & 0xFFu;
                colorTypeAndFlags = ((uint)typeCode) << (int)kTypeShift | r | (g << 8) | (b << 16);
            }
            else
            {
                logRadiance = 0;
                colorTypeAndFlags = ((uint)typeCode) << (int)kTypeShift;
            }
        }
    };


    public class GPUScene : System.IDisposable
    {
        // MeshLight 缓存
        private static HashSet<MeshLight> MeshLightCache = new HashSet<MeshLight>();

        public static void RegisterMeshLight(MeshLight ml)
        {
            MeshLightCache.Add(ml);
            Instance?.MarkSceneDirty();
            Debug.Log($"Registered MeshLight: {ml.name}. Total count: {MeshLightCache.Count}");
        }

        public static void UnregisterMeshLight(MeshLight ml)
        {
            MeshLightCache.Remove(ml);
            Instance?.MarkSceneDirty();
            Debug.Log($"Unregistered MeshLight: {ml.name}. Total count: {MeshLightCache.Count}");
        }

        // 单例引用（可选，方便通知）
        public static GPUScene Instance { get; private set; }

        public GPUScene()
        {
            Instance = this;
        }


        public bool isBufferInitialized;

        public void InitBuffer()
        {
            _instanceBuffer = new ComputeBuffer(8192, Marshal.SizeOf<InstanceData>());
            _primitiveBuffer = new ComputeBuffer(327680, Marshal.SizeOf<PrimitiveData>());
            _lightInfoBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 327680, Marshal.SizeOf<PolymorphicLightInfo>());
            _geometryInstanceToLight = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 10240, sizeof(uint));
            isBufferInitialized = true;
        }

        // 存储最终传给 BindlessPlugin 的总列表
        public List<Texture2D> globalTexturePool = new List<Texture2D>();

        // 缓存已添加的纹理组，避免重复上传相同的材质纹理组合
        private Dictionary<string, uint> textureGroupCache = new Dictionary<string, uint>();

        // private Dictionary<int, List<uint>> meshPrimitiveCache = new Dictionary<int, List<uint>>();

        // public uint emissiveMeshCount;
        public uint emissiveTriangleCount;
        public uint otherLightCount;

        public uint numLights => emissiveTriangleCount + otherLightCount;
        // public uint instanceCount;

        private uint GetTextureGroupIndex(Material mat)
        {
            if (mat == null) return 0;

            Texture2D texEmission = (Texture2D)mat.GetTexture("_EmissionMap") ?? Texture2D.whiteTexture;

            string key = $"{texEmission.GetInstanceID()}";

            if (textureGroupCache.TryGetValue(key, out uint startIndex))
            {
                return startIndex;
            }

            startIndex = (uint)globalTexturePool.Count;
            globalTexturePool.Add(texEmission); // index + 0

            textureGroupCache.Add(key, startIndex);
            return startIndex;
        }

        public ComputeBuffer _instanceBuffer;
        public ComputeBuffer _primitiveBuffer;
        public GraphicsBuffer _lightInfoBuffer;
        public GraphicsBuffer _geometryInstanceToLight;

        private List<InstanceData> instanceDataList = new List<InstanceData>();
        private List<PrimitiveData> primitiveDataList = new List<PrimitiveData>();


        List<uint> geometryInstanceToLightArray = new List<uint>();

        // Mesh 数据缓存：避免每帧重复分配 vertices/uv/triangles 数组
        private struct MeshCache
        {
            public Vector3[] vertices;
            public Vector2[] uvs;
            public int[][] trianglesPerSubMesh;
        }

        private Dictionary<int, MeshCache> _meshDataCache = new Dictionary<int, MeshCache>();

        // 每个 instance 对应的 Renderer（一个 Renderer 多个自发光 SubMesh 会产生多条记录）
        private List<Renderer> _emissiveInstanceRenderers = new List<Renderer>();

        // private List<Light> _pointLightCache = new List<Light>();

        // 场景拓扑脏标记：true 时完整重建，false 时仅更新 transform
        private bool _sceneTopologyDirty = true;

        /// <summary>场景物体增减、材质/Mesh 变化时调用，触发下一帧完整重建。</summary>
        public void MarkSceneDirty()
        {
            Debug.Log("Scene marked dirty. Will trigger full rebuild on next Build() call.");
            _sceneTopologyDirty = true;
        }

        public void Build(RayTracingAccelerationStructure ras)
        {
            // 收集场景内所有启用的点光源，打包成 PolymorphicLightInfo，追加在三角面光之后
            var currentLights = Object.FindObjectsByType<Light>(FindObjectsSortMode.None)
                .Where(l => l != null && l.enabled && l.type == LightType.Rectangle)
                .ToList();

            otherLightCount = (uint)currentLights.Count;


            if (_sceneTopologyDirty)
            {
                BuildFull();
                UpdateInstanceID(ras);
                _sceneTopologyDirty = false;
            }
            else
            {
                UpdateTransformsOnly();
            }

            if (otherLightCount > 0)
            {
                UpdateAreaLight(currentLights);
            }
        }

        public Dictionary<MeshRenderer, uint> rendererInstanceIdMap = new Dictionary<MeshRenderer, uint>();

        uint instanceId = 0;

        public RTHandle localLightPdfTexture;
        public uint2 localLightPdfTextureSize;

        // 完整重建：刷新 renderer 列表、primitive 数据、instance 数据，上传两个 buffer
        private void BuildFull()
        {
            instanceDataList.Clear();
            primitiveDataList.Clear();
            globalTexturePool.Clear();
            textureGroupCache.Clear();
            _emissiveInstanceRenderers.Clear();
            _meshDataCache.Clear();
            geometryInstanceToLightArray.Clear();

            instanceId = 0;


            // 只遍历 MeshLightCache
            var meshLights = MeshLightCache.ToList();
            foreach (var ml in meshLights)
            {
                if (ml == null || !ml.enabled || ml.Mesh == null || ml.Materials == null)
                    continue;

                rendererInstanceIdMap[ml.Renderer] = instanceId;
                Mesh mesh = ml.Mesh;
                int subMeshCount = mesh.subMeshCount;
                Material[] sharedMaterials = ml.Materials;
                MeshCache mc = GetOrCacheMeshData(mesh);
                Matrix4x4 localToWorld = ml.transform.localToWorldMatrix;
                for (int subIdx = 0; subIdx < subMeshCount; subIdx++)
                {
                    instanceId++;
                    geometryInstanceToLightArray.Add((uint)primitiveDataList.Count);

                    Material mat = subIdx < sharedMaterials.Length ? sharedMaterials[subIdx] : null;
                    if (mat == null && sharedMaterials.Length > 0) mat = sharedMaterials[^1];
                    bool isEmissive = mat != null && mat.IsKeywordEnabled("_EMISSION") && mat.GetColor("_EmissionColor").maxColorComponent > 0.01f;
                    if (!isEmissive)
                        continue;
                    int[] subMeshTriangles = mc.trianglesPerSubMesh[subIdx];
                    uint instanceIndex = (uint)instanceDataList.Count;


                    for (int t = 0; t < subMeshTriangles.Length; t += 3)
                    {
                        int i0 = subMeshTriangles[t];
                        int i1 = subMeshTriangles[t + 1];
                        int i2 = subMeshTriangles[t + 2];
                        primitiveDataList.Add(new PrimitiveData
                        {
                            pos0 = new float3(mc.vertices[i0]),
                            pos1 = new float3(mc.vertices[i1]),
                            pos2 = new float3(mc.vertices[i2]),
                            uv0 = new float2(mc.uvs[i0]),
                            uv1 = new float2(mc.uvs[i1]),
                            uv2 = new float2(mc.uvs[i2]),
                            instanceId = instanceIndex
                        });
                    }

                    uint baseTextureIndex = GetTextureGroupIndex(mat);
                    var emissiveColor = mat.GetColor("_EmissionColor").linear;
                    instanceDataList.Add(new InstanceData
                    {
                        transform = localToWorld,
                        emissiveTextureIndex = baseTextureIndex,
                        emissiveColor = new float3(emissiveColor.r, emissiveColor.g, emissiveColor.b)
                    });
                    _emissiveInstanceRenderers.Add(ml.Renderer);
                }
            }

            if (instanceDataList.Count > _instanceBuffer.count)
                Debug.LogError($"Instance count {instanceDataList.Count} exceeds buffer capacity {_instanceBuffer.count}!");
            if (primitiveDataList.Count > _primitiveBuffer.count)
                Debug.LogError($"Primitive count {primitiveDataList.Count} exceeds buffer capacity {_primitiveBuffer.count}!");

            _instanceBuffer.SetData(instanceDataList);
            _primitiveBuffer.SetData(primitiveDataList);


            emissiveTriangleCount = (uint)primitiveDataList.Count;

            uint maxLocalLights = emissiveTriangleCount + otherLightCount;
            Rtxdi.RtxdiUtils.ComputePdfTextureSize(maxLocalLights, out uint texWidth, out uint texHeight, out uint mipLevels);

            if ((localLightPdfTextureSize.x != texWidth || localLightPdfTextureSize.y != texHeight))
            {
                localLightPdfTexture?.Release();

                localLightPdfTextureSize = new uint2(texWidth, texHeight);

                var textureDesc = new RenderTextureDescriptor((int)texWidth, (int)texHeight, RenderTextureFormat.RFloat)
                {
                    dimension = TextureDimension.Tex2D,
                    enableRandomWrite = true,
                    useMipMap = true,
                    autoGenerateMips = false,
                    useDynamicScale = false,
                    mipCount = (int)mipLevels,
                };

                localLightPdfTexture = RTHandles.Alloc(textureDesc);
            }

            // localLightPdfTexture = RTHandles.Alloc(
            //     name: "LocalLightPDFTexture",
            //     dimension: TextureDimension.Tex2D,
            //     colorFormat: GraphicsFormat.R32_SFloat,
            //     width: (int)texWidth,
            //     height: (int)texHeight,
            //     enableRandomWrite: true,
            //     useMipMap: true,
            //     autoGenerateMips:false,
            //     useDynamicScale: false
            //     );


            // Debug.Log($"BuildFull completed: {instanceDataList.Count} instances, {primitiveDataList.Count} primitives, {globalTexturePool.Count} unique emissive textures.");
        }

        private void UpdatePointLight(List<Light> currentLights)
        {
            var pointLightInfos = new PolymorphicLightInfo[otherLightCount];
            for (int i = 0; i < currentLights.Count; i++)
            {
                Light light = currentLights[i];

                // flux = linear color * intensity（点光源 flux 直接用于 radiance = flux / r²）
                Color linearColor = light.color.linear;
                float3 flux = new float3(linearColor.r, linearColor.g, linearColor.b) * light.intensity;

                pointLightInfos[i] = PackPointLightInfo(light);
            }

            // SetData 支持 offset，将点光源追加在三角灯之后
            _lightInfoBuffer.SetData(pointLightInfos, 0, (int)emissiveTriangleCount, (int)otherLightCount);
        }

        private void UpdateAreaLight(List<Light> currentLights)
        {
            var pointLightInfos = new PolymorphicLightInfo[otherLightCount];
            for (var i = 0; i < currentLights.Count; i++)
            {
                pointLightInfos[i] = PackAreaLightInfo(currentLights[i]);
            }

            // SetData 支持 offset，将点光源追加在三角灯之后
            _lightInfoBuffer.SetData(pointLightInfos, 0, (int)emissiveTriangleCount, (int)otherLightCount);
        }

        public void UpdateInstanceID(RayTracingAccelerationStructure ras)
        {
            foreach (var keyValuePair in rendererInstanceIdMap)
            {
                MeshRenderer renderer = keyValuePair.Key;
                uint instanceId = keyValuePair.Value;

                if (renderer == null)
                    continue;

                ras.UpdateInstanceID(renderer, instanceId);
                Debug.Log($"Updated InstanceID for Renderer {renderer.name} to {instanceId}");
            }


            var otherRenderers = Object.FindObjectsByType<MeshRenderer>(FindObjectsSortMode.None).Where(r => !rendererInstanceIdMap.ContainsKey(r)).ToList();

            foreach (var renderer in otherRenderers)
            {
                if (renderer == null)
                    continue;

                var meshFilter = renderer.GetComponent<MeshFilter>();
                if (meshFilter == null || meshFilter.sharedMesh == null)
                    continue;

                ras.UpdateInstanceID(renderer, instanceId);
                var subMeshCount = meshFilter.sharedMesh.subMeshCount;

                for (int i = 0; i < subMeshCount; i++)
                {
                    geometryInstanceToLightArray.Add(0xffffffffu);
                }

                instanceId += (uint)subMeshCount;
                // Debug.Log($"Assigned new InstanceID {instanceId} to Renderer {renderer.name} (not in MeshLightCache)");
            }


            _geometryInstanceToLight.SetData(geometryInstanceToLightArray);

            for (var i = 0; i < geometryInstanceToLightArray.Count; i++)
            {
                Debug.Log($"{i} starts at Primitive index {geometryInstanceToLightArray[i]}");
            }
        }

        // 仅更新动态 transform，不重建 primitive，每帧开销极小
        private void UpdateTransformsOnly()
        {
            for (int i = 0; i < _emissiveInstanceRenderers.Count; i++)
            {
                if (_emissiveInstanceRenderers[i] == null)
                {
                    // Renderer 已被销毁（如场景切换），回退到完整重建
                    BuildFull();
                    return;
                }

                var inst = instanceDataList[i];
                inst.transform = _emissiveInstanceRenderers[i].transform.localToWorldMatrix;
                instanceDataList[i] = inst;
            }

            _instanceBuffer.SetData(instanceDataList);
        }

        private MeshCache GetOrCacheMeshData(Mesh mesh)
        {
            int id = mesh.GetInstanceID();
            if (!_meshDataCache.TryGetValue(id, out MeshCache cache))
            {
                int subCount = mesh.subMeshCount;
                cache = new MeshCache
                {
                    vertices = mesh.vertices,
                    uvs = mesh.uv,
                    trianglesPerSubMesh = new int[subCount][]
                };
                for (int i = 0; i < subCount; i++)
                    cache.trianglesPerSubMesh[i] = mesh.GetTriangles(i);
                _meshDataCache[id] = cache;
            }

            return cache;
        }

        #region Debugging Helpers

        public void DebugReadback()
        {
            if (_lightInfoBuffer == null || primitiveDataList.Count == 0)
            {
                Debug.LogError("Buffer not initialized yet.");
                return;
            }

            // 1. 准备接收数组
            PolymorphicLightInfo[] debugData = new PolymorphicLightInfo[primitiveDataList.Count];

            // 2. 从 GPU 同步回读数据 (注意：这会阻塞 CPU，仅调试用)
            _lightInfoBuffer.GetData(debugData);

            // 3. 检查数据
            int validCount = 0;
            for (int i = 0; i < debugData.Length; i++)
            {
                var info = debugData[i];

                validCount++;
                // if (validCount <= 5) // 只打印前 5 个有效数据避免刷屏

                // 解码 radiance：logRadiance 低16位存 log 空间强度，colorTypeAndFlags 低24位存 RGB8 归一化颜色
                uint logRad = info.logRadiance & 0xFFFFu;
                float intensity = (logRad == 0) ? 0f : math.exp2(((logRad - 1) / 65534f) * 48f - 8f);

                if (intensity < 0.01f)
                    continue;

                float normR = (info.colorTypeAndFlags & 0xFFu) / 255f;
                float normG = ((info.colorTypeAndFlags >> 8) & 0xFFu) / 255f;
                float normB = ((info.colorTypeAndFlags >> 16) & 0xFFu) / 255f;
                var c = new Color(normR * intensity, normG * intensity, normB * intensity, 1f);

                // center 已经是 world-space 质心
                var center = new Vector3(info.center.x, info.center.y, info.center.z);

                // direction1/direction2 存 oct-encoded normalize(edge1) / normalize(edge2)，叉积得法线
                float3 edge1Dir = OctUnorm32ToDir(info.direction1);
                float3 edge2Dir = OctUnorm32ToDir(info.direction2);
                var normalDir = Vector3.Cross(edge1Dir, edge2Dir).normalized;


                Debug.Log($"[P {i}] C: {center}, R: {c}, " + $"N: {normalDir}, ");

                c.a = 1.0f;


                Debug.DrawLine(center, center + normalDir * intensity / 10, c, 10);
            }

            Debug.Log($"Total primitives with data: {validCount} / {debugData.Length}");
        }

        /// <summary>
        /// 逆变换 ndirToOctUnorm32：将 oct-encoded uint 解码回单位方向向量。
        /// 编码：p = ndirToOctSigned(n); p = saturate(p*0.5+0.5); packed = uint(p.x*0xfffe) | (uint(p.y*0xfffe)&lt;&lt;16)
        /// </summary>
        private static float3 OctUnorm32ToDir(uint packed)
        {
            float px = (packed & 0xFFFFu) / (float)0xFFFEu;
            float py = (packed >> 16) / (float)0xFFFEu;
            // 还原到 signed [-1,1]
            float2 p = new float2(px * 2f - 1f, py * 2f - 1f);
            float3 n = new float3(p.x, p.y, 1f - math.abs(p.x) - math.abs(p.y));
            if (n.z < 0f)
            {
                // octWrap: (1 - |p.yx|) * sign(p.xy)
                float2 wrap = (1f - math.abs(p.yx)) * math.select(-1f, 1f, p.xy >= 0f);
                n.x = wrap.x;
                n.y = wrap.y;
            }

            return math.normalize(n);
        }


        /// <summary>
        /// 将点光源打包成 PolymorphicLightInfo（对应 HLSL 的 packLightColor + type=kPoint）。
        /// 参考 PolymorphicLight.hlsl：packLightColor / PointLight::Store。
        /// kPoint = 7, kPolymorphicLightTypeShift = 24
        /// kPolymorphicLightMinLog2Radiance = -8, kPolymorphicLightMaxLog2Radiance = 40
        /// </summary>
        private static PolymorphicLightInfo PackPointLightInfo(Light point)
        {
            var info = new PolymorphicLightInfo
            {
                center = point.transform.position,
                direction1 = 0,
                direction2 = 0,
                scalars = 0,
                logRadiance = 0,
                iesProfileIndex = 0,
                primaryAxis = 0,
                cosConeAngleAndSoftness = 0,
                padding = 0,
            };

            info.SetColorAndType(point.color * point.intensity, PolymorphicLightType.kPoint);

            return info;
        }

        [StructLayout(LayoutKind.Explicit)]
        private struct FloatUIntUnion
        {
            [FieldOffset(0)] public uint ui;
            [FieldOffset(0)] public float f;
        }

        private static readonly FloatUIntUnion Multiple = new FloatUIntUnion { ui = 0x07800000 }; // 2^-112

        public static ushort Fp32ToFp16(float v)
        {
            FloatUIntUnion biasedFloat = new FloatUIntUnion {   };
            biasedFloat.f = v * Multiple.f;
            uint u = biasedFloat.ui;

            uint sign = u & 0x80000000;
            uint body = u & 0x0fffffff;

            // C# 中的 uint 右移是逻辑右移（高位补0），与 C++ unsigned 行为一致
            return (ushort)((sign >> 16 | body >> 13) & 0xFFFF);
        }
        /// <summary>
        /// 将单位向量压缩为 32 位整数 (2x16-bit 格式)
        /// </summary>
        public static uint PackNormalizedVector(Vector3 x)
        {
            // 1. 将单位向量投影到八面体
            Vector2 xy = UnitVectorToOctahedron(x);

            // 2. 将范围从 [-1, 1] 映射到 [0, 1]
            xy.x = xy.x * 0.5f + 0.5f;
            xy.y = xy.y * 0.5f + 0.5f;

            // 3. 量化为 16 位无符号整数 (0 - 65535)
            uint ux = FloatToUInt(Saturate(xy.x), (1 << 16) - 1);
            uint uy = FloatToUInt(Saturate(xy.y), (1 << 16) - 1);

            // 4. 打包：X 占低 16 位，Y 占高 16 位
            return ux | (uy << 16);
        }

        /// <summary>
        /// 八面体投影实现 (Octahedral Mapping)
        /// 这是原 C++ 中 unitVectorToOctahedron 的典型实现
        /// </summary>
        private static Vector2 UnitVectorToOctahedron(Vector3 v)
        {
            float invL1Norm = 1.0f / (MathF.Abs(v.x) + MathF.Abs(v.y) + MathF.Abs(v.z));
            Vector2 res = new Vector2(v.x * invL1Norm, v.y * invL1Norm);

            if (v.z < 0)
            {
                // 如果在下半球，进行特殊的折叠处理
                float oldX = res.x;
                res.x = (1.0f - MathF.Abs(res.y)) * (res.x >= 0 ? 1.0f : -1.0f);
                res.y = (1.0f - MathF.Abs(oldX)) * (res.y >= 0 ? 1.0f : -1.0f);
            }
            return res;
        }

        /// <summary>
        /// 模拟 HLSL 的 saturate 函数，限制在 [0, 1]
        /// </summary>
        private static float Saturate(float val) => Math.Clamp(val, 0f, 1f);

        /// <summary>
        /// 将 0-1 的浮点数转为指定范围的整数
        /// </summary>
        private static uint FloatToUInt(float val, uint maxInt)
        {
            return (uint)(val * maxInt + 0.5f);
        }

        private static PolymorphicLightInfo PackAreaLightInfo(Light rect)
        {
            float surfaceArea = rect.areaSize.x * rect.areaSize.y;

            var radiance = rect.color * rect.intensity / surfaceArea;

            var transform = rect.transform;
            var right = transform.right;
            var up = transform.up;

            var info = new PolymorphicLightInfo();
            info.SetColorAndType(radiance, PolymorphicLightType.kRect);
            info.center = transform.position;
            info.scalars = (uint) (Fp32ToFp16(rect.areaSize.x) |  (Fp32ToFp16(rect.areaSize.y) << 16));
            info.direction1 = PackNormalizedVector(right);
            info.direction2 = PackNormalizedVector(up);
            
            return info;
        }

        #endregion

        // ~GPUScene()
        // {
        //     // Debug.LogWarning(" ~GPUScene ");
        //     // Dispose();
        // }

        public void Dispose()
        {
            _instanceBuffer?.Dispose();
            _primitiveBuffer?.Dispose();
            _lightInfoBuffer?.Dispose();
            _meshDataCache.Clear();

            localLightPdfTexture?.Release();
        }
    }
}