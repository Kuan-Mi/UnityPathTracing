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

    [StructLayout(LayoutKind.Sequential, Pack = 16)]
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
    };

 
    public class GPUScene : IDisposable
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
            _primitiveBuffer = new ComputeBuffer(127680, Marshal.SizeOf<PrimitiveData>());
            _lightInfoBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 127680, Marshal.SizeOf<PolymorphicLightInfo>());
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

            uint maxLocalLights = emissiveTriangleCount;
            Rtxdi.RtxdiUtils.ComputePdfTextureSize(maxLocalLights, out uint texWidth, out uint texHeight, out uint mipLevels);

            localLightPdfTextureSize = new uint2(texWidth, texHeight);
            localLightPdfTexture?.Release();

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


            Debug.Log($"BuildFull completed: {instanceDataList.Count} instances, {primitiveDataList.Count} primitives, {globalTexturePool.Count} unique emissive textures.");
        }

        public void UpdateInstanceID(RayTracingAccelerationStructure ras)
        {
            if (_meshDataCache.Count == 0)
            {
                return;
            }

            foreach (var keyValuePair in rendererInstanceIdMap)
            {
                MeshRenderer renderer = keyValuePair.Key;
                uint instanceId = keyValuePair.Value;

                if (renderer == null)
                    continue;

                ras.UpdateInstanceID(renderer, instanceId);
                // Debug.Log($"Updated InstanceID for Renderer {renderer.name} to {instanceId}");
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

            // for (var i = 0; i < geometryInstanceToLightArray.Count; i++)
            // {
            //     Debug.Log($"{i} starts at Primitive index {geometryInstanceToLightArray[i]}");
            // }
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

        public static Color UnpackRadiance(uint2 packed)
        {
            // packed.x 包含 R (低16位) 和 G (高16位)
            // packed.y 包含 B (低16位) 和 A (高16位) - 虽然你的 HLSL 填的是 0，但为了完整性解出来是 Alpha

            // 1. 提取 R (x 的低 16 位)
            ushort r_bits = (ushort)(packed.x & 0xFFFF);

            // 2. 提取 G (x 的高 16 位)
            ushort g_bits = (ushort)((packed.x >> 16) & 0xFFFF);

            // 3. 提取 B (y 的低 16 位)
            ushort b_bits = (ushort)(packed.y & 0xFFFF);

            // 4. 提取 A (y 的高 16 位)
            ushort a_bits = (ushort)((packed.y >> 16) & 0xFFFF);

            // 5. 转换回 float
            float r = Mathf.HalfToFloat(r_bits);
            float g = Mathf.HalfToFloat(g_bits);
            float b = Mathf.HalfToFloat(b_bits);
            float a = Mathf.HalfToFloat(a_bits);

            return new Color(r, g, b, a);
        }

        private float3 UnpackOctDirection(uint packed)
        {
            // 1. 提取低16位和高16位
            uint ux = packed & 0xFFFF;
            uint uy = (packed >> 16) & 0xFFFF;

            // 2. 归一化回 [0, 1] 范围
            // Shader 中乘以了 0xfffe (65534.0)，所以这里除以 65534.0
            float u = ux / 65534.0f;
            float v = uy / 65534.0f;

            // 3. 映射回 [-1, 1] (Signed Octahedron Space)
            float x = u * 2.0f - 1.0f;
            float y = v * 2.0f - 1.0f;

            // 4. 重建 Z 轴
            // 在八面体表面 |x| + |y| + |z| = 1 => |z| = 1 - (|x| + |y|)
            float z = 1.0f - (math.abs(x) + math.abs(y));

            // 5. 处理背面 (Z < 0) 的折叠 (Wrap)
            // 对应 Shader 中的 octWrap
            if (z < 0)
            {
                float tempX = x;
                // HLSL: (1.f - abs(v.yx)) * select(v.xy >= 0.f, 1.f, -1.f);
                // 注意：HLSL的 select(>=0) 对 0 返回 1，而 math.sign(0) 返回 0，所以这里需手写判断
                x = (1.0f - math.abs(y)) * (tempX >= 0 ? 1.0f : -1.0f);
                y = (1.0f - math.abs(tempX)) * (y >= 0 ? 1.0f : -1.0f);
            }

            // 6. 归一化 (因为 Octahedral 映射不保长，只保方向)
            return math.normalize(new float3(x, y, z));
        }

        // 辅助函数：解包 2x fp16 标量 (Shader 中的 scalars)
        private void UnpackScalars(uint packed, out float s1, out float s2)
        {
            // 这是一个 uint 包含两个 f16
            ushort h1 = (ushort)(packed & 0xFFFF);
            ushort h2 = (ushort)(packed >> 16);
            s1 = math.f16tof32(h1);
            s2 = math.f16tof32(h2);
        }

        public class TriangleLight
        {
            public float3 baseO;
            public float3 edge1;
            public float3 edge2;
            public float3 radiance;
            public float3 normal;
            public float surfaceArea;


            const float kPolymorphicLightMinLog2Radiance = -8.0f;
            const float kPolymorphicLightMaxLog2Radiance = 40.0f;

            static float3 octToNdirUnorm32(uint pUnorm)
            {
                float2 p;
                p.x = Math.Clamp((float)(pUnorm & 0xffff) / 0xfffe, 0.0f, 1.0f);
                p.y = Math.Clamp((float)(pUnorm >> 16) / 0xfffe, 0.0f, 1.0f);
                p.x = p.x * 2.0f - 1.0f;
                p.y = p.y * 2.0f - 1.0f;
                return octToNdirSigned(p);
            }

            static float3 octToNdirSigned(float2 p)
            {
                float3 n = new float3(p.x, p.y, 1.0f - Math.Abs(p.x) - Math.Abs(p.y));
                float t = Math.Max(0, -n.z);

                n.x += n.x >= 0.0f ? -t : t;
                n.y += n.y >= 0.0f ? -t : t;

                return normalize(n);
            }

            static float f16tof32(uint x)
            {
                // 对应 F16_M_BITS=10, F16_E_BITS=5, F16_S_MASK=0x8000 的实现
                uint E_MASK = (1 << 5) - 1;
                int BIAS = (int)E_MASK >> 1;
                float DENORM_SCALE = 1.0f / (1 << (14 + 10));
                float NORM_SCALE = 1.0f / (float)(1 << 10);

                uint s = (x & 0x8000) << 16; // 移位到 32位 float 的符号位
                uint e = (x >> 10) & E_MASK;
                uint m = x & ((1 << 10) - 1);

                float result;
                if (e == 0)
                {
                    result = DENORM_SCALE * m;
                }
                else if (e == E_MASK)
                {
                    // 处理 Infinity 或 NaN
                    uint resI = s | 0x7F800000 | (m << (23 - 10));
                    byte[] bytes = BitConverter.GetBytes(resI);
                    result = BitConverter.ToSingle(bytes, 0);
                }
                else
                {
                    result = 1.0f + (float)m * NORM_SCALE;
                    if (e < BIAS)
                        result /= (float)(1 << (BIAS - (int)e));
                    else
                        result *= (float)(1 << ((int)e - BIAS));
                }

                return (s != 0) ? -result : result;
            }
            static float3 unpackLightColor(PolymorphicLightInfo lightInfo)
            {
                float3 color = Unpack_R8G8B8_UFLOAT(lightInfo.colorTypeAndFlags);
                float radianceValue = unpackLightRadiance(lightInfo.logRadiance & 0xffff);
                return color * radianceValue;
            }

            static float3 Unpack_R8G8B8_UFLOAT(uint rgb)
            {
                float r = (float)(rgb & 0xFF) / 255.0f;
                float g = (float)((rgb >> 8) & 0xFF) / 255.0f;
                float b = (float)((rgb >> 16) & 0xFF) / 255.0f;
                return new float3(r, g, b);
            }
            static float3 cross(float3 a, float3 b)
            {
                return new float3(
                    a.y * b.z - a.z * b.y,
                    a.z * b.x - a.x * b.z,
                    a.x * b.y - a.y * b.x
                );
            }

            static float length(float3 v)
            {
                return (float)Math.Sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
            }

            static float3 normalize(float3 v)
            {
                float len = length(v);
                return len > 0 ? v / len : new float3(0, 0, 0);
            }
            static float unpackLightRadiance(uint logRadiance)
            {
                if (logRadiance == 0) return 0.0f;
                float t = (float)(logRadiance - 1) / 65534.0f;
                float exponent = t * (kPolymorphicLightMaxLog2Radiance - kPolymorphicLightMinLog2Radiance) + kPolymorphicLightMinLog2Radiance;
                return (float)Math.Pow(2.0, exponent);
            }

            public static TriangleLight Create(PolymorphicLightInfo lightInfo)
            {
                TriangleLight triLight = new TriangleLight();

                triLight.edge1 = octToNdirUnorm32(lightInfo.direction1) ;
                triLight.edge2 = octToNdirUnorm32(lightInfo.direction2) ;
                triLight.baseO = lightInfo.center;
                
                
                
                triLight.radiance = unpackLightColor(lightInfo);

                float3 lightNormal = cross(triLight.edge1, triLight.edge2);
                float lightNormalLength = length(lightNormal);

                if (lightNormalLength > 0.0)
                {
                    triLight.surfaceArea = 0.5f * lightNormalLength;
                    triLight.normal = lightNormal / lightNormalLength;
                }
                else
                {
                    triLight.surfaceArea = 0.0f;
                    triLight.normal = 0.0f;
                }

                return triLight;
            }
        }


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
                var lightInfo = debugData[i];
                
                var triLight = TriangleLight.Create(lightInfo);
                
                // 简单检查：如果 radiance 或 center 有非零值，说明 Shader 跑通了
                // 注意：lightInfo.radiance 是 uint2 (fp16 packed)，检查 raw value 即可
                bool hasData =     math.lengthsq(triLight.radiance) > 0;

                if (hasData)
                {
                    validCount++;
                    // if (validCount <= 5) // 只打印前 5 个有效数据避免刷屏
                    { 

                        var c = new Color(triLight.radiance.x, triLight.radiance.y, triLight.radiance.z);
                        
                        Debug.DrawLine(triLight.baseO, triLight.baseO + triLight.normal, c, 10);
                    }
                }
            }

            Debug.Log($"Total primitives with data: {validCount} / {debugData.Length}");

            if (validCount == 0)
            {
                Debug.LogWarning("Shader execution finished, but all data is ZERO. Possible causes:\n" +
                                 "1. Shader didn't run (Barrier issue?)\n" +
                                 "2. Emissive color/texture is black\n" +
                                 "3. Transform matrix placed objects at (0,0,0)");
            }
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