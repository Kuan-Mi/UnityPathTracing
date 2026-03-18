using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;
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

    [StructLayout(LayoutKind.Sequential)]
    public struct InstanceData
    {
        public float4x4 transform;

        public float3 emissiveColor;
        public uint emissiveTextureIndex;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct RAB_LightInfo
    {
        // uint4[0]
        public float3 center;
        public uint scalars; // 2x float16

        // uint4[1]
        public uint2 radiance; // fp16x4
        public uint direction1; // oct-encoded
        public uint direction2; // oct-encoded
    };

    public class GPUScene : System.IDisposable
    {
        public bool isBufferInitialized;

        public void InitBuffer()
        {
            
#if UNITY_EDITOR
            UnityEditor.ObjectChangeEvents.changesPublished += (ref UnityEditor.ObjectChangeEventStream stream) =>
            {
                MarkSceneDirty();
            };
#endif
            
            
            _instanceBuffer = new ComputeBuffer(8192, Marshal.SizeOf<InstanceData>());
            _primitiveBuffer = new ComputeBuffer(32768, Marshal.SizeOf<PrimitiveData>());
            _lightInfoBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 32768, Marshal.SizeOf<RAB_LightInfo>());
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

        private List<InstanceData> instanceDataList = new List<InstanceData>();
        private List<PrimitiveData> primitiveDataList = new List<PrimitiveData>();

        // Mesh 数据缓存：避免每帧重复分配 vertices/uv/triangles 数组
        private struct MeshCache
        {
            public Vector3[] vertices;
            public Vector2[] uvs;
            public int[][] trianglesPerSubMesh;
        }
        private Dictionary<int, MeshCache> _meshDataCache = new Dictionary<int, MeshCache>();

        // Renderer 缓存 + 自发光实例引用：避免每帧 FindObjectsByType 和重建 primitive
        private Renderer[] _cachedRenderers;
        private struct EmissiveInstanceRef { public Renderer renderer; }
        private List<EmissiveInstanceRef> _emissiveInstanceRefs = new List<EmissiveInstanceRef>();

        // 场景拓扑脏标记：true 时完整重建，false 时仅更新 transform
        private bool _sceneTopologyDirty = true;

        /// <summary>场景物体增减、材质/Mesh 变化时调用，触发下一帧完整重建。</summary>
        public void MarkSceneDirty() => _sceneTopologyDirty = true;

        public void Build()
        {
            if (_sceneTopologyDirty)
            {
                BuildFull();
                _sceneTopologyDirty = false;
            }
            else
            {
                UpdateTransformsOnly();
            }
        }

        // 完整重建：刷新 renderer 列表、primitive 数据、instance 数据，上传两个 buffer
        private void BuildFull()
        {
            instanceDataList.Clear();
            primitiveDataList.Clear();
            globalTexturePool.Clear();
            textureGroupCache.Clear();
            _emissiveInstanceRefs.Clear();

            _cachedRenderers = Object.FindObjectsByType<Renderer>(FindObjectsSortMode.None);

            foreach (var r in _cachedRenderers)
            {
                MeshFilter mf = r.GetComponent<MeshFilter>();
                if (mf == null || mf.sharedMesh == null)
                    continue;

                Mesh mesh = mf.sharedMesh;
                int subMeshCount = mesh.subMeshCount;
                Matrix4x4 localToWorld = r.transform.localToWorldMatrix;
                Material[] sharedMaterials = r.sharedMaterials;

                // 从缓存获取 Mesh 数据，避免重复分配托管数组
                MeshCache mc = GetOrCacheMeshData(mesh);

                for (int subIdx = 0; subIdx < subMeshCount; subIdx++)
                {
                    Material mat = subIdx < sharedMaterials.Length ? sharedMaterials[subIdx] : null;
                    if (mat == null && sharedMaterials.Length > 0) mat = sharedMaterials[^1];

                    bool isEmissive = mat != null && mat.IsKeywordEnabled("_EMISSION") && mat.GetColor("_EmissionColor").maxColorComponent > 0.01f;
                    if (!isEmissive)
                        continue;

                    // 构造 Primitive Data（本地空间，不随 transform 变化）
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
                            uv0  = new float2(mc.uvs[i0]),
                            uv1  = new float2(mc.uvs[i1]),
                            uv2  = new float2(mc.uvs[i2]),
                            instanceId = instanceIndex
                        });
                    }

                    // 构造 Instance Data
                    uint baseTextureIndex = GetTextureGroupIndex(mat);
                    var emissiveColor = mat.GetColor("_EmissionColor").linear;
                    instanceDataList.Add(new InstanceData
                    {
                        transform            = localToWorld,
                        emissiveTextureIndex = baseTextureIndex,
                        emissiveColor        = new float3(emissiveColor.r, emissiveColor.g, emissiveColor.b)
                    });

                    // 记录引用，供后续逐帧 transform 更新使用
                    _emissiveInstanceRefs.Add(new EmissiveInstanceRef { renderer = r });
                }
            }

            if (instanceDataList.Count > _instanceBuffer.count)
                Debug.LogError($"Instance count {instanceDataList.Count} exceeds buffer capacity {_instanceBuffer.count}!");
            if (primitiveDataList.Count > _primitiveBuffer.count)
                Debug.LogError($"Primitive count {primitiveDataList.Count} exceeds buffer capacity {_primitiveBuffer.count}!");

            _instanceBuffer.SetData(instanceDataList);
            _primitiveBuffer.SetData(primitiveDataList);

            emissiveTriangleCount = (uint)primitiveDataList.Count;
            // Debug.Log($"BuildFull completed: {instanceDataList.Count} instances, {primitiveDataList.Count} primitives, {globalTexturePool.Count} unique emissive textures.");
        }

        // 仅更新动态 transform，不重建 primitive，每帧开销极小
        private void UpdateTransformsOnly()
        {
            for (int i = 0; i < _emissiveInstanceRefs.Count; i++)
            {
                var inst = instanceDataList[i];
                inst.transform = _emissiveInstanceRefs[i].renderer.transform.localToWorldMatrix;
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
                    vertices            = mesh.vertices,
                    uvs                 = mesh.uv,
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

        public void DebugReadback()
        {
            if (_lightInfoBuffer == null || primitiveDataList.Count == 0)
            {
                Debug.LogError("Buffer not initialized yet.");
                return;
            }

            // 1. 准备接收数组
            RAB_LightInfo[] debugData = new RAB_LightInfo[primitiveDataList.Count];

            // 2. 从 GPU 同步回读数据 (注意：这会阻塞 CPU，仅调试用)
            _lightInfoBuffer.GetData(debugData);

            // 3. 检查数据
            int validCount = 0;
            for (int i = 0; i < debugData.Length; i++)
            {
                // 简单检查：如果 radiance 或 center 有非零值，说明 Shader 跑通了
                // 注意：lightInfo.radiance 是 uint2 (fp16 packed)，检查 raw value 即可
                bool hasData = debugData[i].radiance.x != 0 ||
                               debugData[i].radiance.y != 0 ||
                               math.lengthsq(debugData[i].center) > 0;

                if (hasData)
                {
                    validCount++;
                    // if (validCount <= 5) // 只打印前 5 个有效数据避免刷屏
                    {
                        var c = UnpackRadiance(debugData[i].radiance);

                        var vv = c.r + c.g + c.b;
                        if (vv < 0.01f)
                        {
                            continue;
                        }

                        float3 decodedDir1 = UnpackOctDirection(debugData[i].direction1);
                        float3 decodedDir2 = UnpackOctDirection(debugData[i].direction2);

                        var normalDir = math.cross(decodedDir1, decodedDir2);

                        // --- 解包 Scalars (Edge Lengths) ---
                        UnpackScalars(debugData[i].scalars, out float len1, out float len2);


                        Debug.Log($"[Primitive {i}] Center: {debugData[i].center}, Radiance: {c}, " +
                                  $"Dir1: {decodedDir1}, Dir2: {decodedDir2}, Normal: {normalDir}, " +
                                  $"EdgeLengths: ({len1}, {len2})");

                        c.a = 1.0f;

                        if (c is { r: < 0.01f, g: < 0.01f, b: < 0.01f })
                            continue;
                        Debug.DrawLine(debugData[i].center, debugData[i].center + normalDir, c, 10);
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

        public void Dispose()
        {
            _instanceBuffer?.Dispose();
            _primitiveBuffer?.Dispose();
            _lightInfoBuffer?.Dispose();
            _meshDataCache.Clear();
        }
    }
}