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

    public class GPUScene
    {
        public static GPUScene instance;

        public GPUScene()
        {
            instance = this;
        }
        
        public bool isBufferInitialized = false;

        public void InitBuffer()
        {
            _instanceBuffer = new ComputeBuffer(10, Marshal.SizeOf<InstanceData>());
            _primitiveBuffer = new ComputeBuffer(8192, Marshal.SizeOf<PrimitiveData>());
            _lightInfoBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 8192, Marshal.SizeOf<RAB_LightInfo>());
            isBufferInitialized = true;
        }

        // 预定义默认纹理，防止材质缺失纹理导致索引错位
        public Texture2D defaultBlack; // 用于 Emission

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

            Texture2D texEmission = (Texture2D)mat.GetTexture("_EmissionMap") ?? defaultBlack;

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

        public List<InstanceData> instanceDataList = new List<InstanceData>();
        public List<PrimitiveData> primitiveDataList = new List<PrimitiveData>();


        [ContextMenu("Build RTAS and Buffers")]
        public void Build()
        {
            defaultBlack = Texture2D.blackTexture;

            instanceDataList.Clear();
            primitiveDataList.Clear();

            globalTexturePool.Clear();
            textureGroupCache.Clear();

            // meshPrimitiveCache.Clear();

            var renderers = Object.FindObjectsByType<Renderer>(FindObjectsSortMode.None);
            // Debug.Log($"Found {renderers.Length} renderers in scene.");

            foreach (var r in renderers)
            {
                MeshFilter mf = r.GetComponent<MeshFilter>();

                if (mf == null || mf.sharedMesh == null)
                    continue;

                Mesh mesh = mf.sharedMesh;
                int subMeshCount = mesh.subMeshCount;
                // int meshInstanceID = mesh.GetInstanceID(); // 获取 Mesh 唯一 ID

                Matrix4x4 localToWorld = r.transform.localToWorldMatrix;

                // bool isMeshCached = meshPrimitiveCache.TryGetValue(meshInstanceID, out List<uint> cachedOffsets);
                // List<uint> currentMeshOffsets = isMeshCached ? cachedOffsets : new List<uint>();
                List<uint> currentMeshOffsets =  new List<uint>();

                Vector3[] vertices = mesh.vertices;
                Vector2[] uvs = mesh.uv;
                Material[] sharedMaterials = r.sharedMaterials;

                // 【关键修改 3】遍历 SubMesh
                for (int subIdx = 0; subIdx < subMeshCount; subIdx++)
                {
                    // 获取当前 SubMesh 对应的材质
                    Material mat = null;
                    if (subIdx < sharedMaterials.Length)
                    {
                        mat = sharedMaterials[subIdx];
                    }

                    // 如果材质索引超出了（比如 Mesh 有 3 个 SubMesh 但 Renderer 只填了 1 个材质），通常取最后一个或默认
                    if (mat == null && sharedMaterials.Length > 0) mat = sharedMaterials[^1];

                    var isEmissive = mat != null && mat.IsKeywordEnabled("_EMISSION");

                    if (!isEmissive)
                    {
                        // 非自发光材质跳过
                        continue;
                    }

                    uint thisSubMeshPrimitiveOffset = 0;

                    // if (isMeshCached)
                    // {
                    //     if (subIdx < currentMeshOffsets.Count)
                    //     {
                    //         thisSubMeshPrimitiveOffset = currentMeshOffsets[subIdx];
                    //     }
                    //     else
                    //     {
                    //         Debug.LogError($"Mesh Cache mismatch for {r.name}");
                    //         thisSubMeshPrimitiveOffset = 0;
                    //     }
                    // }
                    // else
                    {
                        thisSubMeshPrimitiveOffset = (uint)primitiveDataList.Count;

                        // 记录到缓存列表
                        currentMeshOffsets.Add(thisSubMeshPrimitiveOffset);

                        // 获取当前 SubMesh 的三角形索引
                        // 注意：GetTriangles 返回的是顶点索引，不需要偏移，直接对应 mesh.vertices
                        int[] subMeshTriangles = mesh.GetTriangles(subIdx);

                        // --- 构造 Primitive Data ---
                        for (int t = 0; t < subMeshTriangles.Length; t += 3)
                        {
                            PrimitiveData prim = new PrimitiveData();
                            int i0 = subMeshTriangles[t];
                            int i1 = subMeshTriangles[t + 1];
                            int i2 = subMeshTriangles[t + 2];

                            prim.pos0 = new float3(vertices[i0]);
                            prim.pos1 = new float3(vertices[i1]);
                            prim.pos2 = new float3(vertices[i2]);

                            // 增加安全检查，防止 UV 数组越界（有些 Mesh 可能没有 UV）
                            prim.uv0 = new float2(uvs[i0]);
                            prim.uv1 = new float2(uvs[i1]);
                            prim.uv2 = new float2(uvs[i2]);

                            prim.instanceId = (uint)instanceDataList.Count; // 当前 Instance 的索引

                            primitiveDataList.Add(prim);
                        }
                    }


                    // --- 构造 Instance Data (每个 SubMesh 一个) ---
                    // 处理材质纹理
                    uint baseTextureIndex = GetTextureGroupIndex(mat);
                    var emissiveColor = mat.GetColor("_EmissionColor").linear;

                    InstanceData inst = new InstanceData
                    {
                        transform = localToWorld,
                        emissiveTextureIndex = baseTextureIndex,
                        emissiveColor = new float3(emissiveColor.r, emissiveColor.g, emissiveColor.b)
                    };

                    // 添加到列表
                    instanceDataList.Add(inst);
                }

                // if (!isMeshCached)
                // {
                //     meshPrimitiveCache.Add(meshInstanceID, currentMeshOffsets);
                // }
            }

            _instanceBuffer.SetData(instanceDataList.ToArray());
            _primitiveBuffer.SetData(primitiveDataList.ToArray());

            emissiveTriangleCount = (uint)primitiveDataList.Count;
            // Debug.Log($"Renderers: {renderers.Length}, Instances: {instanceDataList.Count}, Primitives: {primitiveDataList.Count}");

            Debug.Log($"emissiveTriangleCount: {emissiveTriangleCount}");
        }

        public bool IsEmpty()
        {
            return instanceDataList.Count == 0 || primitiveDataList.Count == 0;
        }

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


        [ContextMenu("Debug Readback Buffer")]
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
    }
}