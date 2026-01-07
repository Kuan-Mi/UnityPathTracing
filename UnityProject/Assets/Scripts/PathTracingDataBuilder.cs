using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using PathTracing;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using Object = UnityEngine.Object;

namespace DefaultNamespace
{
    [StructLayout(LayoutKind.Sequential)]
    public struct PrimitiveData
    {
        public half2 uv0;
        public half2 uv1;
        public half2 uv2;
        public float worldArea;

        public half2 n0;
        public half2 n1;
        public half2 n2;
        public float uvArea;

        public half2 t0;
        public half2 t1;
        public half2 t2;
        public float bitangentSign;
    }
 
    [StructLayout(LayoutKind.Sequential)]
    public struct InstanceData
    {
        // 对应 HLSL 中的 float4
        public float4 mOverloadedMatrix0;
        public float4 mOverloadedMatrix1;
        public float4 mOverloadedMatrix2;

        public half4 baseColorAndMetalnessScale;
        public half4 emissionAndRoughnessScale;

        public half2 normalUvScale;
        public uint textureOffsetAndFlags;
        public uint primitiveOffset;
        public float scale;

        public uint morphPrimitiveOffset;
        public uint unused1;
        public uint unused2;
        public uint unused3;
    }

    public class PathTracingDataBuilder
    {
        // 位偏移定义
        private const int FLAG_FIRST_BIT = 24;
        private const uint NON_FLAG_MASK = (1u << FLAG_FIRST_BIT) - 1;

// 具体的 Flag 位定义 (对应 HLSL 的 0x01, 0x02...)
        private const uint FLAG_NON_TRANSPARENT = 0x01;
        private const uint FLAG_TRANSPARENT = 0x02;
        private const uint FLAG_FORCED_EMISSION = 0x04;
        private const uint FLAG_STATIC = 0x08;
        private const uint FLAG_HAIR = 0x10;
        private const uint FLAG_LEAF = 0x20;
        private const uint FLAG_SKIN = 0x40;
        private const uint FLAG_MORPH = 0x80;


        // 预定义默认纹理，防止材质缺失纹理导致索引错位
        public Texture2D defaultWhite; // 用于 BaseColor
        public Texture2D defaultBlack; // 用于 Emission
        public Texture2D defaultNormal; // 用于 Normal (0.5, 0.5, 1.0)
        public Texture2D defaultMask; // 用于 Roughness/Metalness (R=0, G=Roughness, B=Metal)

        // 存储最终传给 BindlessPlugin 的总列表
        public List<Texture2D> globalTexturePool = new List<Texture2D>();


        // 缓存已添加的纹理组，避免重复上传相同的材质纹理组合
        private Dictionary<string, uint> textureGroupCache = new Dictionary<string, uint>();

        private uint GetTextureGroupIndex(Material mat)
        {
            if (mat == null) return 0;

            // 获取四张纹理，如果为空则使用默认值
            Texture2D texBase = (Texture2D)mat.GetTexture("_BaseMap") ?? defaultWhite;
            Texture2D texMask = (Texture2D)mat.GetTexture("_MetallicGlossMap") ?? defaultMask;
            Texture2D texNormal = (Texture2D)mat.GetTexture("_BumpMap") ?? defaultNormal;
            Texture2D texEmission = (Texture2D)mat.GetTexture("_EmissionMap") ?? defaultBlack;

            // 生成唯一 Key 判断这四张图是否已经成组添加过
            string key = $"{texBase.GetInstanceID()}_{texMask.GetInstanceID()}_{texNormal.GetInstanceID()}_{texEmission.GetInstanceID()}";

            if (textureGroupCache.TryGetValue(key, out uint startIndex))
            {
                return startIndex;
            }

            // 如果没添加过，则按顺序连续存入 4 张
            startIndex = (uint)globalTexturePool.Count;
            globalTexturePool.Add(texBase); // index + 0
            globalTexturePool.Add(texMask); // index + 1
            globalTexturePool.Add(texNormal); // index + 2
            globalTexturePool.Add(texEmission); // index + 3

            textureGroupCache.Add(key, startIndex);
            return startIndex;
        }

        public RayTracingAccelerationStructure accelerationStructure;

        public RayTracingAccelerationStructure.Settings settings;
        public ComputeBuffer _instanceBuffer;
        public ComputeBuffer _primitiveBuffer;

        public Texture2DArray textureArray;

        public List<InstanceData> instanceDataList = new List<InstanceData>();
        public List<PrimitiveData> primitiveDataList = new List<PrimitiveData>();

        [ContextMenu("Build RTAS and Buffers")]
        public void Build()
        {
            defaultWhite = Texture2D.whiteTexture;
            defaultBlack = Texture2D.blackTexture;
            defaultNormal = Texture2D.normalTexture;
            defaultMask = Texture2D.whiteTexture;


            if (accelerationStructure != null)
            {
                accelerationStructure.Release();
                accelerationStructure = null;
            }

            instanceDataList.Clear();
            primitiveDataList.Clear();

            globalTexturePool.Clear();
            textureGroupCache.Clear();

            
            settings = new RayTracingAccelerationStructure.Settings
            {
                managementMode = RayTracingAccelerationStructure.ManagementMode.Automatic,
                rayTracingModeMask = RayTracingAccelerationStructure.RayTracingModeMask.Everything
            };

            accelerationStructure = new RayTracingAccelerationStructure(settings);
            accelerationStructure.Build();

            var renderers = Object.FindObjectsByType<Renderer>(FindObjectsSortMode.None);
            Debug.Log($"Found {renderers.Length} renderers in scene.");


            uint currentPrimitiveOffset = 0;

            // 【关键修改 1】全局 Instance 索引计数器，用于对应 InstanceBuffer 的下标
            int globalInstanceIndexCounter = 0;

            for (int i = 0; i < renderers.Length; i++)
            {
                Renderer r = renderers[i];
                MeshFilter mf = r.GetComponent<MeshFilter>();

                if (mf == null || mf.sharedMesh == null)
                    continue;

                Mesh mesh = mf.sharedMesh;
                int subMeshCount = mesh.subMeshCount;

                Matrix4x4 localToWorld = renderers[i].transform.localToWorldMatrix;

                // --- 构造 Primitive Data (每个三角形一个) ---
                int[] triangles = mesh.triangles;
                Vector3[] vertices = mesh.vertices;
                Vector2[] uvs = mesh.uv;
                Vector3[] normals = mesh.normals;

                mesh.RecalculateTangents();
                Vector4[] tangents = mesh.tangents;
                Material[] sharedMaterials = r.sharedMaterials;


                // 【关键修改 2】设置 RTAS 中该 Renderer 的 InstanceID 为当前的全局计数器基数
                // 这样 shader 中: Index = BaseID + GeometryIndex(SubMeshIndex) 就能对齐了
                accelerationStructure.UpdateInstanceID(r, (uint)globalInstanceIndexCounter);
                Debug.Log($"Updated InstanceID for Renderer '{r.name}' to {globalInstanceIndexCounter}.");


                // 【关键修改 3】遍历 SubMesh
                for (int subIdx = 0; subIdx < subMeshCount; subIdx++)
                {
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

                        prim.n0 = EncodeUnitVector(normals[i0], true);
                        prim.n1 = EncodeUnitVector(normals[i1], true);
                        prim.n2 = EncodeUnitVector(normals[i2], true);

                        // 增加安全检查，防止 UV 数组越界（有些 Mesh 可能没有 UV）
                        prim.uv0 = new half2(uvs[i0]);
                        prim.uv1 = new half2(uvs[i1]);
                        prim.uv2 = new half2(uvs[i2]);

                        // 计算面积
                        Vector3 p0 = vertices[i0];
                        Vector3 p1 = vertices[i1];
                        Vector3 p2 = vertices[i2];

                        Vector3 edge20 = p2 - p0;
                        Vector3 edge10 = p1 - p0;
                        float worldArea = Vector3.Cross(edge20, edge10).magnitude * 0.5f;
                        prim.worldArea = Math.Max(worldArea, 1e-9f);

                        // UV 面积
                        if (uvs.Length > 0)
                        {
                            // 3. 计算 UV 面积 (原版代码逻辑)
                            Vector3 uvEdge20 = uvs[i2] - uvs[i0];
                            Vector3 uvEdge10 = uvs[i1] - uvs[i0];
                            float uvArea = Vector3.Cross(uvEdge20, uvEdge10).magnitude * 0.5f;
                            prim.uvArea = Math.Max(uvArea, 1e-9f);
                        }
                        else
                        {
                            prim.uvArea = 1e-9f;
                        }

                        // 切线
                        {
                            Vector3 tang0 = tangents[i0];
                            Vector3 tang1 = tangents[i1];
                            Vector3 tang2 = tangents[i2];


                            prim.t0 = EncodeUnitVector(tang0, true);
                            prim.t1 = EncodeUnitVector(tang1, true);
                            prim.t2 = EncodeUnitVector(tang2, true);
                            prim.bitangentSign = tangents[i0].w;
                        }

                        primitiveDataList.Add(prim);
                    }


                    // --- 构造 Instance Data (每个 SubMesh 一个) ---
                    InstanceData inst = new InstanceData();


                    // 矩阵部分
                    inst.mOverloadedMatrix0 = new float4(localToWorld.m00, localToWorld.m01, localToWorld.m02, localToWorld.m03);
                    inst.mOverloadedMatrix1 = new float4(localToWorld.m10, localToWorld.m11, localToWorld.m12, localToWorld.m13);
                    inst.mOverloadedMatrix2 = new float4(localToWorld.m20, localToWorld.m21, localToWorld.m22, localToWorld.m23);

                    inst.primitiveOffset = currentPrimitiveOffset;
                    inst.morphPrimitiveOffset = 0;

                    // 获取当前 SubMesh 对应的材质
                    Material mat = null;
                    if (subIdx < sharedMaterials.Length)
                    {
                        mat = sharedMaterials[subIdx];
                    }

                    // 如果材质索引超出了（比如 Mesh 有 3 个 SubMesh 但 Renderer 只填了 1 个材质），通常取最后一个或默认
                    if (mat == null && sharedMaterials.Length > 0) mat = sharedMaterials[sharedMaterials.Length - 1];

                    // 处理材质纹理
                    uint baseTextureIndex = GetTextureGroupIndex(mat);

                    // 处理 Flags
                    uint currentFlags = 0;
                    if (mat != null)
                    {
                        bool isTransparent = mat.renderQueue >= 3000 || mat.IsKeywordEnabled("_SURFACE_TYPE_TRANSPARENT");
                        currentFlags |= isTransparent ? FLAG_TRANSPARENT : FLAG_NON_TRANSPARENT;
                        if (r.gameObject.isStatic) currentFlags |= FLAG_STATIC;
                    }
                    else
                    {
                        currentFlags |= FLAG_NON_TRANSPARENT; // 默认不透明
                    }

                    inst.textureOffsetAndFlags = ((currentFlags & 0xFF) << FLAG_FIRST_BIT) | (baseTextureIndex & NON_FLAG_MASK);

                    // 处理材质属性 Scale
                    if (mat != null)
                    {
                        Color col = mat.HasProperty("_BaseColor") ? mat.GetColor("_BaseColor") : Color.white;
                        // 注意：如果使用的是 Standard Shader，属性名可能是 _Color, _MainTex 等，需根据项目实际 Shader 调整
                        float metalScale = mat.HasProperty("_Metallic") ? mat.GetFloat("_Metallic") : 0.0f;
                        inst.baseColorAndMetalnessScale = new half4(new half(col.r), new half(col.g), new half(col.b), new half(metalScale));

                        Color emi = mat.HasProperty("_EmissionColor") ? mat.GetColor("_EmissionColor") : Color.black;
                        if (mat.IsKeywordEnabled("_EMISSION"))
                        {
                            // 有些 Shader 需要开启 Keyword 才有 Emission
                        }

                        float roughScale = mat.HasProperty("_Smoothness") ? (1.0f - mat.GetFloat("_Smoothness")) : 0.5f;
                        // 如果 Shader 属性叫 _Roughness 直接取即可

                        inst.emissionAndRoughnessScale = new half4(new half(emi.r), new half(emi.g), new half(emi.b), new half(roughScale));

                        Vector2 tiling = mat.mainTextureScale;
                        inst.normalUvScale = new half2(new half(tiling.x), new half(tiling.y));
                    }
                    else
                    {
                        inst.baseColorAndMetalnessScale = new half4(new float4(1, 1, 1, 0));
                        inst.emissionAndRoughnessScale = new half4(new float4(0, 0, 0, 0.5f));
                        inst.normalUvScale = new half2(new half(1), new half(1));
                    }

                    inst.scale = r.transform.lossyScale.x;

                    // 添加到列表
                    instanceDataList.Add(inst);
                    
                    Debug.Log($"Added Instance {instanceDataList.Count - 1}: Renderer '{r.name}', SubMesh {subIdx}, Material '{(mat != null ? mat.name : "null")}', PrimitiveOffset {currentPrimitiveOffset}, Triangles {subMeshTriangles.Length / 3}");

                    // 更新 Offset
                    currentPrimitiveOffset += (uint)(subMeshTriangles.Length / 3);

                    // 更新全局索引
                    globalInstanceIndexCounter++;
                }
            }

 


            _instanceBuffer?.Release();
            _instanceBuffer = new ComputeBuffer(instanceDataList.Count, Marshal.SizeOf<InstanceData>());
            _instanceBuffer.SetData(instanceDataList.ToArray());

            _primitiveBuffer?.Release();
            _primitiveBuffer = new ComputeBuffer(primitiveDataList.Count, Marshal.SizeOf<PrimitiveData>());
            _primitiveBuffer.SetData(primitiveDataList.ToArray());

            accelerationStructure.Build();


            Debug.Log($"Built RTAS. Renderers: {renderers.Length}, Instances: {instanceDataList.Count}, Primitives: {primitiveDataList.Count}");
        }

        float SafeSign(float x)
        {
            return x >= 0.0f ? 1.0f : -1.0f;
        }

        float2 SafeSign(float2 v)
        {
            return new float2(SafeSign(v.x), SafeSign(v.y));
        }


        half2 EncodeUnitVector(float3 v, bool bSigned = false)
        {
            v /= math.dot(math.abs(v), 1.0f);

            float2 octWrap = (1.0f - math.abs(v.yx)) * SafeSign(v.xy);
            v.xy = v.z >= 0.0f ? v.xy : octWrap;

            return new half2(bSigned ? v.xy : 0.5f * v.xy + 0.5f);
        }

        public bool IsEmpty()
        {
            return instanceDataList.Count == 0 || primitiveDataList.Count == 0;
        }
    }
}