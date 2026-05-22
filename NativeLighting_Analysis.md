# RTXPT 灯光系统分析文档

> 分析来源：`RenderingPlugin/External/RTXPT/Rtxpt/Lighting/`  
> 目标：为 Unity 端实现灯光收集与烘焙流程提供参考

---

## 1. 总体架构

RTXPT 的灯光系统由以下几个模块组成：

| 模块 | 文件 | 职责 |
|------|------|------|
| **LightsBaker** | `Lighting/LightsBaker.cpp/h` | 核心：收集所有场景灯光、生成 GPU 采样结构、管理 NEE-AT 时间反馈 |
| **EnvMapBaker** | `Lighting/Distant/EnvMapBaker.cpp/h` | 将 HDR 天空盒或程序天空烘焙成 Cubemap，含平行光 |
| **EnvMapImportanceSamplingBaker** | `Lighting/Distant/EnvMapImportanceSamplingBaker.cpp/h` | 从 Cubemap 生成重要性采样图（MIP Importance Map），供 Path Tracer 使用 |
| **LightsBaker.hlsl** | `Lighting/LightsBaker.hlsl` | GPU 端的所有 Compute Shader 实现 |
| **LightingTypes.hlsli** | `Shaders/PathTracer/Lighting/LightingTypes.hlsli` | CPU/GPU 共享数据结构定义 |
| **LightingConfig.h** | `Shaders/PathTracer/Lighting/LightingConfig.h` | 容量上限等编译常数 |

---

## 2. 灯光类型

所有灯光统一打包为 `PolymorphicLightInfo`（**32 字节**，2× uint4）+ `PolymorphicLightInfoEx`（**16 字节**，4× uint）：

### 2.1 PolymorphicLightType 枚举

| 类型 | 说明 |
|------|------|
| `kPoint` | 点光源（radius=0 的精确点） |
| `kSphere` | 球形面积光（radius>0 的点/聚光） |
| `kEnvironmentQuad` | 环境贴图四叉树节点（GPU 动态细分） |
| 发射三角形 | Emissive Triangle（每个三角面片一个灯光条目） |

### 2.2 灯光缓冲区布局（每帧重建）

```
[0 .. EnvmapQuadNodeCount-1]                     : 环境光四叉树节点（GPU 细分填充）
[EnvmapQuadNodeCount .. AnalyticLightCount-1]    : 解析光源（Point/Spot），CPU 端打包
[AnalyticLightCount .. TotalLightCount-1]        : 发射三角形，GPU 端（BakeEmissiveTriangles）
```

最大灯光数：`RTXPT_LIGHTING_MAX_LIGHTS = 512 * 1024`（RTXPT 原版）；Unity 工程中目前设置为 `MaxLights = 8192`

---

## 3. GPU 缓冲区清单

| 缓冲区 | 类型 | 大小 | 说明 |
|--------|------|------|------|
| `m_controlBuffer` | `StructuredBuffer<LightingControlData>` | 1 元素 | 全局控制数据，含计数、采样模式等 |
| `m_lightsBuffer` | `StructuredBuffer<PolymorphicLightInfo>` | MAX_LIGHTS | 灯光主数据（32B/元素） |
| `m_lightsExBuffer` | `StructuredBuffer<PolymorphicLightInfoEx>` | MAX_LIGHTS | 灯光扩展数据（16B/元素，方向、IES 等） |
| `m_lightWeights` | `TypedBuffer<float>` | 2 × WEIGHTS_COUNT_HALF | Ping-pong 权重缓冲（当前帧/历史帧） |
| `m_perLightProxyCounters` | `TypedBuffer<uint>` | MAX_LIGHTS | 每个灯的采样代理计数 |
| `m_lightSamplingProxies` | `TypedBuffer<uint>` | MAX_SAMPLING_PROXIES | 灯光采样代理索引列表（排序后） |
| `m_historyRemapCurrentToPast` | `TypedBuffer<uint>` | MAX_LIGHTS | 当前帧→上一帧索引映射 |
| `m_historyRemapPastToCurrent` | `TypedBuffer<uint>` | MAX_LIGHTS | 上一帧→当前帧索引映射 |
| `m_scratchBuffer` | `RawBuffer` | LLB_SCRATCH_BUFFER_SIZE | 存放发射三角形处理任务 |
| `m_envLightLookupMap` | `Texture2D<uint>` | envMapResolution² | 方向→环境光索引查找图（用于 MIS） |
| `m_NEE_AT_FeedbackTotalWeight` | `Texture2D<float>` | renderRes | NEE-AT 时间反馈：每像素采样权重 |
| `m_NEE_AT_FeedbackCandidates` | `Texture2D<uint>` | renderRes | NEE-AT 时间反馈：每像素候选灯光索引 |
| `m_NEE_AT_LocalSamplingBuffer` | `TypedBuffer<uint>` | tileW × tileH × LOCAL_PROXY_COUNT | 屏幕空间分块的本地灯光采样代理 |

---

## 4. 每帧更新流程

### 4.1 调用顺序（Sample.cpp 中）

```
// 阶段1：在 BVH 建立、denoiser 准备等工作之前/之后皆可
LightsBaker::UpdateBegin(...)      ← 需要发射三角形材质数据可访问

// ... 中间可执行 Path Tracing 相关准备工作 ...

// 阶段2：必须在 Path Tracing 主通道（NEE）之前，且在运动向量可用后
LightsBaker::UpdateEnd(...)        ← 需要深度图 + 运动向量

// 传递给 Path Tracer 的绑定：
//  t12: ControlBuffer (SRV)
//  t13: LightsBuffer (SRV)
//  t14: LightsExBuffer (SRV)
//  t15: PerLightProxyCounters (SRV)
//  t16: LightSamplingProxies (SRV)
//  t17: LocalSamplingBuffer (SRV)
//  t18: EnvLightLookupMap (SRV)
//  u20: FeedbackTotalWeight (UAV, Path Tracer 写入反馈)
//  u21: FeedbackCandidates (UAV, Path Tracer 写入反馈)
```

### 4.2 UpdateBegin 内部阶段

```
1. CPU 端数据收集
   ├── CollectEnvmapLightPlaceholders()  → 在灯光缓冲区头部插入 N 个占位符
   ├── CollectAnalyticLightsCPU()        → 遍历场景 Point/Spot 灯光，打包为 PolymorphicLightInfo
   └── ProcessEmissiveGeometry()         → 为发射材质的每个几何体建立 GPU 批处理任务

2. GPU 上传
   ├── writeBuffer(controlBuffer)         : 写入 LightingControlData（含计数、常数）
   ├── writeBuffer(lightsBuffer)          : 上传环境光占位符 + 解析光源
   ├── writeBuffer(lightsExBuffer)        : 上传对应扩展数据
   ├── writeBuffer(historyRemapCurrentToPast)
   ├── writeBuffer(historyRemapPastToCurrent)
   └── writeBuffer(scratchBuffer)        : 上传发射三角形处理任务

3. GPU Compute Passes（顺序执行）
   ├── ResetLightProxyCounters           : 清零采样代理计数
   ├── ResetPastToCurrentHistory         : 重置历史索引映射
   ├── EnvLightsBackupPast               : 备份上一帧环境光节点（用于时间映射）
   ├── EnvLightsSubdivideBase            : 按环境贴图重要性对四叉树基础层细分
   ├── EnvLightsSubdivideBoost           : 对重要区域额外 boost 细分
   ├── BakeEmissiveTriangles             : 为每个发射三角形计算辐照度、历史映射
   ├── EnvLightFillLookupMap             : 填充方向→环境光索引的查找贴图
   ├── EnvLightsMapPastToCurrent         : 建立环境光四叉树的帧间索引映射
   ├── [若有历史反馈] ProcessFeedbackHistoryPreFilter : 对上帧反馈预滤波
   ├── [若有历史反馈] ProcessFeedbackHistoryP0        : 反馈→代理计数统计
   ├── ComputeWeights                    : 为每个灯光计算采样权重（含重要性 boost）
   ├── ComputeProxyCounts                : 根据权重决定每个灯有几个采样代理
   ├── ComputeProxyBaselineOffsets       : 前缀和，确定每个灯代理在代理缓冲区的起始偏移
   ├── CreateProxyJobs                   : 生成代理构建任务列表
   └── ExecuteProxyJobs                  : 执行任务，填充 m_lightSamplingProxies
```

### 4.3 UpdateEnd 内部阶段（NEE-AT 模式专用）

在运动向量可用之后执行，利用深度图+运动向量做时间重投影：

```
├── ProcessFeedbackHistoryP1a  : 生成低分辨率混合反馈（早期终止优化）
├── ProcessFeedbackHistoryP1b  : 全分辨率时间融合
├── ProcessFeedbackHistoryP2   : 反馈→本地 tile 候选灯光汇总
├── ProcessFeedbackHistoryP3   : 构建本地 tile 采样分布（LocalSamplingBuffer 填充）
└── ClearFeedbackHistory       : 清空反馈缓冲区，等待新一帧 Path Tracer 写入
```

---

## 5. 灯光收集详解

### 5.1 解析光源（Analytic Lights）—— CPU 端

**`CollectAnalyticLightsCPU()`** 遍历场景图中所有 `LightType_Spot` / `LightType_Point`：

- **Point Light（radius=0）**：打包为 `kPoint`，颜色 = flux（`color × intensity`）
- **Point Light（radius>0）**：打包为 `kSphere`，颜色 = radiance（`flux / (π × r²)`）
- **Spot Light（radius=0）**：打包为 `kPoint` + Shaping（cone angle）
- **Spot Light（radius>0）**：打包为 `kSphere` + Shaping（direction、cos cone angle、softness）

颜色打包为 **R8G8B8（归一化色度）+ R16（log2 亮度）**，通过 `packLightColor()` 函数编码。

帧间追踪：通过 `LightSamplerLink`（挂在引擎灯光节点上）存储上一帧的索引，实现时间映射。

### 5.2 环境光（Environment Quad Lights）—— GPU 端细分

**`CollectEnvmapLightPlaceholders()`** 仅在 CPU 端预留 `RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT` 个占位符（最大 8192 个）。

实际数据由以下 GPU Pass 填充：
- **EnvLightsSubdivideBase**：从重要性贴图自顶向下细分四叉树
- **EnvLightsSubdivideBoost**：对高亮区域额外增加子节点（boost 细分）
- **EnvLightFillLookupMap**：将每个叶节点覆盖的方向范围写入 `envLightLookupMap`（供 MIS 使用）

### 5.3 发射三角形（Emissive Triangle Lights）—— GPU 端烘焙

**`ProcessEmissiveGeometry()`** 在 CPU 端为每个发射材质的几何体生成 `EmissiveTrianglesProcTask`：

```cpp
struct EmissiveTrianglesProcTask {
    uint InstanceIndex;
    uint GeometryIndex;
    uint TriangleIndexFrom;
    uint TriangleIndexTo;
    uint DestinationBufferOffset;  // 在 lightsBuffer 中的写入偏移
    uint HistoricBufferOffset;     // 上一帧对应偏移，RTXPT_INVALID_LIGHT_INDEX 表示新增
};
```

GPU 端 **BakeEmissiveTriangles** Compute Shader：
- 读取顶点/索引缓冲区 + 材质（发射颜色、贴图）
- 计算每个三角形的世界空间面积和辐照度
- 写入 `lightsBuffer[DestinationBufferOffset + triangleIndex]`
- 同时填充 `historyRemapCurrentToPast`

**Analytic Light Proxy 机制**：  
如果某网格的材质标记了 `EnableAsAnalyticLightProxy`，则其 `subInstanceData.AnalyticProxyLightIndex` 会被设置为对应解析光的索引。命中该面片时不使用面积光，而直接引用解析光，实现光源网格与解析光的统一采样。

---

## 6. 采样代理（Sampling Proxy）机制

权重→代理数量 是灯光采样均匀化的核心：

1. **ComputeWeights**：按辐照度 × 重要性 boost（视锥内倍增、强度变化倍增）计算每个灯的权重
2. **ComputeProxyCounts**：权重归一化后分配代理数（权重越大，代理越多）
3. **ComputeProxyBaselineOffsets**：并行前缀和，确定每个灯在 `m_lightSamplingProxies` 中的起始位置
4. **CreateProxyJobs + ExecuteProxyJobs**：将各灯的索引填充进代理列表

Path Tracer 在全局采样时，**均匀采样代理列表而非灯光列表**，实现权重加权采样。

---

## 7. NEE-AT（Next Event Estimation with Adaptive Temporal）

NEE-AT 是 RTXPT 最复杂的特性，分为全局采样和本地屏幕空间采样：

### 7.1 反馈写入（Path Tracer 执行时）

Path Tracer 在 NEE 命中光源时，向反馈缓冲区写入：
- `u_LightFeedbackTotalWeight[pixel]`：采样权重（float）
- `u_LightFeedbackCandidates[pixel]`：采样到的灯光索引（uint，带 Screen-Space-Coherent 标记位）

使用**加权蓄水池采样（Weighted Reservoir Sampling）**，每像素只保留权重最高的候选。

### 7.2 反馈处理（UpdateEnd 中）

```
上帧反馈图（全分辨率）
  → ProcessFeedbackHistoryPreFilter: 使用 disocclusion/深度/运动向量剔除无效像素
  → ProcessFeedbackHistoryP0: 更新 perLightProxyCounters（统计各灯被选中的次数）
  → ProcessFeedbackHistoryP1a: 生成低分辨率 tile 级别的早期反馈（用于快速采样）
  → ProcessFeedbackHistoryP1b: 全分辨率时间融合到 scratch 缓冲
  → ProcessFeedbackHistoryP2: 按 tile 汇总候选灯光（使用 Jitter 抖动防止规律化）
  → ProcessFeedbackHistoryP3: 构建每个 tile 的本地采样分布表（LocalSamplingBuffer）
  → ClearFeedbackHistory: 清空，等待本帧 Path Tracer 重新写入
```

### 7.3 本地采样（LocalSamplingBuffer）

屏幕按 `RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE` 分块，每个 tile 存储 `RTXPT_LIGHTING_LOCAL_PROXY_COUNT`（128/256/512）个重要灯光的加权分布，Path Tracer 优先从本地列表采样。

---

## 8. 环境贴图流程（EnvMapBaker + EnvMapImportanceSamplingBaker）

```
输入：等矩阵 HDR 图或程序天空 + 平行光（最多 EMB_MAXDIRLIGHTS 个）
  ↓ EnvMapBaker::Update()
处理后 Cubemap（含平行光烘焙、Mip 链）
  ↓ EnvMapImportanceSamplingBaker::Update()
  ├── 生成 Radiance+Importance 合并贴图（GetRadianceAndImportanceMap()）
  │    格式：各 MIP 存储辐亮度（RGB）+ 重要性权重（A）
  └── 传入 LightsBaker::UpdateBegin() 的 envMapProcessed 参数
```

`EnvMapBaker` 还负责：
- 生成 GGX 预滤波 Cubemap（IBL 高光用）
- 生成 SH 漫反射 Cubemap
- BC6H 压缩
- BRDF LUT 生成

---

## 9. BakeSettings 主要参数（Unity 端需对应提供）

| 参数 | 类型 | 说明 |
|------|------|------|
| `ImportanceSamplingType` | uint | 0=Uniform, 1=Power, 2=NEE-AT |
| `CameraPosition` | float3 | 相机世界位置 |
| `CameraDirection` | float3 | 相机朝向 |
| `AverageContentsDistance` | float | 场景平均深度（FPS 约 10m） |
| `ViewProjMatrix` | float4x4 | 用于视锥裁剪 boost |
| `ViewportSize` / `PrevViewportSize` | float2 | 当前/上一帧分辨率 |
| `FrameIndex` | int64_t | 帧序号（不能为 -1） |
| `ResetFeedback` | bool | 场景变化时重置所有时间历史 |
| `GlobalTemporalFeedbackWeight` | float | 0~0.95，历史反馈权重 |
| `LocalToGlobalSampleRatio` | float | 0~1，本地 vs 全局采样比例 |
| `EnvMapParams` | struct | 天空盒变换、颜色倍增、是否启用 |
| `DistantVsLocalImportanceScale` | float | 环境光 vs 局部光的相对重要性 |

---

## 10. Unity 端实现指南

### 10.1 实际采用的架构（GPU Compute Pipeline）

Unity C# 侧直接使用预编译好的 `.computeshader` 资产（`Assets/RTXPT/Shaders/Lighting/`），通过 `NativeComputePipeline` 驱动完整的代理构建管线，与 C++ `LightsBaker::UpdateBegin()` 逻辑等价：

```
NativeRtxptLightingPass.Setup() [C# 主线程]
  ├── FindObjectsByType<Light>()         → 收集 Point/Spot
  ├── PackPointLight / PackSpotLight     → 填充 s_lightsStaging[]
  │    ├── PackLightColor()              → ColorTypeAndFlags + LogRadiance
  │    ├── NDirToOctUnorm32()            → 方向 oct 编码
  │    └── Fp32ToFp16()                 → fp16 打包
  └── GraphicsBuffer.SetData() × 3
       ├── LightControlBuffer   (TotalLightCount, AnalyticLightCount, ImportanceSamplingType=1,
       │                         WeightsSumUINT=0, _paddingBK[28]=CurrentWeightsOffset,
       │                         _paddingBK[29]=HistoricWeightsOffset)
       ├── LightBuffer
       └── LightExBuffer

NativeRtxptLightingPass.RecordRenderGraph() + ExecutePass() [Render Graph]
  └── AddUnsafePass "NativeRtxpt.LightsBaker"
       ├── 1. ResetLightProxyCounters
       ├── 2. ResetPastToCurrentHistory
       ├── 3. ComputeWeights
       ├── 4. ComputeProxyCounts
       ├── 5. ComputeProxyBaselineOffsets
       ├── 6. CreateProxyJobs
       └── 7. ExecuteProxyJobs
            → GPU 写入 SamplingProxyCount、LightSamplingProxies → Path Tracer 可用
```

`ImportanceSamplingType = 1`（Power-based 权重比例采样），GPU 自动计算权重并分配代理数。

---

### 10.2 GPU Compute Shader 详解

所有 Shader 位于 `Assets/RTXPT/Shaders/Lighting/`，由 `NativeComputeShaderImporter` 在导入时编译为 DXIL。  
所有 Shader 共用同一套 UAV 槽位绑定（u0～u9），不同 Shader 只使用其中一个子集。

**公共 UAV 绑定槽位表：**

| 槽位 | 名称 | 类型 | 描述 |
|------|------|------|------|
| u0 | `u_controlBuffer` | `RWStructuredBuffer<LightingControlData>` | 全局控制数据（计数、权重和、代理数等）|
| u1 | `u_lightsBuffer` | `RWStructuredBuffer<PolymorphicLightInfo>` | 灯光主数据（32B/元素）|
| u2 | `u_lightsExBuffer` | `RWStructuredBuffer<PolymorphicLightInfoEx>` | 灯光扩展数据（16B/元素）|
| u3 | `u_scratchBuffer` | `RWByteAddressBuffer` | 代理构建任务存储（32MB 原始内存）|
| u4 | `u_scratchList` | `RWBuffer<uint>` | 每组代理计数的局部前缀和暂存 |
| u5 | `u_lightWeights` | `RWBuffer<float>` | Ping-pong 权重缓冲（2 × WeightsCountHalf 个 float）|
| u6 | `u_historyRemapCurrentToPast` | `RWBuffer<uint>` | 当前帧→上一帧索引映射 |
| u7 | `u_historyRemapPastToCurrent` | `RWBuffer<uint>` | 上一帧→当前帧索引映射 |
| u8 | `u_perLightProxyCounters` | `RWBuffer<uint>` | 每个灯的代理计数（以及 InterlockedAdd 目标）|
| u9 | `u_lightSamplingProxies` | `RWBuffer<uint>` | 代理索引列表（排序后供 Path Tracer 使用）|

---

#### 10.2.1 ResetLightProxyCounters

**作用：** 清零代理计数缓冲，为本帧代理构建准备干净的初始状态。  
**numthreads：** `[128, 1, 1]`  
**Dispatch：** `ceil((TotalLightCount + 1) / 128, 1, 1)`（+1 是为 "invalid light" 预留的末位槽）

| 方向 | 缓冲区 | 说明 |
|------|--------|------|
| 读 | u0 `u_controlBuffer` | 读取 `TotalLightCount` 作为循环上界 |
| 写 | u8 `u_perLightProxyCounters` | 将 `[0 .. TotalLightCount]` 全部清零 |

```hlsl
if (lightIndex > lightCount) return;   // > 而非 >= 是为了包含 invalid slot
u_perLightProxyCounters[lightIndex] = 0;
```

---

#### 10.2.2 ResetPastToCurrentHistory

**作用：** 重置上一帧→当前帧索引映射，确保当前帧任何未更新的映射不会错误复用历史索引。  
**numthreads：** `[128, 1, 1]`  
**Dispatch：** `ceil(max(HistoricTotalLightCount, TotalLightCount) / 128, 1, 1)`

| 方向 | 缓冲区 | 说明 |
|------|--------|------|
| 读 | u0 `u_controlBuffer` | 读取 `HistoricTotalLightCount` 和 `TotalLightCount` |
| 写 | u7 `u_historyRemapPastToCurrent` | 将覆盖范围内所有槽设为 `RTXPT_INVALID_LIGHT_INDEX` |

---

#### 10.2.3 ComputeWeights

**作用：** 为每个灯光计算采样权重（基于辐照度），并将所有灯光的权重总和写入控制缓冲的 `WeightsSumUINT`（通过 `InterlockedAdd` 进行浮点累加）。  
**numthreads：** `[128, 1, 1]`（每个线程处理 `LLB_LOCAL_BLOCK_SIZE = 32` 个灯光，共 128×32 = 4096 个/组）  
**Dispatch：** `ceil(TotalLightCount / (128 × 32), 1, 1)`，至少 1 组

| 方向 | 缓冲区 | 说明 |
|------|--------|------|
| 读 | u0 `u_controlBuffer` | 读取 `TotalLightCount`、`BakerConstants.CurrentWeightsBufferOffset`、`HistoricWeightsBufferOffset`、`LastFrameTemporalFeedbackAvailable` |
| 读 | u1 `u_lightsBuffer` | 读取每个灯的 `PolymorphicLightInfo`（计算辐照度） |
| 读 | u2 `u_lightsExBuffer` | 读取每个灯的 `PolymorphicLightInfoEx` |
| 读 | u6 `u_historyRemapCurrentToPast` | 找到历史帧对应灯索引，读取历史权重做强度变化 Boost |
| 读/写 | u5 `u_lightWeights` | **写**：`weights[CurrentWeightsBufferOffset + lightIndex]`；读历史半区作 ImportanceBoost |
| 写 | u0 `u_controlBuffer` | `InterlockedAdd` 累加 `WeightsSumUINT`（线程组聚合后 thread0 写一次）|

**核心逻辑：**
```
weight = ComputeWeight(lightInfo)          // 基于 Luminance（log-radiance 解码）
weight = ImportanceBooster(weight, ...)   // 视锥 Boost + 历史强度变化 Boost
u_lightWeights[CurrentOffset + lightIdx] = weight
```
`WeightsSumUINT` 是一个 uint 存储的 float 位模式，借助 `InterlockedAdd` 实现并发浮点累加（RTXPT 自定义技巧）。

---

#### 10.2.4 ComputeProxyCounts

**作用：** 根据每个灯的权重分配代理数量（权重越大代理越多），并进行线程组内局部前缀和，把局部累积结果写入 `u_scratchList` 和 `u_lightSamplingProxies`，同时将全局代理数用 `InterlockedAdd` 累积到 `SamplingProxyCount`。  
**numthreads：** `[128, 1, 1]`  
**Dispatch：** `ceil(TotalLightCount / 128, 1, 1)`，至少 1 组

| 方向 | 缓冲区 | 说明 |
|------|--------|------|
| 读 | u0 `u_controlBuffer` | 读取 `TotalLightCount`、`WeightsSum()`、`BakerConstants.CurrentWeightsBufferOffset` |
| 读 | u5 `u_lightWeights` | 读取 `weights[CurrentWeightsBufferOffset + lightIndex]` |
| 读/写 | u8 `u_perLightProxyCounters` | 写入每个灯分配到的代理数 |
| 写 | u4 `u_scratchList` | 写入线程组内局部代理计数偏移（供 ComputeProxyBaselineOffsets 使用）|
| 写 | u9 `u_lightSamplingProxies` | 写入每组（128灯/组）的代理计数（供全局前缀和使用）|
| 写 | u0 `u_controlBuffer` | `InterlockedAdd` 更新 `SamplingProxyCount` |

**代理数公式：**
```
budget = RTXPT_LIGHTING_SAMPLING_PROXY_RATIO * max(TotalLightCount, MaxLights/10)
proxyCount = clamp(round(weight / totalWeight * budget), 1, MAX_PROXIES_PER_LIGHT)
```

---

#### 10.2.5 ComputeProxyBaselineOffsets

**作用：** 全局前缀和（parallel prefix-sum），将每个灯在 `u_lightSamplingProxies` 中的代理起始偏移写入该缓冲的头部槽位。这是**单线程组**的扫描算法，必须以 `(1, 1, 1)` 分发。  
**numthreads：** `[32, 1, 1]`（或 64 根据编译配置）  
**Dispatch：** `(1, 1, 1)`（单组，利用 groupshared 内存做完整前缀和）

| 方向 | 缓冲区 | 说明 |
|------|--------|------|
| 读/写 | u0 `u_controlBuffer` | 读 `TotalLightCount`、`SamplingProxyCount`；写最终确认的 `SamplingProxyCount` |
| 读/写 | u9 `u_lightSamplingProxies` | 前几个元素存放各组代理数，扫描后原地改写为前缀和（每个灯的起始偏移）|

**前缀和结果含义：**  
`u_lightSamplingProxies[groupIdx]` 从"第 groupIdx 组有多少代理"变为"第 groupIdx 组之前共有多少代理"，供 CreateProxyJobs 算出每个灯的 `ProxyIndexBase`。

---

#### 10.2.6 CreateProxyJobs

**作用：** 每个线程处理一个灯光，基于 `u_perLightProxyCounters` 和 `u_lightSamplingProxies` 中的前缀和偏移，计算该灯需要填充的代理范围 `[FillFrom, FillTo)`，然后将任务分拆为若干 `SamplingProxyBuildProcTask` 并写入 `u_scratchBuffer`，最后将任务总数 `InterlockedAdd` 到 `ProxyBuildTaskCount`。  
**numthreads：** `[128, 1, 1]`  
**Dispatch：** `ceil(TotalLightCount / 128, 1, 1)`，至少 1 组

| 方向 | 缓冲区 | 说明 |
|------|--------|------|
| 读 | u0 `u_controlBuffer` | 读取 `TotalLightCount` |
| 读 | u8 `u_perLightProxyCounters` | 读取每个灯分配到的代理数 |
| 读 | u9 `u_lightSamplingProxies` | 读取前缀和偏移（`ProxyIndexBase`）|
| 读 | u4 `u_scratchList` | 读取线程组内局部偏移修正量 |
| 写 | u3 `u_scratchBuffer` | 写入 `SamplingProxyBuildProcTask` 数组（每个任务最多负责 `LLB_MAX_PROXIES_PER_TASK=32` 个代理）|
| 写 | u0 `u_controlBuffer` | `InterlockedAdd` 更新 `ProxyBuildTaskCount` |

```cpp
struct SamplingProxyBuildProcTask {
    uint LightIndex;       // 属于哪个灯
    uint ProxyIndexBase;   // 该灯在代理列表中的绝对起始位置
    uint FillProxyIndexFrom; // 本任务负责填充的代理起始
    uint FillProxyIndexTo;   // 本任务负责填充的代理结束（不含）
};
```

---

#### 10.2.7 ExecuteProxyJobs

**作用：** 逐任务执行代理填充——将灯光索引 `LightIndex` 写入 `u_lightSamplingProxies` 的 `[FillFrom, FillTo)` 区间，从而完成代理列表的最终填充。  
**numthreads：** `[128, 1, 1]`  
**Dispatch：** `ceil(LLB_MAX_PROXY_PROC_TASKS / 128, 1, 1)`（以最大任务数估算；Shader 内部以 `ProxyBuildTaskCount` 作早退守卫）

| 方向 | 缓冲区 | 说明 |
|------|--------|------|
| 读 | u0 `u_controlBuffer` | 读取 `ProxyBuildTaskCount`（GPU写入值，作超界守卫）|
| 读 | u3 `u_scratchBuffer` | 读取 `SamplingProxyBuildProcTask` 数组 |
| 写 | u9 `u_lightSamplingProxies` | 将 `LightIndex` 填入 `[FillFrom .. FillTo)` 范围 |

**执行后状态：**  
- `u_lightSamplingProxies[0 .. SamplingProxyCount-1]` 已填满灯光索引（权重大的灯出现次数更多）
- `u_controlBuffer[0].SamplingProxyCount` 含有真实代理数（GPU 写入），`LightSampler::IsEmpty()` 检查此值

---

### 10.3 GPU 代理管线数据流图

```
CPU SetData
  LightBuffer / LightExBuffer / LightControlBuffer (TotalLightCount, Weights Offsets)
         │
         ▼
[1] ResetLightProxyCounters
         │  u_perLightProxyCounters = 0
         ▼
[2] ResetPastToCurrentHistory
         │  u_historyRemapPastToCurrent = INVALID
         ▼
[3] ComputeWeights
         │  u_lightWeights[CurrentOffset + i] = weight(i)
         │  u_controlBuffer.WeightsSumUINT += Σ weight
         ▼
[4] ComputeProxyCounts
         │  u_perLightProxyCounters[i] = proxyCount(i)
         │  u_scratchList[group] = local offset prefix-sum
         │  u_lightSamplingProxies[group+1] = group total count
         │  u_controlBuffer.SamplingProxyCount += total
         ▼
[5] ComputeProxyBaselineOffsets  (1 thread group)
         │  u_lightSamplingProxies[group] → converted to exclusive prefix-sum
         ▼
[6] CreateProxyJobs
         │  u_scratchBuffer[taskId] = SamplingProxyBuildProcTask { lightIdx, proxyBase, from, to }
         │  u_controlBuffer.ProxyBuildTaskCount += taskCount
         ▼
[7] ExecuteProxyJobs
         │  u_lightSamplingProxies[proxyBase + k] = lightIndex  (for k in [from, to))
         ▼
Path Tracer 使用:
  t15: u_perLightProxyCounters  → 每灯代理数（MIS 权重）
  t16: u_lightSamplingProxies   → 代理索引列表（均匀采样 → 权重比例效果）
```

---

### 10.4 权重 Ping-Pong 机制

`u_lightWeights` 缓冲区分为两个半区（各 `WeightsCountHalf = MaxLights + 1` 个 float）：

- **当前帧半区**（`CurrentWeightsBufferOffset`）：本帧 ComputeWeights 写入
- **历史帧半区**（`HistoricWeightsBufferOffset`）：ImportanceBoost 读取上一帧权重做强度变化检测

C# 端每帧交替 `_ping = !_ping`，通过 `_paddingBK[28/29]` 传递偏移给 Shader：

```csharp
uint currentOffset  = _ping ? 0u : WeightsCountHalf;   // 写入目标
uint historicOffset = _ping ? WeightsCountHalf : 0u;   // 历史读取源
// 写入 ctrl._paddingBK[28] / [29]（对应 LightsBakerConstants 中 offset=112 处的两个 uint）
```

---

### 10.5 C# GPU 结构体对齐说明

| C# 结构体 | 大小 | 对应 HLSL | 备注 |
|-----------|------|-----------|------|
| `RtxptPolymorphicLightInfo` | 32 B | `PolymorphicLightInfo` | CenterXYZ + ColorTypeAndFlags + Direction1/2 + Scalars + LogRadiance |
| `RtxptPolymorphicLightInfoEx` | 16 B | `PolymorphicLightInfoEx` | IesProfileIndex + PrimaryAxis + CosConeAngleAndSoftness + UniqueID |
| `RtxptLightingControlData` | 576 B | `LightingControlData` | 112 B 实际字段 + 464 B `_paddingBK[116]`（LightsBakerConstants 占位）|

`_paddingBK` 字段偏移对应：  
- `_paddingBK[28]` = `BakerConstants.CurrentWeightsBufferOffset`（`LightsBakerConstants` 偏移 112 字节处）
- `_paddingBK[29]` = `BakerConstants.HistoricWeightsBufferOffset`

---

### 10.6 灯光采样完整绑定槽（Path Tracer 使用）

```hlsl
StructuredBuffer<LightingControlData>    t_LightsCB                  : register(t12);
StructuredBuffer<PolymorphicLightInfo>   t_Lights                    : register(t13);
StructuredBuffer<PolymorphicLightInfoEx> t_LightsEx                  : register(t14);
Buffer<uint>                             t_LightProxyCounters        : register(t15);
Buffer<uint>                             t_LightProxyIndices         : register(t16);  // ← LightSamplingProxies
Buffer<uint>                             t_LightLocalSamplingBuffer  : register(t17);
Texture2D<uint>                          t_EnvLookupMap              : register(t18);
RWTexture2D<float>                       u_LightFeedbackTotalWeight  : register(u20);
RWTexture2D<uint>                        u_LightFeedbackCandidates   : register(u21);
```

---

### 10.7 NEE-AT 额外要求（暂未实现，供后续参考）

- `ImportanceSamplingType = 2`
- 需要深度图 + 屏幕空间运动向量传入 `UpdateEnd()`
- `ProcessFeedbackHistory` 系列 Compute Pass 需要 Native 端 `LightsBaker::UpdateEnd` 支持
- `LocalSamplingBuffer` 由 ProcessFeedbackHistoryP3 填充，Path Tracer 才能使用本地 tile 采样

### 10.8 SubInstanceData 字段说明

```cpp
uint EmissiveLightMappingOffset;  // 发射三角形在 lightsBuffer 中的起始偏移（0xFFFFFFFF = 无）
uint AnalyticProxyLightIndex;     // 解析光代理索引（0xFFFFFFFF = 无）
```
这两个字段由 Native `ProcessEmissiveGeometry()` 填充；当前最小实现中均为 0xFFFFFFFF（无发射三角形）。

---

## 11. 当前实现状态

### ✅ 已完成

| 文件 | 内容 |
|------|------|
| `NativeRtxptSceneStructs.cs` | 新增 `RtxptPolymorphicLightInfo`（32B）、`RtxptPolymorphicLightInfoEx`（16B）、`RtxptLightingControlData`（576B）及 `RtxptLightType` 枚举 |
| `NativeRtxptBufferResources.cs` | 修正三个缓冲区 stride；新增 `LightWeightsBuffer`（16386 float，Ping-Pong）和 `ScratchListBuffer`（8192 uint）；`WeightsCountHalf = MaxLights + 1 = 8193` |
| `NativeRtxptLightingPass.cs` | **GPU Compute Pipeline**：构造 7 个 `NativeComputePipeline`；`Setup()` CPU 打包并 SetData；`ExecutePass()` 按序 Dispatch 7 个 Shader |
| `NativeRtxptFeature.cs` | 新增 7 个 `NativeComputeShader` 公开字段；`AutoFillShaders()` 从 `Assets/RTXPT/Shaders/Lighting/` 自动加载；`_lightingPass` 构造传入全部 7 个 Shader |
| `NativeRtxptPathTracerPass.cs` | 补绑缺失的 `t_LightProxyIndices`（`LightSamplingProxies`） |

### ⏳ 待完成（超出最小实现范围）

| 优先级 | 内容 | 说明 |
|--------|------|------|
| 高 | **运行时测试** | 进入 Unity Play Mode，用 RenderDoc/PIX 验证 LightControlBuffer 内容及 NEE 采样结果 |
| 中 | **Spot 内锥角精确化** | Unity 2022.2+ 有 `light.innerSpotAngle`，替换当前 `outerRad * 0.8f` 近似 |
| 中 | **发射三角形支持** | 遍历 `MeshRenderer` 发射材质，生成 `EmissiveTrianglesProcTask`，`TriangleLightCount > 0` |
| 低 | **环境光对接** | 将 Unity 天空盒 Cubemap 传入 `EnvMapBaker`，`EnvmapQuadNodeCount > 0` |
| 低 | **NEE-AT 支持** | 需要 Native `LightsBaker::UpdateEnd`、深度图、运动向量，`ImportanceSamplingType = 2` |

---

*文档更新时间：2026-05-22*  
*分析版本：RTXPT commit 对应 RenderingPlugin/External/RTXPT/.git*
