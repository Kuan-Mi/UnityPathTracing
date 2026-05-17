# UnityNRD 项目架构文档

## 一、项目总览

本项目是一个 Unity 原生 DXR 路径追踪渲染插件，核心思路是：**用 C# 编写的 ScriptableRendererFeature 管理多个 Render Pass，每个 Pass 通过 Native Plugin（C++ DLL）直接调用 D3D12 API 执行 Compute Shader 或 DXR Ray Tracing Shader**，完全绕开 Unity 的标准 Shader/Material 系统以获得 D3D12 完整特性（inline ray tracing、lib_6_x、descriptor heap 自由控制等）。

---

## 二、目录结构

```
f:\UnityNRD\
├── RenderingPlugin/               # C++ Native Plugin 源码
│   ├── NativeRenderPlugin/        # 核心插件（Compute + RayTrace dispatch）
│   ├── Denoiser/                  # NRD 降噪器原生封装
│   ├── ShaderCompilerPlugin/      # DXC 着色器编译器插件
│   ├── OMMBakerPlugin/            # Opacity Micro-Map 烘焙
│   ├── PrepareLight/              # 光源预处理
│   ├── D3D12HeapHook/             # D3D12 堆 Hook
│   └── External/
│       └── RTXPT/Rtxpt/           # NVIDIA RTXPT 原始 C++ 示例代码（参考用，不编译进插件）
│           ├── Sample.cpp         # 主渲染循环（pass 执行顺序、binding set 参考）
│           ├── SampleCommon/RenderTargets.cpp  # 所有 GPU 资源的格式和尺寸定义
│           └── PathTracer/        # PathTracer HLSL 源码
└── UnityProject/
    ├── Assets/
    │   ├── Scripts/Rendering/
    │   │   ├── Features/          # ScriptableRendererFeature 主控器
    │   │   │   ├── NativeNrdFeature.cs     # NRD 全路径追踪 Feature
    │   │   │   └── NativeRtxdiFeature.cs   # RTXDI Feature
    │   │   ├── Passes/            # ScriptableRenderPass 实现
    │   │   │   ├── NativeNRD/     # NRD 专属 pass（Opaque/Transparent/SHARC/TAA/Final...）
    │   │   │   └── NativeRtxdi/   # RTXDI 专属 pass（26个，按_Common/DI/GI/GBuffer/PT/Presampling分组）
    │   │   └── Data/              # 数据结构（PassContext、Resources）
    │   ├── RTXPT/Shaders/         # 已适配 Unity 的 HLSL 计算着色器
    │   │   ├── ProcessingPasses/  # AccumulationPass, ExportVisibilityBuffer, DenoisingGuidesBaker...
    │   │   ├── Lighting/          # LightsBaker 系列着色器
    │   │   └── Lighting/Distant/  # 环境光贴图烘焙着色器
    │   └── RTXDI/Shaders/         # RTXDI 专属着色器
    └── Packages/
        └── top.kuanmi.native-ray-tracing/  # 内部 Package（核心封装层）
            └── Runtime/Scripts/RayTracing/
                ├── NativeComputePipeline.cs       # CS pipeline 封装（PSO + root sig）
                ├── NativeComputeDescriptorSet.cs  # CS descriptor set 封装（ring buffer）
                ├── NativeRayTracePipeline.cs      # DXR pipeline 封装
                └── NativeRenderPlugin.cs          # P/Invoke 入口（所有 NR_* 函数）
```

---

## 三、核心封装层（Package 层）

### NativeComputePipeline.cs
**路径**: `UnityProject/Packages/top.kuanmi.native-ray-tracing/Runtime/Scripts/RayTracing/NativeComputePipeline.cs`

- 封装 D3D12 Compute PSO + Root Signature
- 从 `NativeComputeShader` asset 读取 DXIL 字节码，调用 `NR_CreateComputeShader(Ex)` 创建本地 handle
- 反射 HLSL 绑定 slot（name → slot index），供 `NativeComputeDescriptorSet` 使用
- 支持 `RootConstantsHint`（将 CBV 提升为 root 32-bit constants）和 `rootSRVHints`（SRV 提升为 inline root descriptor）
- 热重载：shader 重编译后自动触发 `OnRebuilt` 事件，关联的 DescriptorSet 自动重建

### NativeComputeDescriptorSet.cs
**路径**: `UnityProject/Packages/top.kuanmi.native-ray-tracing/Runtime/Scripts/RayTracing/NativeComputeDescriptorSet.cs`

- 持有一个 `NativeComputePipeline` 的所有资源绑定（CBV/SRV/UAV/TLAS 等）
- 内部维护 **RingSize=8** 的 pinned NativeArray ring buffer，解耦主线程绑定与渲染线程执行
- 每帧 Pass 调用 `Set*` 方法（SetTexture/SetRWTexture/SetBuffer/SetRWBuffer/SetConstantBuffer/SetRootConstants/SetBindlessTexture/SetAccelerationStructure 等）设置绑定
- `Dispatch(cmd, ds, groupsX, groupsY, groupsZ)` 提交到命令缓冲区

### NativeRenderPlugin.cs（P/Invoke 入口）
**路径**: `UnityProject/Packages/top.kuanmi.native-ray-tracing/Runtime/Scripts/RayTracing/NativeRenderPlugin.cs`

所有与 C++ Plugin DLL 的互操作入口：

| 函数 | 用途 |
|------|------|
| `NR_CreateComputeShader(dxil, size, name)` | 创建 CS pipeline，返回 ulong handle |
| `NR_CreateComputeShaderEx(dxil, size, name, hintsJson)` | 带 root constants/SRV 提示 |
| `NR_CS_CreateDescriptorSet(pipelineHandle)` | 创建 descriptor set |
| `NR_CS_GetRenderEventFunc()` | 获取 dispatch 回调函数指针 |
| `NR_CreateRayTraceShaderFromBytes(dxil, size, name)` | 创建 DXR pipeline |
| `NR_RTS_GetRenderEventFunc()` | 获取 DXR dispatch 回调 |
| `NR_GetFrameTickEventFunc()` | 获取帧同步事件 |

---

## 四、C++ Plugin 层

### NativeRenderPlugin（核心）
**路径**: `RenderingPlugin/NativeRenderPlugin/`

- **ComputeShader.cpp/h**: D3D12 CS PSO 创建、root signature 生成、descriptor heap 分配、Dispatch 执行
- **RayTraceShader.cpp/h**: D3D12 DXR 管线（ray gen/miss/hit group shader table 构建）、DispatchRays 执行
- **Plugin.cpp**: 所有 `NR_*` 函数导出（line 357-520 = RTS exports，line 746-889 = CS exports）

#### Dispatch 数据结构
**Compute Shader**:
```cpp
struct CS_RenderEventData {
    uint64_t shaderHandle;    // +0 (8B)
    uint32_t threadGroupX;    // +8 (4B)
    uint32_t threadGroupY;    // +12 (4B)
    uint32_t threadGroupZ;    // +16 (4B)
};  // 20 bytes
```

**Ray Tracing**:
```cpp
struct RTS_RenderEventData {
    uint64_t shaderHandle;  // +0 (8B)
    uint32_t width;         // +8 (4B)
    uint32_t height;        // +12 (4B)
};  // 16 bytes
```

### ShaderCompilerPlugin
**路径**: `RenderingPlugin/ShaderCompilerPlugin/src/ShaderCompilerPlugin.cpp`

- 封装 DXC（DirectX Shader Compiler）
- API: `NR_SC_Compile(hlslPath, includeDirs, extraArgs, outBytes*, outSize*)`
- Ray tracing shader 硬编码使用 `lib_6_9` profile
- Compute shader 需调用方在 extraArgs 中指定 target（如 `cs_6_6`）

### Denoiser Plugin
**路径**: `RenderingPlugin/Denoiser/`

- 封装 NVIDIA NRD（Ray-Tracing Denoiser）
- 提供 REBLUR_DIFFUSE_SPECULAR 等降噪模式
- C# 侧通过 `NrdDenoiser` 类 + `NrdPass` 调用

---

## 五、C# 管理层

### Feature：主控器（ScriptableRendererFeature）

#### NativeNrdFeature.cs
**路径**: `UnityProject/Assets/Scripts/Rendering/Features/NativeNrdFeature.cs`

完整 NRD 路径追踪 Feature（基于 TraceOpaque/TraceTransparent + SHARC + NRD Denoiser）：

**Pass 执行顺序**:
```
NRDTlasUpdatePass → NRDSharcPass (Resolve+Update) → NRDConfidenceBlurPass (5次) →
NRDOpaquePass → [NrdDenoiser - shadow/opaque] → NRDTransparentPass →
NRDCompositionPass → NRDTaaPass → NRDFinalPass → OutputBlit → NativeFrameTick
```

**资源管理**: 每个摄像机独立维护 `NativeNrdTextureResources` 资源池（`_resourcePools` dict，key = camera instanceID + eyeIndex×100000）

#### NativeRtxdiFeature.cs
**路径**: `UnityProject/Assets/Scripts/Rendering/Features/NativeRtxdiFeature.cs`

完整 RTXDI Feature（ReSTIR DI + GI + PT + NRD Denoiser）：

**Pass 执行顺序**:
```
BuildAccelerationStructure → PrepareLights → RaytracedGBuffer → PostprocessGBuffer →
PdfMips → PresampleLights → PresampleReGir →
[DI: GenerateInitialSamples → TemporalResampling → SpatialResampling → ShadeSamples] →
[GI: BrdfRayTracing → ShadeSecondarySurfaces → GITemporalResampling → GISpatialResampling → GIFinalShading] →
[PT: GenerateInitialSamples → [FillSampleID? ComputeDuplicationMap?] → TemporalResampling → SpatialResampling → FinalShading] →
[NRD Denoising via NrdPass] → CompositingPass → OutputBlit → NativeFrameTick
```

---

### 关键 Pass 文件

#### NativeNRD Passes
| 文件 | 着色器 | Group Size | 作用 |
|------|--------|-----------|------|
| NRDOpaquePass.cs | TraceOpaque.computeshader | 16×16 | 追踪不透明表面，输出 MV/ViewZ/Normal_Roughness/DirectLighting/PsrThroughput/Diff/Spec |
| NRDTransparentPass.cs | TraceTransparent.computeshader | 16×16 | 追踪透明表面 |
| NRDSharcPass.cs | nrdSharcResolve + nrdSharcUpdate | 256/16×16 | SHARC 辐射缓存（Resolve+Update 双 pass，ping-pong） |
| NRDConfidenceBlurPass.cs | ConfidenceBlur.computeshader | 16×16 | SHARC 梯度模糊（5次 ping-pong 迭代） |
| NRDCompositionPass.cs | Composition.computeshader | 16×16 | 合并 Diff/Spec → ComposedDiff/ComposedSpec_ViewZ |
| NRDTaaPass.cs | Taa.computeshader | 16×16 | TAA（ping-pong history） |
| NRDFinalPass.cs | Final.computeshader | 16×16 | PostAA + PreAA → 最终输出 |
| NRDDlssBeforePass.cs | DlssBefore.computeshader | 16×16 | 准备 DLSS-RR 引导缓冲 |
| NRDDlssAfterPass.cs | DlssAfter.computeshader | 16×16 | DLSS 输出后处理 |
| NRDTlasUpdatePass.cs | (无 shader) | — | 重建 TLAS + 蒙皮更新 |
| NativeFrameTick.cs | (无 shader) | — | 帧边界通知 Native 层 |

#### NativeRtxdi Passes（部分关键文件）
| 文件 | 着色器 | Group Size | 作用 |
|------|--------|-----------|------|
| NativeRtxdiBuildAccelerationStructurePass.cs | — | — | 构建 TLAS |
| NativeRtxdiRaytracedGBufferPass.cs | RaytracedGBuffer.computeshader | 16×16 | 主光线 GBuffer（9纹理输出） |
| NativeRtxdiPostprocessGBufferPass.cs | PostprocessGBuffer.computeshader | 16×16 | 生成 NRD 兼容 NormalRoughness |
| NativeRtxdiPrepareLightsPass.cs | PrepareLights.computeshader | 256 | CPU 光源收集 + GPU 光源数据准备 |
| NativeRtxdiPdfMipsPass.cs | GenerateMips.compute | — | LocalLightPdfTexture + EnvPdfTexture mip 链 |
| NativeRtxdiCompositingPass.cs | CompositingPass.computeshader | — | noisy+denoised Diffuse/Specular → DirectLighting |
| NativeRtxdiBindings.cs | — | — | 公共绑定辅助（BindRabCommon，绑定 RAB_Buffers.hlsli 超集） |

---

## 六、RTXPT 着色器（UnityProject/Assets/RTXPT/Shaders/）

这些是为实现 `NativeRtxptFeature` 而准备的、已适配 Unity 的 HLSL 计算着色器：

### ProcessingPasses/
| 着色器文件 | numthreads | 关键绑定 | 作用 |
|-----------|-----------|---------|------|
| AccumulationPass.computeshader | [8,8,1] | SRV t0=InputColor; UAV u0=AccumulatedColor, u1=OutputColor; CBV b0=g_Const | 多帧累积（参考模式） |
| ExportVisibilityBuffer.computeshader | [8,8,1] | UAV u5=MotionVectors, u6=Depth; CBV b0=g_Const | 从内联可见性缓存导出深度和运动向量 |
| PostProcess_DenoiserPrepareInputs.computeshader | [8,8,1] | UAV u0=OutputColor, u7=SpecHitT, u31-u37=Denoiser inputs, u40=StablePlanesHeader, u42=StablePlanesBuffer, u44=StableRadiance | 从 StablePlanes 生成 NRD 输入 |
| PostProcess_DenoiserFinalMerge.computeshader | [8,8,1] | SRV t2=DiffRadiance, t3=SpecRadiance, t5=Validation, t6=ViewspaceZ, t7=DisocclusionMix, t10=StablePlanesBuffer; UAV u0=OutputColor | 将降噪结果合并为最终颜色 |
| PostProcess_NoDenoiserFinalMerge.computeshader | [8,8,1] | UAV u0=OutputColor, u40=StablePlanesHeader, u42=StablePlanesBuffer, u44=StableRadiance | 无降噪时的 FinalMerge |
| DenoisingGuidesBaker_DenoiseSpecHitT.computeshader | [8,8,1] | UAV u7=SpecularHitT; MiniConst.Ping | SpecularHitT 双边滤波（2遍 ping-pong） |
| DenoisingGuidesBaker_ComputeAvgLayerRadiance.computeshader | [8,8,1] | StablePlanesHeader/Buffer/StableRadiance → DenoiserAvgLayerRadianceHalfRes | 半分辨率平均辐射量 |
| BakeEmissiveTriangles.computeshader | [256,1,1] | SRV: SubInstance/Instance/Geometry/Material data; UAV u0-u3=Light buffers, u6-u7=HistoryRemap | 发光三角形光源烘焙 |

### Lighting/ 和 Lighting/Distant/
LightsBaker 系列着色器（代理光源计算、环境光贴图烘焙、反馈历史处理等）。

---

## 七、关键数据结构

### StablePlane（RTXPT 核心）
- 大小：**80 bytes/元素**（来自 Config.h）
- 总缓冲：`W × H × cStablePlaneCount(3)` 个元素
- 类型：StructuredBuffer（在 HLSL 中为 RWByteAddressBuffer 或 StructuredBuffer<StablePlane>）
- `StablePlanesHeader`：`R32_UINT Texture2DArray[4]`（slice 0/1/2 = 分支ID，slice 3 = 首次命中距离）

### SampleConstants（PathTracer 主 CBV，b0）
包含：`PathTracerConstants`（PT 参数）、视图/投影矩阵、环境光参数、帧序号等。

### SampleMiniConstants（push constant，b1）
轻量级每pass常量（用于 ping-pong 标志、累积权重等）。

### NativeRtxdiPassContext（RTXDI）
每帧 context，传递给所有 NativeRtxdi* pass：
- CBV handles（ResamplingConstants、GBufferConstants、PerPassConstants）
- GBuffer 纹理（当前帧 + 上一帧，ping-pong）
- TLAS handle
- 分辨率信息

---

## 八、常用编码模式

### 1. Pass 典型结构
```csharp
public class MyPass : ScriptableRenderPass
{
    private NativeComputePipeline     _cs;
    private NativeComputeDescriptorSet _ds;

    public MyPass(NativeComputeShader shader)
    {
        _cs = new NativeComputePipeline(shader);
        _ds = new NativeComputeDescriptorSet(_cs);
    }

    public override void Execute(ScriptableRenderContext ctx, ref RenderingData data)
    {
        CommandBuffer cmd = CommandBufferPool.Get();
        _ds.SetTexture("g_InputTex", inputPtr);
        _ds.SetRWTexture("g_OutputTex", outputPtr);
        _ds.SetConstantBuffer("g_Const", cbuf);
        _cs.Dispatch(cmd, _ds, groupsX, groupsY, 1);
        ctx.ExecuteCommandBuffer(cmd);
        CommandBufferPool.Release(cmd);
    }
}
```

### 2. 每摄像机资源池（防止多摄像机冲突）
```csharp
long key = cam.GetInstanceID() + (eyeIndex * 100000L);
if (!_resourcePools.TryGetValue(key, out var pool))
{
    pool = new MyResourcePool();
    _resourcePools.Add(key, pool);
}
```

### 3. Ping-Pong 帧缓冲
```csharp
bool isOdd = (Time.frameCount % 2) == 1;
RTHandle current  = isOdd ? pool.PingBuffer : pool.PongBuffer;
RTHandle previous = isOdd ? pool.PongBuffer : pool.PingBuffer;
```

### 4. Root Constants（MiniConst）
轻量级按帧变化的常量，通过 `RootConstantsHint` 提升为 root 32-bit constants，避免 CBV 创建开销：
```csharp
_cs = new NativeComputePipeline(shader, new RootConstantsHint[]
{
    new RootConstantsHint { Name = "g_MiniConst", Count = 4 }
});
// 绑定时：
_ds.SetRootConstants("g_MiniConst", miniConstBytes);
```

---

## 九、RTXPT 原始 C++ 参考文件

以下文件**不参与编译**，仅作为理解 HLSL 绑定和 pass 逻辑的参考：

| 文件 | 用途 |
|------|------|
| `RenderingPlugin/External/RTXPT/Rtxpt/Sample.cpp` | 主渲染循环：`RecreateBindingSet()` 定义所有 UAV/SRV slot 编号；`PathTrace()` 定义 pass 执行顺序 |
| `RenderingPlugin/External/RTXPT/Rtxpt/SampleCommon/RenderTargets.cpp` | 所有 GPU 资源的 DXGI 格式和尺寸（**是确认 texture format 的权威来源**） |
| `RenderingPlugin/External/RTXPT/Rtxpt/PathTracer/Config.h` | `cStablePlaneCount=3`、StablePlane 大小等关键常量 |
| `RenderingPlugin/External/RTXPT/Rtxpt/PathTracer/PathTracerSample.hlsl` | 主 DXR shader（lib_6_9），含 BUILD_STABLE_PLANES / REFERENCE / FILL_STABLE_PLANES 变体 |

---

## 十、待实现：NativeRtxptFeature

基于以上架构，下一个要实现的 Feature 是 `NativeRtxptFeature`，参考 `NativeNrdFeature` 和 `NativeRtxdiFeature` 的结构，实现 RTXPT（NVIDIA Path Tracing with Stable Planes + NRD Denoising）管线。

详见 [NativeRtxptFeature_Plan.md](NativeRtxptFeature_Plan.md)。

**关键区别（相比 NativeNrdFeature）**:
- 使用 **DXR 主路径追踪**（`PathTracerSample.hlsl`，lib_6_9），而非 Compute Shader inline RT
- 引入 **StablePlanes** 机制（3平面，每像素80B buffer + Header Texture2DArray）
- **DenoiserPrepareInputs** → **NRD（×3 planes）** → **DenoiserFinalMerge** 三阶段降噪合并
- PT_USE_RESTIR_DI=0，PT_USE_RESTIR_GI=0（不使用 RTXDI 重采样）
