# NativeRtxptFeature 实现计划

> **降噪策略（已更新）**：不使用 NRD，改为在管线最后用一次 **DLSS Ray Reconstruction（DLSS-RR）** 完成去噪与超分辨率。

## 一、资源清单

### 纹理（渲染分辨率 W×H，除非特别说明）

| 名称 | 格式 | 用途 |
|------|------|------|
| OutputColor | RGBA16_FLOAT | PT 主输出噪声图像，作为 DLSS-RR 输入 |
| Depth | R32_FLOAT | 屏幕深度（PT 输出）|
| ScreenMotionVectors | RGBA16_FLOAT | 屏幕空间运动向量（PT 输出）|
| Throughput | R32_UINT（fp16×2 打包）| PSR 路径 throughput |
| SpecularHitT | R32_FLOAT | 高光命中距离（双边滤波后）|
| ScratchFloat1 | R32_FLOAT | 双边滤波 ping-pong 临时缓冲 |
| StableRadiance | RGBA16_FLOAT | 稳定平面辐射量（不需降噪部分）|
| StablePlanesHeader | R32_UINT, Texture2DArray[4] | 每像素稳定平面头 ([0..2]=分支ID, [3]=首次命中距离）|
| DenoiserNormalRoughness | A2B10G10R10_UNORM | 法线+粗糙度（PT 输出，复用为 DLSS-RR guide 源）|
| BaseColor | B10G11R11_UFloat | GBuffer 基础色 |
| SpecNormal | R32_UINT | GBuffer 高光法线（打包）|
| RoughnessMetal | RG16_FLOAT | GBuffer 粗糙度+金属度 |
| MaterialInfo | R32_UINT | GBuffer 材质标志 |
| **DlssRrDiffAlbedo** | A2B10G10R10_UNORM | DLSS-RR 漫反射 Albedo guide |
| **DlssRrSpecAlbedo** | A2B10G10R10_UNORM | DLSS-RR 高光 Albedo guide |
| **DlssRrSpecHitDistance** | R16_FLOAT | DLSS-RR 高光命中距离 guide |
| **DlssRrNormalRoughness** | RGBA16_FLOAT | DLSS-RR 法线+粗糙度 guide |
| **DlssRrOutput** | RGBA16_FLOAT，**显示分辨率** | DLSS-RR 输出（去噪+超分辨率）|
| AccumulatedRadiance | RGBA32_FLOAT | 参考模式帧累积缓冲 |
| ProcessedOutputColor | RGBA16_FLOAT，显示分辨率 | 参考模式输出 |

> ❌ **已移除的 NRD 专用纹理**：DenoiserViewspaceZ、DenoiserMotionVectors、DenoiserDiffRadianceHitDist、DenoiserSpecRadianceHitDist、DenoiserDisocclusionThresholdMix、CombinedHistoryClampRelax、DenoiserAvgLayerRadianceHalfRes、DenoiserOutDiff0/1/2、DenoiserOutSpec0/1/2、DenoiserValidation。

### Buffer（无变化）

| 名称 | 元素类型/大小 | 数量 | 用途 |
|------|-------------|------|------|
| StablePlanesBuffer | StablePlane (80B) | W×H×3 | 稳定平面逐像素数据 |
| SurfaceDataBuffer | PackedPathTracerSurfaceData | W×H×2 | 表面 GBuffer 命中数据 |
| LightControlBuffer | 结构体 | 1 | 光源管理控制数据 |
| LightBuffer | LightData | maxLights | 光源数据 |
| LightExBuffer | LightExData | maxLights | 扩展光源数据 |
| LightScratchBuffer | mixed | 按需 | 光源处理临时缓冲 |
| HistoryRemapCurrentToPast | uint | maxLights | 光源历史映射（当前→上帧）|
| HistoryRemapPastToCurrent | uint | maxLights | 光源历史映射（上帧→当前）|
| LightProxyCounters | uint | varies | 代理光源计数器 |
| LightSamplingProxies | uint | varies | 代理光源索引 |
| LocalSamplingBuffer | mixed | varies | 本地采样缓冲 |

---

## 二、Pass 列表及着色器绑定

### Phase 0: TLAS 构建（Native API）
- `NativeRtxptBuildTlasPass` — FlushPendingCopies + BuildAccelerationStructures + RecordSkinnedMorphUpdate

---

### Phase 1: 光源更新（LightsBaker，多个 CS）
（与原计划相同，略）

---

### Phase 2: 主路径追踪（RayTracing Shader，lib_6_9）
- 宏：`PATH_TRACER_MODE=PATH_TRACER_MODE_BUILD_STABLE_PLANES`（实时）/ `REFERENCE`
- 写入：OutputColor, Depth, ScreenMotionVectors, Throughput, SpecularHitT, StablePlanesHeader, StablePlanesBuffer, StableRadiance, DenoiserNormalRoughness, BaseColor, SpecNormal, RoughnessMetal, MaterialInfo

---

### Phase 3: ExportVisibilityBuffer（CS, 8×8）
- 写入：ScreenMotionVectors, Depth（全屏导出）

---

### Phase 4: DenoiseSpecHitT（CS, 8×8，×2 ping-pong）
- 输入/输出：SpecularHitT ↔ ScratchFloat1（双边滤波）

---

### Phase 5: NoDenoiserFinalMerge（CS, 8×8）

- **着色器**: `PostProcess_NoDenoiserFinalMerge.computeshader`
- **输入**: StablePlanesHeader, StablePlanesBuffer, StableRadiance
- **输出**: OutputColor（稳定平面合并为最终噪声图像）
- **作用**: 代替 NRD DenoiserFinalMerge，直接将各平面辐射量合并，不做 NRD 降噪

---

### Phase 6: DLSS-RR Guide Buffer 准备（CS, 8×8）

对应原来的 `NRDDlssBeforePass`，从 PT GBuffer 数据中生成 DLSS-RR 所需的 guide buffers：

- **输入(SRV)**: BaseColor, RoughnessMetal, SpecularHitT, DenoiserNormalRoughness
- **输出(UAV)**: DlssRrDiffAlbedo, DlssRrSpecAlbedo, DlssRrSpecHitDistance, DlssRrNormalRoughness
- **作用**: 将 BaseColor×(1-Metalness)→DiffAlbedo，BaseColor×Metalness→SpecAlbedo，重采样法线为 DLSS-RR 格式

---

### Phase 7: DLSS Ray Reconstruction（DlssRRPass）

- **接口**: `DlssRRPass.Setup(dlrr.GetInteropDataPtr(frameInput, resources, 1.0f, setting.upscalerMode), new DlssRRPass.Settings { tmpDisableRR = setting.tmpDisableDlssRR })`
- **DlrrResources** 绑定：
  | DlrrResources 字段 | 来源纹理 |
  |-------------------|---------|
  | input | OutputColor（噪声 PT 输出）|
  | output | DlssRrOutput（显示分辨率）|
  | mv | ScreenMotionVectors |
  | depth | Depth |
  | diffAlbedo | DlssRrDiffAlbedo |
  | specAlbedo | DlssRrSpecAlbedo |
  | normalRoughness | DlssRrNormalRoughness |
  | specHitDistance | DlssRrSpecHitDistance |
- **输出**: DlssRrOutput（去噪 + 超分辨率，显示分辨率）

---

### Phase 8: AccumulationPass（CS, 8×8）[仅参考模式]
（与原计划相同）

---

## 三、各 Pass 数据流图

```
Phase 0: TLAS Build
    ↓
Phase 1: LightsBaker (env map + emissive + proxies + feedback)
    ↓
Phase 2: PathTracer RT Shader
    → OutputColor (noisy), Depth, ScreenMotionVectors, Throughput
    → StablePlanesHeader, StablePlanesBuffer, StableRadiance
    → SpecularHitT, DenoiserNormalRoughness
    → BaseColor, RoughnessMetal, SpecNormal, MaterialInfo
    ↓
Phase 3: ExportVisibilityBuffer
    → MotionVectors, Depth (full-screen)
    ↓
Phase 4: DenoiseSpecHitT (×2 ping-pong)
    → SpecularHitT (filtered)
    ↓
Phase 5: NoDenoiserFinalMerge
    → OutputColor (stable planes merged, still noisy)
    ↓
Phase 6: DLSS-RR Guide Buffer Prep (DlssBeforePass equivalent)
    → DlssRrDiffAlbedo, DlssRrSpecAlbedo
    → DlssRrNormalRoughness, DlssRrSpecHitDistance
    ↓
Phase 7: DLSS Ray Reconstruction (DlssRRPass)
    input  = OutputColor  (noisy)
    guides = DlssRrDiff/SpecAlbedo, NormalRoughness, SpecHitDistance
    mv     = ScreenMotionVectors
    depth  = Depth
    → DlssRrOutput  (denoised + upscaled, display resolution)
    ↓
Phase 8 (opt): AccumulationPass [reference mode only]
    → AccumulatedRadiance, ProcessedOutputColor
    ↓
OutputBlit → Screen
```

---

## 四、关键设计决策

1. **不实现 RTXDI**：PT_USE_RESTIR_DI=0, PT_USE_RESTIR_GI=0
2. **稳定平面数量 = 3**（cStablePlaneCount=3）
3. **降噪策略：NRD → DLSS-RR**  
   不再使用 NrdDenoiser 实例（原来 3 个/相机），改为单个 `DlrrDenoiser` 实例 + `DlssRRPass`，减少纹理内存约 30%，并实现超分辨率（QUALITY 模式 1.5× 上采样）
4. **超分辨率质量模式**：由 `NativeRtxptSetting.upscalerMode`（`UpscalerMode` 枚举）控制；NATIVE=1:1，QUALITY=1.5×，PERFORMANCE=2×
5. **常量缓冲结构**: SampleConstants 作为 g_Const (b0)，SampleMiniConstants 作为 g_MiniConst (b1)
6. **参考模式**: AccumulationPass 仅在 `RtxptPathTracerMode.Reference` 下使用