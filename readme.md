# UnityPathTracing

[English](#english) | [中文](#中文)

---

<a name="english"></a>

A real-time path tracing rendering system built in Unity URP, replicating the core features of NVIDIA NRDSample with optimizations and extensions for the Unity environment.

Blog post: [NRDSample Implementation in Unity](https://www.kuanmi.top/2026/01/22/UnityNRD)

---

## Features

- [x] **Path Tracing**: DXR-based path tracing pipeline
- [x] **SHARC**: Spatial Hash Radiance Cache
- [x] **NRD Denoising**: NVIDIA NRD integrated via native C++ plugin, supporting REBLUR and SIGMA
- [x] **DLSS Ray Reconstruction**: DLSS RR integrated via native plugin for upscaling and ray reconstruction
- [x] **Multiple Light Types**: Point lights, spot lights, and area lights
- [x] **Dynamic Scenes**: Skinned mesh animation and dynamic objects
- [x] **Primary Surface Replacement**: High-quality specular reflection via PSR
- [x] **VR Support**: Path tracing in VR mode
- [x] **TextMeshPro**: World-space TextMeshPro rendering support
- [ ] **ReSTIR DI**: Planned
- [ ] **Volumetric Lighting**: Planned

---

## Requirements

- **GPU**: NVIDIA RTX GPU with DXR support (RTX 3060 or above recommended)

---

## Tech Stack

| Component | Description |
|-----------|-------------|
| Engine | Unity URP, Render Graph |
| Graphics API | DirectX 12 (DXR) |
| Denoising | NVIDIA NRD (REBLUR, SIGMA) |
| Upscaling / Reconstruction | NVIDIA DLSS Ray Reconstruction |
| Radiance Cache | SHARC |

---

<a name="中文"></a>

在 Unity URP 中实现的实时路径追踪渲染系统，复刻了 NVIDIA NRDSample 的核心功能，并针对 Unity 环境进行了优化和扩展。

详见博客：[NRDSample 在 Unity 中的实现](https://www.kuanmi.top/2026/01/22/UnityNRD)

---

## 功能特性

- [x] **路径追踪**：基于 DXR 的路径追踪管线
- [x] **SHARC**：空间哈希辐射缓存
- [x] **NRD 降噪**：通过原生 C++ 插件集成 NVIDIA NRD，支持 REBLUR 和 SIGMA
- [x] **DLSS Ray Reconstruction**：通过原生插件集成 DLSS RR，实现超分辨率和重建功能
- [x] **多种光源支持**：点光源、聚光灯、区域光源支持
- [x] **动态场景**：支持动态物体和蒙皮动画
- [x] **主表面替换**：通过主表面替换实现高质量的镜面反射
- [x] **VR 支持**：支持 VR 模式下的路径追踪渲染
- [x] **TMP 支持**：世界空间下的 TextMeshPro 文本渲染支持
- [ ] **ReSTIR DI**：待实现
- [ ] **体积光**：待实现

---

## 硬件要求

- **GPU**：支持 DXR 的 NVIDIA RTX 显卡（如 RTX 3060 及以上）

---

## 技术栈

| 组件 | 说明 |
|------|------|
| 引擎 | Unity URP，Render Graph |
| 图形 API | DirectX 12 (DXR) |
| 降噪 | NVIDIA NRD（REBLUR、SIGMA） |
| 超分 / 重建 | NVIDIA DLSS Ray Reconstruction |
| 辐射缓存 | SHARC |

---

![ShaderBalls](images/0004.png)
![Bistro](images/0005.png)
![动画](images/0001.png)
![动态自发光](images/0002.png)
![镜面反射](images/0003.png)
