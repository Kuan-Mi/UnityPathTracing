# UnityPathTracing

[English](#english) | [中文](#中文)

---

<a name="english"></a>

A real-time path tracing rendering system built in Unity URP, replicating the core features of NVIDIA NRDSample, RTXDI, and other RTX technologies, with optimizations and extensions for the Unity environment. Primarily serves as an experimental playground for various RTX-related technologies in Unity.

Due to the lack of Bindless and SER support, there are still gaps in performance and quality compared to the original. For demonstration purposes only.

Blog post: [NRDSample Implementation in Unity](https://www.kuanmi.top/2026/01/22/UnityNRD)

---

## Features

- [x] **Path Tracing**: DXR-based path tracing pipeline
- [x] **SHARC**: Spatial Hash Radiance Cache
- [x] **NRD Denoising**: NVIDIA NRD integrated via native C++ plugin, supporting REBLUR and SIGMA
- [x] **DLSS Ray Reconstruction**: DLSS RR integrated via native plugin for upscaling and ray reconstruction
- [x] **ReSTIR DI/GI**: Compute shader version matches the original in performance but cannot handle alpha clipping. Ray tracing version supports alpha clipping but incurs a 4ms performance cost.
- [x] **Multiple Light Types**: Point lights, spot lights, and area lights
- [x] **Dynamic Scenes**: Skinned mesh animation and dynamic objects
- [x] **Primary Surface Replacement**: High-quality specular reflection via PSR
- [x] **VR Support**: Path tracing in VR mode
- [x] **TMP Support**: World-space TextMeshPro rendering support
- [x] **Auto Exposure**: Histogram-based auto exposure
- [x] **Subsurface Scattering**: RTXCR integrated, supports transmission (direct light)
- [ ] **Volumetric Lighting**: Planned

---

## Requirements

- **GPU**: NVIDIA RTX GPU with DXR support (RTX 3060 or above recommended)
- **Unity Version**: 6000.3.2. Versions 6000.3.4 and above currently have a bug in Frame Debugger that causes crashes.

---

## Acknowledgements

Thanks to [inedelcu](https://github.com/INedelcu) for the great help with writing ray tracing shaders and handling acceleration structures.

## References
[NRD-Sample](https://github.com/NVIDIA-RTX/NRD-Sample)

[RTXGI](https://github.com/NVIDIA-RTX/RTXGI)

[RTXDI](https://github.com/NVIDIA-RTX/RTXDI)

---

<a name="中文"></a>

在 Unity URP 中实现的实时路径追踪渲染系统，复刻了 NVIDIA NRDSample、RTXDI等核心功能，并针对 Unity 环境进行了优化和扩展。主要是试验各种RTX相关技术在Unity中的使用。

受限于没有Bindless、SER等特性支持，在性能和质量方面和原版仍有差距，仅用于演示。

详见博客：[NRDSample 在 Unity 中的实现](https://www.kuanmi.top/2026/01/22/UnityNRD)

---

## 功能特性

- [x] **路径追踪**：基于 DXR 的路径追踪管线
- [x] **SHARC**：空间哈希辐射缓存
- [x] **NRD 降噪**：通过原生 C++ 插件集成 NVIDIA NRD，支持 REBLUR 和 SIGMA
- [x] **DLSS Ray Reconstruction**：通过原生插件集成 DLSS RR，实现超分辨率和重建功能
- [x] **ReSTIR DI/GI**：计算着色器版本和原版性能一致，但无法处理透明度裁切。光追版本可以处理透明度裁切，但性能差了4ms。
- [x] **多种光源支持**：点光源、聚光灯、区域光源支持
- [x] **动态场景**：支持动态物体和蒙皮动画
- [x] **主表面替换**：通过主表面替换实现高质量的镜面反射
- [x] **VR 支持**：支持 VR 模式下的路径追踪渲染
- [x] **TMP 支持**：世界空间下的 TextMeshPro 文本渲染支持
- [x] **自动曝光**：基于直方图的自动曝光
- [x] **次表面散射**：集成 RTXCR，支持透射（直接光）
- [ ] **体积光**：待实现

---

## 要求

- **GPU**：支持 DXR 的 NVIDIA RTX 显卡（如 RTX 3060 及以上）
- **Untiy版本**: 6000.3.2 6000.3.4以上版本FrameDebug目前有Bug，会闪退。

---


## 致谢

感谢 [inedelcu](https://github.com/INedelcu) 的帮助，在编写光追着色器和处理加速结构方面帮了我很多。

## 参考
[NRD-Sample](https://github.com/NVIDIA-RTX/NRD-Sample)

[RTXGI](https://github.com/NVIDIA-RTX/RTXGI)

[RTXDI](https://github.com/NVIDIA-RTX/RTXDI)

---

![ShaderBalls](images/0004.png)
![Bistro](images/0005.png)
![动画](images/0001.png)
![动态自发光](images/0002.png)
![镜面反射](images/0003.png)
