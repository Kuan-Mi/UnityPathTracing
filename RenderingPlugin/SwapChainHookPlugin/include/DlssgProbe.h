// DlssgProbe.h
// Persistent DLSS Frame Generation (DLSS-G / NGX FrameGeneration) context used
// as a log-only feasibility probe. It:
//   * inits NGX on Unity's D3D12 device and queries FrameGeneration capability,
//   * creates the FG feature ONCE and keeps it alive,
//   * allocates its own (plugin-owned) color/depth/mvec/output textures, and
//   * runs NGX_D3D12_EVALUATE_DLSSG repeatedly on those resources.
//
// It never reads Unity's swapchain/backbuffer and never alters presentation, so
// it cannot disturb Unity rendering — it only proves the GPU side of FG (create
// once, evaluate many) runs stably in the live process. Real inputs (hudless /
// mvec / depth) and present pacing are separate later steps.
//
// Lives in the standalone SwapChainHookPlugin so it can be toggled independently.
#pragma once

struct ID3D12Device;

namespace DlssgProbe
{
    // level: 0=info, 1=warn, 2=error.
    using LogFn = void (*)(int level, const char* msg);

    // One-time init: NGX + capability query + persistent FG feature create +
    // resource allocation. Call only with a non-null device (the device is null
    // on the first Unity device-init events). A null device is ignored WITHOUT
    // consuming the one-shot latch, so callers may retry until a device appears.
    // bbWidth/bbHeight/bbFormatDXGI describe the intended FG output (back buffer).
    void Init(ID3D12Device* device,
              unsigned bbWidth, unsigned bbHeight, unsigned bbFormatDXGI,
              LogFn log);

    // True once Init() actually executed against a valid device.
    bool InitAttempted();

    // True once the FG feature was created and resources are ready to Evaluate.
    bool IsReady();

    // Record + submit + wait one NGX_D3D12_EVALUATE_DLSSG on the plugin-owned
    // resources, logging the result. No-op unless IsReady(). Safe to call from a
    // worker thread; uses its own command queue/list/fence.
    void Evaluate();

    // Release the FG feature + resources and shut NGX down. Idempotent.
    void Shutdown();
}
