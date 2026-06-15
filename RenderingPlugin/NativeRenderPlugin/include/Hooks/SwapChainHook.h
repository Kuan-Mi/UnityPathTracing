// SwapChainHook.h
// Diagnostic-only vtable hooks for the DXGI swapchain path Unity uses.
//
// Unity exposes no callback for swapchain creation or Present. To observe what
// Unity actually does in a player build, this module patches:
//   * IDXGIFactory::CreateSwapChain        (vtable idx 10)
//   * IDXGIFactory2::CreateSwapChainForHwnd (vtable idx 15)
//   * IDXGISwapChain::Present              (vtable idx 8)
//   * IDXGISwapChain1::Present1            (vtable idx 22)
// Every hook logs its arguments and calls straight through — no behavior change.
// This is the reconnaissance step for a possible DLSS-FG present interception.
//
// Same vtable-patch technique as D3D12HeapHook (adapted from DX12BindlessUnity).
// Editor note: Unity's GetSwapChain() returns nullptr in the editor, so the
// Present hook only installs in player builds; the factory hook installs in
// both but only fires if a swapchain is (re)created after install.
#pragma once

struct IDXGISwapChain;
struct ID3D12Device;
struct ID3D12CommandQueue;

namespace SwapChainHook
{
    // level: 0=info, 1=warn, 2=error. Safe to call from any thread.
    using LogFn = void (*)(int level, const char* msg);

    // Optional logger. If unset, messages go to OutputDebugStringA.
    void SetLogger(LogFn fn);

    // ----- DLSS-FG present-pacing stub (Test, hardware-independent) ----------
    // Enable the frame-generation present-pacing HARNESS. This does NOT run AI
    // frame generation; it stands up the exact present plumbing FG will need so
    // it can be validated on any GPU (incl. the pre-Ada RTX 3060 dev box):
    // retain the real frame, present it, pace ~half a frame, then present a
    // second time. On the 4060 the second (duplicate) present is replaced by the
    // EvaluateFeature output. Content-preserving — it presents the real frame
    // twice, so there is no visual corruption; only the Present1 rate doubles.
    //
    // Provide Unity's D3D12 device and the queue the swapchain presents on
    // (Unity's command queue). Off until enabled; the plugin enables it only
    // when the NR_FG_PACING_STUB env var is set. Idempotent.
    void EnablePacingStub(ID3D12Device* device, ID3D12CommandQueue* presentQueue);

    // True once the pacing stub is active.
    bool IsPacingStubEnabled();

    // Patch the DXGI factory vtable (CreateSwapChain / CreateSwapChainForHwnd)
    // by creating a throwaway IDXGIFactory2 — the vtable is shared by all
    // factories of that concrete type, so this also covers the factory Unity
    // uses. Install as early as possible (UnityPluginLoad) to have a chance of
    // catching the initial swapchain creation. Returns true on success.
    // No-op after the first successful install.
    bool InstallFactoryHook();

    // Patch Present / Present1 on the given live swapchain's vtable. Obtain the
    // swapchain from IUnityGraphicsD3D12v8::GetSwapChain() (player only). The
    // vtable is shared across swapchains of the same type, so one patch covers
    // any later-created swapchain too. Idempotent; safe to call every frame
    // until it succeeds. Returns true once the hook is installed.
    bool TryInstallPresentHook(IDXGISwapChain* swapChain);

    // True once the Present hook has been installed.
    bool IsPresentHookInstalled();
}
