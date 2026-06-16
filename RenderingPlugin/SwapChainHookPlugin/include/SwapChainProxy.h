// SwapChainProxy.h
// A COM proxy that wraps Unity's real IDXGISwapChain and is handed to Unity in
// its place (from the CreateSwapChain[ForHwnd] factory hook). Every method
// forwards 1:1 to the real swapchain — Stage 1 is a faithful pass-through whose
// only purpose is to prove Unity runs identically when its swapchain is wrapped.
//
// Why a proxy instead of a passive Present vtable hook: DLSS-FG must present an
// EXTRA (generated) frame per real frame, but Unity owns the back-buffer
// rotation and assumes exactly one Present per frame. Injecting an extra Present
// from a side hook desyncs Unity's GetCurrentBackBufferIndex view and
// device-removes the GPU (confirmed). Whoever paces FG must therefore OWN the
// buffer bookkeeping — i.e. BE the swapchain. This proxy is that ownership point.
// Stage 2 will intercept Present/GetBuffer/GetCurrentBackBufferIndex/ResizeBuffers
// to insert the generated frame while keeping Unity's index view consistent.
#pragma once

struct IDXGISwapChain1;
struct ID3D12CommandQueue;

namespace SwapChainProxy
{
    // level: 0=info, 1=warn, 2=error.
    using LogFn = void (*)(int level, const char* msg);

    // Wrap a freshly-created real swapchain. Returns a proxy (as IDXGISwapChain1*,
    // ref count 1 — hand it straight to the caller's out-param) or nullptr if the
    // real object cannot be promoted to IDXGISwapChain4 (then don't wrap). The
    // proxy takes its own reference on the real swapchain, so the caller should
    // Release its local reference to the real one after a successful wrap.
    //
    // presentQueue is the ID3D12CommandQueue the swapchain was created against
    // (Unity's present/graphics queue) — the Stage 2 frame-doubler records its
    // copy onto this same queue so it's ordered after Unity's per-frame render
    // without needing Unity's fence. May be null (doubler then stays disabled).
    IDXGISwapChain1* Wrap(IDXGISwapChain1* real, ID3D12CommandQueue* presentQueue, LogFn log);
}
