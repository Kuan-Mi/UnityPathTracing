// StreamlineContext.h
// Thin wrapper around the NVIDIA Streamline SDK (v2.9.0) for the DLSS Frame
// Generation bring-up. This is the Streamline analogue of NgxContext "Test 1":
// it initializes Streamline against Unity's existing D3D12 device and queries
// DLSS-G (Frame Generation) support, so the SDK integration can be validated on
// any GPU before tackling the swapchain-proxy work.
//
// Why Streamline (vs. our own NGX path): inserting generated frames requires
// owning the swapchain's buffer rotation. Our in-place double-present test
// proved Unity desyncs and the device is removed. Streamline provides the proxy
// swapchain + frame pacing + Reflex, which is the NVIDIA-recommended route.
//
// Hardware note: like NGX FG, DLSS-G needs Ada (RTX 40-series) or newer. On the
// RTX 3060 dev box IsDLSSGSupported() is expected to return false — that is a
// PASS for this test (it confirms Streamline loaded and reported correctly).
//
// This module is initialized only when the NR_STREAMLINE env var is set, so it
// never perturbs normal runs. It uses manual hooking, so slInit does NOT touch
// Unity's existing swapchain/device — only loads plugins and answers the query.
#pragma once

struct ID3D12Device;

namespace StreamlineContext
{
    // level: 0=info, 1=warn, 2=error. Safe to call from any thread.
    using LogFn = void (*)(int level, const char* msg);

    // Optional logger. If unset, messages go to OutputDebugStringA.
    void SetLogger(LogFn fn);

    // slInit (manual hooking, DLSS-G + Reflex + PCL features), hand Streamline
    // Unity's device, then query DLSS-G support + requirements and log a summary.
    // Idempotent. Returns true if Streamline initialized (independent of whether
    // DLSS-G is available on this GPU).
    bool Initialize(ID3D12Device* device);

    // True once slInit succeeded.
    bool IsInitialized();

    // True only on hardware/driver that supports DLSS-G (Ada+). False before init.
    bool IsDLSSGSupported();

    // slShutdown. Safe to call even if Initialize was never called/failed.
    void Shutdown();
}
