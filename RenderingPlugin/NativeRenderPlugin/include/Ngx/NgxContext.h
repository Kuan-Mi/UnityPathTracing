// NgxContext.h
// Thin wrapper around NVIDIA NGX SDK init/shutdown and the DLSS Frame
// Generation (DLSS-FG) support query. This is "Test 1" of the FG bring-up:
// it stands NGX up on Unity's *existing* D3D12 device (NGX is normally
// initialized at device-creation time, so attaching mid-session is the first
// unknown to validate) and reports whether DLSS-FG is available.
//
// Hardware note: NGX core init + capability query are GPU-agnostic and succeed
// on any RTX card (incl. the RTX 3060 dev box, which also supports DLSS-SR).
// DLSS Frame Generation itself requires Ada (RTX 40-series) or newer, so on the
// 3060 IsFrameGenerationAvailable() is expected to return false with a
// hardware-not-supported reason — that is a PASS for this test, not a failure.
//
// Same logger-bridge convention as D3D12HeapHook / SwapChainHook.
#pragma once

struct ID3D12Device;

namespace NgxContext
{
    // level: 0=info, 1=warn, 2=error. Safe to call from any thread.
    using LogFn = void (*)(int level, const char* msg);

    // Optional logger. If unset, messages go to OutputDebugStringA.
    void SetLogger(LogFn fn);

    // Initialize the NGX SDK against Unity's existing D3D12 device, allocate a
    // capability parameter map, and query DLSS-FG support. Logs a summary of the
    // result (availability, driver requirement, init result code). Idempotent;
    // a second call after success is a no-op. Returns true if NGX itself
    // initialized (independent of whether FG is available on this GPU).
    bool Initialize(ID3D12Device* device);

    // True once NGX has been initialized successfully.
    bool IsInitialized();

    // True only on hardware/driver that supports DLSS Frame Generation (Ada+).
    // Always false before Initialize().
    bool IsFrameGenerationAvailable();

    // Destroy the capability parameter map and shut down NGX. Safe to call even
    // if Initialize() was never called or failed.
    void Shutdown(ID3D12Device* device);
}
