// SLHooks.h
// Streamline manual-hooking plumbing shared by DLSS-G and Reflex/PCL.
#pragma once

#include <cstdint>

class SLHooks
{
public:
    // Player-only. Installs the device queue hook first, then the DXGI factory hook,
    // before Unity creates its device/queues/swapchain.
    static void InstallPresentPathHooks();

    static bool IsQueueProxyActive();

    // The proxy swapchain's current back-buffer index (the one the next Present will flip), or
    // 0xFFFFFFFF if no proxy swapchain exists yet. Used to key present-marker tokens by back buffer.
    static uint32_t CurrentBackBufferIndex();

    static void Shutdown();
};
