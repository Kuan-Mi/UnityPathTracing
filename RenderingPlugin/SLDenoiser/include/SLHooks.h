// SLHooks.h
// Streamline manual-hooking plumbing shared by DLSS-G and Reflex/PCL.
#pragma once

class SLHooks
{
public:
    // Player-only. Installs the device queue hook first, then the DXGI factory hook,
    // before Unity creates its device/queues/swapchain.
    static void InstallPresentPathHooks();

    static bool IsQueueProxyActive();
    static void Shutdown();
};
