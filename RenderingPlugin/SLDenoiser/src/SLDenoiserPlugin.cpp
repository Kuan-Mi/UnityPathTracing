// SLDenoiserPlugin.cpp
// Unity native plugin entry for DLSS Ray Reconstruction (evaluate-time) AND DLSS-G Frame
// Generation (present-path) via Streamline. One shared slInit via SLCore (loads
// DLSS_RR + DLSS_G + Reflex + PCL). SLCore owns the SL lifecycle + logging so future
// features (DLSS-SR, NIS, …) can be added without touching init/device/shutdown.
//
// LOAD MODEL: this must be a LOAD-ON-STARTUP plugin so the DLSS-G queue/swapchain hooks are
// installed before Unity creates its device/swapchain. The FG hooks are installed ONLY in a
// standalone PLAYER — in the editor they crash, so we auto-detect the host process (the
// editor runs as Unity.exe) and skip them there. DLSS-RR (evaluate-only, no hooks) works in
// both editor and player.
//
// Keep this plugin and the legacy StreamlineProbePlugin mutually exclusive (one slInit/process).

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_5.h>

#include "IUnityInterface.h"
#include "IUnityGraphics.h"
#include "IUnityGraphicsD3D12.h"
#include "IUnityLog.h"

#include "SLCore.h"
#include "SLDlssrr.h"
#include "SLDlssrrFrameData.h"
#include "SLDlssg.h"
#include "SLHooks.h"
#include "SLReflex.h"

namespace sl { struct FrameToken; }

namespace
{
    IUnityInterfaces*      s_Interfaces = nullptr;
    IUnityGraphics*        s_Graphics   = nullptr;
    IUnityLog*             s_Log        = nullptr;
    IUnityGraphicsD3D12v7* s_D3D12      = nullptr;
    bool                   s_IsPlayer   = false; // false in the editor (Unity.exe)

    void LogBridge(int level, const char* msg)
    {
        if (s_Log)
        {
            UnityLogType type = (level == 2) ? kUnityLogTypeError
                              : (level == 1) ? kUnityLogTypeWarning
                                             : kUnityLogTypeLog;
            s_Log->Log(type, msg, __FILE__, __LINE__);
        }
        else { OutputDebugStringA(msg); OutputDebugStringA("\n"); }
    }

    // Editor (Unity.exe) vs standalone player (<Project>.exe). The FG present/queue hooks
    // are player-only; the editor gets DLSS-RR evaluate only.
    bool DetectIsPlayer()
    {
        char path[MAX_PATH] = {};
        DWORD n = GetModuleFileNameA(nullptr, path, (DWORD)sizeof(path));
        if (n == 0) return false; // unknown → treat as editor (safe: no hooks)
        const char* slash = strrchr(path, '\\');
        const char* exe = slash ? slash + 1 : path;
        return _stricmp(exe, "Unity.exe") != 0;
    }

    void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType)
    {
        if (eventType == kUnityGfxDeviceEventInitialize)
        {
            s_D3D12 = s_Interfaces->Get<IUnityGraphicsD3D12v7>();
            if (s_D3D12 && s_D3D12->GetDevice())
                SLCore::SetDevice(s_D3D12->GetDevice());
            // (Device may not exist yet at plugin-load time; the real init event sets it.)
        }
        else if (eventType == kUnityGfxDeviceEventShutdown)
        {
            SLReflex::Shutdown();
            SLHooks::Shutdown();
            SLDlssg::Shutdown();
            SLDlssrr::Shutdown();
            SLCore::Shutdown();
            s_D3D12 = nullptr;
        }
    }

    // DLSS-RR render-event-and-data (evaluate on Unity's command list).
    void UNITY_INTERFACE_API OnRRRenderEventAndData(int /*eventId*/, void* data)
    {
        if (!data || !s_D3D12) return;
        UnityGraphicsD3D12RecordingState state{};
        if (!s_D3D12->CommandRecordingState(&state) || !state.commandList) return;
        SLDlssrr::Dispatch(reinterpret_cast<SLDlssrrFrameData*>(data), state.commandList);
    }

    // Render-thread submit-start event: data is the FrameToken* minted on the main thread by
    // SL_GetNewFrameToken, forwarded verbatim via IssuePluginEventAndData. Emits the
    // eRenderSubmitStart PCL marker for that exact token. The latency-critical Reflex sleep is
    // NOT here — it runs on the main thread.
    void UNITY_INTERFACE_API OnRenderSubmitStart(int /*eventId*/, void* data)
    {
        sl::FrameToken* token = reinterpret_cast<sl::FrameToken*>(data);
        // Start of CPU render submission for this frame. Universal (PCL works on every adapter,
        // FG or not).
        if (token) SLReflex::MarkRenderSubmitStart(*token);
    }

    // Render-thread submit-end event: data is the FrameToken* for the frame whose command-stream
    // work is complete. Emits eRenderSubmitEnd before the present hook emits ePresentStart, then
    // records this frame's token in the present slot for the back buffer it rendered into — the
    // next Present flips that buffer, so the present hook resolves this exact token for it.
    // Player-only: the present hook (the resolver) only exists in the player.
    void UNITY_INTERFACE_API OnRenderSubmitEnd(int /*eventId*/, void* data)
    {
        sl::FrameToken* token = reinterpret_cast<sl::FrameToken*>(data);
        if (!token) return;
        SLReflex::MarkRenderSubmitEnd(*token);
        if (s_IsPlayer)
            SLCore::RegisterPresentToken(token, SLHooks::CurrentBackBufferIndex());
    }

    // DLSS-G per-frame inputs (render thread): tag depth/mvec + set constants.
    void UNITY_INTERFACE_API OnFGFrameInputs(int /*eventId*/, void* data)
    {
        if (data) SLDlssg::ConsumeFrameInputs(*reinterpret_cast<const SLDlssg::FrameInputs*>(data));
    }
}

extern "C"
{
    void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces* unityInterfaces)
    {
        s_Interfaces = unityInterfaces;
        s_Graphics   = s_Interfaces->Get<IUnityGraphics>();
        s_Log        = s_Interfaces->Get<IUnityLog>();
        s_IsPlayer   = DetectIsPlayer();

        s_Graphics->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);

        // Shared slInit (RR + G + Reflex + PCL) + log bridge — before the device/swapchain exist.
        SLCore::Init(&LogBridge);

        if (s_IsPlayer)
        {
            // Player: install the present-path hooks before Unity creates its device/queues/
            // swapchain. Queue hook FIRST (the present queue must be SL-proxied). The proxy
            // device/queue/swapchain are SL's common interposer (required so presentCommon() runs
            // every frame under manual hooking) and are installed on every adapter; Frame
            // Generation is only enabled as a *mode* on top when the adapter supports it.
            SLHooks::InstallPresentPathHooks();
            LogBridge(0, "[NR/SLDlssg] Player detected: present-path hooks installed (Reflex/PCL on "
                         "every adapter; DLSS-G frame generation enabled only when supported).");
        }
        else
        {
            LogBridge(0, "[NR/SLDlssg] Editor detected: present hooks skipped (DLSS-RR evaluate only).");
        }

        // If the device already exists (on-demand load), set it now; otherwise the registered
        // callback will (load-on-startup, before device creation).
        OnGraphicsDeviceEvent(kUnityGfxDeviceEventInitialize);

        LogBridge(0, "[NR/SLDlssg] SLDenoiser plugin loaded (DLSS-RR + DLSS-G via Streamline).");
    }

    void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload()
    {
        if (s_Graphics)
            s_Graphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
        SLReflex::Shutdown();
        SLHooks::Shutdown();
        SLDlssg::Shutdown();
        SLDlssrr::Shutdown();
        SLCore::Shutdown();
        LogBridge(0, "[NR/SLDlssg] SLDenoiser plugin unloaded.");
    }

    // ---- DLSS-RR (evaluate) ----
    UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
    GetSLDlssrrRenderEventAndDataFunc() { return OnRRRenderEventAndData; }

    int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API CreateSLDlssrrInstance() { return SLDlssrr::CreateInstance(); }
    void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DestroySLDlssrrInstance(int id) { SLDlssrr::DestroyInstance(id); }

    bool UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SLDlssrr_QueryOptimalRenderSize(
        unsigned outputWidth, unsigned outputHeight, unsigned char mode,
        unsigned* outRenderWidth, unsigned* outRenderHeight)
    {
        return SLDlssrr::QueryOptimalRenderSize(outputWidth, outputHeight, mode, outRenderWidth, outRenderHeight);
    }

    // ---- Streamline render-submit PCL markers ----
    // Issue with the token from SL_GetNewFrameToken as data:
    //   cmd.IssuePluginEventAndData(GetSLRenderSubmitStartEventFunc(), 0, frameToken).
    UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
    GetSLRenderSubmitStartEventFunc() { return OnRenderSubmitStart; }

    UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
    GetSLRenderSubmitEndEventFunc() { return OnRenderSubmitEnd; }

    // ---- DLSS-G (frame generation) ----
    // Issue per frame: cmd.IssuePluginEventAndData(GetSLFGFrameInputsFunc(), 0, ptrToFrameInputs).
    UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
    GetSLFGFrameInputsFunc() { return OnFGFrameInputs; }

    void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SL_SetFrameGeneration(int enable)
    {
        SLDlssg::SetFrameGeneration(enable != 0);
    }

    int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SL_IsFrameGenerationOn()
    {
        return SLDlssg::IsFrameGenerationOn() ? 1 : 0;
    }

    // ---- Streamline frame token + Reflex simulation markers (main thread, not plugin events) ----
    typedef void* SLFrameTokenHandle;

    // SL_GetNewFrameToken: mint this frame's shared Streamline token. C# forwards this opaque
    // handle to render-thread events and to every main-thread marker call for the same frame.
    SLFrameTokenHandle UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SL_GetNewFrameToken()
    {
        // Still mint the per-frame token in the editor — DLSS-RR (evaluate-only, editor + player)
        // forwards it to the render thread for tagging.
        return SLCore::GetNewFrameToken();
    }

    // SL_ReflexSleep: main thread, top of frame BEFORE input. Applies Reflex options and calls
    // slReflexSleep for this token. Player-only; no-op in the editor.
    void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SL_ReflexSleep(void* frameToken)
    {
        if (!s_IsPlayer || !frameToken) return;
        SLReflex::Sleep(*reinterpret_cast<sl::FrameToken*>(frameToken));
    }

    // SL_MarkSimulationStart: main thread, immediately after Reflex sleep, before input/sim.
    void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SL_MarkSimulationStart(void* frameToken)
    {
        if (!s_IsPlayer || !frameToken) return;
        SLReflex::MarkSimulationStart(*reinterpret_cast<sl::FrameToken*>(frameToken));
    }

    // SL_MarkSimulationEnd: main thread, after game logic (before rendering). eSimulationEnd.
    // frameToken is this frame's token from SL_GetNewFrameToken. Player-only (Reflex/PCL).
    void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SL_MarkSimulationEnd(void* frameToken)
    {
        if (!s_IsPlayer || !frameToken) return;
        SLReflex::MarkSimulationEnd(*reinterpret_cast<sl::FrameToken*>(frameToken));
    }

    // SL_ConsumePclPingCount: main thread, after SL_GetNewFrameToken minted the token for the
    // frame that is about to sample input. Returns how many PCL stats pings the WndProc saw
    // since the previous consume; C# owns the token attribution and calls SL_MarkPclLatencyPing.
    unsigned UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SL_ConsumePclPingCount()
    {
        if (!s_IsPlayer) return 0;
        return SLReflex::ConsumePclPingCount();
    }

    void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SL_MarkPclLatencyPing(void* frameToken, unsigned count)
    {
        if (!s_IsPlayer || !frameToken || count == 0) return;
        SLReflex::MarkPclLatencyPing(*reinterpret_cast<sl::FrameToken*>(frameToken), count);
    }

    // SL_MarkTriggerFlash: main thread, on the frame whose input sampled a trigger (click/fire).
    // Drives the Reflex Latency Analyzer flash indicator. Player-only.
    void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SL_MarkTriggerFlash(void* frameToken)
    {
        if (!s_IsPlayer || !frameToken) return;
        SLReflex::MarkTriggerFlash(*reinterpret_cast<sl::FrameToken*>(frameToken));
    }

    // ---- Reflex (low latency) ----
    // mode: 0 = Off, 1 = On (Low Latency), 2 = On + Boost. fpsCapUs: 0 = uncapped.
    // Player-only: tied to the DLSS-G present path, so it's a no-op in the editor (Unity.exe).
    void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SL_SetReflexMode(int mode, unsigned fpsCapUs)
    {
        if (!s_IsPlayer) return; // Reflex is player-only.
        SLReflex::SetMode(mode, fpsCapUs);
    }

    int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SL_GetReflexMode()
    {
        if (!s_IsPlayer) return 0; // Off in the editor — Reflex never runs there.
        return SLReflex::GetMode();
    }

    int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SL_IsReflexLowLatencyAvailable()
    {
        if (!s_IsPlayer) return 0; // Reflex is player-only.
        return SLReflex::IsLowLatencyAvailable() ? 1 : 0;
    }

    int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SL_GetReflexStats(SLReflex::Stats* stats)
    {
        if (!s_IsPlayer || !stats) return 0;
        return SLReflex::GetStats(*stats) ? 1 : 0;
    }
}
