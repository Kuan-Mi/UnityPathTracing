// DlssgProbe.cpp — see header. Persistent log-only DLSS-G context.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgiformat.h>
#include <wrl/client.h>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <string>

#include "nvsdk_ngx.h"
#include "nvsdk_ngx_helpers.h"
#include "nvsdk_ngx_defs_dlssg.h"
#include "nvsdk_ngx_helpers_dlssg.h"

#include "DlssgProbe.h"

using Microsoft::WRL::ComPtr;

namespace
{
    DlssgProbe::LogFn g_log = nullptr;
    std::atomic<bool> g_initAttempted{false};
    std::atomic<bool> g_ready{false};

    // Persistent NGX + D3D12 state (created in Init, torn down in Shutdown).
    ID3D12Device*        g_device   = nullptr;   // borrowed (Unity-owned)
    NVSDK_NGX_Parameter* g_params   = nullptr;
    NVSDK_NGX_Handle*    g_handle   = nullptr;

    ComPtr<ID3D12CommandQueue>        g_queue;
    ComPtr<ID3D12CommandAllocator>    g_alloc;
    ComPtr<ID3D12GraphicsCommandList> g_list;
    ComPtr<ID3D12Fence>               g_fence;
    UINT64                            g_fenceVal = 0;

    ComPtr<ID3D12Resource> g_color;     // FG color input  (back buffer stand-in)
    ComPtr<ID3D12Resource> g_depth;     // FG depth input
    ComPtr<ID3D12Resource> g_mvec;      // FG motion vectors
    ComPtr<ID3D12Resource> g_output;    // FG interpolated-frame output (UAV)

    unsigned g_w = 0, g_h = 0;
    unsigned g_evalCount = 0;

    void Logf(int level, const char* fmt, ...)
    {
        char buf[640];
        va_list ap; va_start(ap, fmt);
        int n = _vsnprintf_s(buf, sizeof(buf), _TRUNCATE, fmt, ap);
        va_end(ap);
        if (n < 0) buf[sizeof(buf) - 1] = '\0';
        char prefixed[704];
        _snprintf_s(prefixed, sizeof(prefixed), _TRUNCATE, "[NR/DlssgProbe] %s", buf);
        if (g_log) g_log(level, prefixed);
        else { OutputDebugStringA(prefixed); OutputDebugStringA("\n"); }
    }

    void NVSDK_CONV NgxLogCallback(const char* message, NVSDK_NGX_Logging_Level,
                                   NVSDK_NGX_Feature)
    {
        if (!message) return;
        char line[704];
        _snprintf_s(line, sizeof(line), _TRUNCATE, "[NR/DlssgProbe][NGX] %s", message);
        for (int i = (int)strnlen_s(line, sizeof(line)) - 1;
             i >= 0 && (line[i] == '\n' || line[i] == '\r'); --i)
            line[i] = '\0';
        if (g_log) g_log(0, line);
    }

    std::string Narrow(const wchar_t* w)
    {
        if (!w) return {};
        int len = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
        if (len <= 0) return {};
        std::string s(len - 1, '\0');
        WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), len, nullptr, nullptr);
        return s;
    }
    const char* Res(NVSDK_NGX_Result r) { static std::string s; s = Narrow(GetNGXResultAsString(r)); return s.c_str(); }

    std::wstring ModuleDir()
    {
        HMODULE h = nullptr;
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&ModuleDir), &h);
        wchar_t path[MAX_PATH] = {};
        GetModuleFileNameW(h, path, MAX_PATH);
        std::wstring s(path);
        size_t p = s.find_last_of(L"\\/");
        if (p != std::wstring::npos) s.resize(p);
        return s;
    }

    int GetI(NVSDK_NGX_Parameter* p, const char* key)  { int v = -1; NVSDK_NGX_Parameter_GetI(p, key, &v); return v; }
    unsigned GetUI(NVSDK_NGX_Parameter* p, const char* key) { unsigned v = 0; NVSDK_NGX_Parameter_GetUI(p, key, &v); return v; }

    // Committed texture in a default heap.
    bool CreateTex(DXGI_FORMAT fmt, unsigned w, unsigned h, bool uav,
                   D3D12_RESOURCE_STATES state, ComPtr<ID3D12Resource>& out, const char* name)
    {
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;

        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width            = w;
        rd.Height           = h;
        rd.DepthOrArraySize = 1;
        rd.MipLevels        = 1;
        rd.Format           = fmt;
        rd.SampleDesc.Count = 1;
        rd.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        rd.Flags            = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                                  : D3D12_RESOURCE_FLAG_NONE;

        HRESULT hr = g_device->CreateCommittedResource(
            &hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr, IID_PPV_ARGS(&out));
        if (FAILED(hr)) { Logf(2, "CreateTex(%s) failed hr=0x%08lx", name, (unsigned long)hr); return false; }
        return true;
    }

    void WaitGpu()
    {
        const UINT64 v = ++g_fenceVal;
        g_queue->Signal(g_fence.Get(), v);
        if (g_fence->GetCompletedValue() < v)
        {
            HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            g_fence->SetEventOnCompletion(v, ev);
            WaitForSingleObject(ev, 10000);
            CloseHandle(ev);
        }
    }

    void Identity(float m[4][4]) { for (int i=0;i<4;++i) for (int j=0;j<4;++j) m[i][j] = (i==j)?1.0f:0.0f; }
}

namespace DlssgProbe
{
    void Init(ID3D12Device* device, unsigned bbWidth, unsigned bbHeight,
              unsigned bbFormatDXGI, LogFn log)
    {
        g_log = log;
        if (!device) return;  // retry-able: don't latch on null device
        bool expected = false;
        if (!g_initAttempted.compare_exchange_strong(expected, true)) return;

        g_device = device;
        g_w = bbWidth; g_h = bbHeight;

        const std::wstring dir = ModuleDir();
        Logf(0, "Init. device=%p target=%ux%u fmt=%u; nvngx dir=%s",
             (void*)device, bbWidth, bbHeight, bbFormatDXGI, Narrow(dir.c_str()).c_str());

        const wchar_t* paths[] = { dir.c_str() };
        NVSDK_NGX_FeatureCommonInfo info = {};
        info.PathListInfo.Path   = paths;
        info.PathListInfo.Length = 1;
        info.LoggingInfo.LoggingCallback     = &NgxLogCallback;
        info.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_ON;

        NVSDK_NGX_Result r = NVSDK_NGX_D3D12_Init_with_ProjectID(
            "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0", dir.c_str(), device, &info);
        if (NVSDK_NGX_FAILED(r)) { Logf(2, "NGX init FAILED (0x%08x %s)", r, Res(r)); return; }
        Logf(0, "NGX init OK");

        r = NVSDK_NGX_D3D12_GetCapabilityParameters(&g_params);
        if (NVSDK_NGX_FAILED(r) || !g_params)
        { Logf(2, "GetCapabilityParameters FAILED (0x%08x %s)", r, Res(r)); NVSDK_NGX_D3D12_Shutdown1(device); return; }

        const int available = GetI(g_params, NVSDK_NGX_Parameter_FrameGeneration_Available);
        Logf(available ? 0 : 1,
             "FrameGeneration: Available=%d NeedsUpdatedDriver=%d MinDriver=%d.%d MultiFrameCountMax=%u",
             available,
             GetI(g_params, NVSDK_NGX_Parameter_FrameGeneration_NeedsUpdatedDriver),
             GetI(g_params, NVSDK_NGX_Parameter_FrameGeneration_MinDriverVersionMajor),
             GetI(g_params, NVSDK_NGX_Parameter_FrameGeneration_MinDriverVersionMinor),
             GetUI(g_params, NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax));
        if (!available) { Logf(1, "FG not available; aborting init."); NVSDK_NGX_D3D12_DestroyParameters(g_params); g_params=nullptr; NVSDK_NGX_D3D12_Shutdown1(device); return; }

        // Persistent command objects for recording Create + Evaluate.
        D3D12_COMMAND_QUEUE_DESC qd = {}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&g_queue))) ||
            FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g_alloc))) ||
            FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_alloc.Get(), nullptr, IID_PPV_ARGS(&g_list))) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_fence))))
        { Logf(2, "Failed to create command objects"); Shutdown(); return; }

        // Plugin-owned FG resources (never touch Unity's swapchain).
        if (!CreateTex(DXGI_FORMAT_R8G8B8A8_UNORM, g_w, g_h, false, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, g_color, "color") ||
            !CreateTex(DXGI_FORMAT_R32_FLOAT,      g_w, g_h, false, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, g_depth, "depth") ||
            !CreateTex(DXGI_FORMAT_R16G16_FLOAT,   g_w, g_h, false, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, g_mvec,  "mvec")  ||
            !CreateTex(DXGI_FORMAT_R8G8B8A8_UNORM, g_w, g_h, true,  D3D12_RESOURCE_STATE_UNORDERED_ACCESS,          g_output,"output"))
        { Shutdown(); return; }

        // Create the FG feature (records init commands on g_list).
        NVSDK_NGX_DLSSG_Create_Params cp = {};
        cp.Width = g_w; cp.Height = g_h; cp.NativeBackbufferFormat = bbFormatDXGI;
        cp.RenderWidth = g_w; cp.RenderHeight = g_h; cp.DynamicResolutionScaling = false;

        r = NGX_D3D12_CREATE_DLSSG(g_list.Get(), 1, 1, &g_handle, g_params, &cp);
        g_list->Close();
        if (NVSDK_NGX_FAILED(r) || !g_handle)
        { Logf(2, "NGX_D3D12_CREATE_DLSSG FAILED (0x%08x %s)", r, Res(r)); Shutdown(); return; }

        ID3D12CommandList* lists[] = { g_list.Get() };
        g_queue->ExecuteCommandLists(1, lists);
        WaitGpu();

        g_ready.store(true);
        Logf(0, "FG feature created (handle id=%u) + resources ready. Will run Evaluate smoke tests.", g_handle->Id);
    }

    void Evaluate()
    {
        if (!g_ready.load()) return;

        if (FAILED(g_alloc->Reset()) || FAILED(g_list->Reset(g_alloc.Get(), nullptr)))
        { Logf(2, "Evaluate: command list reset failed"); return; }

        NVSDK_NGX_D3D12_DLSSG_Eval_Params ep = {};
        ep.pBackbuffer        = g_color.Get();
        ep.pDepth             = g_depth.Get();
        ep.pMVecs             = g_mvec.Get();
        ep.pOutputInterpFrame = g_output.Get();

        NVSDK_NGX_DLSSG_Opt_Eval_Params opt = {};
        opt.multiFrameCount = 1;
        opt.multiFrameIndex = 1;
        Identity(opt.cameraViewToClip);
        Identity(opt.clipToCameraView);
        Identity(opt.clipToLensClip);
        Identity(opt.clipToPrevClip);
        Identity(opt.prevClipToClip);
        opt.mvecScale[0] = 1.0f; opt.mvecScale[1] = 1.0f;
        opt.cameraNear = 0.1f; opt.cameraFar = 1000.0f; opt.cameraFOV = 1.0f;
        opt.cameraAspectRatio = (float)g_w / (float)g_h;
        opt.depthInverted = false;
        opt.cameraMotionIncluded = true;
        opt.reset = (g_evalCount == 0);   // first eval has no valid history

        NVSDK_NGX_Result r = NGX_D3D12_EVALUATE_DLSSG(g_list.Get(), g_handle, g_params, &ep, &opt);
        g_list->Close();
        if (NVSDK_NGX_FAILED(r))
        {
            Logf(2, "NGX_D3D12_EVALUATE_DLSSG #%u FAILED (0x%08x %s)", g_evalCount + 1, r, Res(r));
            // Stop after a failure to avoid log spam; feature stays alive.
            g_ready.store(false);
            return;
        }

        ID3D12CommandList* lists[] = { g_list.Get() };
        g_queue->ExecuteCommandLists(1, lists);
        WaitGpu();

        ++g_evalCount;
        Logf(0, "NGX_D3D12_EVALUATE_DLSSG #%u SUCCESS (reset=%d) — FG evaluate runs in-process.",
             g_evalCount, (int)opt.reset);
    }

    void Shutdown()
    {
        g_ready.store(false);
        if (g_handle)  { NVSDK_NGX_D3D12_ReleaseFeature(g_handle); g_handle = nullptr; }
        if (g_params)  { NVSDK_NGX_D3D12_DestroyParameters(g_params); g_params = nullptr; }
        g_output.Reset(); g_mvec.Reset(); g_depth.Reset(); g_color.Reset();
        g_list.Reset(); g_alloc.Reset(); g_fence.Reset(); g_queue.Reset();
        if (g_device) { NVSDK_NGX_D3D12_Shutdown1(g_device); g_device = nullptr; }
    }

    bool InitAttempted() { return g_initAttempted.load(); }
    bool IsReady()       { return g_ready.load(); }
}
