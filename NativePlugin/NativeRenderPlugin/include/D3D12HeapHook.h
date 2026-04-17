// D3D12HeapHook.h
// VTable hook for ID3D12GraphicsCommandList::SetDescriptorHeaps.
// Caches the heaps Unity binds so the plugin can restore them after
// running its own dispatches that bind a private descriptor heap.
//
// Technique adapted from DX12BindlessUnity (meetem) — see
// DX12BindlessUnity/NativePlugin.Bindless/PluginSource/source/RenderAPI_D3D12.cpp
#pragma once

struct ID3D12GraphicsCommandList;
struct ID3D12Device;

namespace D3D12HeapHook
{
    // Logger callback. level: 0=info, 1=warn, 2=error. Safe to call from any thread.
    using LogFn = void (*)(int level, const char* msg);

    // Optional: install a logger. If not set, messages go to OutputDebugStringA.
    void SetLogger(LogFn fn);

    // Install the SetDescriptorHeaps vtable hook using the given command list's
    // vtable. Idempotent and thread-safe: only the first successful call patches.
    // Safe to call every render event.
    void InstallHook(ID3D12GraphicsCommandList* cmdList);

    // Install the hook using a freshly created throwaway command list. Use this
    // at plugin init so the hook is active BEFORE Unity issues its first
    // SetDescriptorHeaps call. The vtable is shared across all cmd lists of
    // the same interface, so patching once is sufficient. Returns true on
    // success.
    bool InstallHookFromDevice(ID3D12Device* device);

    // Mark the start / end of the plugin's own rendering work on this thread.
    // While inside this scope, calls to SetDescriptorHeaps are passed through
    // but are NOT cached as "Unity's heaps". Pair these calls around plugin
    // Dispatch so RestoreUnityHeaps can replay Unity's real heaps afterwards.
    void BeginPluginDispatch();
    void EndPluginDispatch();

    // Re-issue the most recent heaps Unity bound on this thread by calling the
    // original (un-hooked) SetDescriptorHeaps. No-op if the hook was never
    // installed or no heaps have been captured yet on this thread.
    void RestoreUnityHeaps(ID3D12GraphicsCommandList* cmdList);
}
