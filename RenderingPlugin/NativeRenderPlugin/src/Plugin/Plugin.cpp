/*
 * Plugin.cpp Unity Native DX12 Ray Tracing Plugin
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <list>
#include <mutex>
#include <string>

#include "IUnityInterface.h"
#include "IUnityGraphics.h"
#include "IUnityGraphicsD3D12.h"
#include "IUnityLog.h"

#include "AccelerationStructure.h"
#include "RayTraceShader.h"
#include "RayTraceDescriptorSet.h"
#include "ComputeShader.h"
#include "ComputeDescriptorSet.h"
#include "RasterShader.h"
#include "RasterDescriptorSet.h"
#include "DescriptorHeapAllocator.h"
#include "BindlessTexture.h"
#include "BindlessBuffer.h"
#include "BindlessUAVTexture.h"
#include "NativeBuffer.h"
#include "ResourceStateTracker.h"
#include "D3D12HeapHook.h"
#include "SwapChainHook.h"
#include "NgxContext.h"
#include "PluginInternal.h"
#include "DeferredDeleteQueue.h"
#include <map>
#include <vector>
#include <functional>
#include <mutex>

using Microsoft::WRL::ComPtr;


// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
static IUnityInterfaces*         s_Interfaces  = nullptr;
static IUnityGraphics*           s_Graphics    = nullptr;
static IUnityGraphicsD3D12v7*    s_D3D12       = nullptr;
static IUnityGraphicsD3D12v8*    s_D3D12v8     = nullptr;
static IUnityLog*                s_Log         = nullptr;
static DescriptorHeapAllocator   s_DescHeap;   // global shared GPU-visible CBV/SRV/UAV heap
static ComPtr<ID3D12DescriptorHeap> s_ClearCpuHeap; // 1-slot CPU-only heap for ClearUnorderedAccessViewUint
static bool                      s_RendererReady = false;

// Event used by NR_WaitForGpuFlush: set by GpuFlushAndSignalCallback on the render thread
// after FlushGpuAndWait() completes, so the main thread can safely free ring buffers.
static HANDLE                    s_gpuFlushDoneEvent = nullptr;

// Global triple-buffer frame index — advanced once per frame by AccelerationStructure::BuildOrUpdate().
// All subsystems (AccelerationStructure, ComputeDescriptorSet, …) index into their
// per-frame ring buffers using this value so they stay in sync.
uint32_t g_frameIndex = 0;

// Monotonic frame serial — see PluginInternal.h. Advanced with g_frameIndex.
uint64_t g_frameSerial = 0;

// Shared transient descriptor ring (see PluginInternal.h). Reserves a contiguous
// sub-range of s_DescHeap at renderer init; descriptor-set dispatches bump-allocate
// SRV/UAV tables out of it and the ring reclaims them once the frame fence completes.
TransientDescriptorRing g_transientRing;

// Capacity of the transient ring in descriptor slots. Sized so it can absorb several
// frames-in-flight worth of dispatches across all descriptor sets without exhaustion.
// Must be <= DescriptorHeapAllocator::kCapacity minus expected bindless usage.
static constexpr uint32_t kTransientRingCapacity = 32768u;

// Shared UPLOAD-heap chunk pool (see PluginInternal.h). Stages CPU writes into
// DEFAULT-heap buffers; chunks are recycled once the frame fence completes.
SharedUploadPool g_uploadPool;

// Shared DEFAULT-heap UAV scratch pool (see PluginInternal.h). Transient build
// scratch for acceleration-structure builds; recycled by the frame fence.
SharedUploadPool g_scratchPool;

// ---------------------------------------------------------------------------
// Deferred deletion — delays destruction of GPU-facing objects until the GPU has
// finished any commands that may reference them. The mechanism lives in
// DeferredDeleteQueue; the free functions below (EnqueueCleanup / SafeReleaseResource
// / NR_EnqueueDescriptorRangeFree, plus RetireObject<T> in PluginInternal.h) are thin
// wrappers over the single instance so resource code stays decoupled from it.
// ---------------------------------------------------------------------------
static DeferredDeleteQueue s_DeleteQueue;

// Forward declarations
static void PluginLog(UnityLogType type, const char* msg, const char* file, int line);
static void PluginLogFmt(UnityLogType type, const char* file, int line, const char* fmt, ...);

#define NR_LOG(fmt, ...)   PluginLogFmt(kUnityLogTypeLog,     __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define NR_WARN(fmt, ...)  PluginLogFmt(kUnityLogTypeWarning, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define NR_ERROR(fmt, ...) PluginLogFmt(kUnityLogTypeError,   __FILE__, __LINE__, fmt, ##__VA_ARGS__)

// kDeleteDelay: number of frames to wait before freeing.
// Unity D3D12 uses up to 2 frames in flight; 3 provides a safe margin.
static constexpr int kDeleteDelay = 3;

void EnqueueCleanup(std::function<void()>&& cleanupTask)
{
    s_DeleteQueue.Enqueue(std::move(cleanupTask));
}

void SafeReleaseResource(ComPtr<ID3D12Resource> resource)
{
    if (!resource) return;

    
        // wchar_t name[128]; UINT size = sizeof(name);
        // resource->GetPrivateData(WKPDID_D3DDebugObjectNameW, &size, name);
        // char logMsg[256];
        // snprintf(logMsg, sizeof(logMsg), "Deferred Resource Released: %ls", name);
        // PluginLog(kUnityLogTypeLog, logMsg, __FILE__, __LINE__);


    // 利用 Lambda 捕获 ComPtr，增加引用计数，等 Lambda 执行完自动释放
    EnqueueCleanup([res = std::move(resource)]() {
        // res 在此处超出作用域，自动调用 Release()
    });
}

void NR_EnqueueDescriptorRangeFree(DescriptorHeapAllocator* alloc, uint32_t base, uint32_t count)
{
    if (!alloc || count == 0) return;
    EnqueueCleanup([alloc, base, count]() {
        alloc->Free(base, count);
    });
}

static void DrainDeferredDeletes(bool force = false)
{
    // Advance the global triple-buffer frame index at the start of each frame tick.
    // Skipped when force=true (shutdown path) to avoid disturbing the index during
    // teardown. (Frame-index advancement is a plugin-wide concern, not the queue's.)
    if (!force)
    {
        g_frameIndex = (g_frameIndex + 1) % kGlobalNumFrames;
        ++g_frameSerial;
    }

    s_DeleteQueue.Drain(force);
}

// ---------------------------------------------------------------------------
// Logging helpers - fall back to printf when IUnityLog isn't available yet
// ---------------------------------------------------------------------------
static void PluginLog(UnityLogType type, const char* msg, const char* file, int line)
{
    if (s_Log)
        s_Log->Log(type, msg, file, line);
    else
        printf("[NativeRender] %s\n", msg);
}

static void PluginLogFmt(UnityLogType type, const char* file, int line, const char* fmt, ...)
{
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    PluginLog(type, buf, file, line);
}

static void FlushGpuAndWait();

// Bridge D3D12HeapHook logs into Unity's log (called from any thread).
static void HeapHookLogBridge(int level, const char* msg)
{
    UnityLogType type = (level == 2) ? kUnityLogTypeError
                      : (level == 1) ? kUnityLogTypeWarning
                                     : kUnityLogTypeLog;
    PluginLog(type, msg, __FILE__, __LINE__);
}

// Bridge SwapChainHook diagnostics into Unity's log (called from any thread).
static void SwapChainHookLogBridge(int level, const char* msg)
{
    UnityLogType type = (level == 2) ? kUnityLogTypeError
                      : (level == 1) ? kUnityLogTypeWarning
                                     : kUnityLogTypeLog;
    PluginLog(type, msg, __FILE__, __LINE__);
}

// Bridge NgxContext diagnostics into Unity's log (called from any thread).
static void NgxLogBridge(int level, const char* msg)
{
    UnityLogType type = (level == 2) ? kUnityLogTypeError
                      : (level == 1) ? kUnityLogTypeWarning
                                     : kUnityLogTypeLog;
    PluginLog(type, msg, __FILE__, __LINE__);
}

// Try to grab Unity's swapchain (player-only) and patch its Present vtable.
// Cheap no-op once installed; safe to call from render callbacks every frame.
static void TryHookUnitySwapChainPresent()
{
    if (!SwapChainHook::IsPresentHookInstalled())
    {
        IDXGISwapChain* sc = s_D3D12v8 ? s_D3D12v8->GetSwapChain() : nullptr;
        if (sc) SwapChainHook::TryInstallPresentHook(sc);
    }

    // Once the Present hook is live, optionally enable the DLSS-FG pacing stub.
    // Env-gated (NR_FG_PACING_STUB=1), default OFF, so normal runs are unaffected.
    // Needs Unity's device + the queue the swapchain presents on.
    if (SwapChainHook::IsPresentHookInstalled() && !SwapChainHook::IsPacingStubEnabled())
    {
        static const bool wantStub = []{
            char buf[8] = {};
            DWORD n = GetEnvironmentVariableA("NR_FG_PACING_STUB", buf, sizeof(buf));
            return n > 0 && buf[0] == '1';
        }();
        if (wantStub && s_D3D12)
        {
            ID3D12Device*       dev = s_D3D12->GetDevice();
            ID3D12CommandQueue* q   = s_D3D12->GetCommandQueue();
            if (dev && q) SwapChainHook::EnablePacingStub(dev, q);
        }
    }
}

// ---------------------------------------------------------------------------
// Device lifecycle callback
// ---------------------------------------------------------------------------
static void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType)
{
    if (eventType == kUnityGfxDeviceEventInitialize)
    {
        // Attach our heap-hook logger to Unity's log (best-effort; falls back
        // to OutputDebugStringA if IUnityLog is unavailable).
        D3D12HeapHook::SetLogger(&HeapHookLogBridge);

        // Acquire v7 for core API (GetDevice, CommandRecordingState).
        // Also try v8 for NotifyResourceState to keep Unity's resource state tracker in sync.
        s_D3D12   = s_Interfaces->Get<IUnityGraphicsD3D12v7>();
        s_D3D12v8 = s_Interfaces->Get<IUnityGraphicsD3D12v8>();
        if (!s_D3D12)
        {
            NR_ERROR("IUnityGraphicsD3D12v7 not available - is D3D12 the active graphics API?");
            return;
        }

        // Wire the deferred-delete queue to Unity's frame fence. The getters read the
        // live s_D3D12 each call, so they stay valid across device re-init / shutdown.
        s_DeleteQueue.Initialize(
            []() -> uint64_t { ID3D12Fence* f = s_D3D12 ? s_D3D12->GetFrameFence() : nullptr; return f ? f->GetCompletedValue() : 0; },
            []() -> uint64_t { return s_D3D12 ? s_D3D12->GetNextFrameFenceValue() : 0; },
            kDeleteDelay);

        ID3D12Device* device = s_D3D12->GetDevice();
        if (!device)
        {
            NR_ERROR("GetDevice() returned nullptr");
            return;
        }

        // Patch the SetDescriptorHeaps vtable slot NOW, before Unity records
        // any more commands. If we wait until our first render callback, we
        // will miss Unity's earlier SetDescriptorHeaps calls in the same frame.
        D3D12HeapHook::InstallHookFromDevice(device);

        // Diagnostic: try to patch Present on Unity's swapchain. Returns null in
        // the editor (no game swapchain); retried later from the render callback.
        TryHookUnitySwapChainPresent();

        // DLSS-FG bring-up Test 1: stand NGX up on Unity's existing device and
        // query DLSS-FG support. Init is GPU-agnostic (succeeds on the RTX 3060
        // dev box); FG availability will report false on pre-Ada hardware, which
        // is the expected result there. Best-effort — never gate the renderer.
        NgxContext::SetLogger(&NgxLogBridge);
        NgxContext::Initialize(device);

        // Check DXR (ID3D12Device5) support
        ComPtr<ID3D12Device5> dev5;
        s_RendererReady = SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&dev5)));
        if (s_RendererReady)
        {
            if (!s_DescHeap.Initialize(device))
                NR_ERROR("DescriptorHeapAllocator initialization failed");

            // Reserve the transient descriptor ring from the shared heap *before*
            // any other allocations so it gets a fixed range at the bottom.
            if (!g_transientRing.Initialize(&s_DescHeap, kTransientRingCapacity))
                NR_ERROR("TransientDescriptorRing initialization failed");

            // Shared UPLOAD-heap chunk pool for staging CPU writes into DEFAULT buffers.
            if (!g_uploadPool.Initialize(device))
                NR_ERROR("SharedUploadPool initialization failed");

            // Shared DEFAULT-heap UAV scratch pool for acceleration-structure builds.
            if (!g_scratchPool.Initialize(device, (4ull << 20), /*isScratch*/ true))
                NR_ERROR("Scratch SharedUploadPool initialization failed");

            // 1-slot CPU-only heap used by ClearNativeGpuBufferCallback
            {
                D3D12_DESCRIPTOR_HEAP_DESC hd = {};
                hd.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
                hd.NumDescriptors = 1;
                hd.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
                if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&s_ClearCpuHeap))))
                    NR_ERROR("Failed to create CPU-only UAV heap for buffer clear");
            }

            NR_LOG("Plugin initialized (DXR device confirmed)");
        }
        else
            NR_ERROR("Device does not support DXR (ID3D12Device5 unavailable)");

        // Create the manual-reset event used by NR_WaitForGpuFlush / GpuFlushAndSignalCallback.
        if (!s_gpuFlushDoneEvent)
            s_gpuFlushDoneEvent = CreateEventW(nullptr, /*bManualReset=*/TRUE, /*bInitialState=*/FALSE, nullptr);
    }
    else if (eventType == kUnityGfxDeviceEventShutdown)
    {
        NR_LOG("Plugin shutdown - BEGIN");
        NR_LOG("Plugin shutdown - calling FlushGpuAndWait to sync GPU...");
        // CRITICAL: Wait for all GPU operations to complete before releasing resources.
        // This prevents D3D12 validation errors when resources are released while still
        // referenced by in-flight GPU commands (especially BLAS builds for skinned meshes).
        FlushGpuAndWait();
        NR_LOG("Plugin shutdown - GPU sync complete");

        // Force-drain any pending deferred deletes before tearing down.
        // CRITICAL: Use force=true to release ALL resources regardless of fence value,
        // because we're about to reset s_DeletionFenceValue to 0. If we don't force-drain,
        // resources with high safeAfterValue will remain in the queue and be incorrectly
        // released when the fence value wraps around on the next play mode entry.
        NR_LOG("Plugin shutdown - draining deferred deletes (force=true)...");
        DrainDeferredDeletes(true);
        NR_LOG("Plugin shutdown - deferred deletes drained");

        // Shut NGX down while Unity's device is still alive.
        NgxContext::Shutdown(s_D3D12 ? s_D3D12->GetDevice() : nullptr);

        g_transientRing.Shutdown();
        g_uploadPool.Shutdown();
        g_scratchPool.Shutdown();
        s_DescHeap.Shutdown();
        s_ClearCpuHeap.Reset();
        s_RendererReady = false;
        s_D3D12         = nullptr;
        s_D3D12v8       = nullptr;

        if (s_gpuFlushDoneEvent) { CloseHandle(s_gpuFlushDoneEvent); s_gpuFlushDoneEvent = nullptr; }
        NR_LOG("Plugin shutdown - COMPLETE");
    }
}

// ---------------------------------------------------------------------------
// UnityPluginLoad / UnityPluginUnload
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
UnityPluginLoad(IUnityInterfaces* unityInterfaces)
{
    s_Interfaces = unityInterfaces;
    s_Log        = unityInterfaces->Get<IUnityLog>();   // may be nullptr in very old versions
    s_Graphics   = unityInterfaces->Get<IUnityGraphics>();

    if (!s_Graphics)
    {
        NR_ERROR("IUnityGraphics not available");
        return;
    }

    // Install the DXGI factory hook as early as possible — this is our only
    // chance to catch a swapchain *creation* call (CreateSwapChain[ForHwnd]).
    // Diagnostic-only: it logs and calls through. Returns the Present hook for
    // any swapchain created after this point; the steady-state present path is
    // also captured lazily via TryHookUnitySwapChainPresent() (GetSwapChain).
    SwapChainHook::SetLogger(&SwapChainHookLogBridge);
    SwapChainHook::InstallFactoryHook();

    s_Graphics->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);

    // The device may already exist (e.g., plugin loaded late in editor).
    OnGraphicsDeviceEvent(kUnityGfxDeviceEventInitialize);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
UnityPluginUnload()
{
    if (s_Graphics)
        s_Graphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
}

// ---------------------------------------------------------------------------
// NR_CreateAccelerationStructure
//   Allocates a new AccelerationStructure and returns an opaque
//   uint64_t handle (pointer cast). Caller owns the lifetime and must call
//   NR_DestroyAccelerationStructure when done.
// ---------------------------------------------------------------------------
extern "C" uint64_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CreateAccelerationStructure()
{
    if (!s_RendererReady)
    {
        NR_WARN("NR_CreateAccelerationStructure called before renderer is ready");
        return 0;
    }
    // Obtain ID3D12Device5 (DXR support already confirmed during plugin init).
    ID3D12Device5* dev = nullptr;
    ID3D12Device*  devBase = s_D3D12->GetDevice();
    if (!devBase || FAILED(devBase->QueryInterface(IID_PPV_ARGS(&dev))))
    {
        NR_ERROR("NR_CreateAccelerationStructure: failed to obtain ID3D12Device5");
        return 0;
    }
    auto* as = new AccelerationStructure(dev, s_Log);
    dev->Release(); // QueryInterface added a ref, release it now
    if (s_D3D12v8)
        as->SetUnityGraphics(s_D3D12v8);
    return reinterpret_cast<uint64_t>(as);
}

// ---------------------------------------------------------------------------
// NR_DestroyAccelerationStructure
//   Destroys a previously created acceleration structure.
//   The object is destroyed after a kDeleteDelay-frame delay so the GPU
//   finishes all commands that may reference it before the memory is freed.
//   The AccelerationStructure's own deferred-delete queue handles internal
//   BLAS/TLAS resources safely.
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_DestroyAccelerationStructure(uint64_t handle)
{
    if (!handle) return;
    RetireObject(reinterpret_cast<AccelerationStructure*>(handle));
}

// ---------------------------------------------------------------------------
// NR_AS_Clear  -  remove all instances from an acceleration structure.
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_AS_Clear(uint64_t handle)
{
    if (!handle) { NR_WARN("NR_AS_Clear: null handle"); return; }
    reinterpret_cast<AccelerationStructure*>(handle)->Clear();
}

// ---------------------------------------------------------------------------
// NR_AS_AddInstance
//   Add one GameObject instance with all its sub-meshes.
//
//   vbPtr / ibPtr      : ID3D12Resource* native GPU buffer pointers
//   vertexCount/Stride : total vertex count and stride for the shared VB
//   indexStride        : 2 or 4
//   submeshDescs       : pointer to array of NR_SubmeshDesc (one per sub-mesh)
//   submeshCount       : length of submeshDescs
//   ommDescs           : optional pointer to array of NR_SubmeshOMMDesc (same length);
//                        pass nullptr if no OMM. Per-entry arrayData == nullptr skips OMM
//                        for that submesh.
//
//   Returns the 0-based instance index, or 0xFFFFFFFF on error.
// ---------------------------------------------------------------------------
extern "C" bool UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_AS_AddInstance(
    uint64_t handle,
    const NR_AddInstanceDesc* desc)
{
    if (!handle || !desc || !desc->vbPtr || !desc->ibPtr || !desc->submeshDescs || desc->submeshCount == 0)
    {
        NR_WARN("NR_AS_AddInstance: invalid arguments");
        return false;
    }
    auto* as = reinterpret_cast<AccelerationStructure*>(handle);
    return as->AddInstance(*desc);
}

// ---------------------------------------------------------------------------
// NR_AS_SetInstanceTransform
//   Update the world transform of an existing instance.
//   transform3x4 : pointer to 12 floats, row-major 3x4 object-to-world matrix.
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_AS_SetInstanceTransform(uint64_t handle, uint32_t instanceIndex, const float* transform3x4)
{
    if (!handle || !transform3x4) return;
    reinterpret_cast<AccelerationStructure*>(handle)
        ->SetInstanceTransform(instanceIndex, transform3x4);
}

// ---------------------------------------------------------------------------
// NR_AS_SetInstanceMask
//   Set the per-instance visibility mask (8 bits). Default = 0xFF.
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_AS_SetInstanceMask(uint64_t handle, uint32_t instanceIndex, uint8_t mask)
{
    if (!handle) return;
    reinterpret_cast<AccelerationStructure*>(handle)
        ->SetInstanceMask(instanceIndex, mask);
}

// ---------------------------------------------------------------------------
// NR_AS_SetInstanceID
//   Set the custom InstanceID returned by InstanceID() intrinsic in HLSL.
//   Use this to align InstanceID() with an index into a structured buffer.
//   instanceHandle is the value passed to NR_AS_AddInstance (e.g. MeshRenderer.GetInstanceID()).
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_AS_SetInstanceID(uint64_t handle, uint32_t instanceHandle, uint32_t id)
{
    if (!handle) return;
    reinterpret_cast<AccelerationStructure*>(handle)
        ->SetInstanceID(instanceHandle, id);
}

// ---------------------------------------------------------------------------
// NR_AS_SetInstanceOrderIndex
//   Set the TLAS emission order for an instance. BuildOrUpdate emits TLAS
//   instances sorted ascending by this value (stable sort; the 0xFFFFFFFF
//   default keeps legacy slot-order emission). Required by callers whose
//   shaders index per-instance buffers via InstanceIndex().
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_AS_SetInstanceOrderIndex(uint64_t handle, uint32_t instanceHandle, uint32_t order)
{
    if (!handle) return;
    reinterpret_cast<AccelerationStructure*>(handle)
        ->SetInstanceOrderIndex(instanceHandle, order);
}

// ---------------------------------------------------------------------------
// NR_AS_SetInstanceHitGroupContribution
//   Update InstanceContributionToHitGroupIndex for an existing instance (its
//   base offset in the caller's flat shader table, which shifts for surviving
//   instances whenever the scene's geometry layout changes).
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_AS_SetInstanceHitGroupContribution(uint64_t handle, uint32_t instanceHandle, uint32_t contribution)
{
    if (!handle) return;
    reinterpret_cast<AccelerationStructure*>(handle)
        ->SetInstanceHitGroupContribution(instanceHandle, contribution);
}

// ---------------------------------------------------------------------------
// NR_AS_RemoveInstance
//   Remove an instance previously added via NR_AS_AddInstance.
//   instanceHandle is the value returned by NR_AS_AddInstance.
//   Decrements the BLAS ref-count; GPU resources are freed after 3 frames.
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_AS_RemoveInstance(uint64_t handle, uint32_t instanceHandle)
{
    if (!handle) { NR_WARN("NR_AS_RemoveInstance: null handle"); return; }
    reinterpret_cast<AccelerationStructure*>(handle)
        ->RemoveInstance(instanceHandle);
}

// ---------------------------------------------------------------------------
// NR_AS_UpdateDynamicVertexBuffer
//   For SkinnedMeshRenderer instances: provide the current-frame GPU vertex
//   buffer (from SkinnedMeshRenderer.GetVertexBuffer) so the BLAS is rebuilt
//   with up-to-date skinned geometry on the next BuildOrUpdate call.
//   vbPtr       : ID3D12Resource* returned by GraphicsBuffer.GetNativeBufferPtr()
//   vertexCount : current vertex count (may match original registration)
//   vertexStride: vertex stride in bytes
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_AS_UpdateDynamicVertexBuffer(uint64_t handle, uint32_t instanceHandle,
                                 void* vbPtr, uint32_t vertexCount, uint32_t vertexStride)
{
    if (!handle || !vbPtr) return;
    reinterpret_cast<AccelerationStructure*>(handle)->UpdateDynamicVertexBuffer(instanceHandle, vbPtr);
}

// ===========================================================================
// RayTraceShader  -  multi-shader, per-instance ray tracing pipeline
// ===========================================================================

// ---------------------------------------------------------------------------
// RTS_RenderEventData
//   Passed from C# via IssuePluginEventAndData for per-descriptor-set dispatches.
//   Must match NativeRenderPlugin.RTS_RenderEventData exactly (Pack=4).
//
//   New layout (32 bytes):
//     descriptorSetHandle — pointer to RayTraceDescriptorSet
//     bindingSlotsPtr     — pointer to CS_BindingSlot[] array (marshalled by C#)
//     bindingCount        — number of entries in bindingSlotsPtr
//     width, height       — dispatch dimensions
//     _pad                — reserved / struct alignment
// ---------------------------------------------------------------------------
#pragma pack(push, 4)
struct RTS_RenderEventData
{
    uint64_t descriptorSetHandle;  // +0  (8B): RayTraceDescriptorSet*
    uint64_t bindingSlotsPtr;      // +8  (8B): CS_BindingSlot*
    uint32_t bindingCount;         // +16 (4B)
    uint32_t width;                // +20 (4B)
    uint32_t height;               // +24 (4B)
    uint32_t _pad;                 // +28 (4B)
};  // Total: 32 bytes
#pragma pack(pop)

// AS build event data - passed to AsBuildRenderCallback
#pragma pack(push, 4)
struct AS_BuildEventData
{
    uint64_t asHandle;     // pointer to AccelerationStructure
};  // 8 bytes
#pragma pack(pop)

// Shader hit-group table rebuild event data - passed to ShtRebuildRenderCallback
// C# side pre-computes the flat variant-index array and passes a pointer + count.
// Layout: pointer (8B) then two u32s (8B) = 16 bytes, Pack=4 avoids struct-end pad.
#pragma pack(push, 4)
struct ShtRebuildEventData
{
    uint64_t        shaderHandle;   // pointer to RayTraceShader
    const uint32_t* variantIndices; // per-geometry hitgroup variant index (C# NativeArray ptr)
    uint32_t        count;          // total number of entries
    uint32_t        _pad;
};  // 24 bytes
#pragma pack(pop)

// NativeBuffer volatile-upload blob: [ NbUploadHeader ][ payload bytes ].
// Self-contained and plugin-owned (malloc'd via NR_NSB_AllocFlushBuffer on the C#
// main thread, freed by the render-thread callback after the data is copied into
// a fresh upload-pool slice). Because nothing persistent is shared across the
// main/render threads, the C# side needs no triple-buffered source array.
#pragma pack(push, 4)
struct NbUploadHeader
{
    uint64_t bufferHandle; // NativeBuffer*
    uint32_t bytes;
    uint32_t _pad;
};
#pragma pack(pop)

// ---------------------------------------------------------------------------
// FlushGpuAndWait - signal a value on the graphics queue and block until done
// ---------------------------------------------------------------------------
static void FlushGpuAndWait()
{
    NR_LOG("FlushGpuAndWait - START");
    if (!s_D3D12) {
        NR_WARN("FlushGpuAndWait - s_D3D12 is null");
        return;
    }
    ID3D12CommandQueue* queue = s_D3D12->GetCommandQueue();
    if (!queue) {
        NR_WARN("FlushGpuAndWait - queue is null");
        return;
    }

    ID3D12Device* device = s_D3D12->GetDevice();
    if (!device) {
        NR_WARN("FlushGpuAndWait - device is null");
        return;
    }

    ComPtr<ID3D12Fence> fence;
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
    {
        NR_ERROR("FlushGpuAndWait - CreateFence failed");
        return;
    }

    if (FAILED(queue->Signal(fence.Get(), 1)))
    {
        NR_ERROR("FlushGpuAndWait - Signal failed");
        return;
    }

    NR_LOG("FlushGpuAndWait - waiting for GPU completion...");
    if (fence->GetCompletedValue() < 1)
    {
        HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (ev)
        {
            fence->SetEventOnCompletion(1, ev);
            // Use a 10-second timeout instead of INFINITE to avoid hanging forever
            DWORD waitResult = WaitForSingleObject(ev, 10000);
            CloseHandle(ev);
            if (waitResult == WAIT_OBJECT_0)
            {
                NR_LOG("FlushGpuAndWait - GPU wait completed");
            }
            else if (waitResult == WAIT_TIMEOUT)
            {
                NR_ERROR("FlushGpuAndWait - GPU wait TIMEOUT after 10 seconds!");
            }
            else
            {
                NR_ERROR("FlushGpuAndWait - GPU wait FAILED");
            }
        }
        else
        {
            NR_ERROR("FlushGpuAndWait - CreateEvent failed");
        }
    }
    else
    {
        NR_LOG("FlushGpuAndWait - GPU already completed");
    }
    NR_LOG("FlushGpuAndWait - END");
}

// ---------------------------------------------------------------------------
// NR_DestroyRayTraceShader
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_DestroyRayTraceShader(uint64_t handle)
{
    if (!handle) return;
    RetireObject(reinterpret_cast<RayTraceShader*>(handle));
}

// ---------------------------------------------------------------------------
// NR_CreateRayTraceShaderFromBytes
//   Builds a DXR pipeline from pre-compiled DXIL bytes.  Returns an opaque
//   uint64 handle on success, 0 on failure.
//   flags: bit 0 = D3D12_RAYTRACING_PIPELINE_FLAG_ALLOW_OPACITY_MICROMAPS (only pass when target profile >= lib_6_9).
//   maxPayloadSizeInBytes: MaxPayloadSizeInBytes for D3D12_RAYTRACING_SHADER_CONFIG.
//   rayGenName: Name of the RayGeneration entry point to use for DispatchRays. Null or empty = first discovered.
//   Caller owns the lifetime; call NR_DestroyRayTraceShader when done.
// ---------------------------------------------------------------------------
extern "C" uint64_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CreateRayTraceShaderFromBytes(const uint8_t* dxilBytes, uint32_t size, const char* name, uint32_t flags = 0, uint32_t maxPayloadSizeInBytes = 4, const char* rayGenName = nullptr)
{
    if (!s_RendererReady)
    {
        NR_WARN("NR_CreateRayTraceShaderFromBytes: renderer not ready");
        return 0;
    }
    ID3D12Device5* dev5 = nullptr;
    ID3D12Device*  base = s_D3D12->GetDevice();
    if (!base || FAILED(base->QueryInterface(IID_PPV_ARGS(&dev5))))
    {
        NR_ERROR("NR_CreateRayTraceShaderFromBytes: failed to obtain ID3D12Device5");
        return 0;
    }
    auto* shader = new RayTraceShader();
    if (!shader->Initialize(dev5, s_Log, s_D3D12v8) ||
        !shader->LoadShaderFromBytes(dxilBytes, size, name, flags, maxPayloadSizeInBytes, rayGenName))
    {
        delete shader;
        dev5->Release();
        return 0;
    }
    dev5->Release();
    return reinterpret_cast<uint64_t>(shader);
}

// ---------------------------------------------------------------------------
// NR_CreateRayTracePipelineFromBlobs
//   Builds a DXR pipeline from N pre-compiled DXIL blobs merged into one RTPSO.
//
//   blobDataPtrs   : array of pointers to DXIL bytes (length = blobCount)
//   blobSizes      : array of sizes in bytes (length = blobCount)
//   blobCount      : number of blobs
//     blob[0]      : raygen + miss shaders
//     blob[1..N-1] : additional hit-group blobs (per-material permutations)
//
//   All other parameters match NR_CreateRayTraceShaderFromBytes.
//   Returns an opaque uint64 handle on success, 0 on failure.
//   Caller owns the lifetime; call NR_DestroyRayTraceShader when done.
// ---------------------------------------------------------------------------
extern "C" uint64_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CreateRayTracePipelineFromBlobs(
    const uint8_t** blobDataPtrs,
    const uint32_t* blobSizes,
    uint32_t        blobCount,
    const char*     name,
    uint32_t        flags,
    uint32_t        maxPayloadSizeInBytes,
    const char*     rayGenName)
{
    if (!s_RendererReady)
    {
        NR_WARN("NR_CreateRayTracePipelineFromBlobs: renderer not ready");
        return 0;
    }
    if (!blobDataPtrs || !blobSizes || blobCount == 0)
    {
        NR_WARN("NR_CreateRayTracePipelineFromBlobs: invalid arguments");
        return 0;
    }

    ID3D12Device5* dev5 = nullptr;
    ID3D12Device*  base = s_D3D12->GetDevice();
    if (!base || FAILED(base->QueryInterface(IID_PPV_ARGS(&dev5))))
    {
        NR_ERROR("NR_CreateRayTracePipelineFromBlobs: failed to obtain ID3D12Device5");
        return 0;
    }

    // Build BlobDesc array
    std::vector<RayTraceShader::BlobDesc> descs(blobCount);
    for (uint32_t i = 0; i < blobCount; ++i)
        descs[i] = { blobDataPtrs[i], blobSizes[i] };

    auto* shader = new RayTraceShader();
    if (!shader->Initialize(dev5, s_Log, s_D3D12v8) ||
        !shader->LoadShaderFromMultipleBlobs(descs.data(), blobCount, name, flags,
                                             maxPayloadSizeInBytes, rayGenName))
    {
        delete shader;
        dev5->Release();
        return 0;
    }
    dev5->Release();
    return reinterpret_cast<uint64_t>(shader);
}
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// NR_RTS_GetRenderEventFunc / NR_RTS_GetRenderEventDataSize
//   Use with CommandBuffer.IssuePluginEventAndData for per-shader ray trace.
// ---------------------------------------------------------------------------
static void UNITY_INTERFACE_API RtsRenderCallback(int /*eventId*/, void* data)
{
    if (!s_RendererReady || !s_D3D12 || !data) return;

    // Diagnostic: in a player build GetSwapChain() becomes valid on the render
    // thread; install the Present hook the first time it does. No-op afterwards.
    TryHookUnitySwapChainPresent();

    auto* ed = static_cast<RTS_RenderEventData*>(data);
    if (!ed->descriptorSetHandle) return;

    UnityGraphicsD3D12RecordingState recordingState = {};
    if (!s_D3D12->CommandRecordingState(&recordingState) || !recordingState.commandList) return;

    auto* ds      = reinterpret_cast<::RayTraceDescriptorSet*>(ed->descriptorSetHandle);
    auto* slots   = reinterpret_cast<const BindingSlot*>(static_cast<uintptr_t>(ed->bindingSlotsPtr));
    auto* cmdList = static_cast<ID3D12GraphicsCommandList4*>(recordingState.commandList);

    D3D12HeapHook::BeginPluginDispatch();
    ds->Dispatch(cmdList, ed->width, ed->height, slots, ed->bindingCount);
    D3D12HeapHook::EndPluginDispatch();

    // Restore Unity's heaps so subsequent Unity SetComputeRootDescriptorTable
    // does not trip D3D12 validation (category 9, id 708).
    D3D12HeapHook::RestoreUnityHeaps(cmdList);
}

// ---------------------------------------------------------------------------
// AS build callback - called via CommandBuffer.IssuePluginEventAndData
// ---------------------------------------------------------------------------
static void UNITY_INTERFACE_API AsBuildRenderCallback(int /*eventId*/, void* data)
{
    if (!s_RendererReady || !s_D3D12 || !data) return;
    auto* ed = static_cast<AS_BuildEventData*>(data);
    if (!ed->asHandle) return;
    UnityGraphicsD3D12RecordingState recordingState = {};
    if (!s_D3D12->CommandRecordingState(&recordingState) || !recordingState.commandList) return;
    auto* as      = reinterpret_cast<AccelerationStructure*>(ed->asHandle);
    auto* cmdList = static_cast<ID3D12GraphicsCommandList4*>(recordingState.commandList);
    as->BuildOrUpdate(cmdList);
}

// ---------------------------------------------------------------------------
// Shader hit-group table rebuild callback.
// C# passes a pre-computed flat variant-index array; we write one shader-table
// entry per element.  Fully decoupled from AccelerationStructure.
// ---------------------------------------------------------------------------
static void UNITY_INTERFACE_API ShtRebuildRenderCallback(int /*eventId*/, void* data)
{
    if (!s_RendererReady || !data) return;
    auto* ed = static_cast<ShtRebuildEventData*>(data);
    if (!ed->shaderHandle || !ed->variantIndices || ed->count == 0)
    {
        NR_WARN("ShtRebuildRenderCallback: invalid event data (shaderHandle/variantIndices/count)");
        return;
    }
    auto* shader = reinterpret_cast<RayTraceShader*>(ed->shaderHandle);
    if (!shader->RebuildHitGroupTable(ed->variantIndices, ed->count))
        NR_WARN("ShtRebuildRenderCallback: RebuildHitGroupTable failed");
}

extern "C" UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RTS_GetRenderEventFunc()
{
    return RtsRenderCallback;
}

extern "C" uint32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RTS_GetRenderEventDataSize()
{
    return static_cast<uint32_t>(sizeof(RTS_RenderEventData));
}

// ---------------------------------------------------------------------------
// NR_RTS_CreateDescriptorSet / NR_RTS_DestroyDescriptorSet
//   Create a RayTraceDescriptorSet bound to a previously-created RayTraceShader.
//   Returns 0 on failure.  The descriptor set owns its own heap slice.
// ---------------------------------------------------------------------------
extern "C" uint64_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RTS_CreateDescriptorSet(uint64_t shaderHandle)
{
    if (!shaderHandle || !s_RendererReady) return 0;
    auto* shader = reinterpret_cast<RayTraceShader*>(shaderHandle);
    ID3D12Device* dev = s_D3D12->GetDevice();
    if (!dev) return 0;
    auto* ds = new ::RayTraceDescriptorSet(shader, dev, s_Log, &s_DescHeap, s_D3D12v8);
    return reinterpret_cast<uint64_t>(ds);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RTS_DestroyDescriptorSet(uint64_t handle)
{
    if (!handle) return;
    RetireObject(reinterpret_cast<::RayTraceDescriptorSet*>(handle));
}

// ---------------------------------------------------------------------------
// NR_RTS_GetBindingCount / NR_RTS_GetSlotIndex / NR_RTS_GetBindingName
//   Slot-layout queries on the RayTraceShader.  C# uses these to build its
//   NativeRayTraceDescriptorSet slot map at construction time.
// ---------------------------------------------------------------------------
extern "C" uint32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RTS_GetBindingCount(uint64_t shaderHandle)
{
    if (!shaderHandle) return 0;
    return reinterpret_cast<RayTraceShader*>(shaderHandle)->GetBindingCount();
}

extern "C" uint32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RTS_GetSlotIndex(uint64_t shaderHandle, const char* name)
{
    if (!shaderHandle || !name) return UINT32_MAX;
    return reinterpret_cast<RayTraceShader*>(shaderHandle)->GetSlotIndex(name);
}

extern "C" UNITY_INTERFACE_EXPORT const char* UNITY_INTERFACE_API
NR_RTS_GetBindingName(uint64_t shaderHandle, uint32_t index)
{
    if (!shaderHandle) return nullptr;
    return reinterpret_cast<RayTraceShader*>(shaderHandle)->GetBindingName(index);
}

// ---------------------------------------------------------------------------
// NR_RTS_SetRootConstantsHint / NR_RTS_SetRootSRVHint
//   Must be called BEFORE NR_CreateRayTraceShaderFromBytes so the reflection
//   pass applies the correct binding types.
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RTS_SetRootConstantsHint(uint64_t shaderHandle, const char* name, uint32_t num32BitValues)
{
    if (!shaderHandle || !name) return;
    reinterpret_cast<RayTraceShader*>(shaderHandle)->SetRootConstantsHint(name, num32BitValues);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RTS_SetRootSRVHint(uint64_t shaderHandle, const char* name)
{
    if (!shaderHandle || !name) return;
    reinterpret_cast<RayTraceShader*>(shaderHandle)->SetRootSRVHint(name);
}

static void UNITY_INTERFACE_API FrameTickCallback(int /*eventId*/)
{
    DrainDeferredDeletes();

    // Reclaim transient descriptor-ring slots whose frame fence has completed,
    // and record this frame's high-water mark against the value the fence will
    // reach once this frame's GPU work finishes. kDeleteDelay matches the safety
    // margin used by the deferred-delete queue so the lifetime semantics align.
    ID3D12Fence* tickFence = s_D3D12 ? s_D3D12->GetFrameFence() : nullptr;
    if (tickFence && g_transientRing.IsInitialized())
    {
        const uint64_t completedValue = tickFence->GetCompletedValue();
        // Clamp against completedValue for the same reason as EnqueueCleanup: a
        // stale-low GetNextFrameFenceValue() (observed 1 vs completed 8) would tag
        // this frame's slots/chunks for reuse below current GPU progress, recycling
        // them while still in flight. See EnqueueCleanup for details.
        const uint64_t nextValue  = s_D3D12->GetNextFrameFenceValue();
        const uint64_t base       = nextValue > completedValue ? nextValue : completedValue;
        const uint64_t frameFence = base + kDeleteDelay;
        g_transientRing.EndFrame(frameFence, completedValue);

        // Recycle shared upload chunks under the same frame-fence discipline.
        if (g_uploadPool.IsInitialized())
            g_uploadPool.EndFrame(frameFence, completedValue);

        // Recycle acceleration-structure build scratch under the same discipline.
        if (g_scratchPool.IsInitialized())
            g_scratchPool.EndFrame(frameFence, completedValue);
    }
}

// ---------------------------------------------------------------------------
// NR_GetGpuFlushEventFunc / NR_WaitForGpuFlush
//   Allows the main thread to synchronise with the render thread before
//   freeing NativeArray ring buffers that in-flight render-thread callbacks
//   may still be reading (e.g. after a hot-reload / Recompile).
//
//   Usage from C#:
//     1. cmd.IssuePluginEvent(NR_GetGpuFlushEventFunc(), 0);
//        Graphics.ExecuteCommandBuffer(cmd);   // queues callback on render thread
//     2. NR_WaitForGpuFlush();                 // main thread blocks here until done
// ---------------------------------------------------------------------------
static void UNITY_INTERFACE_API GpuFlushAndSignalCallback(int /*eventId*/)
{
    // Called on the render thread — all previously-enqueued render callbacks
    // have already executed before this one runs.
    FlushGpuAndWait();
    if (s_gpuFlushDoneEvent)
        SetEvent(s_gpuFlushDoneEvent);
}

extern "C" UnityRenderingEvent UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_GetGpuFlushEventFunc()
{
    return GpuFlushAndSignalCallback;
}

// Blocks the calling thread (main thread) until GpuFlushAndSignalCallback has
// run and signalled s_gpuFlushDoneEvent.  Resets the event afterwards so it
// can be reused.  Timeout: 15 seconds.
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_WaitForGpuFlush()
{
    if (!s_gpuFlushDoneEvent) return;
    DWORD result = WaitForSingleObject(s_gpuFlushDoneEvent, 15000);
    if (result == WAIT_TIMEOUT)
        NR_ERROR("NR_WaitForGpuFlush: timed out after 15 seconds!");
    else if (result != WAIT_OBJECT_0)
        NR_ERROR("NR_WaitForGpuFlush: WaitForSingleObject failed");
    ResetEvent(s_gpuFlushDoneEvent);
}

extern "C" UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_AS_GetBuildRenderEventFunc()
{
    return AsBuildRenderCallback;
}

extern "C" UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RTS_GetRebuildHitGroupTableEventFunc()
{
    return ShtRebuildRenderCallback;
}

extern "C" uint32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RTS_GetRebuildHitGroupTableEventDataSize()
{
    return static_cast<uint32_t>(sizeof(ShtRebuildEventData));
}

extern "C" UnityRenderingEvent UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_GetFrameTickEventFunc()
{
    return FrameTickCallback;
}

extern "C" uint32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_AS_GetBuildEventDataSize()
{
    return static_cast<uint32_t>(sizeof(AS_BuildEventData));
}

extern "C" intptr_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_AS_GetTLASNativePtr(uint64_t asHandle)
{
    if (!asHandle) return 0;
    return reinterpret_cast<intptr_t>(
        reinterpret_cast<AccelerationStructure*>(asHandle)->GetTLAS());
}

// ===========================================================================
// BindlessTexture  -  independent GPU-visible texture array
// ===========================================================================

// ---------------------------------------------------------------------------
// NR_CreateBindlessTexture
//   Allocates a BindlessTexture with |capacity| slots in the shared GPU heap.
//   Returns an opaque uint64 handle. Caller owns lifetime; call
//   NR_DestroyBindlessTexture when done.
// ---------------------------------------------------------------------------
extern "C" uint64_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CreateBindlessTexture(uint32_t capacity)
{
    if (!s_RendererReady || !s_DescHeap.IsInitialized())
    {
        NR_WARN("NR_CreateBindlessTexture: renderer not ready");
        return 0;
    }
    ID3D12Device* device = s_D3D12->GetDevice();
    if (!device) return 0;

    auto* bt = new BindlessTexture();
    if (!bt->Initialize(device, &s_DescHeap, capacity, s_Log))
    {
        delete bt;
        return 0;
    }
    return reinterpret_cast<uint64_t>(bt);
}

// ---------------------------------------------------------------------------
// NR_DestroyBindlessTexture
//   Enqueues destruction after a kDeleteDelay-frame GPU fence delay.
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_DestroyBindlessTexture(uint64_t handle)
{
    if (handle) RetireObject(reinterpret_cast<BindlessTexture*>(handle));
}

// ---------------------------------------------------------------------------
// NR_BT_SetTexture
//   Sets the texture at |index| within the BindlessTexture array.
//   Pass IntPtr.Zero (d3d12ResourcePtr == nullptr) to write a null SRV.
//   Returns 1 on success, 0 if index is out of range.
// ---------------------------------------------------------------------------
extern "C" int32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_BT_SetTexture(uint64_t handle, uint32_t index, void* d3d12ResourcePtr)
{
    if (!handle) return 0;
    auto* bt = reinterpret_cast<BindlessTexture*>(handle);
    if (index >= bt->Capacity()) return 0;
    bt->SetTexture(index, static_cast<ID3D12Resource*>(d3d12ResourcePtr));
    return 1;
}

// ---------------------------------------------------------------------------
// NR_BT_Resize
//   Resizes the BindlessTexture to |newCapacity| slots.
//   If the GPU handle changes (always true when capacity differs), any shader
//   that references this BindlessTexture must call NR_RTS_SetBindlessTexture
//   again before the next dispatch.
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_BT_Resize(uint64_t handle, uint32_t newCapacity)
{
    if (!handle) return;
    reinterpret_cast<BindlessTexture*>(handle)->Resize(newCapacity);
}

// ---------------------------------------------------------------------------
// NR_BT_GetCapacity
//   Returns the current capacity of the BindlessTexture.
// ---------------------------------------------------------------------------
extern "C" uint32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_BT_GetCapacity(uint64_t handle)
{
    if (!handle) return 0;
    return reinterpret_cast<BindlessTexture*>(handle)->Capacity();
}

// ===========================================================================
// BindlessBuffer  -  independent GPU-visible buffer (ByteAddressBuffer) array
// ===========================================================================

// ---------------------------------------------------------------------------
// NR_CreateBindlessBuffer
//   Allocates a BindlessBuffer with |capacity| slots in the shared GPU heap.
//   Returns an opaque uint64 handle. Caller owns lifetime; call
//   NR_DestroyBindlessBuffer when done.
// ---------------------------------------------------------------------------
extern "C" uint64_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CreateBindlessBuffer(uint32_t capacity)
{
    if (!s_RendererReady || !s_DescHeap.IsInitialized())
    {
        NR_WARN("NR_CreateBindlessBuffer: renderer not ready");
        return 0;
    }
    ID3D12Device* device = s_D3D12->GetDevice();
    if (!device) return 0;

    auto* bb = new BindlessBuffer();
    if (!bb->Initialize(device, &s_DescHeap, capacity, s_Log))
    {
        delete bb;
        return 0;
    }
    return reinterpret_cast<uint64_t>(bb);
}

// ---------------------------------------------------------------------------
// NR_DestroyBindlessBuffer
//   Enqueues destruction after a kDeleteDelay-frame GPU fence delay.
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_DestroyBindlessBuffer(uint64_t handle)
{
    if (handle) RetireObject(reinterpret_cast<BindlessBuffer*>(handle));
}

// ---------------------------------------------------------------------------
// NR_BB_SetBuffer
//   Sets the buffer at |index| within the BindlessBuffer array.
//   Pass IntPtr.Zero (d3d12ResourcePtr == nullptr) to write a null SRV.
//   Returns 1 on success, 0 if index is out of range.
// ---------------------------------------------------------------------------
extern "C" int32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_BB_SetBuffer(uint64_t handle, uint32_t index, void* d3d12ResourcePtr)
{
    if (!handle) return 0;
    auto* bb = reinterpret_cast<BindlessBuffer*>(handle);
    if (index >= bb->Capacity()) return 0;
    bb->SetBuffer(index, static_cast<ID3D12Resource*>(d3d12ResourcePtr));
    return 1;
}

// ---------------------------------------------------------------------------
// NR_BB_Resize
//   Resizes the BindlessBuffer to |newCapacity| slots.
//   If the GPU handle changes, any shader that references this BindlessBuffer
//   must call NR_RTS_SetBindlessBuffer again before the next dispatch.
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_BB_Resize(uint64_t handle, uint32_t newCapacity)
{
    if (!handle) return;
    reinterpret_cast<BindlessBuffer*>(handle)->Resize(newCapacity);
}

// ---------------------------------------------------------------------------
// NR_BB_GetCapacity
//   Returns the current capacity of the BindlessBuffer.
// ---------------------------------------------------------------------------
extern "C" uint32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_BB_GetCapacity(uint64_t handle)
{
    if (!handle) return 0;
    return reinterpret_cast<BindlessBuffer*>(handle)->Capacity();
}

// ===========================================================================
// BindlessUAVTexture  -  GPU-visible RWTexture2D[] UAV descriptor array
// ===========================================================================

// ---------------------------------------------------------------------------
// NR_CreateBindlessUAVTexture
//   Allocates a BindlessUAVTexture with |capacity| slots in the shared GPU heap.
//   Returns an opaque uint64 handle. Caller owns lifetime; call
//   NR_DestroyBindlessUAVTexture when done.
// ---------------------------------------------------------------------------
extern "C" uint64_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CreateBindlessUAVTexture(uint32_t capacity)
{
    if (!s_RendererReady || !s_DescHeap.IsInitialized())
    {
        NR_WARN("NR_CreateBindlessUAVTexture: renderer not ready");
        return 0;
    }
    ID3D12Device* device = s_D3D12->GetDevice();
    if (!device) return 0;

    auto* uav = new BindlessUAVTexture();
    if (!uav->Initialize(device, &s_DescHeap, capacity, s_Log))
    {
        delete uav;
        return 0;
    }
    return reinterpret_cast<uint64_t>(uav);
}

// ---------------------------------------------------------------------------
// NR_DestroyBindlessUAVTexture
//   Enqueues destruction after a kDeleteDelay-frame GPU fence delay.
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_DestroyBindlessUAVTexture(uint64_t handle)
{
    if (handle) RetireObject(reinterpret_cast<BindlessUAVTexture*>(handle));
}

// ---------------------------------------------------------------------------
// NR_BUAV_SetTexture
//   Sets the UAV descriptor at |index| within the BindlessUAVTexture array.
//   |d3d12ResourcePtr|  – native ID3D12Resource* (pass nullptr to write null UAV)
//   |mipSlice|          – which mip level to expose as UAV (0-based)
//   |dxgiFormat|        – DXGI_FORMAT for the view; 0 = derive from resource
//   Returns 1 on success, 0 if index is out of range.
// ---------------------------------------------------------------------------
extern "C" int32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_BUAV_SetTexture(uint64_t handle, uint32_t index, void* d3d12ResourcePtr,
                   uint32_t mipSlice, uint32_t dxgiFormat)
{
    if (!handle) return 0;
    auto* uav = reinterpret_cast<BindlessUAVTexture*>(handle);
    if (index >= uav->Capacity()) return 0;
    uav->SetTexture(index,
                    static_cast<ID3D12Resource*>(d3d12ResourcePtr),
                    mipSlice,
                    static_cast<DXGI_FORMAT>(dxgiFormat));
    return 1;
}

// ---------------------------------------------------------------------------
// NR_BUAV_Resize
//   Resizes the BindlessUAVTexture to |newCapacity| slots.
//   Any shader that references this object must call SetBindlessRWTexture again.
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_BUAV_Resize(uint64_t handle, uint32_t newCapacity)
{
    if (!handle) return;
    reinterpret_cast<BindlessUAVTexture*>(handle)->Resize(newCapacity);
}

// ---------------------------------------------------------------------------
// NR_BUAV_GetCapacity
//   Returns the current capacity of the BindlessUAVTexture.
// ---------------------------------------------------------------------------
extern "C" uint32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_BUAV_GetCapacity(uint64_t handle)
{
    if (!handle) return 0;
    return reinterpret_cast<BindlessUAVTexture*>(handle)->Capacity();
}

// ===========================================================================
// NativeBuffer  -  triple-buffered upload-heap constant buffer
// ===========================================================================

// ---------------------------------------------------------------------------
// NR_CreateNativeBuffer
//   Unified buffer allocation (nvrhi Buffer analog). The desc flags select the
//   heap and the active upload/readback path — see NativeBufferDesc:
//     * isVolatile + maxVersions>1 -> per-frame UPLOAD ring (CPU writes via Upload).
//     * DEFAULT + structStride     -> GPU-resident, staged CPU writes / readback.
//     * canHaveUAVs                 -> ALLOW_UNORDERED_ACCESS on the DEFAULT resource.
//   Returns opaque handle (NativeBuffer*) or 0 on failure.
// ---------------------------------------------------------------------------
extern "C" uint64_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CreateNativeBuffer(uint64_t byteSize, uint32_t structStride, uint32_t maxVersions,
                      uint32_t canHaveUAVs, uint32_t isConstantBuffer, uint32_t isVolatile,
                      const char* debugName)
{
    if (!s_D3D12)
    {
        NR_WARN("NR_CreateNativeBuffer: renderer not ready");
        return 0;
    }
    ID3D12Device* device = s_D3D12->GetDevice();
    if (!device)
    {
        NR_WARN("NR_CreateNativeBuffer: no D3D12 device");
        return 0;
    }

    NativeBufferDesc desc{};
    desc.byteSize         = byteSize;
    desc.structStride     = structStride;
    desc.maxVersions      = maxVersions ? maxVersions : 1u;
    desc.canHaveUAVs      = canHaveUAVs != 0;
    desc.isConstantBuffer = isConstantBuffer != 0;
    desc.isVolatile       = isVolatile != 0;
    if (debugName && debugName[0]) desc.debugName = debugName;

    auto* nb = new NativeBuffer();
    if (!nb->Initialize(device, desc, s_Log))
    {
        delete nb;
        NR_WARN("NR_CreateNativeBuffer: Initialize failed");
        return 0;
    }
    return reinterpret_cast<uint64_t>(nb);
}

// ---------------------------------------------------------------------------
// NR_DestroyNativeBuffer
//   Enqueues destruction after a GPU fence delay (same as other objects).
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_DestroyNativeBuffer(uint64_t handle)
{
    if (handle) RetireObject(reinterpret_cast<NativeBuffer*>(handle));
}
 
static void UNITY_INTERFACE_API NativeBufferUploadCallback(int /*eventId*/, void* data)
{
    if (!data) return;

    const auto* hdr = static_cast<const NbUploadHeader*>(data);
    if (hdr->bufferHandle && hdr->bytes)
    {
        auto* nb = reinterpret_cast<NativeBuffer*>(hdr->bufferHandle);
        nb->Upload(static_cast<const uint8_t*>(data) + sizeof(NbUploadHeader), hdr->bytes);
    }

    std::free(data);
}

extern "C" UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
GetNativeBufferUploadCallbackPtr()
{
    return NativeBufferUploadCallback;
}

// ===========================================================================
// NativeBuffer GPU clear (DEFAULT-heap buffers with ALLOW_UNORDERED_ACCESS)
// ===========================================================================

// ---------------------------------------------------------------------------
// NR_ClearNativeGpuBufferCallback
//   Render-thread callback (IssuePluginEventAndData).
//   data = NativeBuffer*  (the Handle value from C#).
//   Zeroes the entire buffer via ClearUnorderedAccessViewUint using:
//     - a 1-slot CPU-only descriptor heap (s_ClearCpuHeap) for the CPU handle
//     - a temporarily allocated slot from s_DescHeap (shader-visible) for the
//       GPU handle required by the API.
//   State management is delegated to Unity's RequestResourceState /
//   NotifyResourceState so barriers compose correctly with surrounding passes.
// ---------------------------------------------------------------------------
static void UNITY_INTERFACE_API NR_ClearNativeGpuBufferCallback(int /*eventId*/, void* data)
{
    if (!s_RendererReady || !s_D3D12 || !data || !s_ClearCpuHeap) return;

    auto* buf = static_cast<NativeBuffer*>(data);
    ID3D12Resource* res = buf->GetResource();
    if (!res) return;

    UnityGraphicsD3D12RecordingState rs = {};
    if (!s_D3D12->CommandRecordingState(&rs) || !rs.commandList) return;
    auto* cmdList = static_cast<ID3D12GraphicsCommandList*>(rs.commandList);

    ID3D12Device* device = s_D3D12->GetDevice();
    if (!device) return;

    // Allocate one temporary slot from the shared shader-visible heap.
    uint32_t tempSlot = s_DescHeap.Allocate(1);

    // Build a UAV desc covering the whole buffer as R32_UINT.
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.ViewDimension      = D3D12_UAV_DIMENSION_BUFFER;
    uavDesc.Format             = DXGI_FORMAT_R32_UINT;
    uavDesc.Buffer.NumElements = buf->SizeInBytes() / 4;

    // CPU-only handle (required by ClearUnorderedAccessViewUint alongside the GPU handle).
    D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = s_ClearCpuHeap->GetCPUDescriptorHandleForHeapStart();
    device->CreateUnorderedAccessView(res, nullptr, &uavDesc, cpuHandle);

    // GPU-visible handle in the currently-bound shader-visible heap.
    D3D12_CPU_DESCRIPTOR_HANDLE gpuCpuHandle = s_DescHeap.GetCPUHandle(tempSlot);
    device->CreateUnorderedAccessView(res, nullptr, &uavDesc, gpuCpuHandle);
    D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = s_DescHeap.GetGPUHandle(tempSlot);

    // Transition to UAV state via Unity's barrier system.
    ResourceStateTracker tracker(s_D3D12v8);
    tracker.Require(res, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // BeginPluginDispatch marks that we are in a plugin-owned dispatch so the
    // vtable hook does not overwrite the cache with our heap binding.
    D3D12HeapHook::BeginPluginDispatch();

    // Manually bind s_DescHeap so the GPU handle is valid on the currently-set heap.
    ID3D12DescriptorHeap* heaps[] = { s_DescHeap.GetHeap() };
    cmdList->SetDescriptorHeaps(1, heaps);

    // Clear to zero.
    const UINT zeros[4] = { 0, 0, 0, 0 };
    cmdList->ClearUnorderedAccessViewUint(gpuHandle, cpuHandle, res, zeros, 0, nullptr);

    D3D12HeapHook::EndPluginDispatch();

    // Notify Unity that the resource is in UAV state with a UAV access (triggers UAV barrier
    // before next use, even if the state doesn't change — write-after-write hazard).
    tracker.Notify(res, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, /*uavAccess=*/true);

    // Return the temporary slot to the pool.
    s_DescHeap.Free(tempSlot, 1);

    // Restore Unity's descriptor heaps so subsequent Unity commands are not broken.
    D3D12HeapHook::RestoreUnityHeaps(cmdList);
}

extern "C" UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_GetClearNativeGpuBufferCallbackPtr()
{
    return NR_ClearNativeGpuBufferCallback;
}

// ===========================================================================
// Texture UAV clear (ClearUnorderedAccessViewFloat)
//   Replicates nvrhi CommandList::clearTextureFloat for UAV textures (d3d12-
//   texture.cpp): one ClearUnorderedAccessViewFloat per mip, no pipeline, no
//   draw — the path the original RTXPT uses for e.g. DebugVizOutput.
// ===========================================================================

// Must match the blob layout written by C# NativeTextureClear.ClearUavFloat.
#pragma pack(push, 4)
struct ClearTextureUavFloatEventData
{
    uint64_t textureResource; // ID3D12Resource* (Unity GetNativeTexturePtr)
    uint32_t format;          // DXGI_FORMAT for the UAV; UNKNOWN derives from the resource desc
    float    color[4];
};
#pragma pack(pop)

// ---------------------------------------------------------------------------
// NR_ClearTextureUavFloatCallback
//   Render-thread callback (IssuePluginEventAndData).
//   data = ClearTextureUavFloatEventData blob from NR_NSB_AllocFlushBuffer;
//   freed here. Descriptor handling mirrors NR_ClearNativeGpuBufferCallback,
//   except the shader-visible slots come from the fence-reclaimed transient
//   ring: the GPU reads them at execution, so they must not be recycled within
//   the frame. The CPU-only slot is consumed at record time and is safe to
//   rewrite per mip.
// ---------------------------------------------------------------------------
static void UNITY_INTERFACE_API NR_ClearTextureUavFloatCallback(int /*eventId*/, void* data)
{
    if (!data) return;
    const ClearTextureUavFloatEventData ed = *static_cast<ClearTextureUavFloatEventData*>(data);
    std::free(data);

    auto* res = reinterpret_cast<ID3D12Resource*>(ed.textureResource);
    if (!s_RendererReady || !s_D3D12 || !res || !s_ClearCpuHeap) return;

    UnityGraphicsD3D12RecordingState rs = {};
    if (!s_D3D12->CommandRecordingState(&rs) || !rs.commandList) return;
    auto* cmdList = static_cast<ID3D12GraphicsCommandList*>(rs.commandList);

    ID3D12Device* device = s_D3D12->GetDevice();
    if (!device) return;

    const D3D12_RESOURCE_DESC rd = res->GetDesc();
    if (rd.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D)
    {
        NR_WARN("NR_ClearTextureUavFloatCallback: resource is not a Texture2D");
        return;
    }

    const uint32_t    mipLevels = rd.MipLevels ? rd.MipLevels : 1u;
    const DXGI_FORMAT fmt       = (ed.format != DXGI_FORMAT_UNKNOWN)
                                      ? static_cast<DXGI_FORMAT>(ed.format) : rd.Format;

    uint32_t   slotBase = g_transientRing.IsInitialized() ? g_transientRing.Allocate(mipLevels)
                                                          : UINT32_MAX;
    const bool fromRing = slotBase != UINT32_MAX;
    if (!fromRing) slotBase = s_DescHeap.Allocate(mipLevels);

    ResourceStateTracker tracker(s_D3D12v8);
    tracker.Require(res, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    D3D12HeapHook::BeginPluginDispatch();

    // The GPU handle must live in the heap currently set on the command list.
    ID3D12DescriptorHeap* heaps[] = { s_DescHeap.GetHeap() };
    cmdList->SetDescriptorHeaps(1, heaps);

    for (uint32_t mip = 0; mip < mipLevels; ++mip)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC u = {};
        u.Format = fmt;
        if (rd.DepthOrArraySize > 1)
        {
            u.ViewDimension                  = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            u.Texture2DArray.MipSlice        = mip;
            u.Texture2DArray.FirstArraySlice = 0;
            u.Texture2DArray.ArraySize       = rd.DepthOrArraySize;
        }
        else
        {
            u.ViewDimension      = D3D12_UAV_DIMENSION_TEXTURE2D;
            u.Texture2D.MipSlice = mip;
        }

        D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = s_ClearCpuHeap->GetCPUDescriptorHandleForHeapStart();
        device->CreateUnorderedAccessView(res, nullptr, &u, cpuHandle);
        device->CreateUnorderedAccessView(res, nullptr, &u, s_DescHeap.GetCPUHandle(slotBase + mip));

        cmdList->ClearUnorderedAccessViewFloat(s_DescHeap.GetGPUHandle(slotBase + mip),
                                               cpuHandle, res, ed.color, 0, nullptr);
    }

    D3D12HeapHook::EndPluginDispatch();

    tracker.Notify(res, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, /*uavAccess=*/true);

    if (!fromRing) s_DescHeap.Free(slotBase, mipLevels);

    D3D12HeapHook::RestoreUnityHeaps(cmdList);
}

extern "C" UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_GetClearTextureUavFloatCallbackPtr()
{
    return NR_ClearTextureUavFloatCallback;
}


// ---------------------------------------------------------------------------
// NR_NSB_AllocFlushBuffer
//   Allocates a flush snapshot blob that C# fills (header + ranges + payload) and
//   passes as the data pointer of IssuePluginEventAndData. The render-thread
//   NsbFlushCallback parses and frees it. Allocated/freed by the plugin so the
//   allocator matches across the managed/native boundary.
// ---------------------------------------------------------------------------
extern "C" intptr_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_NSB_AllocFlushBuffer(uint32_t sizeBytes)
{
    return sizeBytes ? reinterpret_cast<intptr_t>(std::malloc(sizeBytes)) : 0;
}

// ---------------------------------------------------------------------------
// NR_NSB_GetCapacity
// ---------------------------------------------------------------------------
extern "C" uint32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_NSB_GetCapacity(uint64_t handle)
{
    if (!handle) return 0;
    return reinterpret_cast<NativeBuffer*>(handle)->GetCapacity();
}

// ---------------------------------------------------------------------------
// NsbFlushCallback render-event callback
//   Called via CommandBuffer.IssuePluginEventAndData.
//   data = flush snapshot blob (NR_NSB_AllocFlushBuffer) carrying the target
//   handle, stride, ranges and packed payload. Records the GPU copy and frees the
//   blob. C# only issues this event for buffers that actually changed.
// ---------------------------------------------------------------------------
static void UNITY_INTERFACE_API NsbFlushCallback(int /*eventId*/, void* data)
{
    if (!data) return;

    const auto* blob = static_cast<const uint8_t*>(data);
    const auto* hdr  = reinterpret_cast<const NsbFlushHeader*>(blob);

    UnityGraphicsD3D12RecordingState recordingState = {};
    if (s_RendererReady && s_D3D12 && hdr->bufferHandle &&
        s_D3D12->CommandRecordingState(&recordingState) && recordingState.commandList)
    {
        auto* nsb     = reinterpret_cast<NativeBuffer*>(hdr->bufferHandle);
        auto* cmdList = static_cast<ID3D12GraphicsCommandList*>(recordingState.commandList);

        const auto*    ranges  = reinterpret_cast<const NsbFlushRange*>(blob + sizeof(NsbFlushHeader));
        const uint8_t* payload = blob + sizeof(NsbFlushHeader) + static_cast<size_t>(hdr->rangeCount) * sizeof(NsbFlushRange);

        nsb->UploadSnapshot(cmdList, ResourceStateTracker(s_D3D12v8), ranges, hdr->rangeCount, payload);
    }

    std::free(data);
}

/// <summary>Returns the render-event function pointer for Flush.</summary>
extern "C" UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_NSB_GetFlushEventFunc()
{
    return NsbFlushCallback;
}

// ---------------------------------------------------------------------------
// NativeStructuredBuffer GPU->CPU readback
//
//   Async, fence-gated (mirrors AsyncGPUReadback): a render-thread event records
//   the DEFAULT->READBACK copy into a slot of the buffer's staging ring and stamps
//   the frame-fence value at which it will be done; the main thread later polls
//   NR_NSB_TryReadback, which copies out the newest slot the GPU has finished.
// ---------------------------------------------------------------------------

// Frame fence is created once by Unity and stable; cache it on the render thread so the
// main-thread poll can read GetCompletedValue() (free-threaded on the fence object).
// Atomic: written on the render thread, read on the main thread.
static std::atomic<ID3D12Fence*> s_FrameFenceCached{ nullptr };

static void UNITY_INTERFACE_API NsbReadbackCallback(int /*eventId*/, void* data)
{
    if (!data) return;

    const auto* req = static_cast<const NsbReadbackRequest*>(data);

    UnityGraphicsD3D12RecordingState rs = {};
    if (s_RendererReady && s_D3D12 && req->bufferHandle &&
        s_D3D12->CommandRecordingState(&rs) && rs.commandList)
    {
        auto* nsb     = reinterpret_cast<NativeBuffer*>(req->bufferHandle);
        auto* cmdList = static_cast<ID3D12GraphicsCommandList*>(rs.commandList);

        if (!s_FrameFenceCached.load(std::memory_order_relaxed))
            s_FrameFenceCached.store(s_D3D12->GetFrameFence(), std::memory_order_release);
        const uint64_t fenceTarget = s_D3D12->GetNextFrameFenceValue();

        nsb->RequestReadback(cmdList, ResourceStateTracker(s_D3D12v8),
                             req->srcByteOffset, req->bytes, fenceTarget);
    }

    std::free(data);
}

/// <summary>Returns the render-event function pointer for the readback request.</summary>
extern "C" UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_NSB_GetReadbackEventFunc()
{
    return NsbReadbackCallback;
}

// ---------------------------------------------------------------------------
// NR_NSB_TryReadback
//   Main thread: returns 1 and fills |dst| (up to |dstBytes|) with the newest pending
//   readback that has completed on the GPU; returns 0 while all requests are still in
//   flight or none is pending.
// ---------------------------------------------------------------------------
extern "C" uint32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_NSB_TryReadback(uint64_t handle, void* dst, uint64_t dstBytes)
{
    ID3D12Fence* fence = s_FrameFenceCached.load(std::memory_order_acquire);
    if (!handle || !dst || !fence) return 0;
    const uint64_t completed = fence->GetCompletedValue();
    auto* nsb = reinterpret_cast<NativeBuffer*>(handle);
    return nsb->TryReadback(completed, dst, dstBytes) ? 1u : 0u;
}

// ===========================================================================
// ComputeShader  -  generic compute pipeline (cs_6_x)
// ===========================================================================

// ---------------------------------------------------------------------------
// CS_BindingSlot
//   One slot per reflected binding. Mirrors ComputeShader.h CS_BindingSlot.
//   (Redeclared here for Plugin.cpp; the authoritative definition is in ComputeShader.h)
// ---------------------------------------------------------------------------
// (CS_BindingSlot and CS_BindingObjectKind are included via ComputeShader.h)

// ---------------------------------------------------------------------------
// CS_RenderEventData
//   Passed from C# via IssuePluginEventAndData for per-shader dispatches.
//   Must match NativeRenderPlugin.CS_RenderEventData exactly (Pack=4).
// ---------------------------------------------------------------------------
#pragma pack(push, 4)
struct CS_RenderEventData
{
    uint64_t descriptorSetHandle; // +0  (8B): pointer to ComputeDescriptorSet
    uint32_t threadGroupX;    // +8  (4B)
    uint32_t threadGroupY;    // +12 (4B)
    uint32_t threadGroupZ;    // +16 (4B)
    uint32_t bindingCount;    // +20 (4B): number of CS_BindingSlot entries
    uint64_t bindingSlotsPtr; // +24 (8B): pointer to CS_BindingSlot[]
};  // Total: 32 bytes
#pragma pack(pop)

// ---------------------------------------------------------------------------
// NR_CreateComputeShader
//   Builds a compute pipeline from pre-compiled DXIL bytes (cs_6_x).
//   Returns an opaque uint64 handle on success, 0 on failure.
// ---------------------------------------------------------------------------
extern "C" uint64_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CreateComputeShader(const uint8_t* dxilBytes, uint32_t size, const char* name)
{
    if (!s_RendererReady)
    {
        NR_WARN("NR_CreateComputeShader: renderer not ready");
        return 0;
    }
    ID3D12Device* device = s_D3D12->GetDevice();
    if (!device)
    {
        NR_ERROR("NR_CreateComputeShader: failed to obtain ID3D12Device");
        return 0;
    }
    ID3D12Device5* dev5 = nullptr;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&dev5))))
    {
        NR_ERROR("NR_CreateComputeShader: failed to obtain ID3D12Device5");
        return 0;
    }
    auto* cs = new ComputeShader();
    if (!cs->Initialize(dev5, s_Log, s_D3D12v8) ||
        !cs->LoadShaderFromBytes(dxilBytes, size, name))
    {
        delete cs;
        dev5->Release();
        return 0;
    }
    dev5->Release();
    return reinterpret_cast<uint64_t>(cs);
}

// ---------------------------------------------------------------------------
// NR_CreateComputeShaderEx
//   Like NR_CreateComputeShader but accepts a hintsJson string that promotes
//   selected CBV bindings to root 32-bit constants.
//   hintsJson format: [{"name":"MyConstants","count":4}, ...]
//   Must be called before Unity sets up resource bindings.
// ---------------------------------------------------------------------------
static void ApplyRootConstantsHints(ShaderBase* shader, const char* hintsJson)
{
    if (!hintsJson || hintsJson[0] == '\0') return;
    // Lightweight JSON array parser: find {"name":"...","count":N} objects
    const char* p = hintsJson;
    while (*p)
    {
        const char* nameStart = strstr(p, "\"name\"");
        if (!nameStart) break;
        const char* q1 = strchr(nameStart + 6, '"');
        if (!q1) break;
        const char* q2 = strchr(q1 + 1, '"');
        if (!q2) break;
        std::string name(q1 + 1, q2);

        const char* countTag = strstr(q2 + 1, "\"count\"");
        if (!countTag) break;
        const char* colon = strchr(countTag + 7, ':');
        if (!colon) break;
        uint32_t count = static_cast<uint32_t>(atoi(colon + 1));

        shader->SetRootConstantsHint(name.c_str(), count);
        p = colon + 1;
    }
}

// Parses "rootSRV":["name1","name2",...] from a JSON object.
static void ApplyRootSRVHints(ShaderBase* shader, const char* hintsJson)
{
    if (!hintsJson || hintsJson[0] == '\0') return;
    const char* tag = strstr(hintsJson, "\"rootSRV\"");
    if (!tag) return;
    const char* arrStart = strchr(tag + 9, '[');
    if (!arrStart) return;
    const char* p = arrStart + 1;
    while (*p)
    {
        const char* q1 = strchr(p, '"');
        if (!q1) break;
        const char* q2 = strchr(q1 + 1, '"');
        if (!q2) break;
        std::string name(q1 + 1, q2);
        if (!name.empty())
            shader->SetRootSRVHint(name.c_str());
        p = q2 + 1;
        // stop at end of array
        const char* nextComma    = strchr(p, ',');
        const char* arrEnd       = strchr(p, ']');
        if (!arrEnd) break;
        if (!nextComma || arrEnd < nextComma) break;
    }
}

// Parses "samplers":[{"name":"...","filter":N,"addressU":N,"addressV":N,"addressW":N,
// "mips":N,"aniso":N}] from the hints JSON. Each entry overrides the static-sampler
// attributes for the matching HLSL sampler variable (see ShaderBase::SetSamplerHint).
// Missing numeric fields default to 0, except aniso which defaults to 16. A single
// "address" key (if present) is used as the fallback for any missing per-axis field.
static void ApplySamplerHints(ShaderBase* shader, const char* hintsJson)
{
    if (!hintsJson || hintsJson[0] == '\0') return;
    const char* tag = strstr(hintsJson, "\"samplers\"");
    if (!tag) return;
    const char* arrStart = strchr(tag + 10, '[');
    if (!arrStart) return;
    const char* arrEnd = strchr(arrStart, ']');
    if (!arrEnd) return;

    // Reads an integer field within [objBegin, objEnd); returns fallback if absent.
    auto ReadInt = [](const char* objBegin, const char* objEnd,
                      const char* key, int fallback) -> int {
        const char* k = strstr(objBegin, key);
        if (!k || k >= objEnd) return fallback;
        const char* colon = strchr(k, ':');
        if (!colon || colon >= objEnd) return fallback;
        return atoi(colon + 1);
    };

    const char* p = arrStart + 1;
    while (p < arrEnd)
    {
        const char* objStart = strchr(p, '{');
        if (!objStart || objStart >= arrEnd) break;
        const char* objEnd = strchr(objStart, '}');
        if (!objEnd) break;

        // name
        const char* nameTag = strstr(objStart, "\"name\"");
        std::string name;
        if (nameTag && nameTag < objEnd)
        {
            const char* q1 = strchr(nameTag + 6, '"');
            const char* q2 = q1 ? strchr(q1 + 1, '"') : nullptr;
            if (q1 && q2 && q2 < objEnd) name.assign(q1 + 1, q2);
        }

        if (!name.empty())
        {
            int filter   = ReadInt(objStart, objEnd, "\"filter\"",   1);
            int address  = ReadInt(objStart, objEnd, "\"address\"",  0); // legacy single-axis fallback
            int addressU = ReadInt(objStart, objEnd, "\"addressU\"", address);
            int addressV = ReadInt(objStart, objEnd, "\"addressV\"", address);
            int addressW = ReadInt(objStart, objEnd, "\"addressW\"", address);
            int mips     = ReadInt(objStart, objEnd, "\"mips\"",     0);
            int aniso    = ReadInt(objStart, objEnd, "\"aniso\"",    16);
            shader->SetSamplerHint(name.c_str(),
                                   static_cast<uint32_t>(filter),
                                   static_cast<uint32_t>(addressU),
                                   static_cast<uint32_t>(addressV),
                                   static_cast<uint32_t>(addressW),
                                   mips != 0,
                                   static_cast<uint32_t>(aniso));
        }
        p = objEnd + 1;
    }
}

extern "C" uint64_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CreateRayTraceShaderFromBytesEx(
    const uint8_t* dxilBytes,
    uint32_t size,
    const char* name,
    uint32_t flags,
    uint32_t maxPayloadSizeInBytes,
    const char* rayGenName,
    const char* hintsJson)
{
    if (!s_RendererReady)
    {
        NR_WARN("NR_CreateRayTraceShaderFromBytesEx: renderer not ready");
        return 0;
    }
    ID3D12Device5* dev5 = nullptr;
    ID3D12Device*  base = s_D3D12->GetDevice();
    if (!base || FAILED(base->QueryInterface(IID_PPV_ARGS(&dev5))))
    {
        NR_ERROR("NR_CreateRayTraceShaderFromBytesEx: failed to obtain ID3D12Device5");
        return 0;
    }
    auto* shader = new RayTraceShader();
    if (!shader->Initialize(dev5, s_Log, s_D3D12v8))
    {
        delete shader;
        dev5->Release();
        return 0;
    }
    ApplyRootConstantsHints(shader, hintsJson);
    ApplyRootSRVHints(shader, hintsJson);
    ApplySamplerHints(shader, hintsJson);
    if (!shader->LoadShaderFromBytes(dxilBytes, size, name, flags, maxPayloadSizeInBytes, rayGenName))
    {
        delete shader;
        dev5->Release();
        return 0;
    }
    dev5->Release();
    return reinterpret_cast<uint64_t>(shader);
}

extern "C" uint64_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CreateRayTracePipelineFromBlobsEx(
    const uint8_t** blobDataPtrs,
    const uint32_t* blobSizes,
    uint32_t        blobCount,
    const char*     name,
    uint32_t        flags,
    uint32_t        maxPayloadSizeInBytes,
    const char*     rayGenName,
    const char*     hintsJson)
{
    if (!s_RendererReady)
    {
        NR_WARN("NR_CreateRayTracePipelineFromBlobsEx: renderer not ready");
        return 0;
    }
    if (!blobDataPtrs || !blobSizes || blobCount == 0)
    {
        NR_WARN("NR_CreateRayTracePipelineFromBlobsEx: invalid arguments");
        return 0;
    }

    ID3D12Device5* dev5 = nullptr;
    ID3D12Device*  base = s_D3D12->GetDevice();
    if (!base || FAILED(base->QueryInterface(IID_PPV_ARGS(&dev5))))
    {
        NR_ERROR("NR_CreateRayTracePipelineFromBlobsEx: failed to obtain ID3D12Device5");
        return 0;
    }

    std::vector<RayTraceShader::BlobDesc> descs(blobCount);
    for (uint32_t i = 0; i < blobCount; ++i)
        descs[i] = { blobDataPtrs[i], blobSizes[i] };

    auto* shader = new RayTraceShader();
    if (!shader->Initialize(dev5, s_Log, s_D3D12v8))
    {
        delete shader;
        dev5->Release();
        return 0;
    }
    ApplyRootConstantsHints(shader, hintsJson);
    ApplyRootSRVHints(shader, hintsJson);
    ApplySamplerHints(shader, hintsJson);
    if (!shader->LoadShaderFromMultipleBlobs(descs.data(), blobCount, name, flags,
                                             maxPayloadSizeInBytes, rayGenName))
    {
        delete shader;
        dev5->Release();
        return 0;
    }
    dev5->Release();
    return reinterpret_cast<uint64_t>(shader);
}

extern "C" uint64_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CreateComputeShaderEx(const uint8_t* dxilBytes, uint32_t size, const char* name, const char* hintsJson)
{
    if (!s_RendererReady)
    {
        NR_WARN("NR_CreateComputeShaderEx: renderer not ready");
        return 0;
    }
    ID3D12Device* device = s_D3D12->GetDevice();
    if (!device)
    {
        NR_ERROR("NR_CreateComputeShaderEx: failed to obtain ID3D12Device");
        return 0;
    }
    ID3D12Device5* dev5 = nullptr;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&dev5))))
    {
        NR_ERROR("NR_CreateComputeShaderEx: failed to obtain ID3D12Device5");
        return 0;
    }
    auto* cs = new ComputeShader();
    if (!cs->Initialize(dev5, s_Log, s_D3D12v8))
    {
        delete cs;
        dev5->Release();
        return 0;
    }
    ApplyRootConstantsHints(cs, hintsJson);
    ApplyRootSRVHints(cs, hintsJson);
    ApplySamplerHints(cs, hintsJson);
    if (!cs->LoadShaderFromBytes(dxilBytes, size, name))
    {
        delete cs;
        dev5->Release();
        return 0;
    }
    dev5->Release();
    return reinterpret_cast<uint64_t>(cs);
}
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_DestroyComputeShader(uint64_t handle)
{
    if (!handle) return;
    RetireObject(reinterpret_cast<ComputeShader*>(handle));
}

// ---------------------------------------------------------------------------
// NR_CS_CreateDescriptorSet / NR_CS_DestroyDescriptorSet
//   Each NativeComputeDescriptorSet on the C# side owns a ComputeDescriptorSet
//   here.  The descriptor set holds its own GPU-heap slice so multiple
//   dispatches of the same ComputeShader per frame are fully independent.
// ---------------------------------------------------------------------------
extern "C" uint64_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CS_CreateDescriptorSet(uint64_t shaderHandle)
{
    if (!shaderHandle || !s_RendererReady) return 0;
    auto* cs = reinterpret_cast<ComputeShader*>(shaderHandle);
    auto* ds = new ComputeDescriptorSet(cs, s_D3D12->GetDevice(), s_Log, &s_DescHeap, s_D3D12v8);
    return reinterpret_cast<uint64_t>(ds);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CS_DestroyDescriptorSet(uint64_t handle)
{
    if (!handle) return;
    RetireObject(reinterpret_cast<ComputeDescriptorSet*>(handle));
}

// ---------------------------------------------------------------------------
// Resource binding queries (slot layout discovery)
// ---------------------------------------------------------------------------
extern "C" uint32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CS_GetBindingCount(uint64_t handle)
{
    if (!handle) return 0;
    return reinterpret_cast<ComputeShader*>(handle)->GetBindingCount();
}

extern "C" uint32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CS_GetSlotIndex(uint64_t handle, const char* name)
{
    if (!handle) return UINT32_MAX;
    return reinterpret_cast<ComputeShader*>(handle)->GetSlotIndex(name);
}

extern "C" UNITY_INTERFACE_EXPORT const char* UNITY_INTERFACE_API
NR_CS_GetBindingName(uint64_t handle, uint32_t index)
{
    if (!handle) return nullptr;
    return reinterpret_cast<ComputeShader*>(handle)->GetBindingName(index);
}

// ---------------------------------------------------------------------------
// NR_CS_GetRenderEventFunc / NR_CS_GetRenderEventDataSize
//   Use with CommandBuffer.IssuePluginEventAndData for compute dispatch.
// ---------------------------------------------------------------------------
static void UNITY_INTERFACE_API CsDispatchCallback(int /*eventId*/, void* data)
{
    if (!s_RendererReady || !s_D3D12 || !data) return;

    auto* ed = static_cast<CS_RenderEventData*>(data);
    if (!ed->descriptorSetHandle) return;

    UnityGraphicsD3D12RecordingState recordingState = {};
    if (!s_D3D12->CommandRecordingState(&recordingState) || !recordingState.commandList) return;

    auto* ds      = reinterpret_cast<ComputeDescriptorSet*>(ed->descriptorSetHandle);
    auto* cmdList = static_cast<ID3D12GraphicsCommandList*>(recordingState.commandList);
    auto* slots   = reinterpret_cast<BindingSlot*>(ed->bindingSlotsPtr);

    D3D12HeapHook::BeginPluginDispatch();
    ds->Dispatch(cmdList, ed->threadGroupX, ed->threadGroupY, ed->threadGroupZ,
                 slots, ed->bindingCount);
    D3D12HeapHook::EndPluginDispatch();

    // Restore Unity's descriptor heaps (our Dispatch rebinds a private heap).
    D3D12HeapHook::RestoreUnityHeaps(cmdList);
}

extern "C" UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CS_GetRenderEventFunc()
{
    return CsDispatchCallback;
}

extern "C" uint32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CS_GetRenderEventDataSize()
{
    return static_cast<uint32_t>(sizeof(CS_RenderEventData));
}

// ===========================================================================
// BC6H cubemap  -  plugin-owned compressed env cube + reinterpret copy
//
//   Unity RenderTextures cannot be BC6H, so the EnvMapBaker's compressed output
//   cube (the original m_cubemapBC6H, sampled as t_EnvironmentMap when compression
//   is enabled) is created here as a plugin-owned committed resource and handed
//   back to C# as a raw ID3D12Resource* — bindable exactly like a Unity texture
//   pointer (objectKind None). The compressor writes an RGBA32_UINT scratch cube
//   (a normal Unity UAV RenderTexture); NR_BC6HCopy reinterpret-copies that scratch
//   into the BC6H cube per subresource (RGBA32_UINT and BC6H_UF16 are both 128-bit
//   and copy-compatible), mirroring EnvMapBaker.cpp's per-slice copyTexture loop.
// ===========================================================================

// ---------------------------------------------------------------------------
// NR_CreateBC6HCube
//   Creates a BC6H_UFLOAT TextureCube (6-slice 2D array) committed resource with
//   |mipLevels| mips. Returned as the raw ID3D12Resource* (uint64); 0 on failure.
//   Released via NR_DestroyBC6HCube.
// ---------------------------------------------------------------------------
extern "C" uint64_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CreateBC6HCube(uint32_t dim, uint32_t mipLevels)
{
    if (!s_RendererReady) { NR_WARN("NR_CreateBC6HCube: renderer not ready"); return 0; }
    ID3D12Device* device = s_D3D12 ? s_D3D12->GetDevice() : nullptr;
    if (!device)          { NR_ERROR("NR_CreateBC6HCube: no ID3D12Device"); return 0; }
    if (dim == 0 || mipLevels == 0) { NR_ERROR("NR_CreateBC6HCube: invalid dim/mips"); return 0; }

    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width            = dim;
    desc.Height           = dim;
    desc.DepthOrArraySize = 6;                       // cube faces (SRV becomes TEXTURECUBE)
    desc.MipLevels        = static_cast<UINT16>(mipLevels);
    desc.Format           = DXGI_FORMAT_BC6H_UF16;   // matches nvrhi BC6H_UFLOAT
    desc.SampleDesc.Count = 1;
    desc.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags            = D3D12_RESOURCE_FLAG_NONE; // BC formats cannot be UAV; copy-dest only

    ID3D12Resource* res = nullptr;
    HRESULT hr = device->CreateCommittedResource(
        &heapProps, D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_COMMON, // first Require()/Notify() through Unity's tracker composes from COMMON
        nullptr, IID_PPV_ARGS(&res));
    if (FAILED(hr) || !res)
    {
        NR_ERROR("NR_CreateBC6HCube: CreateCommittedResource failed (hr=0x%08x)", static_cast<unsigned>(hr));
        return 0;
    }
    res->SetName(L"EnvMapBakerMainCubeBC6H");
    return reinterpret_cast<uint64_t>(res);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_DestroyBC6HCube(uint64_t handle)
{
    if (!handle) return;
    // Defer the Release until the GPU is past any in-flight frame that may still
    // reference the cube (same retire-bucket discipline as other plugin resources).
    auto* res = reinterpret_cast<ID3D12Resource*>(handle);
    EnqueueCleanup([res]{ res->Release(); });
}

// ---------------------------------------------------------------------------
// BC6HCopy_RenderEventData
//   Must match NativeRenderPlugin.BC6HCopy_RenderEventData exactly (Pack=4).
// ---------------------------------------------------------------------------
#pragma pack(push, 4)
struct BC6HCopy_RenderEventData
{
    uint64_t srcScratch;  // +0  (8B): ID3D12Resource* RGBA32_UINT scratch cube
    uint64_t dstBC6H;     // +8  (8B): ID3D12Resource* BC6H cube (from NR_CreateBC6HCube)
    uint32_t mipLevels;   // +16 (4B)
    uint32_t arraySize;   // +20 (4B): always 6 for a cube
};  // Total: 24 bytes
#pragma pack(pop)

static void UNITY_INTERFACE_API BC6HCopyCallback(int /*eventId*/, void* data)
{
    if (!s_RendererReady || !s_D3D12 || !data) return;
    auto* ed = static_cast<BC6HCopy_RenderEventData*>(data);
    auto* src = reinterpret_cast<ID3D12Resource*>(ed->srcScratch);
    auto* dst = reinterpret_cast<ID3D12Resource*>(ed->dstBC6H);
    if (!src || !dst || ed->mipLevels == 0 || ed->arraySize == 0) return;

    UnityGraphicsD3D12RecordingState recordingState = {};
    if (!s_D3D12->CommandRecordingState(&recordingState) || !recordingState.commandList) return;
    auto* cmdList = static_cast<ID3D12GraphicsCommandList*>(recordingState.commandList);

    // Route both transitions through Unity's tracker so they compose with surrounding
    // passes (scratch was last left in UAV; the BC6H cube's next use is an SRV read,
    // whose Require will transition it out of COPY_DEST).
    ResourceStateTracker tracker(s_D3D12v8);
    tracker.Require(src, D3D12_RESOURCE_STATE_COPY_SOURCE);
    tracker.Require(dst, D3D12_RESOURCE_STATE_COPY_DEST);

    // Per-subresource reinterpret copy: scratch texel (RGBA32_UINT) ↦ one BC6H 4×4 block.
    // Matches EnvMapBaker.cpp:623-628 (copyTexture per mip × face).
    for (uint32_t m = 0; m < ed->mipLevels; ++m)
        for (uint32_t f = 0; f < ed->arraySize; ++f)
        {
            // subresource = MipSlice + ArraySlice*MipLevels + PlaneSlice*MipLevels*ArraySize
            // (PlaneSlice 0). Same as D3D12CalcSubresource, inlined to avoid the d3dx12 dependency.
            const UINT sub = m + f * ed->mipLevels;
            D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
            dstLoc.pResource        = dst;
            dstLoc.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dstLoc.SubresourceIndex = sub;
            D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
            srcLoc.pResource        = src;
            srcLoc.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            srcLoc.SubresourceIndex = sub;
            cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
        }

    tracker.Notify(src, D3D12_RESOURCE_STATE_COPY_SOURCE);
    tracker.Notify(dst, D3D12_RESOURCE_STATE_COPY_DEST);
}

extern "C" UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_GetBC6HCopyEventFunc()
{
    return BC6HCopyCallback;
}

extern "C" uint32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_GetBC6HCopyEventDataSize()
{
    return static_cast<uint32_t>(sizeof(BC6HCopy_RenderEventData));
}

// ===========================================================================
// RasterShader  -  generic graphics pipeline (vs_6_x + ps_6_x)
// ===========================================================================

// ---------------------------------------------------------------------------
// RAS_RenderEventData
//   Passed from C# via IssuePluginEventAndData for per-descriptor-set draws.
//   Must match NativeRenderPlugin.RAS_RenderEventData exactly (Pack=4).
//   Render targets / depth target travel here (they are not reflected
//   resources); SRV/CBV/sampler inputs travel via bindingSlotsPtr.
// ---------------------------------------------------------------------------
#pragma pack(push, 4)
struct RAS_RenderEventData
{
    uint64_t descriptorSetHandle;  // RasterDescriptorSet*
    uint64_t bindingSlotsPtr;      // BindingSlot*
    uint32_t bindingCount;
    uint32_t numRenderTargets;
    uint64_t rtvResources[8];      // ID3D12Resource*
    uint32_t rtvFormats[8];        // DXGI_FORMAT
    uint64_t dsvResource;          // ID3D12Resource* or 0
    uint32_t dsvFormat;            // DXGI_FORMAT
    uint32_t clearFlags;           // bit0 = clear color, bit1 = clear depth
    float    clearColor[4];
    float    clearDepth;
    float    viewportX, viewportY, viewportW, viewportH;
    uint32_t vertexCount;
    uint32_t instanceCount;
    float    blendFactor;          // OMSetBlendFactor constant (blendMode==4 constant-color); was _pad
};
#pragma pack(pop)

// ---------------------------------------------------------------------------
// NR_CreateRasterShader / NR_CreateRasterShaderEx
//   Build a graphics pipeline from pre-compiled VS + PS DXIL blobs and a
//   RasterPipelineStateDesc.  The Ex form also accepts a hintsJson string
//   (rootConstants / rootSRV), identical to the compute / ray-tracing Ex forms.
// ---------------------------------------------------------------------------
static uint64_t CreateRasterShaderImpl(
    const uint8_t* vsDxil, uint32_t vsSize,
    const uint8_t* psDxil, uint32_t psSize,
    const RasterPipelineStateDesc* state,
    const char* name, const char* hintsJson)
{
    if (!s_RendererReady)
    {
        NR_WARN("NR_CreateRasterShader: renderer not ready");
        return 0;
    }
    if (!state)
    {
        NR_WARN("NR_CreateRasterShader: null state desc");
        return 0;
    }
    ID3D12Device* device = s_D3D12->GetDevice();
    if (!device)
    {
        NR_ERROR("NR_CreateRasterShader: failed to obtain ID3D12Device");
        return 0;
    }
    ID3D12Device5* dev5 = nullptr;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&dev5))))
    {
        NR_ERROR("NR_CreateRasterShader: failed to obtain ID3D12Device5");
        return 0;
    }
    auto* rs = new RasterShader();
    if (!rs->Initialize(dev5, s_Log, s_D3D12v8))
    {
        delete rs;
        dev5->Release();
        return 0;
    }
    ApplyRootConstantsHints(rs, hintsJson);
    ApplyRootSRVHints(rs, hintsJson);
    ApplySamplerHints(rs, hintsJson);
    if (!rs->LoadShaderFromBlobs(vsDxil, vsSize, psDxil, psSize, *state, name))
    {
        delete rs;
        dev5->Release();
        return 0;
    }
    dev5->Release();
    return reinterpret_cast<uint64_t>(rs);
}

extern "C" uint64_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CreateRasterShader(const uint8_t* vsDxil, uint32_t vsSize,
                      const uint8_t* psDxil, uint32_t psSize,
                      const RasterPipelineStateDesc* state, const char* name)
{
    return CreateRasterShaderImpl(vsDxil, vsSize, psDxil, psSize, state, name, nullptr);
}

extern "C" uint64_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CreateRasterShaderEx(const uint8_t* vsDxil, uint32_t vsSize,
                        const uint8_t* psDxil, uint32_t psSize,
                        const RasterPipelineStateDesc* state, const char* name,
                        const char* hintsJson)
{
    return CreateRasterShaderImpl(vsDxil, vsSize, psDxil, psSize, state, name, hintsJson);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_DestroyRasterShader(uint64_t handle)
{
    if (!handle) return;
    RetireObject(reinterpret_cast<RasterShader*>(handle));
}

// ---------------------------------------------------------------------------
// NR_RAS_CreateDescriptorSet / NR_RAS_DestroyDescriptorSet
// ---------------------------------------------------------------------------
extern "C" uint64_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RAS_CreateDescriptorSet(uint64_t shaderHandle)
{
    if (!shaderHandle || !s_RendererReady) return 0;
    auto* rs = reinterpret_cast<RasterShader*>(shaderHandle);
    auto* ds = new RasterDescriptorSet(rs, s_D3D12->GetDevice(), s_Log, &s_DescHeap, s_D3D12v8);
    return reinterpret_cast<uint64_t>(ds);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RAS_DestroyDescriptorSet(uint64_t handle)
{
    if (!handle) return;
    RetireObject(reinterpret_cast<RasterDescriptorSet*>(handle));
}

// ---------------------------------------------------------------------------
// Slot-layout discovery (mirrors NR_CS_*)
// ---------------------------------------------------------------------------
extern "C" uint32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RAS_GetBindingCount(uint64_t handle)
{
    if (!handle) return 0;
    return reinterpret_cast<RasterShader*>(handle)->GetBindingCount();
}

extern "C" uint32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RAS_GetSlotIndex(uint64_t handle, const char* name)
{
    if (!handle) return UINT32_MAX;
    return reinterpret_cast<RasterShader*>(handle)->GetSlotIndex(name);
}

extern "C" UNITY_INTERFACE_EXPORT const char* UNITY_INTERFACE_API
NR_RAS_GetBindingName(uint64_t handle, uint32_t index)
{
    if (!handle) return nullptr;
    return reinterpret_cast<RasterShader*>(handle)->GetBindingName(index);
}

// ---------------------------------------------------------------------------
// RasDrawCallback — issued via CommandBuffer.IssuePluginEventAndData.
// ---------------------------------------------------------------------------
static void UNITY_INTERFACE_API RasDrawCallback(int /*eventId*/, void* data)
{
    if (!s_RendererReady || !s_D3D12 || !data) return;

    auto* ed = static_cast<RAS_RenderEventData*>(data);
    if (!ed->descriptorSetHandle) return;

    UnityGraphicsD3D12RecordingState recordingState = {};
    if (!s_D3D12->CommandRecordingState(&recordingState) || !recordingState.commandList) return;

    auto* ds      = reinterpret_cast<RasterDescriptorSet*>(ed->descriptorSetHandle);
    auto* cmdList = static_cast<ID3D12GraphicsCommandList*>(recordingState.commandList);
    auto* slots   = reinterpret_cast<const BindingSlot*>(ed->bindingSlotsPtr);

    RasterDrawDesc draw = {};
    draw.numRenderTargets = ed->numRenderTargets;
    for (uint32_t i = 0; i < 8; ++i)
    {
        draw.rtvResources[i] = reinterpret_cast<ID3D12Resource*>(ed->rtvResources[i]);
        draw.rtvFormats[i]   = ed->rtvFormats[i];
    }
    draw.dsvResource  = reinterpret_cast<ID3D12Resource*>(ed->dsvResource);
    draw.dsvFormat    = ed->dsvFormat;
    draw.clearFlags   = ed->clearFlags;
    draw.clearColor[0] = ed->clearColor[0]; draw.clearColor[1] = ed->clearColor[1];
    draw.clearColor[2] = ed->clearColor[2]; draw.clearColor[3] = ed->clearColor[3];
    draw.clearDepth   = ed->clearDepth;
    draw.viewportX = ed->viewportX; draw.viewportY = ed->viewportY;
    draw.viewportW = ed->viewportW; draw.viewportH = ed->viewportH;
    draw.vertexCount   = ed->vertexCount;
    draw.instanceCount = ed->instanceCount;
    draw.blendFactor   = ed->blendFactor;

    D3D12HeapHook::BeginPluginDispatch();
    ds->Draw(cmdList, draw, slots, ed->bindingCount);
    D3D12HeapHook::EndPluginDispatch();

    // Restore Unity's descriptor heaps (our Draw rebinds a private heap).
    D3D12HeapHook::RestoreUnityHeaps(cmdList);
}

extern "C" UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RAS_GetRenderEventFunc()
{
    return RasDrawCallback;
}

extern "C" uint32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RAS_GetRenderEventDataSize()
{
    return static_cast<uint32_t>(sizeof(RAS_RenderEventData));
}
