/*
 * Plugin.cpp Unity Native DX12 Ray Tracing Plugin
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <cstdint>
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
#include "ComputeShader.h"
#include "ComputeDescriptorSet.h"
#include "DescriptorHeapAllocator.h"
#include "BindlessTexture.h"
#include "BindlessBuffer.h"
#include "D3D12HeapHook.h"
#include "PluginInternal.h"
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
static bool                      s_RendererReady = false;

// ---------------------------------------------------------------------------
// Deferred delete queue — delays destruction of GPU-facing objects until the
// GPU has finished executing all commands that may reference them.
//
// Each frame NR_FrameTick() signals s_DeletionFence with an incrementing value.
// Destroy functions push entries tagged with (current fence value + kDeleteDelay)
// instead of immediately deleting.  DrainDeferredDeletes() frees entries whose
// safeAfterValue <= fence->GetCompletedValue().
// ---------------------------------------------------------------------------

using DeletionTask = std::function<void()>;

static std::map<uint64_t, std::vector<DeletionTask>> s_DeletionQueue;
static std::mutex s_DeletionMutex;

// Forward declarations
static void PluginLog(UnityLogType type, const char* msg, const char* file, int line);

// kDeleteDelay: number of frames to wait before freeing.
// Unity D3D12 uses up to 2 frames in flight; 3 provides a safe margin.
static constexpr int kDeleteDelay = 3;


void EnqueueCleanup(std::function<void()>&& cleanupTask)
{
    if (!cleanupTask) return;

    uint64_t fenceValue = s_D3D12->GetNextFrameFenceValue() + kDeleteDelay;

    std::lock_guard<std::mutex> lk(s_DeletionMutex);
    s_DeletionQueue[fenceValue].emplace_back(std::move(cleanupTask));
}

template<typename T>
void SafeDelete(T*& ptr)
{
    if (!ptr) return;
    T* rawPtr = ptr;
    ptr = nullptr; // 立即置空防止野指针
    EnqueueCleanup([rawPtr]() {
        delete rawPtr;
    });
}

void SafeReleaseResource(ComPtr<ID3D12Resource> resource)
{
    if (!resource) return;

    
        wchar_t name[128]; UINT size = sizeof(name);
        resource->GetPrivateData(WKPDID_D3DDebugObjectNameW, &size, name);
        char logMsg[256];
        snprintf(logMsg, sizeof(logMsg), "Deferred Resource Released: %ls", name);
        PluginLog(kUnityLogTypeLog, logMsg, __FILE__, __LINE__);


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
    // 获取当前 GPU 已完成的 Fence 值
    uint64_t completedValue = (s_D3D12 && !force) ? s_D3D12->GetFrameFence()->GetCompletedValue() : UINT64_MAX;

    std::vector<std::vector<DeletionTask>> tasksToExecute;

    {
        std::lock_guard<std::mutex> lk(s_DeletionMutex);
        auto it = s_DeletionQueue.begin();
        while (it != s_DeletionQueue.end())
        {
            // 因为 map 是有序的，如果当前 key > 已完成值，后面所有的都没完成
            if (!force && it->first > completedValue)
                break;

            tasksToExecute.emplace_back(std::move(it->second));
            it = s_DeletionQueue.erase(it);
        }
    }

    char logMsg[256];

    auto countTasks = 0;
    for (const auto& batch : tasksToExecute)
        countTasks += batch.size();

    snprintf(logMsg, sizeof(logMsg), "Draining Deferred Deletes: executing %zu batches, %d tasks", tasksToExecute.size(), countTasks);
    PluginLog(kUnityLogTypeLog, logMsg, __FILE__, __LINE__);


    // 在锁外执行真正的析构操作，防止死锁并减少锁占用时间
    for (auto& batch : tasksToExecute)
    {
        for (auto& task : batch)
        {
            if (task) task();
        }
    }
}

void EnqueueDeferredDelete(void* ptr, DeferredType type)
{
    if (!ptr) return;

    // 根据类型将 void* 强转回具体指针，并包装成 Lambda
    // 这里使用 Lambda 捕获，可以在销毁时正确触发各个类的析构函数
    switch (type)
    {
    case DeferredType::BindlessTexture:
        EnqueueCleanup([p = static_cast<BindlessTexture*>(ptr)] { delete p; });
        break;

    case DeferredType::BindlessBuffer:
        EnqueueCleanup([p = static_cast<BindlessBuffer*>(ptr)] { delete p; });
        break;

    case DeferredType::AccelStruct:
        EnqueueCleanup([p = static_cast<AccelerationStructure*>(ptr)] { delete p; });
        break;

    case DeferredType::RayTraceShader:
        EnqueueCleanup([p = static_cast<RayTraceShader*>(ptr)] { delete p; });
        break;

    case DeferredType::ComputeShader:
        EnqueueCleanup([p = static_cast<ComputeShader*>(ptr)] { delete p; });
        break;

    case DeferredType::ComputeDescriptorSet:
        EnqueueCleanup([p = static_cast<ComputeDescriptorSet*>(ptr)] { delete p; });
        break;

    case DeferredType::AccelStructBlas:
        EnqueueCleanup([p = static_cast<BLASEntry*>(ptr)] { delete p; });
        break;

    default:
        // 如果进入了未定义的类型，为了安全起见，尝试直接 delete 
        // 但注意：void* 是不能直接 delete 的，这里最好记录一个错误日志
        if (s_Log) {
            char buf[256];
            snprintf(buf, sizeof(buf), "Unknown DeferredType %d for ptr 0x%p", (int)type, ptr);
            s_Log->Log(kUnityLogTypeWarning, buf, __FILE__, __LINE__);
        }
        break;
    }

    // 可选：调试用 Log（建议只在 Debug 模式开启，避免性能抖动）
// #ifdef _DEBUG
    char logMsg[128];
    snprintf(logMsg, sizeof(logMsg), "Enqueued Cleanup: ptr=0x%p, type=%d", ptr, (int)type);
    PluginLog(kUnityLogTypeLog, logMsg, __FILE__, __LINE__);
// #endif
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

static void FlushGpuAndWait();

#define NR_LOG(msg)     PluginLog(kUnityLogTypeLog,     (msg), __FILE__, __LINE__)
#define NR_WARN(msg)    PluginLog(kUnityLogTypeWarning, (msg), __FILE__, __LINE__)
#define NR_ERROR(msg)   PluginLog(kUnityLogTypeError,   (msg), __FILE__, __LINE__)

// Bridge D3D12HeapHook logs into Unity's log (called from any thread).
static void HeapHookLogBridge(int level, const char* msg)
{
    UnityLogType type = (level == 2) ? kUnityLogTypeError
                      : (level == 1) ? kUnityLogTypeWarning
                                     : kUnityLogTypeLog;
    PluginLog(type, msg, __FILE__, __LINE__);
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

        // Check DXR (ID3D12Device5) support
        ComPtr<ID3D12Device5> dev5;
        s_RendererReady = SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&dev5)));
        if (s_RendererReady)
        {
            if (!s_DescHeap.Initialize(device))
                NR_ERROR("DescriptorHeapAllocator initialization failed");

            NR_LOG("Plugin initialized (DXR device confirmed)");
        }
        else
            NR_ERROR("Device does not support DXR (ID3D12Device5 unavailable)");
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

        s_DescHeap.Shutdown();
        s_RendererReady = false;
        s_D3D12         = nullptr;
        s_D3D12v8       = nullptr;
        NR_LOG("Plugin shutdown - COMPLETE");
    }
}

// ---------------------------------------------------------------------------
// NR_FrameTick
//   Must be called once per frame from the CPU (main thread) before submitting
//   rendering commands.  Signals the deletion fence with the next value, then
//   drains any deferred-delete entries whose fence value has been passed.
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_FrameTick()
{
    DrainDeferredDeletes();
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
    EnqueueDeferredDelete(reinterpret_cast<void*>(handle), DeferredType::AccelStruct);
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
    reinterpret_cast<AccelerationStructure*>(handle)
        ->UpdateDynamicVertexBuffer(instanceHandle, vbPtr, vertexCount, vertexStride);
}

// ===========================================================================
// RayTraceShader  -  multi-shader, per-instance ray tracing pipeline
// ===========================================================================

// ---------------------------------------------------------------------------
// RTS_RenderEventData
//   Passed from C# via IssuePluginEventAndData for per-shader dispatches.
//   Must match NativeRenderPlugin.RTS_RenderEventData exactly (Pack=4).
// ---------------------------------------------------------------------------
#pragma pack(push, 4)
struct RTS_RenderEventData
{
    uint64_t shaderHandle;  // +0 (8B): pointer to RayTraceShader
    uint32_t width;         // +8 (4B)
    uint32_t height;        // +12 (4B)
};  // Total: 16 bytes
#pragma pack(pop)

// AS build event data - passed to AsBuildRenderCallback
#pragma pack(push, 4)
struct AS_BuildEventData
{
    uint64_t asHandle;  // pointer to AccelerationStructure
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
    EnqueueDeferredDelete(reinterpret_cast<void*>(handle), DeferredType::RayTraceShader);
}

// ---------------------------------------------------------------------------
// NR_CreateRayTraceShaderFromBytes
//   Builds a DXR pipeline from pre-compiled DXIL bytes.  Returns an opaque
//   uint64 handle on success, 0 on failure.
//   Caller owns the lifetime; call NR_DestroyRayTraceShader when done.
// ---------------------------------------------------------------------------
extern "C" uint64_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CreateRayTraceShaderFromBytes(const uint8_t* dxilBytes, uint32_t size, const char* name)
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
    if (!shader->Initialize(dev5, s_Log, &s_DescHeap, s_D3D12v8) ||
        !shader->LoadShaderFromBytes(dxilBytes, size, name))
    {
        delete shader;
        dev5->Release();
        return 0;
    }
    dev5->Release();
    return reinterpret_cast<uint64_t>(shader);
}

// ---------------------------------------------------------------------------
// Resource binding helpers (return 1 on success, 0 if name not found)
// ---------------------------------------------------------------------------
extern "C" int32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RTS_SetBuffer(uint64_t handle, const char* name, void* d3d12ResourcePtr)
{
    if (!handle) return 0;
    return reinterpret_cast<RayTraceShader*>(handle)
        ->SetBuffer(name, static_cast<ID3D12Resource*>(d3d12ResourcePtr)) ? 1 : 0;
}

extern "C" int32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RTS_SetRWBuffer(uint64_t handle, const char* name, void* d3d12ResourcePtr)
{
    if (!handle) return 0;
    return reinterpret_cast<RayTraceShader*>(handle)
        ->SetRWBuffer(name, static_cast<ID3D12Resource*>(d3d12ResourcePtr)) ? 1 : 0;
}

extern "C" int32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RTS_SetRWStructuredBuffer(uint64_t handle, const char* name, void* d3d12ResourcePtr,
                              uint32_t elementCount, uint32_t elementStride)
{
    if (!handle) return 0;
    return reinterpret_cast<RayTraceShader*>(handle)
        ->SetRWStructuredBuffer(name, static_cast<ID3D12Resource*>(d3d12ResourcePtr),
                                elementCount, elementStride) ? 1 : 0;
}

extern "C" int32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RTS_SetTexture(uint64_t handle, const char* name, void* d3d12ResourcePtr)
{
    if (!handle) return 0;
    return reinterpret_cast<RayTraceShader*>(handle)
        ->SetTexture(name, static_cast<ID3D12Resource*>(d3d12ResourcePtr)) ? 1 : 0;
}

extern "C" int32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RTS_SetRWTexture(uint64_t handle, const char* name, void* d3d12ResourcePtr)
{
    if (!handle) return 0;
    return reinterpret_cast<RayTraceShader*>(handle)
        ->SetRWTexture(name, static_cast<ID3D12Resource*>(d3d12ResourcePtr)) ? 1 : 0;
}

extern "C" int32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RTS_SetConstantBuffer(uint64_t handle, const char* name, void* d3d12ResourcePtr)
{
    if (!handle) return 0;
    return reinterpret_cast<RayTraceShader*>(handle)
        ->SetConstantBuffer(name, static_cast<ID3D12Resource*>(d3d12ResourcePtr)) ? 1 : 0;
}

extern "C" int32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RTS_SetStructuredBuffer(uint64_t handle, const char* name, void* d3d12ResourcePtr,
                            uint32_t elementCount, uint32_t elementStride)
{
    if (!handle) return 0;
    return reinterpret_cast<RayTraceShader*>(handle)
        ->SetStructuredBuffer(name, static_cast<ID3D12Resource*>(d3d12ResourcePtr),
                              elementCount, elementStride) ? 1 : 0;
}

extern "C" int32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RTS_SetAccelerationStructure(uint64_t handle, const char* name, void* tlasd3d12Ptr)
{
    if (!handle) return 0;
    return reinterpret_cast<RayTraceShader*>(handle)
        ->SetAccelerationStructure(name, static_cast<ID3D12Resource*>(tlasd3d12Ptr)) ? 1 : 0;
}

// Preferred variant: binds by AccelerationStructure object — TLAS ptr is resolved dynamically at Dispatch time.
// asHandle is the uint64_t returned by NR_CreateAccelerationStructure.
extern "C" int32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RTS_SetAccelerationStructureHandle(uint64_t shaderHandle, const char* name, uint64_t asHandle)
{
    if (!shaderHandle) return 0;
    return reinterpret_cast<RayTraceShader*>(shaderHandle)
        ->SetAccelerationStructureObject(name,
            asHandle ? reinterpret_cast<AccelerationStructure*>(asHandle) : nullptr) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// NR_RTS_GetRenderEventFunc / NR_RTS_GetRenderEventDataSize
//   Use with CommandBuffer.IssuePluginEventAndData for per-shader ray trace.
// ---------------------------------------------------------------------------
static void UNITY_INTERFACE_API RtsRenderCallback(int /*eventId*/, void* data)
{
    if (!s_RendererReady || !s_D3D12 || !data) return;

    auto* ed = static_cast<RTS_RenderEventData*>(data);
    if (!ed->shaderHandle) return;

    UnityGraphicsD3D12RecordingState recordingState = {};
    if (!s_D3D12->CommandRecordingState(&recordingState) || !recordingState.commandList) return;

    auto* shader  = reinterpret_cast<RayTraceShader*>(ed->shaderHandle);
    auto* cmdList = static_cast<ID3D12GraphicsCommandList4*>(recordingState.commandList);

    D3D12HeapHook::BeginPluginDispatch();
    shader->Dispatch(cmdList, ed->width, ed->height);
    D3D12HeapHook::EndPluginDispatch();

    // Our Dispatch bound a private descriptor heap. Restore Unity's heaps so
    // Unity's subsequent SetComputeRootDescriptorTable does not trip D3D12
    // validation (category 9, id 708).
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

extern "C" UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_AS_GetBuildRenderEventFunc()
{
    return AsBuildRenderCallback;
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

// ---------------------------------------------------------------------------
// NR_RTS_SetBindlessTexture
//   Binds a BindlessTexture to an unbounded Texture2D[] variable.
//   Returns 1 on success, 0 if the name is not found or isn't an SRV_ARRAY.
// ---------------------------------------------------------------------------
extern "C" int32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RTS_SetBindlessTexture(uint64_t shaderHandle, const char* name, uint64_t btHandle)
{
    if (!shaderHandle) return 0;
    auto* shader = reinterpret_cast<RayTraceShader*>(shaderHandle);
    auto* bt     = reinterpret_cast<BindlessTexture*>(btHandle); // may be nullptr to unbind
    return shader->SetBindlessTexture(name, bt) ? 1 : 0;
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
    if (handle) EnqueueDeferredDelete(reinterpret_cast<void*>(handle), DeferredType::BindlessTexture);
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
// NR_RTS_SetBindlessBuffer
//   Binds a BindlessBuffer to an unbounded ByteAddressBuffer[] variable.
//   Returns 1 on success, 0 if the name is not found or isn't an SRV_ARRAY.
// ---------------------------------------------------------------------------
extern "C" int32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_RTS_SetBindlessBuffer(uint64_t shaderHandle, const char* name, uint64_t bbHandle)
{
    if (!shaderHandle) return 0;
    auto* shader = reinterpret_cast<RayTraceShader*>(shaderHandle);
    auto* bb     = reinterpret_cast<BindlessBuffer*>(bbHandle); // may be nullptr to unbind
    return shader->SetBindlessBuffer(name, bb) ? 1 : 0;
}

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
    if (handle) EnqueueDeferredDelete(reinterpret_cast<void*>(handle), DeferredType::BindlessBuffer);
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
    auto* cs = new ComputeShader();
    if (!cs->Initialize(device, s_Log, &s_DescHeap, s_D3D12v8) ||
        !cs->LoadShaderFromBytes(dxilBytes, size, name))
    {
        delete cs;
        return 0;
    }
    return reinterpret_cast<uint64_t>(cs);
}

// ---------------------------------------------------------------------------
// NR_CreateComputeShaderEx
//   Like NR_CreateComputeShader but accepts a hintsJson string that promotes
//   selected CBV bindings to root 32-bit constants.
//   hintsJson format: [{"name":"MyConstants","count":4}, ...]
//   Must be called before Unity sets up resource bindings.
// ---------------------------------------------------------------------------
static void ApplyRootConstantsHints(ComputeShader* cs, const char* hintsJson)
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

        cs->SetRootConstantsHint(name.c_str(), count);
        p = colon + 1;
    }
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
    auto* cs = new ComputeShader();
    if (!cs->Initialize(device, s_Log, &s_DescHeap, s_D3D12v8))
    {
        delete cs;
        return 0;
    }
    ApplyRootConstantsHints(cs, hintsJson);
    if (!cs->LoadShaderFromBytes(dxilBytes, size, name))
    {
        delete cs;
        return 0;
    }
    return reinterpret_cast<uint64_t>(cs);
}
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_DestroyComputeShader(uint64_t handle)
{
    if (!handle) return;
    EnqueueDeferredDelete(reinterpret_cast<void*>(handle), DeferredType::ComputeShader);
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
    EnqueueDeferredDelete(reinterpret_cast<void*>(handle), DeferredType::ComputeDescriptorSet);
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
    auto* slots   = reinterpret_cast<CS_BindingSlot*>(ed->bindingSlotsPtr);

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
