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

#include "IUnityInterface.h"
#include "IUnityGraphics.h"
#include "IUnityGraphicsD3D12.h"
#include "IUnityLog.h"

#include "AccelerationStructure.h"
#include "RayTraceShader.h"
#include "ComputeShader.h"
#include "DescriptorHeapAllocator.h"
#include "BindlessTexture.h"
#include "BindlessBuffer.h"

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

// ---------------------------------------------------------------------------
// Device lifecycle callback
// ---------------------------------------------------------------------------
static void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType)
{
    if (eventType == kUnityGfxDeviceEventInitialize)
    {
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
        NR_LOG("Plugin shutdown");
        s_DescHeap.Shutdown();
        s_RendererReady = false;
        s_D3D12         = nullptr;
        s_D3D12v8       = nullptr;
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
//   The object is deleted immediately; ensure the GPU is idle or that the
//   AS was not bound to an in-flight frame before calling this.
//   (The AccelerationStructure's own deferred-delete queue handles
//    internal BLAS/TLAS resources safely.)
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_DestroyAccelerationStructure(uint64_t handle)
{
    if (!handle) return;
    FlushGpuAndWait();
    delete reinterpret_cast<AccelerationStructure*>(handle);
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
//   posOff - tanOff    : byte offsets within each vertex (~0u = absent)
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
    uint32_t instanceHandle,
    void*    vbPtr,
    uint32_t vertexCount,
    uint32_t vertexStride,
    uint32_t positionOffset,
    uint32_t normalOffset,
    uint32_t texCoord1Offset,
    uint32_t tangentOffset,
    void*    ibPtr,
    uint32_t indexStride,
    const NR_SubmeshDesc*    submeshDescs,
    uint32_t                 submeshCount,
    const NR_SubmeshOMMDesc* ommDescs)
{
    if (!handle || !vbPtr || !ibPtr || !submeshDescs || submeshCount == 0)
    {
        NR_WARN("NR_AS_AddInstance: invalid arguments");
        return false;
    }
    auto* as = reinterpret_cast<AccelerationStructure*>(handle);
    auto* vb = static_cast<ID3D12Resource*>(vbPtr);
    auto* ib = static_cast<ID3D12Resource*>(ibPtr);
    return as->AddInstance(instanceHandle, vb, vertexCount, vertexStride,
                           positionOffset, normalOffset, texCoord1Offset, tangentOffset,
                           ib, indexStride,
                           submeshDescs, submeshCount, ommDescs);
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
    if (!s_D3D12) return;
    ID3D12CommandQueue* queue = s_D3D12->GetCommandQueue();
    if (!queue) return;

    ID3D12Device* device = s_D3D12->GetDevice();
    if (!device) return;

    ComPtr<ID3D12Fence> fence;
    if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
        return;

    if (FAILED(queue->Signal(fence.Get(), 1)))
        return;

    if (fence->GetCompletedValue() < 1)
    {
        HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (ev)
        {
            fence->SetEventOnCompletion(1, ev);
            WaitForSingleObject(ev, INFINITE);
            CloseHandle(ev);
        }
    }
}

// ---------------------------------------------------------------------------
// NR_DestroyRayTraceShader
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_DestroyRayTraceShader(uint64_t handle)
{
    if (!handle) return;
    FlushGpuAndWait();
    delete reinterpret_cast<RayTraceShader*>(handle);
}

// ---------------------------------------------------------------------------
// NR_CreateRayTraceShaderFromBytes
//   Builds a DXR pipeline from pre-compiled DXIL bytes.  Returns an opaque
//   uint64 handle on success, 0 on failure.
//   Caller owns the lifetime; call NR_DestroyRayTraceShader when done.
// ---------------------------------------------------------------------------
extern "C" uint64_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CreateRayTraceShaderFromBytes(const uint8_t* dxilBytes, uint32_t size)
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
    if (!shader->Initialize(dev5, s_Log, &s_DescHeap) ||
        !shader->LoadShaderFromBytes(dxilBytes, size))
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

    shader->Dispatch(cmdList, ed->width, ed->height);
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
//   Destroys a BindlessTexture and returns its descriptor slots to the pool.
//   Ensure no in-flight GPU work references it before calling.
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_DestroyBindlessTexture(uint64_t handle)
{
    if (handle) delete reinterpret_cast<BindlessTexture*>(handle);
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
//   Destroys a BindlessBuffer and returns its descriptor slots to the pool.
//   Ensure no in-flight GPU work references it before calling.
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_DestroyBindlessBuffer(uint64_t handle)
{
    if (handle) delete reinterpret_cast<BindlessBuffer*>(handle);
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
// CS_RenderEventData
//   Passed from C# via IssuePluginEventAndData for per-shader dispatches.
//   Must match NativeRenderPlugin.CS_RenderEventData exactly (Pack=4).
// ---------------------------------------------------------------------------
#pragma pack(push, 4)
struct CS_RenderEventData
{
    uint64_t shaderHandle;    // +0 (8B): pointer to ComputeShader
    uint32_t threadGroupX;    // +8 (4B)
    uint32_t threadGroupY;    // +12 (4B)
    uint32_t threadGroupZ;    // +16 (4B)
};  // Total: 20 bytes
#pragma pack(pop)

// ---------------------------------------------------------------------------
// NR_CreateComputeShader
//   Builds a compute pipeline from pre-compiled DXIL bytes (cs_6_x).
//   Returns an opaque uint64 handle on success, 0 on failure.
// ---------------------------------------------------------------------------
extern "C" uint64_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CreateComputeShader(const uint8_t* dxilBytes, uint32_t size)
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
    if (!cs->Initialize(device, s_Log, &s_DescHeap) ||
        !cs->LoadShaderFromBytes(dxilBytes, size))
    {
        delete cs;
        return 0;
    }
    return reinterpret_cast<uint64_t>(cs);
}

// ---------------------------------------------------------------------------
// NR_DestroyComputeShader
// ---------------------------------------------------------------------------
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_DestroyComputeShader(uint64_t handle)
{
    if (!handle) return;
    FlushGpuAndWait();
    delete reinterpret_cast<ComputeShader*>(handle);
}

// ---------------------------------------------------------------------------
// Resource binding helpers
// ---------------------------------------------------------------------------
extern "C" int32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CS_SetBuffer(uint64_t handle, const char* name, void* d3d12ResourcePtr)
{
    if (!handle) return 0;
    return reinterpret_cast<ComputeShader*>(handle)
        ->SetBuffer(name, static_cast<ID3D12Resource*>(d3d12ResourcePtr)) ? 1 : 0;
}

extern "C" int32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CS_SetRWBuffer(uint64_t handle, const char* name, void* d3d12ResourcePtr)
{
    if (!handle) return 0;
    return reinterpret_cast<ComputeShader*>(handle)
        ->SetRWBuffer(name, static_cast<ID3D12Resource*>(d3d12ResourcePtr)) ? 1 : 0;
}

extern "C" int32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CS_SetTexture(uint64_t handle, const char* name, void* d3d12ResourcePtr)
{
    if (!handle) return 0;
    return reinterpret_cast<ComputeShader*>(handle)
        ->SetTexture(name, static_cast<ID3D12Resource*>(d3d12ResourcePtr)) ? 1 : 0;
}

extern "C" int32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CS_SetRWTexture(uint64_t handle, const char* name, void* d3d12ResourcePtr)
{
    if (!handle) return 0;
    return reinterpret_cast<ComputeShader*>(handle)
        ->SetRWTexture(name, static_cast<ID3D12Resource*>(d3d12ResourcePtr)) ? 1 : 0;
}

extern "C" int32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CS_SetConstantBuffer(uint64_t handle, const char* name, void* d3d12ResourcePtr)
{
    if (!handle) return 0;
    return reinterpret_cast<ComputeShader*>(handle)
        ->SetConstantBuffer(name, static_cast<ID3D12Resource*>(d3d12ResourcePtr)) ? 1 : 0;
}

extern "C" int32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CS_SetStructuredBuffer(uint64_t handle, const char* name, void* d3d12ResourcePtr,
                           uint32_t elementCount, uint32_t elementStride)
{
    if (!handle) return 0;
    return reinterpret_cast<ComputeShader*>(handle)
        ->SetStructuredBuffer(name, static_cast<ID3D12Resource*>(d3d12ResourcePtr),
                              elementCount, elementStride) ? 1 : 0;
}

extern "C" int32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CS_SetBindlessTexture(uint64_t handle, const char* name, uint64_t btHandle)
{
    if (!handle) return 0;
    return reinterpret_cast<ComputeShader*>(handle)
        ->SetBindlessTexture(name, reinterpret_cast<BindlessTexture*>(btHandle)) ? 1 : 0;
}

extern "C" int32_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
NR_CS_SetBindlessBuffer(uint64_t handle, const char* name, uint64_t bbHandle)
{
    if (!handle) return 0;
    return reinterpret_cast<ComputeShader*>(handle)
        ->SetBindlessBuffer(name, reinterpret_cast<BindlessBuffer*>(bbHandle)) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// NR_CS_GetRenderEventFunc / NR_CS_GetRenderEventDataSize
//   Use with CommandBuffer.IssuePluginEventAndData for compute dispatch.
// ---------------------------------------------------------------------------
static void UNITY_INTERFACE_API CsDispatchCallback(int /*eventId*/, void* data)
{
    if (!s_RendererReady || !s_D3D12 || !data) return;

    auto* ed = static_cast<CS_RenderEventData*>(data);
    if (!ed->shaderHandle) return;

    UnityGraphicsD3D12RecordingState recordingState = {};
    if (!s_D3D12->CommandRecordingState(&recordingState) || !recordingState.commandList) return;

    auto* cs      = reinterpret_cast<ComputeShader*>(ed->shaderHandle);
    auto* cmdList = static_cast<ID3D12GraphicsCommandList*>(recordingState.commandList);

    cs->Dispatch(cmdList, ed->threadGroupX, ed->threadGroupY, ed->threadGroupZ);
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
