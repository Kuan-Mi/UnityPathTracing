#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <vector>
#include <mutex>

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// DescriptorHeapAllocator
//   Manages one large GPU-visible CBV/SRV/UAV descriptor heap shared by all
//   NativeRender objects (RayTraceShader, BindlessTexture, …).
//
//   D3D12 only allows one shader-visible CBV/SRV/UAV heap bound at a time, so
//   every object that needs shader-visible descriptors must use the same heap.
//   This allocator provides a simple range-based API so each object can own a
//   contiguous slice of that heap.
//
//   Thread-safety: Allocate/Free are mutex-protected; GetHeap/GetCPUHandle/
//   GetGPUHandle are lock-free read-only accessors.
// ---------------------------------------------------------------------------
class DescriptorHeapAllocator
{
public:
    // Total number of CBV/SRV/UAV descriptor slots in the shared heap.
    static constexpr uint32_t kCapacity = 65536u;

    // Create the D3D12 heap.  Must be called once before any other method.
    bool Initialize(ID3D12Device* device);

    // Release the heap and reset all state.
    void Shutdown();

    // Allocate |count| contiguous slots.  Returns the 0-based index of the
    // first slot.  This function never returns the sentinel UINT32_MAX.
    // Triggers an assertion on exhaustion – callers must size kCapacity
    // appropriately.
    uint32_t Allocate(uint32_t count);

    // Return the range [base, base+count) to the free pool.
    void Free(uint32_t base, uint32_t count);

    // Raw heap pointer – passed to SetDescriptorHeaps().
    ID3D12DescriptorHeap* GetHeap() const { return m_heap.Get(); }

    // Increment size for arithmetic on handles.
    UINT GetIncrementSize() const { return m_incSize; }

    // CPU/GPU handles for a given slot index.
    D3D12_CPU_DESCRIPTOR_HANDLE GetCPUHandle(uint32_t slot) const;
    D3D12_GPU_DESCRIPTOR_HANDLE GetGPUHandle(uint32_t slot) const;

    bool IsInitialized() const { return m_heap != nullptr; }

private:
    ComPtr<ID3D12DescriptorHeap> m_heap;
    UINT     m_incSize  = 0;
    uint32_t m_bumpNext = 0;    // next un-allocated slot (bump pointer)

    struct FreeRange { uint32_t base, count; };
    std::vector<FreeRange> m_freeList;
    std::mutex             m_mutex;
};
