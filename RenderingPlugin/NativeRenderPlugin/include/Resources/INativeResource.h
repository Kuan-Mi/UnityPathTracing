#pragma once
#include <d3d12.h>

// ---------------------------------------------------------------------------
// INativeResource
//   Common base for every plugin-owned GPU resource (NativeBuffer,
//   NativeGpuBuffer, NativeStructuredBuffer, ...). Lets the descriptor and
//   state-tracking layers resolve a bound resource without switching on the
//   concrete type: a BindingSlot whose objectKind names one of these wrappers
//   can be reinterpret_cast<INativeResource*> directly.
//
//   Pointer-identity contract: each wrapper must inherit INativeResource as its
//   single / first base, so the pointer C# passes as BindingSlot.objectPtr (the
//   value returned from the native `new`) is the INativeResource subobject.
// ---------------------------------------------------------------------------
class INativeResource
{
public:
    virtual ~INativeResource() = default;

    // The underlying D3D12 resource. May resolve dynamically (e.g. the
    // current frame slot of a multi-buffered upload buffer). Volatile buffers
    // that have no persistent resource identity (their data lives in a shared
    // upload-pool suballocation that changes every write) return nullptr — they
    // are bound by GPU VA via GetGpuVirtualAddress() instead.
    virtual ID3D12Resource* GetResource() const = 0;

    // GPU virtual address for inline-root binding (root CBV / root SRV). The
    // default resolves the resource's base address; a volatile buffer overrides
    // this to return its current suballocation's offset-correct address (the
    // nvrhi volatile-constant-buffer model — bind by VA, no backing resource).
    virtual D3D12_GPU_VIRTUAL_ADDRESS GetGpuVirtualAddress() const
    {
        ID3D12Resource* r = GetResource();
        return r ? r->GetGPUVirtualAddress() : 0;
    }
};
