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
    // current frame slot of a multi-buffered upload buffer).
    virtual ID3D12Resource* GetResource() const = 0;
};
