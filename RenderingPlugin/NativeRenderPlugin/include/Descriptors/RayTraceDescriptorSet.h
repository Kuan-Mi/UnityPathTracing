#pragma once
#include "DescriptorSetBase.h"
#include "RayTraceShader.h"

// ---------------------------------------------------------------------------
// RayTraceDescriptorSet
//   Binds resources and issues one DispatchRays for a RayTraceShader.  Holds no
//   per-frame GPU-heap state of its own: each Dispatch bump-allocates its
//   SRV/UAV descriptor table from the global TransientDescriptorRing.  All
//   common descriptor management is provided by DescriptorSetBase<RayTraceShader>.
//
//   Lifetime: created via NR_RTS_CreateDescriptorSet /
//             destroyed via NR_RTS_DestroyDescriptorSet (both called from C#).
// ---------------------------------------------------------------------------
class RayTraceDescriptorSet : public DescriptorSetBase<RayTraceShader>
{
public:
    using DescriptorSetBase::DescriptorSetBase;  // inherit constructor

    // Execute a DispatchRays call.  All descriptor writing, resource-state
    // requests, and root-parameter setup happen here.
    void Dispatch(ID3D12GraphicsCommandList4* cmdList,
                  UINT width, UINT height,
                  const BindingSlot* slots, uint32_t slotCount);
};
