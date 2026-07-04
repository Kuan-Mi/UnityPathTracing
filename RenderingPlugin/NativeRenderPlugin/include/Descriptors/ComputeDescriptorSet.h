#pragma once
#include "DescriptorSetBase.h"
#include "ComputePipeline.h"

// ---------------------------------------------------------------------------
// ComputeDescriptorSet
//   Binds resources and issues one compute Dispatch for a ComputePipeline. Holds
//   no per-frame GPU-heap state of its own: each Dispatch bump-allocates its
//   SRV/UAV descriptor table from the global TransientDescriptorRing.  All
//   common descriptor management is provided by DescriptorSetBase<ComputePipeline>.
//
//   Lifetime: created via NR_CS_CreateDescriptorSet /
//             destroyed via NR_CS_DestroyDescriptorSet (both called from C#).
// ---------------------------------------------------------------------------
class ComputeDescriptorSet : public DescriptorSetBase<ComputePipeline>
{
public:
    using DescriptorSetBase::DescriptorSetBase;  // inherit constructor

    // Execute the compute dispatch.  All resource binding, descriptor writing,
    // resource-state requests, and root-parameter setup happen here.
    void Dispatch(ID3D12GraphicsCommandList* cmdList,
                  UINT threadGroupX, UINT threadGroupY, UINT threadGroupZ,
                  const BindingSlot* slots, uint32_t slotCount);
};
