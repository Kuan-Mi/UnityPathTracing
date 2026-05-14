#pragma once
#include "DescriptorSetBase.h"
#include "ComputeShader.h"

// ---------------------------------------------------------------------------
// ComputeDescriptorSet
//   Owns the GPU-heap slice (SRV / UAV allocations) for one logical
//   descriptor-set tied to a ComputeShader.  All common descriptor management
//   is provided by DescriptorSetBase<ComputeShader>.
//
//   Lifetime: created via NR_CS_CreateDescriptorSet /
//             destroyed via NR_CS_DestroyDescriptorSet (both called from C#).
// ---------------------------------------------------------------------------
class ComputeDescriptorSet : public DescriptorSetBase<ComputeShader>
{
public:
    using DescriptorSetBase::DescriptorSetBase;  // inherit constructor

    // Execute the compute dispatch.  All resource binding, descriptor writing,
    // resource-state requests, and root-parameter setup happen here.
    void Dispatch(ID3D12GraphicsCommandList* cmdList,
                  UINT threadGroupX, UINT threadGroupY, UINT threadGroupZ,
                  const CS_BindingSlot* slots, uint32_t slotCount);
};
