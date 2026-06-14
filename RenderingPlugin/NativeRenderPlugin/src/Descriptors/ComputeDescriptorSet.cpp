#include "ComputeDescriptorSet.h"

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------
void ComputeDescriptorSet::Dispatch(
    ID3D12GraphicsCommandList* cmdList,
    UINT threadGroupX, UINT threadGroupY, UINT threadGroupZ,
    const BindingSlot* slots, uint32_t slotCount)
{
    if (!m_shader || !m_shader->GetPSO() || !m_shader->GetRootSignature() || !m_allocator) return;
    if (!slots && slotCount > 0) return;
    if (!ValidateBindings(slots, slotCount)) return;

    uint32_t srvBase, uavBase;
    if (!AllocateTransientTables(srvBase, uavBase)) return;
    WriteDescriptors(slots, slotCount, srvBase, uavBase);

    cmdList->SetPipelineState(m_shader->GetPSO());
    RequestResourceStates(slots, slotCount);                 // may record per-subresource SRV reads
    BindRootParams(cmdList, slots, slotCount, srvBase, uavBase);
    EmitSubresourceReadBarriers(cmdList);                    // UAV -> read on the SRV mip(s), if any
    cmdList->Dispatch(threadGroupX, threadGroupY, threadGroupZ);
    RestoreSubresourceUAVStates(cmdList);                    // read -> UAV (hand back as Unity expects)
    NotifyResourceStates(slots, slotCount);
}
