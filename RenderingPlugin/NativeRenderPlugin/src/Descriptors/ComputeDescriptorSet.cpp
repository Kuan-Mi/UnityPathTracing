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

    uint32_t slotIdx;
    AcquireSlot(slotIdx);
    EnsureDescriptors(slots, slotCount, slotIdx);

    cmdList->SetPipelineState(m_shader->GetPSO());
    BindRootParams(cmdList, slots, slotCount, slotIdx);
    RequestResourceStates(slots, slotCount);
    cmdList->Dispatch(threadGroupX, threadGroupY, threadGroupZ);
}