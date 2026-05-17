#include "RayTraceDescriptorSet.h"
#include "PluginInternal.h"

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------
void RayTraceDescriptorSet::Dispatch(
    ID3D12GraphicsCommandList4* cmdList,
    UINT width, UINT height,
    const BindingSlot* slots, uint32_t slotCount)
{
    if (!m_shader || !m_shader->GetPSO() || !m_shader->GetRootSignature() || !m_allocator) return;
    if (!m_shader->GetRayGenTable() || !m_shader->GetMissTable() || !m_shader->GetHitGroupTable()) return;
    if (!slots && slotCount > 0) return;
    if (!ValidateBindings(slots, slotCount)) return;

    uint32_t slotIdx;
    AcquireSlot(slotIdx);
    EnsureDescriptors(slots, slotCount, slotIdx);

    cmdList->SetPipelineState1(m_shader->GetPSO());
    BindRootParams(cmdList, slots, slotCount, slotIdx);
    RequestResourceStates(slots, slotCount);

    // DispatchRays
    const UINT stride = D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT;
    D3D12_DISPATCH_RAYS_DESC drd = {};
    drd.RayGenerationShaderRecord.StartAddress = m_shader->GetRayGenTable()->GetGPUVirtualAddress();
    drd.RayGenerationShaderRecord.SizeInBytes  = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    drd.MissShaderTable.StartAddress           = m_shader->GetMissTable()->GetGPUVirtualAddress();
    drd.MissShaderTable.SizeInBytes            = stride * m_shader->GetMissCount();
    drd.MissShaderTable.StrideInBytes          = stride;
    drd.HitGroupTable.StartAddress             = m_shader->GetHitGroupTable()->GetGPUVirtualAddress();
    drd.HitGroupTable.SizeInBytes              = stride * m_shader->GetHitGroupCount();
    drd.HitGroupTable.StrideInBytes            = stride;
    drd.Width  = width;
    drd.Height = height;
    drd.Depth  = 1;
    cmdList->DispatchRays(&drd);
    NotifyResourceStates(slots, slotCount);
}