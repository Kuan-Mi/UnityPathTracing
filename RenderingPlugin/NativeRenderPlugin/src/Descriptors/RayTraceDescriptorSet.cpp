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
    if (m_shader->GetHitGroupCount() == 0) return; // RebuildHitGroupTable not yet run; skip DispatchRays
    if (!slots && slotCount > 0) return;
    if (!ValidateBindings(slots, slotCount)) return;

    uint32_t srvBase, uavBase, samplerBase;
    if (!AllocateTransientTables(srvBase, uavBase, samplerBase)) return;
    WriteDescriptors(slots, slotCount, srvBase, uavBase);
    WriteSamplerDescriptors(slots, slotCount, samplerBase);

    cmdList->SetPipelineState1(m_shader->GetPSO());
    RequestResourceStates(slots, slotCount);
    BindRootParams(cmdList, slots, slotCount, srvBase, uavBase, samplerBase);

    // DispatchRays
    const UINT stride = 64;
    D3D12_DISPATCH_RAYS_DESC drd = {};
    drd.RayGenerationShaderRecord.StartAddress = m_shader->GetRayGenTable()->GetGPUVirtualAddress();
    drd.RayGenerationShaderRecord.SizeInBytes  = stride;
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
