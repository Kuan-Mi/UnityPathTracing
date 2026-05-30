#include "RasterDescriptorSet.h"

// ---------------------------------------------------------------------------
// Constructor
//   Enables the graphics root-binding path, requests SRV inputs in the
//   pixel-shader-readable state, and creates the small CPU-only RTV / DSV heaps
//   used to spin up render-target views per draw.
// ---------------------------------------------------------------------------
RasterDescriptorSet::RasterDescriptorSet(
    RasterShader*            shader,
    ID3D12Device*            device,
    IUnityLog*               log,
    DescriptorHeapAllocator* allocator,
    IUnityGraphicsD3D12v8*   d3d12v8)
    : DescriptorSetBase<RasterShader>(shader, device, log, allocator, d3d12v8)
{
    m_graphicsRootBinding = true;
    // Inputs may be sampled in either the VS or PS; request the combined state.
    m_srvReadState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                     D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

    D3D12_DESCRIPTOR_HEAP_DESC rtv = {};
    rtv.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtv.NumDescriptors = kMaxRTV;
    rtv.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    device->CreateDescriptorHeap(&rtv, IID_PPV_ARGS(&m_rtvHeap));
    m_rtvInc = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    D3D12_DESCRIPTOR_HEAP_DESC dsv = {};
    dsv.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsv.NumDescriptors = 1;
    dsv.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    device->CreateDescriptorHeap(&dsv, IID_PPV_ARGS(&m_dsvHeap));
    m_dsvInc = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
}

// ---------------------------------------------------------------------------
// Draw
// ---------------------------------------------------------------------------
void RasterDescriptorSet::Draw(
    ID3D12GraphicsCommandList* cmdList,
    const RasterDrawDesc& draw,
    const BindingSlot* slots, uint32_t slotCount)
{
    if (!m_shader || !m_shader->GetPSO() || !m_shader->GetRootSignature() || !m_allocator) return;
    if (!m_rtvHeap || !m_dsvHeap) return;
    if (!slots && slotCount > 0) return;
    if (!ValidateBindings(slots, slotCount)) return;

    const uint32_t numRT = (draw.numRenderTargets <= kMaxRTV) ? draw.numRenderTargets : kMaxRTV;

    // --- SRV/UAV input descriptors ---
    uint32_t srvBase, uavBase;
    if (!AllocateTransientTables(srvBase, uavBase)) return;
    WriteDescriptors(slots, slotCount, srvBase, uavBase);

    // --- Resource states: inputs (shared helper) + render targets ---
    RequestResourceStates(slots, slotCount);
    for (uint32_t i = 0; i < numRT; ++i)
        m_tracker.Require(draw.rtvResources[i], D3D12_RESOURCE_STATE_RENDER_TARGET);
    if (draw.dsvResource)
        m_tracker.Require(draw.dsvResource, D3D12_RESOURCE_STATE_DEPTH_WRITE);

    // --- Create RTV / DSV views (consumed immediately by OMSetRenderTargets) ---
    D3D12_CPU_DESCRIPTOR_HANDLE rtvStart = m_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    for (uint32_t i = 0; i < numRT; ++i)
    {
        D3D12_RENDER_TARGET_VIEW_DESC rd = {};
        rd.Format        = static_cast<DXGI_FORMAT>(draw.rtvFormats[i]);
        rd.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        D3D12_CPU_DESCRIPTOR_HANDLE h = rtvStart;
        h.ptr += static_cast<SIZE_T>(i) * m_rtvInc;
        m_device->CreateRenderTargetView(draw.rtvResources[i],
            draw.rtvFormats[i] ? &rd : nullptr, h);
    }

    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = m_dsvHeap->GetCPUDescriptorHandleForHeapStart();
    const bool haveDSV = draw.dsvResource != nullptr;
    if (haveDSV)
    {
        D3D12_DEPTH_STENCIL_VIEW_DESC dd = {};
        dd.Format        = static_cast<DXGI_FORMAT>(draw.dsvFormat);
        dd.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        m_device->CreateDepthStencilView(draw.dsvResource,
            draw.dsvFormat ? &dd : nullptr, dsvHandle);
    }

    // --- Pipeline + root params ---
    cmdList->SetPipelineState(m_shader->GetPSO());
    BindRootParams(cmdList, slots, slotCount, srvBase, uavBase);

    // Constant blend color for the constant-color blend mode (donut bloom composite).
    // All four channels carry the same factor, matching nvrhi::Color(blendFactor); ignored by
    // pipelines whose blend state does not reference BLEND_FACTOR.
    const float blendConst[4] = { draw.blendFactor, draw.blendFactor, draw.blendFactor, draw.blendFactor };
    cmdList->OMSetBlendFactor(blendConst);

    // --- Output merger / rasterizer fixed state ---
    cmdList->OMSetRenderTargets(numRT, numRT ? &rtvStart : nullptr,
                                /*RTsSingleHandleToDescriptorRange=*/TRUE,
                                haveDSV ? &dsvHandle : nullptr);

    if ((draw.clearFlags & 0x1u) != 0)
        for (uint32_t i = 0; i < numRT; ++i)
        {
            D3D12_CPU_DESCRIPTOR_HANDLE h = rtvStart;
            h.ptr += static_cast<SIZE_T>(i) * m_rtvInc;
            cmdList->ClearRenderTargetView(h, draw.clearColor, 0, nullptr);
        }
    if (haveDSV && (draw.clearFlags & 0x2u) != 0)
        cmdList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, draw.clearDepth, 0, 0, nullptr);

    D3D12_VIEWPORT vp = {};
    vp.TopLeftX = draw.viewportX; vp.TopLeftY = draw.viewportY;
    vp.Width    = draw.viewportW; vp.Height   = draw.viewportH;
    vp.MinDepth = 0.0f;           vp.MaxDepth = 1.0f;
    cmdList->RSSetViewports(1, &vp);

    D3D12_RECT sc = {};
    sc.left   = static_cast<LONG>(draw.viewportX);
    sc.top    = static_cast<LONG>(draw.viewportY);
    sc.right  = static_cast<LONG>(draw.viewportX + draw.viewportW);
    sc.bottom = static_cast<LONG>(draw.viewportY + draw.viewportH);
    cmdList->RSSetScissorRects(1, &sc);

    cmdList->IASetPrimitiveTopology(m_shader->GetPrimitiveTopology());

    cmdList->DrawInstanced(draw.vertexCount,
                           draw.instanceCount ? draw.instanceCount : 1, 0, 0);

    // --- Post-draw state notify ---
    NotifyResourceStates(slots, slotCount);
    for (uint32_t i = 0; i < numRT; ++i)
        m_tracker.Notify(draw.rtvResources[i], D3D12_RESOURCE_STATE_RENDER_TARGET);
    if (draw.dsvResource)
        m_tracker.Notify(draw.dsvResource, D3D12_RESOURCE_STATE_DEPTH_WRITE);
}
