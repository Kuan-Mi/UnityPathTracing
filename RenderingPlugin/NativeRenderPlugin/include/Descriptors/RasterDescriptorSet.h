#pragma once
#include "DescriptorSetBase.h"
#include "RasterShader.h"

// ---------------------------------------------------------------------------
// RasterDrawDesc
//   One graphics draw's render-target / viewport / draw-arg parameters, filled
//   by the plugin event callback from RAS_RenderEventData.  SRV/CBV/sampler
//   inputs travel separately through the reflected BindingSlot[] array (shared
//   with compute / ray tracing); render targets are NOT reflected resources.
// ---------------------------------------------------------------------------
struct RasterDrawDesc
{
    uint32_t        numRenderTargets;     // 0..8
    ID3D12Resource* rtvResources[8];      // Unity render-target resources
    uint32_t        rtvFormats[8];        // DXGI_FORMAT for each RTV view (must match PSO)
    ID3D12Resource* dsvResource;          // depth target or nullptr
    uint32_t        dsvFormat;            // DXGI_FORMAT for the DSV view
    uint32_t        clearFlags;           // bit0 = clear color, bit1 = clear depth
    float           clearColor[4];
    float           clearDepth;
    float           viewportX, viewportY, viewportW, viewportH;
    uint32_t        vertexCount;
    uint32_t        instanceCount;
};

// ---------------------------------------------------------------------------
// RasterDescriptorSet
//   Binds resources and issues one DrawInstanced for a RasterShader.  Reuses
//   all of DescriptorSetBase<RasterShader> (transient SRV/UAV table, root-param
//   binding, resource-state tracking) with the graphics root-binding path
//   enabled, and adds the render-target machinery (RTV/DSV heaps + OM setup)
//   that compute / ray tracing do not need.
//
//   Lifetime: created via NR_RAS_CreateDescriptorSet /
//             destroyed via NR_RAS_DestroyDescriptorSet (both called from C#).
// ---------------------------------------------------------------------------
class RasterDescriptorSet : public DescriptorSetBase<RasterShader>
{
public:
    RasterDescriptorSet(RasterShader*            shader,
                        ID3D12Device*            device,
                        IUnityLog*               log,
                        DescriptorHeapAllocator* allocator,
                        IUnityGraphicsD3D12v8*   d3d12v8);

    // Execute one DrawInstanced.  All descriptor writing, RTV/DSV setup,
    // resource-state requests and root-parameter setup happen here.
    void Draw(ID3D12GraphicsCommandList* cmdList,
              const RasterDrawDesc& draw,
              const BindingSlot* slots, uint32_t slotCount);

private:
    static constexpr uint32_t kMaxRTV = 8;

    ComPtr<ID3D12DescriptorHeap> m_rtvHeap;  // CPU-only, kMaxRTV slots
    ComPtr<ID3D12DescriptorHeap> m_dsvHeap;  // CPU-only, 1 slot
    UINT m_rtvInc = 0;
    UINT m_dsvInc = 0;
};
