#pragma once
#include "ShaderBase.h"  // pulls in all D3D12/DXC/Unity headers + Binding types

// ---------------------------------------------------------------------------
// RasterPipelineStateDesc
//   Fixed-function graphics-pipeline state supplied by C# at pipeline creation.
//   Must match NativeRenderPlugin.RasterPipelineStateDesc exactly (Pack=4).
//
//   Enum fields use the raw D3D12 enum values; a 0 sentinel selects a sensible
//   default (cull NONE, fill SOLID, depth func LESS_EQUAL, topology TRIANGLE),
//   matching the fullscreen-triangle post-process pass this was built for.
// ---------------------------------------------------------------------------
#pragma pack(push, 4)
struct RasterPipelineStateDesc
{
    uint32_t numRenderTargets;       // 0..8
    uint32_t rtvFormats[8];          // DXGI_FORMAT per render target
    uint32_t dsvFormat;              // DXGI_FORMAT, 0 = no depth target
    uint32_t cullMode;               // D3D12_CULL_MODE (1=NONE,2=FRONT,3=BACK); 0 => NONE
    uint32_t fillMode;               // D3D12_FILL_MODE (2=WIREFRAME,3=SOLID);   0 => SOLID
    uint32_t depthTestEnable;        // bool (0/1)
    uint32_t depthWriteEnable;       // bool (0/1)
    uint32_t depthFunc;              // D3D12_COMPARISON_FUNC; 0 => LESS_EQUAL
    uint32_t blendMode;              // 0=opaque, 1=alpha, 2=additive, 3=premultiplied
    uint32_t primitiveTopology;      // D3D_PRIMITIVE_TOPOLOGY for IASetPrimitiveTopology
                                     // (POINTLIST=1, LINELIST=2, LINESTRIP=3, TRIANGLELIST=4,
                                     //  TRIANGLESTRIP=5); 0 => TRIANGLELIST. The PSO's
                                     //  PrimitiveTopologyType is derived from this.
    uint32_t frontCounterClockwise;  // bool (0/1)
    uint32_t sampleCount;            // MSAA sample count; 0 => 1
};
#pragma pack(pop)

// ---------------------------------------------------------------------------
// RasterShader
//   One self-contained graphics pipeline (vertex + pixel stage).  Binding
//   metadata, root-signature build, logging and hints are provided by
//   ShaderBase, exactly like ComputeShader / RayTraceShader; only the
//   two-stage reflection merge and the graphics PSO build are specific here.
//
//   Resources are reflected from BOTH the VS and PS blobs and merged (deduped
//   by HLSL variable name) into the single shared binding table, so the root
//   signature covers every stage.  Render targets are bound separately at draw
//   time (OMSetRenderTargets) and are NOT part of the reflected binding set.
// ---------------------------------------------------------------------------
class RasterShader : public ShaderBase
{
public:
    RasterShader()  = default;
    ~RasterShader() = default;

    bool Initialize(ID3D12Device5* device, IUnityLog* log, IUnityGraphicsD3D12v8* d3d12v8);

    // Build a graphics pipeline from pre-compiled VS + PS DXIL blobs.
    bool LoadShaderFromBlobs(const uint8_t* vsDxil, uint32_t vsSize,
                             const uint8_t* psDxil, uint32_t psSize,
                             const RasterPipelineStateDesc& state,
                             const char* name = nullptr);

    // --- Accessors for RasterDescriptorSet ---
    ID3D12PipelineState* GetPSO() const { return m_pso.Get(); }
    D3D12_PRIMITIVE_TOPOLOGY GetPrimitiveTopology() const { return m_primTopology; }

private:
    // Reflect one DXIL blob (VS or PS) into the shared binding state without
    // resetting it, deduping by name.  Mirrors ComputeShader::ReflectBindings but
    // is additive so VS and PS contributions merge.
    bool ReflectBlobInto(IDxcBlob* shaderBlob);
    bool BuildPipeline(IDxcBlob* vsBlob, IDxcBlob* psBlob);

    ComPtr<ID3D12PipelineState> m_pso;
    RasterPipelineStateDesc     m_state         = {};
    D3D12_PRIMITIVE_TOPOLOGY    m_primTopology  = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
};
