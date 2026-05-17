#pragma once
#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Describes one resource binding discovered via DXC reflection.
// ---------------------------------------------------------------------------
enum class BindingType
{
    SRV,            // single StructuredBuffer<T> or ByteAddressBuffer or Texture (SRV)
    UAV,            // single RWTexture2D / RWStructuredBuffer / RWBuffer (UAV)
    CBV,            // ConstantBuffer<T> bound as inline root CBV descriptor
    SRV_ARRAY,      // unbounded array[] bound via BindlessTexture or BindlessBuffer
    UAV_ARRAY,      // unbounded RWTexture2D[] bound via BindlessUAVTexture
    TLAS,           // RaytracingAccelerationStructure
    ROOT_CONSTANTS, // ConstantBuffer<T> pushed via SetComputeRoot32BitConstants
    ROOT_SRV,       // buffer SRV / TLAS promoted to inline root descriptor
};

struct Binding
{
    std::string        name;            // HLSL variable name
    BindingType        type;            // SRV / UAV / CBV / SRV_ARRAY / TLAS / ROOT_CONSTANTS
    uint32_t           space;           // register space
    uint32_t           registerIndex;   // tn / un / bn number
    uint32_t           heapOffset;      // offset within the shared SRV/UAV alloc range
    uint32_t           rootParam;       // root parameter index
    uint32_t           num32BitValues;  // ROOT_CONSTANTS only: total DWORD count from hint
};

// ---------------------------------------------------------------------------
// BindingSlot
//   One slot per reflected binding, passed from C# via IssuePluginEventAndData.
//   Must match NativeRenderPlugin.BindingSlot exactly (Pack=4, 32 bytes).
// ---------------------------------------------------------------------------
#pragma pack(push, 4)
enum class BindingObjectKind : uint32_t
{
    None               = 0,
    AccelStruct        = 1,
    BindlessTexture    = 2,
    BindlessBuffer     = 3,
    RootConstants      = 4,
    NativeBuffer       = 5,
    BindlessUAVTexture = 6,
    NativeGpuBuffer    = 7,
};

struct BindingSlot
{
    uint64_t             resourcePtr;   // ID3D12Resource* (may be 0)
    uint64_t             objectPtr;     // AccelerationStructure* | BindlessTexture* | BindlessBuffer*
    uint32_t             count;         // element count  (StructuredBuffer or typed buffer)
    uint32_t             stride;        // element stride (StructuredBuffer; 0 = raw/typed)
    BindingObjectKind    objectKind;    // what objectPtr points to
    uint32_t             format;        // DXGI_FORMAT for typed buffer UAV/SRV (0 = raw/structured)
}; // 32 bytes
#pragma pack(pop)
