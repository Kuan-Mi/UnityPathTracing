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
    uint32_t           arrayCount;      // SRV/UAV: descriptors this binding occupies (1 = single,
                                        // N = bounded array e.g. RWTexture2D u[N]). Always 1 for
                                        // CBV/TLAS/ROOT_*; unbounded arrays use SRV_ARRAY/UAV_ARRAY.
};

// ---------------------------------------------------------------------------
// BindingSlot
//   One slot per reflected binding, passed from C# via IssuePluginEventAndData.
//   Must match NativeRenderPlugin.CS_BindingSlot exactly (Pack=4, 24 bytes).
//
//   A single `objectPtr` holds whatever the binding points at; objectKind tells
//   the plugin how to interpret it (see BindingObjectKind).  There is no separate
//   raw-resource field: a raw Unity ID3D12Resource* is carried in the same
//   pointer with objectKind == None.
// ---------------------------------------------------------------------------
#pragma pack(push, 4)
enum class BindingObjectKind : uint32_t
{
    None               = 0, // objectPtr is a raw ID3D12Resource* (Unity texture/buffer; 0 = unset)
    AccelStruct        = 1, // objectPtr is an AccelerationStructure*
    BindlessTexture    = 2, // objectPtr is a BindlessTexture*
    BindlessBuffer     = 3, // objectPtr is a BindlessBuffer*
    RootConstants      = 4, // objectPtr is a raw pointer to the 32-bit-constants payload
    NativeBuffer       = 5, // objectPtr is an INativeResource* (volatile / DEFAULT-staged / UAV buffer)
    BindlessUAVTexture = 6, // objectPtr is a BindlessUAVTexture*
};

struct BindingSlot
{
    uint64_t             objectPtr;     // resource/wrapper/payload pointer, interpreted per objectKind
    uint32_t             count;         // element count  (StructuredBuffer or typed buffer)
    uint32_t             stride;        // element stride (StructuredBuffer; 0 = raw/typed)
    BindingObjectKind    objectKind;    // how objectPtr is interpreted
    uint32_t             format;        // DXGI_FORMAT for typed buffer UAV/SRV (0 = raw/structured)
}; // 24 bytes
#pragma pack(pop)
