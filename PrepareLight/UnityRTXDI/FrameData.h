#pragma once
#include <cstdint>
#include <d3d12.h>
#include <NRIDescs.h>

#pragma pack(push, 1)
namespace nri
{
    struct Texture;
}

struct FrameData
{
    nri::Buffer* primitiveBuffer; // StructuredBuffer<PrimitiveData>
    nri::Buffer* instanceBuffer; // StructuredBuffer<InstanceData>
    nri::Buffer* lightDataBuffer; // RWStructuredBuffer<RAB_LightInfo>

    int numPrimitives; // 用于 Dispatch 大小和 PushConstants
    int InstanceCount; // 用于 Dispatch 大小和 PushConstants

    int instanceId;
};

struct ResourceInput
{
    nri::Texture* texture;
    nri::Format format;
};
#pragma pack(pop)
