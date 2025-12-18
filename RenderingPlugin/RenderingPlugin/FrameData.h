#pragma once
#include <cstdint>
#include <d3d12.h>
#include <NRD.h>
#include <NRDSettings.h>
#include <NRIDescs.h>

#pragma pack(push, 1)
namespace nri
{
    struct Texture;
}

struct FrameData
{
    nrd::CommonSettings commonSettings;
    nrd::SigmaSettings sigmaSettings;
    nrd::ReblurSettings reblurSettings;

    uint16_t width;
    uint16_t height;

    ID3D12Resource* mvPointer;
    ID3D12Resource* normalRoughnessPointer;
    ID3D12Resource* viewZPointer;
    ID3D12Resource* penumbraPointer;
    ID3D12Resource* shadowTranslucencyPointer;
    ID3D12Resource* diffRadiancePointer;
    ID3D12Resource* outDiffRadiancePointer;
    ID3D12Resource* validationPointer;

    int instanceId;
};

struct NriResourceState
{
    nri::AccessBits accessBits;
    uint32_t layout;
    nri::StageBits stageBits;
};

struct NrdResourceInput
{
    nrd::ResourceType type;
    nri::Texture* texture;
    NriResourceState state;
};
#pragma pack(pop)
