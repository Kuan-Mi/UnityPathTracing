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
