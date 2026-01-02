#pragma once
#include <cstdint>
#include <d3d12.h>
#include <NRD.h>
#include <NRDSettings.h>
#include <NRIDescs.h>


#pragma pack(push, 1)

struct RRFrameData
{
    nri::Texture* inputTex;
    nri::Texture* outputTex;
    nri::Texture* mvTex;
    nri::Texture* depthTex;
    nri::Texture* diffuseAlbedoTex;
    nri::Texture* specularAlbedoTex;
    nri::Texture* normalRoughnessTex;
    nri::Texture* specularMvOrHitTex;
    
    float worldToViewMatrix[16];
    float viewToClipMatrix[16];

    uint16_t width;
    uint16_t height;
    
    float cameraJitter[2];

    int instanceId;
};

#pragma pack(pop)
