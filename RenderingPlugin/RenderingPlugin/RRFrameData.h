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

    uint16_t outputWidth;
    uint16_t outputHeight;
    uint16_t currentWidth;
    uint16_t currentHeight;
    float cameraJitter[2];

    int instanceId;
    nri::UpscalerMode upscalerMode;
    
};

#pragma pack(pop)
