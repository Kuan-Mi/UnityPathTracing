#pragma once
#include <cstdint>
#include <d3d12.h>

// ===================================================================================
// SLDlssrrFrameData — interop struct mirrored by SLDlssrr.cs (SLDLRR namespace).
//
// Byte-identical to the C# [StructLayout(Sequential, Pack=1)] mirror. Matches the
// RRFrameData.h / DlrrFrameData.cs convention used by the NRI DLSS-RR path, but the
// texture handles here are RAW ID3D12Resource* (RenderTexture native ptr), not NRI
// textures — Streamline tags native D3D12 resources directly (sl::Resource).
//
// Matrices are stored as raw Unity Matrix4x4 bytes (column-major). The native side
// memcpy's them straight into row-major sl::float4x4, which performs the column->row
// transpose Streamline expects (same trick as StreamlineProbe::FrameInputs).
// ===================================================================================

#pragma pack(push, 1)

struct SLDlssrrFrameData
{
    // --- shared per-frame Streamline token (sl::FrameToken*, minted by SL_FrameBegin) ---
    // Passed in from C# so the plugin stays stateless. nullptr in the editor edit-mode game
    // view (no main-thread Reflex tick) — DLSS-RR mints its own token in that case.
    void* frameToken;

    // --- tagged resources (native ID3D12Resource*) ---
    ID3D12Resource* inputTex;           // noisy ray-traced color (render res)
    ID3D12Resource* outputTex;          // denoised/upscaled color (output res)
    ID3D12Resource* mvTex;              // motion vectors (render res)
    ID3D12Resource* depthTex;           // linear view-Z (render res)
    ID3D12Resource* diffuseAlbedoTex;   // diffuse albedo
    ID3D12Resource* specularAlbedoTex;  // specular albedo
    ID3D12Resource* normalRoughnessTex; // normal + roughness (packed in alpha)
    ID3D12Resource* specularMvOrHitTex; // specular motion vectors OR specular hit distance

    // --- D3D12_RESOURCE_STATES each resource is in when SL reads it (SL transitions) ---
    uint32_t inputState;
    uint32_t outputState;
    uint32_t mvState;
    uint32_t depthState;
    uint32_t diffuseAlbedoState;
    uint32_t specularAlbedoState;
    uint32_t normalRoughnessState;
    uint32_t specularMvOrHitState;

    // --- DLSSDOptions matrices (for the specular-hit-distance path) ---
    float worldToViewMatrix[16]; // -> DLSSDOptions::worldToCameraView
    float viewToWorldMatrix[16]; // -> DLSSDOptions::cameraViewToWorld (inverse of above)

    // --- sl::Constants common camera matrices (no jitter baked in) ---
    float cameraViewToClip[16];
    float clipToCameraView[16];
    float clipToPrevClip[16];
    float prevClipToClip[16];

    // --- sl::Constants camera vectors ---
    float cameraPos[3];
    float cameraUp[3];
    float cameraRight[3];
    float cameraFwd[3];

    float cameraJitter[2]; // pixel space
    float mvecScale[2];    // brings mvec into [-1,1]

    float cameraNear;
    float cameraFar;
    float cameraFOV;    // radians
    float cameraAspect;

    uint16_t outputWidth;
    uint16_t outputHeight;
    uint16_t renderWidth;
    uint16_t renderHeight;

    int32_t instanceId;
    int32_t depthInverted;
    int32_t cameraMotionIncluded;
    int32_t motionVectors3D;
    int32_t reset;

    uint8_t useSpecularMotionVector; // 1: specularMvOrHitTex is specular MV; 0: it is hit distance
    uint8_t upscalerMode;            // maps to sl::DLSSMode (see SLDlssrr.cpp mapMode)
    uint8_t preset;                  // 0=default, 4=PresetD, 5=PresetE (sl::DLSSDPreset)
};

#pragma pack(pop)
