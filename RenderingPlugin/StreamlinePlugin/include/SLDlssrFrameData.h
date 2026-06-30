#pragma once
#include <cstdint>
#include <d3d12.h>

// ===================================================================================
// SLDlssrFrameData — interop struct mirrored by SLDlssr.cs (SLDLRR namespace).
//
// DLSS Super Resolution (kFeatureDLSS) counterpart of SLDlssrrFrameData.h. Byte-
// identical to the C# [StructLayout(Sequential, Pack=1)] mirror. Matches the
// DLSRFrameData.h / DlsrUpscaler.cs convention used by the NRI DLSS-SR path, but the
// texture handles here are RAW ID3D12Resource* (RenderTexture native ptr), not NRI
// textures — Streamline tags native D3D12 resources directly (sl::Resource).
//
// SR needs fewer guides than RR: input color, motion vectors, depth and (optional)
// exposure only — no albedo / normal-roughness / specular-hit-distance buffers.
//
// Matrices are stored as raw Unity Matrix4x4 bytes (column-major). The native side
// memcpy's them straight into row-major sl::float4x4, which performs the column->row
// transpose Streamline expects (same trick as SLDlssrr / StreamlineProbe).
// ===================================================================================

#pragma pack(push, 1)

struct SLDlssrFrameData
{
    // --- shared per-frame Streamline token (sl::FrameToken*, minted by SL_GetNewFrameToken) ---
    // Passed in from C# so the plugin stays stateless. nullptr in the editor edit-mode game
    // view (no main-thread Reflex tick) — DLSS-SR mints its own token in that case.
    void* frameToken;

    // --- tagged resources (native ID3D12Resource*) ---
    ID3D12Resource* inputTex;    // input color (render res)
    ID3D12Resource* outputTex;   // upscaled color (output res)
    ID3D12Resource* mvTex;       // motion vectors (render res)
    ID3D12Resource* depthTex;    // depth (render res)
    ID3D12Resource* exposureTex; // optional 1x1 exposure, may be null

    // --- D3D12_RESOURCE_STATES each resource is in when SL reads it (SL transitions) ---
    uint32_t inputState;
    uint32_t outputState;
    uint32_t mvState;
    uint32_t depthState;
    uint32_t exposureState;

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

    uint8_t upscalerMode; // maps to sl::DLSSMode (see SLDlssr.cpp MapMode)
    uint8_t preset;       // maps to sl::DLSSPreset (see SLDlssr.cpp MapPreset)
};

#pragma pack(pop)
