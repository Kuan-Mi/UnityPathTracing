// SLDlssg.h
// DLSS-G (Frame Generation) via Streamline, hosted in the StreamlinePlugin plugin alongside
// DLSS-RR. Ported from the proven StreamlineProbe machinery, but fed REAL path-tracer
// inputs (depth + motion vectors + camera constants; HUDLessColor omitted for now — SL
// interpolates the presented backbuffer).
//
// Unlike DLSS-RR (evaluate-time), DLSS-G takes over the PRESENT path: it needs the SL
// device-queue + swapchain proxies installed BEFORE Unity creates them, so the plugin
// must be load-on-startup and these hooks are installed only in the PLAYER (the editor
// auto-detect in StreamlinePlugin.cpp skips them — queue proxying crashes the editor).
//
// slInit/slSetD3DDevice/slShutdown + logging are shared via SLCore (which loads
// DLSS_G/Reflex/PCL alongside DLSS_RR). See SLCore.h.
#pragma once

struct IDXGISwapChain3;
namespace sl { struct FrameToken; }

namespace SLDlssg
{
    // Real per-frame DLSS-G inputs from the path tracer. Native ID3D12Resource* pointers;
    // matrices are raw Unity Matrix4x4 bytes (column-major) → memcpy'd into row-major
    // sl::float4x4 (transpose SL expects). Layout MUST match the C# mirror exactly.
    struct FrameInputs
    {
        void*    depth;          // linear view-Z (render res)
        void*    motionVectors;  // screen-space motion vectors (render res)
        void*    frameToken;     // this frame's sl::FrameToken* (from SL_GetNewFrameToken)
        unsigned depthState, mvecState;  // D3D12_RESOURCE_STATES
        unsigned mvecDepthW, mvecDepthH; // render res (extent of depth/mvec)
        unsigned colorW, colorH;         // output/display res (full frame extent)

        float    cameraViewToClip[16];
        float    clipToCameraView[16];
        float    clipToPrevClip[16];
        float    prevClipToClip[16];

        float    cameraPos[3], cameraUp[3], cameraRight[3], cameraFwd[3];
        float    jitterX, jitterY;       // pixels
        float    mvecScaleX, mvecScaleY; // normalize mvec to [-1,1]
        float    cameraNear, cameraFar, cameraFOV, cameraAspect;
        int      depthInverted;
        int      cameraMotionIncluded;
        int      motionVectors3D;
        int      reset;
    };

    // Tag real depth/mvec + set constants for the shared frame token (render thread, after
    // SLCore::GetNewFrameToken). The frame token + Reflex sleep are owned by SLCore/SLReflex now.
    void ConsumeFrameInputs(const FrameInputs& inputs);

    // Called by SLHooks from the SL proxy swapchain Present hook.
    void OnPresentPre(IDXGISwapChain3* proxySwapchain);
    void OnPresentPost();
    void OnSwapChainAdopted(unsigned width, unsigned height);

    // Runtime FG on/off (applied on the present thread).
    void SetFrameGeneration(bool enable);
    bool IsFrameGenerationOn();

    void Shutdown();
}
