#pragma once

#include <atomic>
#include <unordered_map>
#include <iostream>
#include <dxgi1_6.h>
#include <d3d12.h>
#include "d3dx12.h"
#include "FrameData.h"

#include "NRD.h"
#include "NRDDescs.h"
#include "NRI.h"
#include "Extensions/NRIHelper.h" 
#include "Extensions/NRIWrapperD3D12.h"
#include "Extensions/NRIUpscaler.h"
#include "NRDIntegration.h"

#include "dxgi.h"
#include "Unity/IUnityGraphicsD3D12.h"
#include "Unity/IUnityLog.h"

class NrdInstance
{
public:
    NrdInstance(IUnityInterfaces* interfaces,int instanceId);
    ~NrdInstance();

    void DispatchCompute( FrameData* data);
    void UpdateResources(const NrdResourceInput* resources, int count);
    

private:
    static constexpr int kMaxFramesInFlight = 3;

    // void UpdateNrdSettings(const FrameData* data);
    void CreateNrd();
    void initialize_and_create_resources();
    void release_resources();

    IUnityGraphicsD3D12v8* s_d3d12 = nullptr;
    IUnityLog* s_Log = nullptr;
    int id;

    // NRD
    nrd::Integration m_NrdIntegration = {};
    
    std::vector<NrdResourceInput> m_CachedResources;
    
    uint32_t frameIndex = 0;

    UINT TextureWidth = 0;
    UINT TextureHeight = 0;

    nrd::Identifier m_SigmaId = 0;
    nrd::Identifier m_ReblurId = 0;
    std::atomic<bool> m_are_resources_initialized{false};
    nri::Upscaler* m_DLRR = nullptr;
};
