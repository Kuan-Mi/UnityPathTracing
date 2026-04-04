#pragma once

#include <atomic>
#include <unordered_map>
#include <iostream>
#include <dxgi1_6.h>
#include <d3d12.h>
#include "d3dx12.h"
#include "FrameData.h"
 
#include "NRI.h"
#include "Extensions/NRIHelper.h" 
#include "Extensions/NRIWrapperD3D12.h"
#include "Extensions/NRIUpscaler.h" 

#include "dxgi.h"
#include "Unity/IUnityGraphicsD3D12.h"
#include "Unity/IUnityLog.h"

using Microsoft::WRL::ComPtr;
using namespace nri;
#include <dxcapi.h> 
 

class BindlessInstance
{
public:
    void CompileShaderRuntime(const char* source, const wchar_t* entryPoint, const wchar_t* targetProfile);
    BindlessInstance(IUnityInterfaces* interfaces,int instanceId);
    ~BindlessInstance();

    void DispatchCompute( FrameData* data);
    void UpdateResources(const ResourceInput* resources, int count);
    

private:
    static constexpr int kMaxFramesInFlight = 3;

    void CreatePipelineAndLayout();
    void CreateResources();
    void CreateDescriptors();
    void initialize_and_create_resources();
    void release_resources();

    IUnityGraphicsD3D12v8* s_d3d12 = nullptr;
    IUnityLog* s_Log = nullptr;
    int id;
 
    
    std::vector<ResourceInput> m_CachedResources;
    
    uint32_t frameIndex = 0;

    UINT TextureWidth = 0;
    UINT TextureHeight = 0;
 
    std::atomic<bool> m_are_resources_initialized{false};
    
    
    
    ComPtr<IDxcBlob> blob;
    size_t size;
    
    
    
    Device* m_Device;
    CoreInterface* m_NRI;
    
    // Pipeline
    PipelineLayout* m_PipelineLayout = nullptr;
    Pipeline* m_Pipeline = nullptr;
    
    
    // 资源

    // Texture* m_OutputTexture = nullptr;
    
    
    
    // Descriptors
    
    
    nri::Descriptor* m_PrimitiveBufferView = nullptr;
    nri::Descriptor* m_InstanceBufferView = nullptr;
    nri::Descriptor* m_LightOutputView = nullptr;
    
    
    DescriptorPool* m_DescriptorPool = nullptr;
    
    DescriptorSet* m_GlobalDescSets[3];
    
    // DescriptorSet* m_GlobalDescSet = nullptr;   // Set 0: UAV, Sampler
    DescriptorSet* m_BindlessDescSet = nullptr; // Set 1: Bindless Array
    
    
    
    std::vector<nri::Descriptor*> m_InputTextureViews;
    std::vector<ID3D12Resource*> m_InputNativeResources;

    // Descriptor* m_OutputTextureView = nullptr; // UAV
    Descriptor* m_Sampler = nullptr;
    
    const uint32_t MAX_BINDLESS_TEXTURES = 1024;
    const uint32_t TARGET_TEXTURE_INDEX = 42; // 我们将纹理放在这个索引上
    
};
