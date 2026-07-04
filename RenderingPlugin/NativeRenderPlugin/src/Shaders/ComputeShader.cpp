#include "ComputeShader.h"
#include <cstdio>
#include <cstdarg>

bool ComputeShader::Initialize(ID3D12Device5* device, IUnityLog* log,
                               IUnityGraphicsD3D12v8* d3d12v8)
{
    m_log     = log;
    m_device  = device;
    m_d3d12v8 = d3d12v8;
    return true;
}

void ComputeShader::Log(UnityLogType type, const char* msg) const
{
    if (m_log) m_log->Log(type, msg, __FILE__, __LINE__);
    else       printf("[ComputeShader] %s\n", msg);
}

void ComputeShader::Logf(UnityLogType type, const char* fmt, ...) const
{
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    Log(type, buf);
}

bool ComputeShader::LoadShaderFromBytes(const uint8_t* dxilBytes, uint32_t size,
                                        const char* debugName,
                                        const char* entryName)
{
    m_name      = (debugName && debugName[0]) ? debugName : "ComputeShader";
    m_entryName = (entryName && entryName[0]) ? entryName : "main";

    if (!dxilBytes || size == 0)
    {
        Log(kUnityLogTypeError, "ComputeShader::LoadShaderFromBytes: empty input");
        return false;
    }

    ComPtr<IDxcUtils> utils;
    if (FAILED(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils))))
    {
        Log(kUnityLogTypeError, "ComputeShader::LoadShaderFromBytes: failed to create IDxcUtils");
        return false;
    }

    ComPtr<IDxcBlobEncoding> blobEnc;
    if (FAILED(utils->CreateBlob(dxilBytes, size, DXC_CP_ACP, &blobEnc)))
    {
        Log(kUnityLogTypeError, "ComputeShader::LoadShaderFromBytes: failed to create blob");
        return false;
    }

    m_shaderBlob = blobEnc;
    Logf(kUnityLogTypeLog, "ComputeShader '%s': shader handle ready (%u bytes, entry '%s')",
         m_name.c_str(), size, m_entryName.c_str());
    return true;
}
