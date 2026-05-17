#pragma once
#include <d3d12.h>
#include <dxgi1_6.h>
#include <dxcapi.h>
#include <wrl/client.h>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

class ShaderCompilerPlugin
{
public:
    ShaderCompilerPlugin();
    ~ShaderCompilerPlugin();

    bool Initialize();

    // Compile HLSL shader to DXIL bytecode (source must be UTF-8).
    // includeDirs: list of directories to search for #include files (empty = use default DXC handler).
    // extraArgs: additional raw DXC arguments (e.g. L"-disable-payload-qualifiers").
    ComPtr<IDxcBlob> CompileShader(
        const char* source,
        const std::wstring& entryPoint,
        const std::wstring& target,
        const std::vector<std::wstring>& defines = {},
        const std::vector<std::wstring>& includeDirs = {},
        const std::vector<std::wstring>& extraArgs = {}
    );

private:
    ComPtr<IDxcUtils>          m_dxcUtils;
    ComPtr<IDxcCompiler3>      m_dxcCompiler;
    ComPtr<IDxcIncludeHandler> m_includeHandler;
};
