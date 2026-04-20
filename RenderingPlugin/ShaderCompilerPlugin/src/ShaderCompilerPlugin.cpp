/*
 * ShaderCompilerPlugin.cpp  –  ShaderCompilerPlugin
 *
 * Standalone DLL that compiles HLSL to DXIL and exports a pure-C API.
 * Integrates IUnityLog for in-Editor log output when loaded by Unity.
 *
 * Exported API:
 *   bool  NR_SC_Compile(hlslPath, includeDirs, extraArgs, outBytes*, outSize*) – compile HLSL to DXIL
 *   void  NR_SC_Free(ptr)                                           – free the output buffer
 *   (Unity lifecycle) UnityPluginLoad / UnityPluginUnload
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>

#include "ShaderCompilerPlugin.h"

#include "IUnityInterface.h"
#include "IUnityLog.h"

// ---------------------------------------------------------------------------
// Custom include handler: resolves #include by searching a list of directories.
// Falls back to the default DXC include handler for anything not found.
// ---------------------------------------------------------------------------
class CustomIncludeHandler : public IDxcIncludeHandler
{
public:
    CustomIncludeHandler(IDxcUtils* utils, IDxcIncludeHandler* fallback,
                         const std::vector<std::wstring>& includeDirs)
        : m_refCount(1), m_utils(utils), m_fallback(fallback), m_includeDirs(includeDirs)
    {}

    ULONG STDMETHODCALLTYPE AddRef()  override { return ++m_refCount; }
    ULONG STDMETHODCALLTYPE Release() override
    {
        ULONG r = --m_refCount;
        if (r == 0) delete this;
        return r;
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** ppv) override
    {
        if (iid == __uuidof(IDxcIncludeHandler) || iid == __uuidof(IUnknown))
        {
            *ppv = static_cast<IDxcIncludeHandler*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }

    HRESULT STDMETHODCALLTYPE LoadSource(LPCWSTR pFilename, IDxcBlob** ppIncludeSource) override
    {
        for (const auto& dir : m_includeDirs)
        {
            std::filesystem::path path = std::filesystem::path(dir) / pFilename;
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file.is_open()) continue;

            auto size = file.tellg();
            file.seekg(0);
            std::string content(static_cast<size_t>(size), '\0');
            file.read(content.data(), size);

            ComPtr<IDxcBlobEncoding> blob;
            HRESULT hr = m_utils->CreateBlob(content.data(),
                static_cast<UINT32>(content.size()), DXC_CP_UTF8, &blob);
            if (SUCCEEDED(hr))
            {
                *ppIncludeSource = blob.Detach();
                return S_OK;
            }
        }
        return m_fallback->LoadSource(pFilename, ppIncludeSource);
    }

private:
    ULONG                     m_refCount;
    IDxcUtils*                m_utils;    // non-owning
    IDxcIncludeHandler*       m_fallback; // non-owning
    std::vector<std::wstring> m_includeDirs;
};

// ---------------------------------------------------------------------------
// Unity lifecycle – register IUnityLog when loaded by Unity
// ---------------------------------------------------------------------------
static IUnityLog* s_Log = nullptr;

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
UnityPluginLoad(IUnityInterfaces* unityInterfaces)
{
    s_Log = unityInterfaces ? unityInterfaces->Get<IUnityLog>() : nullptr;
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
UnityPluginUnload()
{
    s_Log = nullptr;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static void SCLog(UnityLogType type, const char* msg)
{
    if (s_Log)
    {
        s_Log->Log(type, msg, __FILE__, __LINE__);
        return;
    }
    OutputDebugStringA("[ShaderCompilerPlugin] ");
    OutputDebugStringA(msg);
    OutputDebugStringA("\n");
    printf("[ShaderCompilerPlugin] %s\n", msg);
}

static void SCLogInfo (const char* msg) { SCLog(kUnityLogTypeLog,     msg); }
static void SCLogWarn (const char* msg) { SCLog(kUnityLogTypeWarning, msg); }
static void SCLogError(const char* msg) { SCLog(kUnityLogTypeError,   msg); }

// ---------------------------------------------------------------------------
// ShaderCompilerPlugin methods
// ---------------------------------------------------------------------------

ShaderCompilerPlugin::ShaderCompilerPlugin() = default;
ShaderCompilerPlugin::~ShaderCompilerPlugin() = default;

bool ShaderCompilerPlugin::Initialize()
{
    HRESULT hr = DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&m_dxcUtils));
    if (FAILED(hr))
    {
        SCLogError("Failed to create DxcUtils");
        return false;
    }

    hr = DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&m_dxcCompiler));
    if (FAILED(hr))
    {
        SCLogError("Failed to create DxcCompiler");
        return false;
    }

    hr = m_dxcUtils->CreateDefaultIncludeHandler(&m_includeHandler);
    if (FAILED(hr))
    {
        SCLogError("Failed to create include handler");
        return false;
    }

    return true;
}

ComPtr<IDxcBlob> ShaderCompilerPlugin::CompileShader(
    const char* source,
    const std::wstring& entryPoint,
    const std::wstring& target,
    const std::vector<std::wstring>& defines,
    const std::vector<std::wstring>& includeDirs,
    const std::vector<std::wstring>& extraArgs)
{
    ComPtr<IDxcBlobEncoding> sourceBlob;
    HRESULT hr = m_dxcUtils->CreateBlob(
        source,
        static_cast<UINT32>(strlen(source)),
        DXC_CP_UTF8,
        &sourceBlob
    );
    if (FAILED(hr))
    {
        SCLogError("Failed to create DXC source blob");
        return nullptr;
    }

    std::vector<LPCWSTR> arguments;
    arguments.push_back(L"-E");
    arguments.push_back(entryPoint.c_str());
    arguments.push_back(L"-T");
    arguments.push_back(target.c_str());
    arguments.push_back(DXC_ARG_WARNINGS_ARE_ERRORS);
    arguments.push_back(DXC_ARG_ALL_RESOURCES_BOUND);

#ifdef _DEBUG
    arguments.push_back(DXC_ARG_DEBUG);
    arguments.push_back(DXC_ARG_SKIP_OPTIMIZATIONS);
#else
    arguments.push_back(DXC_ARG_OPTIMIZATION_LEVEL3);
#endif

    for (const auto& define : defines)
    {
        arguments.push_back(L"-D");
        arguments.push_back(define.c_str());
    }

    for (const auto& dir : includeDirs)
    {
        arguments.push_back(L"-I");
        arguments.push_back(dir.c_str());
    }

    for (const auto& arg : extraArgs)
        arguments.push_back(arg.c_str());

    DxcBuffer sourceBuffer;
    sourceBuffer.Ptr      = sourceBlob->GetBufferPointer();
    sourceBuffer.Size     = sourceBlob->GetBufferSize();
    sourceBuffer.Encoding = 0;

    ComPtr<CustomIncludeHandler> customHandler;
    IDxcIncludeHandler* activeHandler = m_includeHandler.Get();
    if (!includeDirs.empty())
    {
        customHandler  = new CustomIncludeHandler(m_dxcUtils.Get(), m_includeHandler.Get(), includeDirs);
        activeHandler  = customHandler.Get();
    }

    ComPtr<IDxcResult> result;
    hr = m_dxcCompiler->Compile(
        &sourceBuffer,
        arguments.data(),
        static_cast<UINT32>(arguments.size()),
        activeHandler,
        IID_PPV_ARGS(&result)
    );

    if (FAILED(hr))
    {
        char buf[512];
        snprintf(buf, sizeof(buf), "DXC Compile() failed (0x%08X)", static_cast<unsigned>(hr));
        SCLogError(buf);
        return nullptr;
    }

    HRESULT compileStatus;
    result->GetStatus(&compileStatus);
    if (FAILED(compileStatus))
    {
        ComPtr<IDxcBlobEncoding> errorBlob;
        result->GetErrorBuffer(&errorBlob);
        if (errorBlob && errorBlob->GetBufferSize() > 0)
            SCLogError(static_cast<const char*>(errorBlob->GetBufferPointer()));
        return nullptr;
    }

    ComPtr<IDxcBlob> shaderBlob;
    result->GetResult(&shaderBlob);
    return shaderBlob;
}

// Convert UTF-8 to wstring
static std::wstring Utf8ToWide(const char* s, int len = -1)
{
    int wlen = MultiByteToWideChar(CP_UTF8, 0, s, len, nullptr, 0);
    std::wstring ws(wlen, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s, len, ws.data(), wlen);
    if (!ws.empty() && ws.back() == L'\0') ws.pop_back();
    return ws;
}

// Parse a semicolon-separated UTF-8 string into vector<wstring>
static std::vector<std::wstring> ParseSemicolonList(const char* list)
{
    std::vector<std::wstring> result;
    if (!list || list[0] == '\0') return result;
    const char* s = list;
    while (*s)
    {
        const char* e = s;
        while (*e && *e != ';') ++e;
        if (e > s)
            result.push_back(Utf8ToWide(s, static_cast<int>(e - s)));
        s = (*e == ';') ? e + 1 : e;
    }
    return result;
}

// Parse semicolon-separated UTF-8 include dirs into vector<wstring>
static std::vector<std::wstring> ParseIncludeDirs(const char* shaderPath, const char* includeDirs)
{
    std::vector<std::wstring> dirs;

    // Auto-prepend the shader file's own directory
    if (shaderPath && shaderPath[0] != '\0')
    {
        std::string p(shaderPath);
        size_t slash = p.find_last_of("/\\");
        if (slash != std::string::npos)
            dirs.push_back(Utf8ToWide(p.substr(0, slash).c_str()));
    }

    if (includeDirs && includeDirs[0] != '\0')
    {
        auto extra = ParseSemicolonList(includeDirs);
        dirs.insert(dirs.end(), extra.begin(), extra.end());
    }
    return dirs;
}

// ---------------------------------------------------------------------------
// Module-level ShaderCompilerPlugin instance (initialized once on first use)
// ---------------------------------------------------------------------------
static ShaderCompilerPlugin s_Plugin;
static bool                 s_Initialized = false;

static bool EnsureInitialized()
{
    if (s_Initialized) return true;
    if (!s_Plugin.Initialize())
    {
        SCLogError("Failed to initialize ShaderCompilerPlugin (DXC unavailable?)");
        return false;
    }
    s_Initialized = true;
    return true;
}

// ---------------------------------------------------------------------------
// Exported API
// ---------------------------------------------------------------------------

extern "C" __declspec(dllexport)
bool NR_SC_Compile(
    const char* hlslPath,
    const char* includeDirs,
    const char* defines,
    const char* extraArgs,
    uint8_t**   outBytes,
    uint32_t*   outSize)
{
    if (!outBytes || !outSize)
    {
        SCLogError("NR_SC_Compile: null output pointers");
        return false;
    }
    *outBytes = nullptr;
    *outSize  = 0;

    if (!hlslPath || hlslPath[0] == '\0')
    {
        SCLogError("NR_SC_Compile: empty hlslPath");
        return false;
    }

    if (!EnsureInitialized()) return false;

    // Read source file
    std::ifstream file(hlslPath, std::ios::binary);
    if (!file.is_open())
    {
        char msg[512];
        snprintf(msg, sizeof(msg), "NR_SC_Compile: cannot open '%s'", hlslPath);
        SCLogError(msg);
        return false;
    }
    std::string source((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    std::vector<std::wstring> dirs     = ParseIncludeDirs(hlslPath, includeDirs);
    std::vector<std::wstring> defs     = ParseSemicolonList(defines);
    std::vector<std::wstring> dxcArgs  = ParseSemicolonList(extraArgs);

    ComPtr<IDxcBlob> blob = s_Plugin.CompileShader(
        source.c_str(), L"", L"lib_6_9", defs, dirs, dxcArgs);

    if (!blob)
    {
        SCLogError("NR_SC_Compile: compilation failed");
        return false;
    }

    const SIZE_T size = blob->GetBufferSize();
    uint8_t* buf = static_cast<uint8_t*>(malloc(size));
    if (!buf)
    {
        SCLogError("NR_SC_Compile: out of memory");
        return false;
    }
    memcpy(buf, blob->GetBufferPointer(), size);

    *outBytes = buf;
    *outSize  = static_cast<uint32_t>(size);
    return true;
}

// ---------------------------------------------------------------------------
// NR_SC_CompileCS
//   Compiles a compute shader HLSL file to DXIL bytecode with a caller-
//   specified entry point and target profile (e.g. "main", "cs_6_6").
//   Otherwise identical to NR_SC_Compile.
// ---------------------------------------------------------------------------
extern "C" __declspec(dllexport)
bool NR_SC_CompileCS(
    const char* hlslPath,
    const char* entryPoint,
    const char* target,
    const char* includeDirs,
    const char* defines,
    const char* extraArgs,
    uint8_t**   outBytes,
    uint32_t*   outSize)
{
    if (!outBytes || !outSize)
    {
        SCLogError("NR_SC_CompileCS: null output pointers");
        return false;
    }
    *outBytes = nullptr;
    *outSize  = 0;

    if (!hlslPath || hlslPath[0] == '\0')
    {
        SCLogError("NR_SC_CompileCS: empty hlslPath");
        return false;
    }
    if (!entryPoint || entryPoint[0] == '\0')
    {
        SCLogError("NR_SC_CompileCS: empty entryPoint");
        return false;
    }
    if (!target || target[0] == '\0')
    {
        SCLogError("NR_SC_CompileCS: empty target");
        return false;
    }

    if (!EnsureInitialized()) return false;

    // Read source file
    std::ifstream file(hlslPath, std::ios::binary);
    if (!file.is_open())
    {
        char msg[512];
        snprintf(msg, sizeof(msg), "NR_SC_CompileCS: cannot open '%s'", hlslPath);
        SCLogError(msg);
        return false;
    }
    std::string source((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    std::vector<std::wstring> dirs    = ParseIncludeDirs(hlslPath, includeDirs);
    std::vector<std::wstring> defs    = ParseSemicolonList(defines);
    std::vector<std::wstring> dxcArgs = ParseSemicolonList(extraArgs);

    std::wstring wEntry  = Utf8ToWide(entryPoint);
    std::wstring wTarget = Utf8ToWide(target);

    ComPtr<IDxcBlob> blob = s_Plugin.CompileShader(
        source.c_str(), wEntry, wTarget, defs, dirs, dxcArgs);

    if (!blob)
    {
        SCLogError("NR_SC_CompileCS: compilation failed");
        return false;
    }

    const SIZE_T size = blob->GetBufferSize();
    uint8_t* buf = static_cast<uint8_t*>(malloc(size));
    if (!buf)
    {
        SCLogError("NR_SC_CompileCS: out of memory");
        return false;
    }
    memcpy(buf, blob->GetBufferPointer(), size);

    *outBytes = buf;
    *outSize  = static_cast<uint32_t>(size);
    return true;
}

extern "C" __declspec(dllexport)
void NR_SC_Free(uint8_t* ptr)
{
    free(ptr);
}
