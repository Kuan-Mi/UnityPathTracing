/*
 * OMMBakerPlugin.cpp  –  Standalone CPU OMM Baker Plugin
 *
 * A minimal, self-contained DLL that wraps the OMM SDK CPU baker.
 * It has NO dependency on D3D12 or the Unity rendering API — it can be
 * loaded in the Editor without an active graphics device.
 *
 * Exported API:
 *   NR_BakeOMMCPU           – run ommCpuBake, store result in static memory
 *   NR_BakeGetArrayDataSize – query byte size of baked array data blob
 *   NR_BakeCopyArrayData    – copy array data blob to caller-supplied buffer
 *   NR_BakeGetDescArrayByteCount – query byte size of desc array blob
 *   NR_BakeGetDescArrayCount     – number of ommCpuOpacityMicromapDesc entries
 *   NR_BakeCopyDescArray    – copy desc array blob
 *   NR_BakeGetIndexCount    – number of per-triangle OMM indices
 *   NR_BakeGetIndexStride   – index stride in bytes (1, 2, or 4)
 *   NR_BakeCopyIndexBuffer  – copy index buffer blob
 *   NR_BakeGetHistogramCount     – number of histogram entries
 *   NR_BakeCopyHistogramFlat     – copy flat histogram (uint32[N*3])
 *   NR_FreeBakeResult       – free static bake result storage
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "omm.h"

// ---------------------------------------------------------------------------
// Logging (console only – no Unity runtime dependency)
// ---------------------------------------------------------------------------
static void BakerLog(const char* level, const char* msg)
{
    printf("[NativeOMMBaker] [%s] %s\n", level, msg);
}

#define BAKER_LOG(msg)   BakerLog("INFO",    msg)
#define BAKER_WARN(msg)  BakerLog("WARN",    msg)
#define BAKER_ERROR(msg) BakerLog("ERROR",   msg)

// ---------------------------------------------------------------------------
// Static baker state  (one instance per process; suitable for Editor use)
// ---------------------------------------------------------------------------
static ommBaker s_Baker = nullptr;

struct BakeResult
{
    std::vector<uint8_t>  arrayData;
    std::vector<uint8_t>  descArray;     // ommCpuOpacityMicromapDesc bytes (8 each)
    uint32_t              descArrayCount = 0;
    std::vector<uint8_t>  indexBuffer;
    uint32_t              indexCount  = 0;
    uint32_t              indexStride = 2;
    std::vector<uint32_t> histogramFlat; // [count, subdivLevel, format] * N
    bool                  valid = false;
};

static BakeResult s_Result;

// ---------------------------------------------------------------------------
// DLL lifecycle
// ---------------------------------------------------------------------------
BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_DETACH)
    {
        if (s_Baker)
        {
            ommDestroyBaker(s_Baker);
            s_Baker = nullptr;
        }
        s_Result = {};
    }
    return TRUE;
}

// ---------------------------------------------------------------------------
// Lazy baker creation
// ---------------------------------------------------------------------------
static bool EnsureBaker()
{
    if (s_Baker) return true;

    ommBakerCreationDesc desc = ommBakerCreationDescDefault();
    desc.type = ommBakerType_CPU;
    if (ommCreateBaker(&desc, &s_Baker) != ommResult_SUCCESS)
    {
        BAKER_ERROR("ommCreateBaker failed");
        s_Baker = nullptr;
        return false;
    }
    BAKER_LOG("CPU baker created");
    return true;
}

// ---------------------------------------------------------------------------
// NR_BakeOMMCPU
//   Runs ommCpuBake synchronously and stores blobs in static storage.
//   alphaPixels : R8_UNORM bytes (texW * texH bytes).
//   uvs         : float2 per vertex (vertexCount * 2 floats), packed.
//                 Indexed by vertex index (same as a standard vertex UV buffer).
//   indices     : CPU index buffer (indexCount * indexStride bytes).
//   indexStride : 2 (16-bit) or 4 (32-bit).
//   alphaCutoff : alpha threshold [0, 1].
//   Returns 1 on success, 0 on failure.
// ---------------------------------------------------------------------------
extern "C" __declspec(dllexport)
int NR_BakeOMMCPU(
    const void* alphaPixels, uint32_t texW, uint32_t texH,
    const void* uvs,
    const void* indices, uint32_t indexCount, uint32_t indexStride,
    float alphaCutoff,
    uint8_t maxSubdivisionLevel,
    float dynamicSubdivisionScale,  // 0 = disabled (uniform), >0 = dynamic
    uint8_t format)                 // 1 = OC1_2_State, 2 = OC1_4_State
{
    if (!alphaPixels || !uvs || !indices || texW == 0 || texH == 0 || indexCount == 0)
    {
        BAKER_ERROR("NR_BakeOMMCPU: invalid arguments");
        return 0;
    }

    if (!EnsureBaker()) return 0;

    s_Result = {};

    // 1. Create texture
    ommCpuTextureMipDesc mipDesc = ommCpuTextureMipDescDefault();
    mipDesc.width       = texW;
    mipDesc.height      = texH;
    mipDesc.rowPitch    = texW;  // R8_UNORM: 1 byte/pixel
    mipDesc.textureData = alphaPixels;

    ommCpuTextureDesc texDesc = ommCpuTextureDescDefault();
    texDesc.format           = ommCpuTextureFormat_UNORM8;
    texDesc.mipCount         = 1;
    texDesc.mips             = &mipDesc;
    texDesc.alphaCutoff      = alphaCutoff;

    ommCpuTexture vmTex = nullptr;
    if (ommCpuCreateTexture(s_Baker, &texDesc, &vmTex) != ommResult_SUCCESS)
    {
        BAKER_ERROR("NR_BakeOMMCPU: ommCpuCreateTexture failed");
        return 0;
    }

    // 2. Bake
    ommCpuBakeInputDesc bakeDesc = ommCpuBakeInputDescDefault();
    bakeDesc.bakeFlags                       = ommCpuBakeFlags_EnableInternalThreads;
    bakeDesc.texture                           = vmTex;
    bakeDesc.alphaMode                         = ommAlphaMode_Test;
    bakeDesc.runtimeSamplerDesc.addressingMode = ommTextureAddressMode_Wrap;
    bakeDesc.runtimeSamplerDesc.filter         = ommTextureFilterMode_Linear;
    bakeDesc.alphaCutoff                       = alphaCutoff;
    bakeDesc.format                            = (format == 1) ? ommFormat_OC1_2_State : ommFormat_OC1_4_State;
    bakeDesc.dynamicSubdivisionScale           = dynamicSubdivisionScale;
    bakeDesc.maxSubdivisionLevel               = maxSubdivisionLevel;
    bakeDesc.texCoordFormat                    = ommTexCoordFormat_UV32_FLOAT;
    bakeDesc.texCoords                         = uvs;
    bakeDesc.texCoordStrideInBytes             = 0;  // packed float2
    bakeDesc.indexFormat                       = (indexStride == 4)
                                                   ? ommIndexFormat_UINT_32
                                                   : ommIndexFormat_UINT_16;
    bakeDesc.indexBuffer                       = indices;
    bakeDesc.indexCount                        = indexCount;

    ommCpuBakeResult bakeResult = nullptr;
    ommResult res = ommCpuBake(s_Baker, &bakeDesc, &bakeResult);
    ommCpuDestroyTexture(s_Baker, vmTex);

    if (res != ommResult_SUCCESS)
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "NR_BakeOMMCPU: ommCpuBake failed (result=%d)", (int)res);
        BAKER_ERROR(buf);
        return 0;
    }

    const ommCpuBakeResultDesc* rd = nullptr;
    ommCpuGetBakeResultDesc(bakeResult, &rd);

    // 3. Copy blobs into static storage
    if (rd->arrayDataSize > 0)
        s_Result.arrayData.assign(
            static_cast<const uint8_t*>(rd->arrayData),
            static_cast<const uint8_t*>(rd->arrayData) + rd->arrayDataSize);

    s_Result.descArrayCount = rd->descArrayCount;
    if (rd->descArrayCount > 0)
    {
        const size_t bytes = rd->descArrayCount * sizeof(ommCpuOpacityMicromapDesc);
        s_Result.descArray.assign(
            reinterpret_cast<const uint8_t*>(rd->descArray),
            reinterpret_cast<const uint8_t*>(rd->descArray) + bytes);
    }

    s_Result.indexCount  = rd->indexCount;
    s_Result.indexStride = (rd->indexFormat == ommIndexFormat_UINT_8)  ? 1u :
                           (rd->indexFormat == ommIndexFormat_UINT_16) ? 2u : 4u;
    if (rd->indexCount > 0)
    {
        const uint32_t idxBytes = rd->indexCount * s_Result.indexStride;
        s_Result.indexBuffer.assign(
            static_cast<const uint8_t*>(rd->indexBuffer),
            static_cast<const uint8_t*>(rd->indexBuffer) + idxBytes);
    }

    s_Result.histogramFlat.reserve(rd->descArrayHistogramCount * 3);
    for (uint32_t j = 0; j < rd->descArrayHistogramCount; ++j)
    {
        s_Result.histogramFlat.push_back(rd->descArrayHistogram[j].count);
        s_Result.histogramFlat.push_back(rd->descArrayHistogram[j].subdivisionLevel);
        s_Result.histogramFlat.push_back(
            static_cast<uint32_t>(rd->descArrayHistogram[j].format));
    }

    ommCpuDestroyBakeResult(bakeResult);
    s_Result.valid = true;

    {
        char buf[256];
        snprintf(buf, sizeof(buf),
            "NR_BakeOMMCPU: OK — arrayData=%zu B, descs=%u, indices=%u, histEntries=%zu",
            s_Result.arrayData.size(), s_Result.descArrayCount,
            s_Result.indexCount, s_Result.histogramFlat.size() / 3);
        BAKER_LOG(buf);
    }
    return 1;
}

// ---------------------------------------------------------------------------
// NR_BakeResultDesc  –  C-compatible descriptor returned to the caller.
// All pointers are valid until NR_FreeBakeResult() is called.
// ---------------------------------------------------------------------------
#pragma pack(push, 4)
struct NR_BakeResultDesc
{
    const void* arrayData;          // OMM array data blob
    uint32_t    arrayDataSize;      // bytes
    const void* descArray;          // ommCpuOpacityMicromapDesc[] blob
    uint32_t    descArrayByteCount; // bytes (descArrayCount * 8)
    uint32_t    descArrayCount;     // number of desc entries
    const void* indexBuffer;        // per-triangle OMM index blob
    uint32_t    indexCount;         // number of indices
    uint32_t    indexStride;        // 1, 2, or 4
    const void* histogramFlat;      // uint32[histogramCount * 3]: [count, subdivLevel, format]
    uint32_t    histogramCount;     // number of histogram entries
};
#pragma pack(pop)

// ---------------------------------------------------------------------------
// NR_GetBakeResult
//   Fills *out with pointers into static storage.
//   Caller must Marshal.Copy all data before calling NR_FreeBakeResult.
//   Returns 1 if a valid result is available, 0 otherwise.
// ---------------------------------------------------------------------------
extern "C" __declspec(dllexport)
int NR_GetBakeResult(NR_BakeResultDesc* out)
{
    if (!out || !s_Result.valid) return 0;
    out->arrayData          = s_Result.arrayData.empty()      ? nullptr : s_Result.arrayData.data();
    out->arrayDataSize      = (uint32_t)s_Result.arrayData.size();
    out->descArray          = s_Result.descArray.empty()      ? nullptr : s_Result.descArray.data();
    out->descArrayByteCount = (uint32_t)s_Result.descArray.size();
    out->descArrayCount     = s_Result.descArrayCount;
    out->indexBuffer        = s_Result.indexBuffer.empty()    ? nullptr : s_Result.indexBuffer.data();
    out->indexCount         = s_Result.indexCount;
    out->indexStride        = s_Result.indexStride;
    out->histogramFlat      = s_Result.histogramFlat.empty()  ? nullptr : s_Result.histogramFlat.data();
    out->histogramCount     = (uint32_t)(s_Result.histogramFlat.size() / 3);
    return 1;
}

extern "C" __declspec(dllexport)
void NR_FreeBakeResult()
    { s_Result = {}; }
