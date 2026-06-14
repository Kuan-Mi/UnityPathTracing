#include "SharedUploadPool.h"
#include <cstdio>

namespace
{
    inline uint64_t AlignUp(uint64_t value, uint64_t alignment)
    {
        return (value + alignment - 1) & ~(alignment - 1);
    }
}

bool SharedUploadPool::Initialize(ID3D12Device* device, uint64_t defaultChunkSize, bool isScratch)
{
    if (!device || defaultChunkSize == 0) return false;
    m_device           = device;
    m_defaultChunkSize = defaultChunkSize;
    m_isScratch        = isScratch;
    m_current          = nullptr;
    m_usedThisFrame.clear();
    m_chunks.clear();
    return true;
}

void SharedUploadPool::Shutdown()
{
    std::lock_guard<std::mutex> lk(m_mutex);
    for (auto& c : m_chunks)
    {
        if (c->buffer && c->cpu)
        {
            c->buffer->Unmap(0, nullptr);
            c->cpu = nullptr;
        }
    }
    m_chunks.clear();
    m_current = nullptr;
    m_usedThisFrame.clear();
    m_device = nullptr;
}

SharedUploadPool::Chunk* SharedUploadPool::CreateChunk(uint64_t size)
{
    auto chunk = std::make_unique<Chunk>();

    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = m_isScratch ? D3D12_HEAP_TYPE_DEFAULT : D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width            = size;
    desc.Height           = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels        = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags            = m_isScratch ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                                        : D3D12_RESOURCE_FLAG_NONE;

    // Scratch chunks live on the GPU (DEFAULT) and start in COMMON; upload chunks
    // are CPU-visible (UPLOAD) and fixed in GENERIC_READ.
    HRESULT hr = m_device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        m_isScratch ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&chunk->buffer));
    if (FAILED(hr)) return nullptr;

    if (!m_isScratch)
    {
        const D3D12_RANGE readRange = { 0, 0 };
        void* mapped = nullptr;
        hr = chunk->buffer->Map(0, &readRange, &mapped);
        if (FAILED(hr)) return nullptr;
        chunk->cpu = static_cast<uint8_t*>(mapped);
    }

    chunk->gpu          = chunk->buffer->GetGPUVirtualAddress();
    chunk->size         = size;
    chunk->writePointer = 0;
    chunk->fence        = kInFlightUnknown; // becomes current immediately

    // Name the chunk so PIX captures and debug-layer messages are readable.
    wchar_t name[64];
    swprintf_s(name, L"NSB %s Chunk %u (%llu B)",
               m_isScratch ? L"Scratch" : L"Upload",
               m_nextChunkId++, static_cast<unsigned long long>(size));
    chunk->buffer->SetName(name);

    Chunk* raw = chunk.get();
    m_chunks.push_back(std::move(chunk));
    return raw;
}

SharedUploadPool::Allocation SharedUploadPool::Allocate(uint64_t size, uint64_t alignment)
{
    if (!m_device || size == 0) return {};
    if (alignment == 0) alignment = 1;

    std::lock_guard<std::mutex> lk(m_mutex);

    // Try the current chunk.
    if (m_current)
    {
        const uint64_t offset = AlignUp(m_current->writePointer, alignment);
        if (offset + size <= m_current->size)
        {
            m_current->writePointer = offset + size;
            return { m_current->buffer.Get(), offset,
                     m_current->cpu ? m_current->cpu + offset : nullptr,
                     m_current->gpu + offset };
        }

        // Full — retire it; it will be stamped with this frame's fence at EndFrame.
        m_usedThisFrame.push_back(m_current);
        m_current = nullptr;
    }

    // Reuse a free chunk (fence == 0) large enough, else create a new one.
    for (auto& c : m_chunks)
    {
        if (c->fence == 0 && c->size >= size)
        {
            m_current = c.get();
            break;
        }
    }
    if (!m_current)
    {
        const uint64_t chunkSize = (size > m_defaultChunkSize) ? size : m_defaultChunkSize;
        m_current = CreateChunk(AlignUp(chunkSize, 256ull));
    }

    if (!m_current) return {}; // allocation failed

    m_current->writePointer = 0;
    m_current->fence        = kInFlightUnknown;

    const uint64_t offset = 0; // fresh chunk
    m_current->writePointer = size;
    return { m_current->buffer.Get(), offset,
             m_current->cpu ? m_current->cpu + offset : nullptr,
             m_current->gpu + offset };
}

void SharedUploadPool::EndFrame(uint64_t frameCompleteFence, uint64_t completedFence)
{
    std::lock_guard<std::mutex> lk(m_mutex);

    // Reclaim chunks whose frame-complete fence has passed.
    for (auto& c : m_chunks)
    {
        if (c->fence != 0 && c->fence != kInFlightUnknown && c->fence <= completedFence)
        {
            c->fence        = 0;
            c->writePointer = 0;
        }
    }

    // Stamp the chunks touched this frame as in-flight until frameCompleteFence.
    for (Chunk* c : m_usedThisFrame)
        c->fence = frameCompleteFence;
    m_usedThisFrame.clear();

    if (m_current)
    {
        m_current->fence = frameCompleteFence;
        m_current        = nullptr;
    }
}
