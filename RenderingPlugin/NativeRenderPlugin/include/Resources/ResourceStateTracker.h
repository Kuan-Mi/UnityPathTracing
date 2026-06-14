#pragma once
#include <d3d12.h>
#include <dxgi1_6.h>          // IUnityGraphicsD3D12.h references IDXGISwapChain
#include "IUnityGraphicsD3D12.h"

// ---------------------------------------------------------------------------
// ResourceStateTracker
//   Thin facade over Unity's IUnityGraphicsD3D12v8 resource-state tracker.
//
//   Unity owns the command list and the authoritative per-resource state map;
//   this plugin records into Unity's list. So the plugin must NOT run its own
//   barrier batcher (that would double-track and fight Unity). Instead every
//   state transition the plugin needs goes through this one chokepoint:
//
//     tracker.Require(res, NEEDED_STATE);  // Unity inserts the barrier
//     ...record commands that read/write res in NEEDED_STATE...
//     tracker.Notify (res, RESULTING_STATE [, uavAccess]);  // Unity records new state
//
//   Centralising the calls here (instead of scattering RequestResourceState /
//   NotifyResourceState across DescriptorSetBase, the upload path, and the
//   plugin event callbacks) keeps state handling consistent and gives one place
//   to add validation/logging later. No-ops safely when the v8 interface is
//   unavailable, mirroring the previous direct-call guards.
// ---------------------------------------------------------------------------
class ResourceStateTracker
{
public:
    ResourceStateTracker() = default;
    explicit ResourceStateTracker(IUnityGraphicsD3D12v8* v8) : m_v8(v8) {}

    bool Valid() const { return m_v8 != nullptr; }

    // Ask Unity to transition |res| into |state| before subsequent commands.
    void Require(ID3D12Resource* res, D3D12_RESOURCE_STATES state) const
    {
        if (m_v8 && res) m_v8->RequestResourceState(res, state);
    }

    // Tell Unity the state |res| is in after the plugin's recorded commands.
    // uavAccess=true forces a UAV barrier before the next use even when the
    // state is unchanged (write-after-write hazard).
    void Notify(ID3D12Resource* res, D3D12_RESOURCE_STATES state, bool uavAccess = false) const
    {
        if (m_v8 && res) m_v8->NotifyResourceState(res, state, uavAccess);
    }

private:
    IUnityGraphicsD3D12v8* m_v8 = nullptr;
};
