/*
 * Closest-hit only wrapper for opaque RTXPT material permutations.
 * Alpha/custom-hit variants include PathTracerMaterialSpecializations.hlsl instead.
 */

#include "PathTracer/Config.h"
#include "PathTracer/PathTracerTypes.hlsli"

#include "Bindings/ShaderResourceBindings.hlsli"
#if PT_USE_RESTIR_GI
#include "Bindings/ReSTIRBindings.hlsli"
#endif

#include "PathTracerBridgeDonut.hlsli"
#include "PathTracer/PathTracer.hlsli"

[shader("closesthit")]
void CLOSESTHIT_ENTRY(inout PathPayload payload : SV_RayPayload, in BuiltInTriangleIntersectionAttributes attrib)
{
    PathState path = PathPayload::unpack(payload);
    PathTracer::HandleHit(path, WorldRayOrigin(), WorldRayDirection(), RayTCurrent(), attrib.barycentrics, GetWorkingContext());
    payload = PathPayload::pack(path);
}
