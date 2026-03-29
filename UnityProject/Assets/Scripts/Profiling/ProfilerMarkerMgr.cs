using Unity.Profiling;

namespace LeetProfiling
{
    public static class ProfilerMarkerMgr
    {
        public static ProfilerMarker PathTracingMarker = new ProfilerMarker("PathTracing");
        public static ProfilerMarker GBufferMarker = new ProfilerMarker("GBufferPass");
        public static ProfilerMarker OpaqueMarker = new ProfilerMarker("OpaquePass");
        public static ProfilerMarker DlssRRMarker = new ProfilerMarker("DlssRRPass");
        
    }
}