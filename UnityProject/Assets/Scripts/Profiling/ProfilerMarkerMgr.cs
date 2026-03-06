using Unity.Profiling;
using Unity.Profiling.LowLevel;

namespace DefaultNamespace
{
    public class ProfilerMarkerMgr
    {
      public static ProfilerMarker sharcUpdateMarker = new ProfilerMarker(ProfilerCategory.Render, "Sharc Update", MarkerFlags.SampleGPU);
      public static ProfilerMarker sharcResolveMarker = new ProfilerMarker(ProfilerCategory.Render, "Sharc Resolve", MarkerFlags.SampleGPU);
      public static ProfilerMarker opaqueTracingMarker = new ProfilerMarker(ProfilerCategory.Render, "Opaque Tracing", MarkerFlags.SampleGPU);
      public static ProfilerMarker nrdDenoise = new ProfilerMarker(ProfilerCategory.Render, "NRD Denoise", MarkerFlags.SampleGPU);
      public static ProfilerMarker compositionMarker = new ProfilerMarker(ProfilerCategory.Render, "Composition", MarkerFlags.SampleGPU);
      public static ProfilerMarker transparentTracingMarker = new ProfilerMarker(ProfilerCategory.Render, "Transparent Tracing", MarkerFlags.SampleGPU);
      public static ProfilerMarker taaMarker = new ProfilerMarker(ProfilerCategory.Render, "TAA", MarkerFlags.SampleGPU);
      public static ProfilerMarker dlssBeforeMarker = new ProfilerMarker(ProfilerCategory.Render, "DLSS Before", MarkerFlags.SampleGPU);
      public static ProfilerMarker dlssDenoiseMarker = new ProfilerMarker(ProfilerCategory.Render, "DLSS Denoise", MarkerFlags.SampleGPU);
      public static ProfilerMarker outputBlitMarker = new ProfilerMarker(ProfilerCategory.Render, "Output Blit", MarkerFlags.SampleGPU);
    }
}