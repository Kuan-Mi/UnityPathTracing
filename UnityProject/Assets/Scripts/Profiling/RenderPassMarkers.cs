using System.Linq;
using System.Reflection;
using Unity.Profiling;
using Unity.Profiling.LowLevel;

/// <summary>
/// ProfilerMarker + 名称的包装，支持隐式转换为 ProfilerMarker，调用方无需修改。
/// </summary>
public readonly struct NamedMarker
{
    public readonly string Name;
    private readonly ProfilerMarker _marker;

    public NamedMarker(ProfilerCategory category, string name, MarkerFlags flags)
    {
        Name    = name;
        _marker = new ProfilerMarker(category, name, flags);
    }

    public static implicit operator ProfilerMarker(NamedMarker nm) => nm._marker;
}

/// <summary>
/// 集中管理所有渲染Pass的 ProfilerMarker，统一由此处引用，便于 GPUProfiler 读取。
/// </summary>
public static class RenderPassMarkers
{
    // ── Prepare ──────────────────────────────────────────────────────────────
    public static readonly NamedMarker PrepareLight = new(ProfilerCategory.Render, "Prepare Light", MarkerFlags.SampleGPU);
    public static readonly NamedMarker PdfTexture = new(ProfilerCategory.Render, "PdfTexture", MarkerFlags.SampleGPU);
    public static readonly NamedMarker GenerateMips = new(ProfilerCategory.Render, "Generate Mips", MarkerFlags.SampleGPU);
    public static readonly NamedMarker Presample = new(ProfilerCategory.Render, "Presample", MarkerFlags.SampleGPU);
    public static readonly NamedMarker ReGir = new(ProfilerCategory.Render, "Regir", MarkerFlags.SampleGPU);

    // ── GBuffer ───────────────────────────────────────────────────────────────
    public static readonly NamedMarker GBufferRay = new(ProfilerCategory.Render, "GBuffer Ray", MarkerFlags.SampleGPU);
    public static readonly NamedMarker GBufferRaster = new(ProfilerCategory.Render, "GBuffer Raster", MarkerFlags.SampleGPU);

    // ── DI ────────────────────────────────────────────────────────────────────
    public static readonly NamedMarker GenInitialSamplesCompute = new(ProfilerCategory.Render, "GenInitialSamples CS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker GenInitialSamples = new(ProfilerCategory.Render, "GenInitialSamples RS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker DiTemporalResamplingCompute = new(ProfilerCategory.Render, "DI TemporalResampling CS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker DiTemporalResampling = new(ProfilerCategory.Render, "DI TemporalResampling RS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker DiSpatialResamplingCompute = new(ProfilerCategory.Render, "DI SpatialResampling CS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker DiSpatialResampling = new(ProfilerCategory.Render, "DI SpatialResampling RS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker DiShadeSamplesCompute = new(ProfilerCategory.Render, "DI ShadeSamples CS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker DiShadeSamples = new(ProfilerCategory.Render, "DI ShadeSamples RS", MarkerFlags.SampleGPU);

    // ── GI ────────────────────────────────────────────────────────────────────
    public static readonly NamedMarker BrdfRayTracing = new(ProfilerCategory.Render, "BrdfRayTracing", MarkerFlags.SampleGPU);
    public static readonly NamedMarker ShadeSecondarySurfaces = new(ProfilerCategory.Render, "ShadeSecondarySurfaces RS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker ShadeSecondarySurfacesCompute = new(ProfilerCategory.Render, "ShadeSecondarySurfaces CS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker GiTemporalResamplingCompute = new(ProfilerCategory.Render, "GI TemporalResampling CS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker GiTemporalResampling = new(ProfilerCategory.Render, "GI TemporalResampling RS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker GiSpatialResamplingCompute = new(ProfilerCategory.Render, "GI SpatialResampling CS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker GiSpatialResampling = new(ProfilerCategory.Render, "GI SpatialResampling RS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker GiFinalShadingCompute = new(ProfilerCategory.Render, "GI FinalShading CS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker GiFinalShading = new(ProfilerCategory.Render, "GI FinalShading RS", MarkerFlags.SampleGPU);

    // ── Main passes ───────────────────────────────────────────────────────────
    public static readonly NamedMarker OpaqueTracing = new(ProfilerCategory.Render, "Opaque Tracing", MarkerFlags.SampleGPU);
    public static readonly NamedMarker TransparentTracing = new(ProfilerCategory.Render, "Transparent Tracing", MarkerFlags.SampleGPU);
    public static readonly NamedMarker ReferencePtTracing = new(ProfilerCategory.Render, "Reference Pt Tracing", MarkerFlags.SampleGPU);
    public static readonly NamedMarker SharcUpdate = new(ProfilerCategory.Render, "Sharc Update", MarkerFlags.SampleGPU);
    public static readonly NamedMarker SharcResolve = new(ProfilerCategory.Render, "Sharc Resolve", MarkerFlags.SampleGPU);

    // ── Denoising / Upscaling ─────────────────────────────────────────────────
    public static readonly NamedMarker NrdDenoise = new(ProfilerCategory.Render, "NRD Denoise", MarkerFlags.SampleGPU);
    public static readonly NamedMarker DlssBefore = new(ProfilerCategory.Render, "DLSS Before", MarkerFlags.SampleGPU);
    public static readonly NamedMarker DlssDenoise = new(ProfilerCategory.Render, "DLSS Denoise", MarkerFlags.SampleGPU);

    // ── Post-process / Output ─────────────────────────────────────────────────
    public static readonly NamedMarker Composition = new(ProfilerCategory.Render, "Composition", MarkerFlags.SampleGPU);
    public static readonly NamedMarker Acc = new(ProfilerCategory.Render, "Acc", MarkerFlags.SampleGPU);
    public static readonly NamedMarker AutoExposure = new(ProfilerCategory.Render, "Auto Exposure", MarkerFlags.SampleGPU);
    public static readonly NamedMarker Taa = new(ProfilerCategory.Render, "TAA", MarkerFlags.SampleGPU);
    public static readonly NamedMarker OutputBlit = new(ProfilerCategory.Render, "Output Blit", MarkerFlags.SampleGPU);

    // ── All marker names (for GPUProfiler inspector population) ──────────────
    public static readonly string[] AllMarkerNames = BuildMarkerNames();

    static string[] BuildMarkerNames()
    {
        var fields = typeof(RenderPassMarkers).GetFields(BindingFlags.Public | BindingFlags.Static);
        return (from f in fields where f.FieldType == typeof(NamedMarker) select (NamedMarker)f.GetValue(null) into nm select nm.Name).ToArray();
    }
}