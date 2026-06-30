namespace PathTracing.Profiling
{
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
    public static readonly NamedMarker PrepareLightsCompute = new(ProfilerCategory.Render, "PrepareLights CS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker PdfTexture = new(ProfilerCategory.Render, "PdfTexture", MarkerFlags.SampleGPU);
    public static readonly NamedMarker GenerateMips = new(ProfilerCategory.Render, "Generate Mips", MarkerFlags.SampleGPU);
    public static readonly NamedMarker Presample = new(ProfilerCategory.Render, "Presample", MarkerFlags.SampleGPU);
    public static readonly NamedMarker ReGir = new(ProfilerCategory.Render, "Regir", MarkerFlags.SampleGPU);

    // ── GBuffer ───────────────────────────────────────────────────────────────
    public static readonly NamedMarker GBufferRay = new(ProfilerCategory.Render, "GBuffer Ray", MarkerFlags.SampleGPU);
    public static readonly NamedMarker GBufferRaster = new(ProfilerCategory.Render, "GBuffer Raster", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RaytracedGBufferCompute = new(ProfilerCategory.Render, "RaytracedGBuffer CS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker PostprocessGBufferCompute = new(ProfilerCategory.Render, "PostprocessGBuffer CS", MarkerFlags.SampleGPU);

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
    public static readonly NamedMarker BrdfRayTracingCompute = new(ProfilerCategory.Render, "BrdfRayTracing CS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker BrdfRayTracing = new(ProfilerCategory.Render, "BrdfRayTracing RS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker ShadeSecondarySurfaces = new(ProfilerCategory.Render, "ShadeSecondarySurfaces RS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker ShadeSecondarySurfacesCompute = new(ProfilerCategory.Render, "ShadeSecondarySurfaces CS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker GiTemporalResamplingCompute = new(ProfilerCategory.Render, "GI TemporalResampling CS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker GiTemporalResampling = new(ProfilerCategory.Render, "GI TemporalResampling RS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker GiSpatialResamplingCompute = new(ProfilerCategory.Render, "GI SpatialResampling CS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker GiSpatialResampling = new(ProfilerCategory.Render, "GI SpatialResampling RS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker GiFinalShadingCompute = new(ProfilerCategory.Render, "GI FinalShading CS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker GiFinalShading = new(ProfilerCategory.Render, "GI FinalShading RS", MarkerFlags.SampleGPU);

    // ── PT (ReSTIR PT) ────────────────────────────────────────────────────────
    public static readonly NamedMarker PtGenerateInitialSamplesCompute = new(ProfilerCategory.Render, "PT GenerateInitialSamples CS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker PtGenerateInitialSamples        = new(ProfilerCategory.Render, "PT GenerateInitialSamples RS", MarkerFlags.SampleGPU);
    public static readonly NamedMarker PtTemporalResamplingCompute     = new(ProfilerCategory.Render, "PT TemporalResampling CS",     MarkerFlags.SampleGPU);
    public static readonly NamedMarker PtTemporalResampling            = new(ProfilerCategory.Render, "PT TemporalResampling RS",     MarkerFlags.SampleGPU);
    public static readonly NamedMarker PtSpatialResamplingCompute      = new(ProfilerCategory.Render, "PT SpatialResampling CS",      MarkerFlags.SampleGPU);
    public static readonly NamedMarker PtSpatialResampling             = new(ProfilerCategory.Render, "PT SpatialResampling RS",      MarkerFlags.SampleGPU);
    public static readonly NamedMarker PtFillSampleIDCompute           = new(ProfilerCategory.Render, "PT FillSampleID CS",           MarkerFlags.SampleGPU);
    public static readonly NamedMarker PtFillSampleID                  = new(ProfilerCategory.Render, "PT FillSampleID RS",           MarkerFlags.SampleGPU);
    public static readonly NamedMarker PtComputeDuplicationMapCompute  = new(ProfilerCategory.Render, "PT ComputeDuplicationMap CS",  MarkerFlags.SampleGPU);
    public static readonly NamedMarker PtFinalShadingCompute           = new(ProfilerCategory.Render, "PT FinalShading CS",           MarkerFlags.SampleGPU);
    public static readonly NamedMarker PtFinalShading                  = new(ProfilerCategory.Render, "PT FinalShading RS",           MarkerFlags.SampleGPU);

    // RTXPT LightingUpdateBegin
    public static readonly NamedMarker RtxptLightingUpdateBegin = new(ProfilerCategory.Render, "LightingUpdateBegin", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptControlDataSetup = new(ProfilerCategory.Render, "ControlDataSetup", MarkerFlags.SampleGPU);
    // Env map baker markers — strings kept byte-for-byte identical to the original RTXPT
    // EnvMapBaker.cpp / EnvMapImportanceSamplingBaker.cpp / donut MipMapGenPass.cpp
    // commandList->beginMarker() labels, so a PIX capture of the replica's bake lines up
    // with the reference hierarchy:
    //   EnvMapBaker
    //   ├─ ProcSkyBaseBake     (BaseLayerCS — writes cube mip 0+1)
    //   ├─ EnvMapBakerMIPs     (MIPReduceCS — solid-angle mip chain)
    //   ├─ BC6UCompression     (BC6UCompress CS → scratch, then copy scratch→BC6H cube)
    //   └─ ISBake              (importance/radiance map build)
    //      ├─ GenIM            (BuildMIPDescentImportanceMapCS)
    //      ├─ MipMapGen::Dispatch  (importance map mip chain)
    //      └─ MipMapGen::Dispatch  (radiance  map mip chain)
    public static readonly NamedMarker RtxptEnvMapBaker = new(ProfilerCategory.Render, "EnvMapBaker", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptEnvMapProcSkyBaseBake = new(ProfilerCategory.Render, "ProcSkyBaseBake", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptEnvMapBakerMIPs = new(ProfilerCategory.Render, "EnvMapBakerMIPs", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptEnvMapBC6UCompression = new(ProfilerCategory.Render, "BC6UCompression", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptEnvMapISBake = new(ProfilerCategory.Render, "ISBake", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptEnvMapGenIM = new(ProfilerCategory.Render, "GenIM", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptEnvMapMipMapGen = new(ProfilerCategory.Render, "MipMapGen::Dispatch", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptResetLightProxyCounters = new(ProfilerCategory.Render, "ResetLightProxyCounters", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptResetPastToCurrentHistory = new(ProfilerCategory.Render, "ResetPastToCurrentHistory", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptEnvLightsBackupPast = new(ProfilerCategory.Render, "EnvLightsBackupPast", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptEnvmapAndAnalyticLightBuffers = new(ProfilerCategory.Render, "EnvmapAndAnalyticLightBuffers", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptEnvLightsSubdivideBase = new(ProfilerCategory.Render, "EnvLightsSubdivideBase", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptEnvLightsSubdivideBoost = new(ProfilerCategory.Render, "EnvLightsSubdivideBoost", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptBakeEmissiveTriangles = new(ProfilerCategory.Render, "BakeEmissiveTriangles", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptEnvLightFillLookupMap = new(ProfilerCategory.Render, "EnvLightFillLookupMap", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptEnvLightsMapPastToCurrent = new(ProfilerCategory.Render, "EnvLightsMapPastToCurrent", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptProcessFeedbackHistoryPreFilter = new(ProfilerCategory.Render, "ProcessFeedbackHistoryPreFilter", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptProcessFeedbackHistoryP0 = new(ProfilerCategory.Render, "ProcessFeedbackHistoryP0", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptComputeWeights = new(ProfilerCategory.Render, "ComputeWeights", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptComputeProxyCounts = new(ProfilerCategory.Render, "ComputeProxyCounts", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptComputeProxyBaselineOffsets = new(ProfilerCategory.Render, "ComputeProxyBaselineOffsets", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptCreateProxyJobs = new(ProfilerCategory.Render, "CreateProxyJobs", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptExecuteProxyJobs = new(ProfilerCategory.Render, "ExecuteProxyJobs", MarkerFlags.SampleGPU);

    // RTXPT LightingUpdateEnd
    public static readonly NamedMarker RtxptLightingUpdateEnd = new(ProfilerCategory.Render, "LightingUpdateEnd", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptProcessFeedbackHistoryP1a = new(ProfilerCategory.Render, "ProcessFeedbackHistoryP1a", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptProcessFeedbackHistoryP1b = new(ProfilerCategory.Render, "ProcessFeedbackHistoryP1b", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptProcessFeedbackHistoryP2 = new(ProfilerCategory.Render, "ProcessFeedbackHistoryP2", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptProcessFeedbackHistoryP3 = new(ProfilerCategory.Render, "ProcessFeedbackHistoryP3", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptClearFeedbackHistory = new(ProfilerCategory.Render, "ClearFeedbackHistory", MarkerFlags.SampleGPU);

    // ── Main passes ───────────────────────────────────────────────────────────
    public static readonly NamedMarker Streamer                 = new(ProfilerCategory.Render, "Streamer", MarkerFlags.SampleGPU);
    public static readonly NamedMarker TLAS                     = new(ProfilerCategory.Render, "TLAS", MarkerFlags.SampleGPU);
    // public static readonly NamedMarker TLAS1                     = new(ProfilerCategory.Render, "TLAS1", MarkerFlags.SampleGPU);
    // public static readonly NamedMarker TLAS2                     = new(ProfilerCategory.Render, "TLAS2", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RecordSkinnedMorphUpdate = new(ProfilerCategory.Render, "RecordSkinnedMorphUpdate", MarkerFlags.SampleGPU);
    
    public static readonly NamedMarker SharcUpdate = new(ProfilerCategory.Render, "Sharc Update", MarkerFlags.SampleGPU);
    public static readonly NamedMarker SharcResolve = new(ProfilerCategory.Render, "Sharc Resolve", MarkerFlags.SampleGPU);
    
    public static readonly NamedMarker OpaqueTracing = new(ProfilerCategory.Render, "Tracing opaque ", MarkerFlags.SampleGPU);
    
    public static readonly NamedMarker NrdDenoise = new(ProfilerCategory.Render, "NRD Denoise", MarkerFlags.SampleGPU);
    public static readonly NamedMarker NrdDenoiseShadow = new(ProfilerCategory.Render, "NRD Denoise Shadow", MarkerFlags.SampleGPU);
    public static readonly NamedMarker NrdDenoiseOpaque = new(ProfilerCategory.Render, "NRD Denoise Opaque", MarkerFlags.SampleGPU);
    public static readonly NamedMarker NrdDenoiseRtxdi  = new(ProfilerCategory.Render, "NRD Denoise RTXDI",  MarkerFlags.SampleGPU);
    
    public static readonly NamedMarker Composition = new(ProfilerCategory.Render, "Composition", MarkerFlags.SampleGPU);
    
    public static readonly NamedMarker TransparentTracing = new(ProfilerCategory.Render, "Tracing transparent ", MarkerFlags.SampleGPU);
    public static readonly NamedMarker ReferencePtTracing = new(ProfilerCategory.Render, "Reference Pt Tracing", MarkerFlags.SampleGPU);

    
    public static readonly NamedMarker DlssBefore  = new(ProfilerCategory.Render, "DLSS Before",   MarkerFlags.SampleGPU);
    public static readonly NamedMarker DlssDenoise = new(ProfilerCategory.Render, "DLSS Denoise",  MarkerFlags.SampleGPU);
    public static readonly NamedMarker DlssUpscale   = new(ProfilerCategory.Render, "DLSS Upscale",    MarkerFlags.SampleGPU);
    public static readonly NamedMarker DlssAfter  = new(ProfilerCategory.Render, "DLSS After",   MarkerFlags.SampleGPU);
    public static readonly NamedMarker NisSharpening = new(ProfilerCategory.Render, "NIS Sharpening", MarkerFlags.SampleGPU);

    
    // RTXPT ToneMappingPasses.cpp marker strings, byte-for-byte: "Luminance" wraps the luminance
    // draw + donut MipMapGen::Dispatch mip chain + capture_cs + read-back copy; "ToneMapping" wraps
    // the tone-map apply draw.
    public static readonly NamedMarker RtxptLuminance = new(ProfilerCategory.Render, "Luminance", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptToneMapping = new(ProfilerCategory.Render, "ToneMapping", MarkerFlags.SampleGPU);

    // donut BloomPass.cpp marker strings, byte-for-byte, nested inside "Bloom": "Downscale" wraps
    // the two half-scale blits, "Blur" the H+V Gaussian draws, "Apply" the composite draw.
    public static readonly NamedMarker RtxptBloomDownscale = new(ProfilerCategory.Render, "Downscale", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptBloomBlur = new(ProfilerCategory.Render, "Blur", MarkerFlags.SampleGPU);
    public static readonly NamedMarker RtxptBloomApply = new(ProfilerCategory.Render, "Apply", MarkerFlags.SampleGPU);

    public static readonly NamedMarker Acc = new(ProfilerCategory.Render, "Acc", MarkerFlags.SampleGPU);
    public static readonly NamedMarker AutoExposure = new(ProfilerCategory.Render, "Auto Exposure", MarkerFlags.SampleGPU);
    public static readonly NamedMarker Taa = new(ProfilerCategory.Render, "TAA", MarkerFlags.SampleGPU);
    public static readonly NamedMarker ToneMapping = new(ProfilerCategory.Render, "Tone Mapping", MarkerFlags.SampleGPU);
    public static readonly NamedMarker Bloom = new(ProfilerCategory.Render, "Bloom", MarkerFlags.SampleGPU);
    public static readonly NamedMarker ConfidenceBlur = new(ProfilerCategory.Render, "Confidence Blur", MarkerFlags.SampleGPU);
    public static readonly NamedMarker Final = new(ProfilerCategory.Render, "Final", MarkerFlags.SampleGPU);
    public static readonly NamedMarker OutputBlit = new(ProfilerCategory.Render, "Output Blit", MarkerFlags.SampleGPU);

    // ── All marker names (for GPUProfiler inspector population) ──────────────
    public static readonly string[] AllMarkerNames = BuildMarkerNames();

    static string[] BuildMarkerNames()
    {
        var fields = typeof(RenderPassMarkers).GetFields(BindingFlags.Public | BindingFlags.Static);
        return (from f in fields where f.FieldType == typeof(NamedMarker) select (NamedMarker)f.GetValue(null) into nm select nm.Name).ToArray();
    }
}

}