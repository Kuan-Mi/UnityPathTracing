// RTXDI C# Verification Test
// To run in Unity: attach this to any GameObject, or call RtxdiTest.RunAllTests() from editor script.
// To run standalone (outside Unity): uncomment the Main() at the bottom and provide a stub for Debug.Assert/Log.

using System;
using System.Text;
using Rtxdi;
using Rtxdi.DI;
using Rtxdi.GI;
using Rtxdi.ReGIR;
using Rtxdi.LightSampling;

#if UNITY_STANDALONE || UNITY_EDITOR
using UnityEngine;
#endif

public class RtxdiTest : MonoBehaviour
{
    static StringBuilder sb = new StringBuilder(32768);

    static void P(string s)
    {
        sb.AppendLine(s);
    }

    static void P(string fmt, params object[] args)
    {
        sb.AppendLine(string.Format(fmt, args));
    }

    static void PrintDIBufferIndices(string label, ReSTIRDI_BufferIndices bi)
    {
        P("  {0}: init={1} tIn={2} tOut={3} sIn={4} sOut={5} shade={6}",
            label,
            bi.initialSamplingOutputBufferIndex,
            bi.temporalResamplingInputBufferIndex,
            bi.temporalResamplingOutputBufferIndex,
            bi.spatialResamplingInputBufferIndex,
            bi.spatialResamplingOutputBufferIndex,
            bi.shadingInputBufferIndex);
    }

    static void PrintGIBufferIndices(string label, ReSTIRGI_BufferIndices bi)
    {
        P("  {0}: sec={1} tIn={2} tOut={3} sIn={4} sOut={5} final={6}",
            label,
            bi.secondarySurfaceReSTIRDIOutputBufferIndex,
            bi.temporalResamplingInputBufferIndex,
            bi.temporalResamplingOutputBufferIndex,
            bi.spatialResamplingInputBufferIndex,
            bi.spatialResamplingOutputBufferIndex,
            bi.finalShadingInputBufferIndex);
    }

    public static string RunAllTests()
    {
        sb.Clear();
        P("===== RTXDI C# Test Output =====");
        P("");

        // ========== 1. JenkinsHash ==========
        P("[1] JenkinsHash");
        uint[] hashInputs = { 0, 1, 2, 42, 100, 0xFFFFFFFF, 12345 };
        foreach (var val in hashInputs)
            P("  JenkinsHash({0}) = {1}", val, RtxdiUtils.JenkinsHash(val));
        P("");

        // ========== 2. CalculateReservoirBufferParameters ==========
        P("[2] CalculateReservoirBufferParameters");
        var rbpTests = new (uint w, uint h, CheckerboardMode cb)[]
        {
            (1920, 1080, CheckerboardMode.Off),
            (1920, 1080, CheckerboardMode.Black),
            (1920, 1080, CheckerboardMode.White),
            (1280, 720, CheckerboardMode.Off),
            (256, 256, CheckerboardMode.Off),
            (1, 1, CheckerboardMode.Off),
            (17, 33, CheckerboardMode.Black),
        };
        foreach (var t in rbpTests)
        {
            var p = RtxdiUtils.CalculateReservoirBufferParameters(t.w, t.h, t.cb);
            P("  ({0},{1},cb={2}): rowPitch={3} arrayPitch={4}",
                t.w, t.h, (uint)t.cb, p.reservoirBlockRowPitch, p.reservoirArrayPitch);
        }

        P("");

        // ========== 3. ComputePdfTextureSize ==========
        P("[3] ComputePdfTextureSize");
        uint[] pdfInputs = { 1, 4, 16, 100, 1000, 10000, 65536, 1000000 };
        foreach (var maxItems in pdfInputs)
        {
            RtxdiUtils.ComputePdfTextureSize(maxItems, out uint w, out uint h, out uint mips);
            P("  maxItems={0}: w={1} h={2} mips={3}", maxItems, w, h, mips);
        }

        P("");

        // ========== 4. FillNeighborOffsetBuffer ==========
        P("[4] FillNeighborOffsetBuffer (first 20 bytes, count=32)");
        byte[] noBuf = new byte[64 * 2];
        RtxdiUtils.FillNeighborOffsetBuffer(noBuf, 32);
        var noSb = new StringBuilder("  ");
        for (int i = 0; i < 20; i++)
            noSb.Append(((sbyte)noBuf[i]).ToString() + " ");
        P(noSb.ToString());
        P("");

        // ========== 5. RISBufferSegmentAllocator ==========
        P("[5] RISBufferSegmentAllocator");
        {
            var alloc = new RISBufferSegmentAllocator();
            P("  initial total = {0}", alloc.GetTotalSizeInElements());
            uint off1 = alloc.AllocateSegment(100);
            P("  alloc(100): offset={0} total={1}", off1, alloc.GetTotalSizeInElements());
            uint off2 = alloc.AllocateSegment(200);
            P("  alloc(200): offset={0} total={1}", off2, alloc.GetTotalSizeInElements());
            uint off3 = alloc.AllocateSegment(50);
            P("  alloc(50): offset={0} total={1}", off3, alloc.GetTotalSizeInElements());
        }
        P("");

        // ========== 6. ReGIRContext (Onion mode - default) ==========
        P("[6] ReGIRContext (Onion, default params)");
        {
            var alloc = new RISBufferSegmentAllocator();
            var sp = new ReGIRStaticParameters
            {
                Mode = ReGIRMode.Onion,
                LightsPerCell = 512,
                gridParameters = new ReGIRGridStaticParameters { GridSize = new Unity.Mathematics.uint3(16, 16, 16) },
                onionParameters = new ReGIROnionStaticParameters { OnionDetailLayers = 5, OnionCoverageLayers = 10 },
            };

            var ctx = new ReGIRContext(sp, alloc);
            P("  cellOffset = {0}", ctx.GetReGIRCellOffset());
            P("  lightSlotCount = {0}", ctx.GetReGIRLightSlotCount());
            P("  gridCalc.lightSlotCount = {0}", ctx.GetReGIRGridCalculatedParameters().lightSlotCount);

            var onionCalc = ctx.GetReGIROnionCalculatedParameters();
            P("  onionCalc.lightSlotCount = {0}", onionCalc.lightSlotCount);
            P("  onionCalc.regirOnionCells = {0}", onionCalc.regirOnionCells);
            P("  onionCalc.layers.size = {0}", onionCalc.regirOnionLayers.Count);
            P("  onionCalc.rings.size = {0}", onionCalc.regirOnionRings.Count);
            P("  onionCalc.cubicRootFactor = {0:F6}", onionCalc.regirOnionCubicRootFactor);
            P("  onionCalc.linearFactor = {0:F6}", onionCalc.regirOnionLinearFactor);

            // Print each layer group
            for (int i = 0; i < onionCalc.regirOnionLayers.Count; i++)
            {
                var lg = onionCalc.regirOnionLayers[i];
                P("  layer[{0}]: innerR={1:F4} outerR={2:F4} invLog={3:F4} count={4} cells={5} ringOff={6} ringCnt={7} cellOff={8}",
                    i, lg.innerRadius, lg.outerRadius, lg.invLogLayerScale,
                    lg.layerCount, lg.cellsPerLayer, lg.ringOffset, lg.ringCount, lg.layerCellOffset);
            }

            // Print first 5 rings
            int ringsToPrint = Math.Min(onionCalc.regirOnionRings.Count, 5);
            for (int i = 0; i < ringsToPrint; i++)
            {
                var r = onionCalc.regirOnionRings[i];
                P("  ring[{0}]: cellAngle={1:F6} invCellAngle={2:F6} cellOffset={3} cellCount={4}",
                    i, r.cellAngle, r.invCellAngle, r.cellOffset, r.cellCount);
            }

            P("  IsLocalLightPowerRISEnable = {0}", ctx.IsLocalLightPowerRISEnable() ? 1 : 0);

            // Change dynamic params to uniform
            var dp = new ReGIRDynamicParameters
            {
                regirCellSize = 2.0f,
                center = new Unity.Mathematics.float3(1.0f, 2.0f, 3.0f),
                fallbackSamplingMode = LocalLightReGIRFallbackSamplingMode.Uniform,
                presamplingMode = LocalLightReGIRPresamplingMode.Uniform,
                regirSamplingJitter = 0.5f,
                regirNumBuildSamples = 16,
            };
            ctx.SetDynamicParameters(dp);
            P("  After Uniform: IsLocalLightPowerRISEnable = {0}", ctx.IsLocalLightPowerRISEnable() ? 1 : 0);

            P("  allocator total = {0}", alloc.GetTotalSizeInElements());
        }
        P("");

        // ========== 7. ReGIRContext (Grid mode) ==========
        P("[7] ReGIRContext (Grid mode)");
        {
            var alloc = new RISBufferSegmentAllocator();
            var sp = new ReGIRStaticParameters
            {
                Mode = ReGIRMode.Grid,
                LightsPerCell = 256,
                gridParameters = new ReGIRGridStaticParameters { GridSize = new Unity.Mathematics.uint3(8, 8, 8) },
                onionParameters = new ReGIROnionStaticParameters { OnionDetailLayers = 5, OnionCoverageLayers = 10 },
            };

            var ctx = new ReGIRContext(sp, alloc);
            P("  cellOffset = {0}", ctx.GetReGIRCellOffset());
            P("  lightSlotCount = {0}", ctx.GetReGIRLightSlotCount());
            P("  gridCalc.lightSlotCount = {0}", ctx.GetReGIRGridCalculatedParameters().lightSlotCount);
            P("  allocator total = {0}", alloc.GetTotalSizeInElements());
        }
        P("");

        // ========== 8. ReGIRContext (Disabled mode) ==========
        P("[8] ReGIRContext (Disabled mode)");
        {
            var alloc = new RISBufferSegmentAllocator();
            alloc.AllocateSegment(500);
            var sp = new ReGIRStaticParameters
            {
                Mode = ReGIRMode.Disabled,
                LightsPerCell = 512,
                gridParameters = new ReGIRGridStaticParameters { GridSize = new Unity.Mathematics.uint3(16, 16, 16) },
                onionParameters = new ReGIROnionStaticParameters { OnionDetailLayers = 5, OnionCoverageLayers = 10 },
            };

            var ctx = new ReGIRContext(sp, alloc);
            P("  cellOffset = {0}", ctx.GetReGIRCellOffset());
            P("  lightSlotCount = {0}", ctx.GetReGIRLightSlotCount());
            P("  allocator total = {0}", alloc.GetTotalSizeInElements());
        }
        P("");

        // ========== 9. ReSTIRDIContext ==========
        P("[9] ReSTIRDIContext");
        {
            var sp = new ReSTIRDIStaticParameters
            {
                NeighborOffsetCount = 8192,
                RenderWidth = 1920,
                RenderHeight = 1080,
                CheckerboardSamplingMode = CheckerboardMode.Off,
            };

            var ctx = new ReSTIRDIContext(sp);
            P("  frameIndex = {0}", ctx.GetFrameIndex());
            P("  resamplingMode = {0}", (uint)ctx.GetResamplingMode());

            var rbp = ctx.GetReservoirBufferParameters();
            P("  reservoirBuffer: rowPitch={0} arrayPitch={1}", rbp.reservoirBlockRowPitch, rbp.reservoirArrayPitch);

            var rtp = ctx.GetRuntimeParams();
            P("  runtime: neighborOffsetMask={0} activeCheckerboard={1}", rtp.neighborOffsetMask, rtp.activeCheckerboardField);

            var isp = ctx.GetInitialSamplingParameters();
            P("  initial: localSamples={0} brdfCutoff={1:F4} mode={2}",
                isp.numPrimaryLocalLightSamples, isp.brdfCutoff, (uint)isp.localLightSamplingMode);

            var tsp = ctx.GetTemporalResamplingParameters();
            P("  temporal: maxHistory={0} depthThr={1:F1} normalThr={2:F1} bias={3} boiling={4:F1} permutThr={5:F1}",
                tsp.maxHistoryLength, tsp.temporalDepthThreshold, tsp.temporalNormalThreshold,
                (uint)tsp.temporalBiasCorrection, tsp.boilingFilterStrength, tsp.permutationSamplingThreshold);

            var ssp = ctx.GetSpatialResamplingParameters();
            P("  spatial: samples={0} radius={1:F1} boost={2} bias={3}",
                ssp.numSpatialSamples, ssp.spatialSamplingRadius, ssp.numDisocclusionBoostSamples, (uint)ssp.spatialBiasCorrection);

            var shp = ctx.GetShadingParameters();
            P("  shading: finalVis={0} maxAge={1} maxDist={2:F1} reuse={3} denoise={4}",
                shp.enableFinalVisibility, shp.finalVisibilityMaxAge, shp.finalVisibilityMaxDistance,
                shp.reuseFinalVisibility, shp.enableDenoiserInputPacking);

            // Frame index progression + buffer indices for each mode
            string[] modeNames = { "None", "Temporal", "Spatial", "TemporalAndSpatial", "FusedSpatiotemporal" };
            ReSTIRDI_ResamplingMode[] modes =
            {
                ReSTIRDI_ResamplingMode.None,
                ReSTIRDI_ResamplingMode.Temporal,
                ReSTIRDI_ResamplingMode.Spatial,
                ReSTIRDI_ResamplingMode.TemporalAndSpatial,
                ReSTIRDI_ResamplingMode.FusedSpatiotemporal
            };

            for (int m = 0; m < 5; m++)
            {
                var ctx2 = new ReSTIRDIContext(sp);
                ctx2.SetResamplingMode(modes[m]);
                P("  Mode={0} frame=0:", modeNames[m]);
                PrintDIBufferIndices("  bi", ctx2.GetBufferIndices());

                ctx2.SetFrameIndex(1);
                P("  Mode={0} frame=1:", modeNames[m]);
                PrintDIBufferIndices("  bi", ctx2.GetBufferIndices());
                P("    uniformRandom={0}", ctx2.GetTemporalResamplingParameters().uniformRandomNumber);

                ctx2.SetFrameIndex(2);
                P("  Mode={0} frame=2:", modeNames[m]);
                PrintDIBufferIndices("  bi", ctx2.GetBufferIndices());
            }
        }
        P("");

        // ========== 10. ReSTIRDIContext with Checkerboard ==========
        P("[10] ReSTIRDIContext (Checkerboard::Black)");
        {
            var sp = new ReSTIRDIStaticParameters
            {
                NeighborOffsetCount = 8192,
                RenderWidth = 1920,
                RenderHeight = 1080,
                CheckerboardSamplingMode = CheckerboardMode.Black,
            };

            var ctx = new ReSTIRDIContext(sp);
            for (uint f = 0; f < 4; f++)
            {
                ctx.SetFrameIndex(f);
                P("  frame={0}: checkerboard={1}", f, ctx.GetRuntimeParams().activeCheckerboardField);
            }
        }
        P("");

        // ========== 11. ReSTIRGIContext ==========
        P("[11] ReSTIRGIContext");
        {
            var sp = new ReSTIRGIStaticParameters
            {
                RenderWidth = 1920,
                RenderHeight = 1080,
                CheckerboardSamplingMode = CheckerboardMode.Off,
            };

            var ctx = new ReSTIRGIContext(sp);
            P("  frameIndex = {0}", ctx.GetFrameIndex());
            P("  resamplingMode = {0}", (uint)ctx.GetResamplingMode());

            var rbp = ctx.GetReservoirBufferParameters();
            P("  reservoirBuffer: rowPitch={0} arrayPitch={1}", rbp.reservoirBlockRowPitch, rbp.reservoirArrayPitch);

            var tsp = ctx.GetTemporalResamplingParameters();
            P("  temporal: maxHistory={0} maxAge={1} depth={2:F1} normal={3:F1} bias={4} boiling={5:F1} fallback={6}",
                tsp.maxHistoryLength, tsp.maxReservoirAge, tsp.depthThreshold, tsp.normalThreshold,
                (uint)tsp.temporalBiasCorrectionMode, tsp.boilingFilterStrength, tsp.enableFallbackSampling);

            var ssp = ctx.GetSpatialResamplingParameters();
            P("  spatial: samples={0} radius={1:F1} depth={2:F1} normal={3:F1} bias={4}",
                ssp.numSpatialSamples, ssp.spatialSamplingRadius, ssp.spatialDepthThreshold,
                ssp.spatialNormalThreshold, (uint)ssp.spatialBiasCorrectionMode);

            var fsp = ctx.GetFinalShadingParameters();
            P("  final: vis={0} mis={1}", fsp.enableFinalVisibility, fsp.enableFinalMIS);

            string[] giModeNames = { "None", "Temporal", "Spatial", "TemporalAndSpatial", "FusedSpatiotemporal" };
            ReSTIRGI_ResamplingMode[] giModes =
            {
                ReSTIRGI_ResamplingMode.None,
                ReSTIRGI_ResamplingMode.Temporal,
                ReSTIRGI_ResamplingMode.Spatial,
                ReSTIRGI_ResamplingMode.TemporalAndSpatial,
                ReSTIRGI_ResamplingMode.FusedSpatiotemporal
            };

            for (int m = 0; m < 5; m++)
            {
                var ctx2 = new ReSTIRGIContext(sp);
                ctx2.SetResamplingMode(giModes[m]);
                P("  Mode={0} frame=0:", giModeNames[m]);
                PrintGIBufferIndices("  bi", ctx2.GetBufferIndices());

                ctx2.SetFrameIndex(1);
                P("  Mode={0} frame=1:", giModeNames[m]);
                PrintGIBufferIndices("  bi", ctx2.GetBufferIndices());
                P("    uniformRandom={0}", ctx2.GetTemporalResamplingParameters().uniformRandomNumber);

                ctx2.SetFrameIndex(2);
                P("  Mode={0} frame=2:", giModeNames[m]);
                PrintGIBufferIndices("  bi", ctx2.GetBufferIndices());
            }
        }
        P("");

        // ========== 12. ImportanceSamplingContext ==========
        P("[12] ImportanceSamplingContext");
        {
            var isParams = new ImportanceSamplingContext_StaticParameters
            {
                localLightRISBufferParams = new RISBufferSegmentParameters { tileSize = 1024, tileCount = 128 },
                environmentLightRISBufferParams = new RISBufferSegmentParameters { tileSize = 1024, tileCount = 128 },
                NeighborOffsetCount = 8192,
                renderWidth = 1920,
                renderHeight = 1080,
                CheckerboardSamplingMode = CheckerboardMode.Off,
                regirStaticParams = new ReGIRStaticParameters
                {
                    Mode = ReGIRMode.Onion,
                    LightsPerCell = 512,
                    gridParameters = new ReGIRGridStaticParameters { GridSize = new Unity.Mathematics.uint3(16, 16, 16) },
                    onionParameters = new ReGIROnionStaticParameters { OnionDetailLayers = 5, OnionCoverageLayers = 10 },
                },
            };

            var ctx = new ImportanceSamplingContext(isParams);

            P("  neighborOffsetCount = {0}", ctx.GetNeighborOffsetCount());
            P("  IsLocalLightPowerRISEnabled = {0}", ctx.IsLocalLightPowerRISEnabled() ? 1 : 0);
            P("  IsReGIREnabled = {0}", ctx.IsReGIREnabled() ? 1 : 0);

            var localRIS = ctx.GetLocalLightRISBufferSegmentParams();
            P("  localRIS: offset={0} tileSize={1} tileCount={2}",
                localRIS.bufferOffset, localRIS.tileSize, localRIS.tileCount);

            var envRIS = ctx.GetEnvironmentLightRISBufferSegmentParams();
            P("  envRIS: offset={0} tileSize={1} tileCount={2}",
                envRIS.bufferOffset, envRIS.tileSize, envRIS.tileCount);

            P("  allocatorTotal = {0}", ctx.GetRISBufferSegmentAllocator().GetTotalSizeInElements());

            // Set initial sampling to ReGIR_RIS mode
            var iss = ctx.GetReSTIRDIContext().GetInitialSamplingParameters();
            iss.localLightSamplingMode = ReSTIRDI_LocalLightSamplingMode.ReGIR_RIS;
            ctx.GetReSTIRDIContext().SetInitialSamplingParameters(iss);
            P("  After set ReGIR_RIS: IsReGIREnabled = {0}", ctx.IsReGIREnabled() ? 1 : 0);
            P("  After set ReGIR_RIS: IsLocalLightPowerRISEnabled = {0}", ctx.IsLocalLightPowerRISEnabled() ? 1 : 0);

            // Set to Power_RIS
            iss.localLightSamplingMode = ReSTIRDI_LocalLightSamplingMode.Power_RIS;
            ctx.GetReSTIRDIContext().SetInitialSamplingParameters(iss);
            P("  After set Power_RIS: IsLocalLightPowerRISEnabled = {0}", ctx.IsLocalLightPowerRISEnabled() ? 1 : 0);

            // LightBufferParams
            var lbp = new RTXDI_LightBufferParameters();
            lbp.localLightBufferRegion.firstLightIndex = 0;
            lbp.localLightBufferRegion.numLights = 100;
            lbp.infiniteLightBufferRegion.firstLightIndex = 100;
            lbp.infiniteLightBufferRegion.numLights = 5;
            lbp.environmentLightParams.lightPresent = 1;
            lbp.environmentLightParams.lightIndex = 105;
            ctx.SetLightBufferParams(lbp);
            var lbpOut = ctx.GetLightBufferParameters();
            P("  lightBuffer: local({0},{1}) infinite({2},{3}) env({4},{5})",
                lbpOut.localLightBufferRegion.firstLightIndex, lbpOut.localLightBufferRegion.numLights,
                lbpOut.infiniteLightBufferRegion.firstLightIndex, lbpOut.infiniteLightBufferRegion.numLights,
                lbpOut.environmentLightParams.lightPresent, lbpOut.environmentLightParams.lightIndex);
        }
        P("");

        P("===== Done =====");
        return sb.ToString();
    }


    [ContextMenu("Run RTXDI Tests")]
    public void RunInUnity()
    {
        string result = RunAllTests();
        Debug.Log(result);
    }
}


