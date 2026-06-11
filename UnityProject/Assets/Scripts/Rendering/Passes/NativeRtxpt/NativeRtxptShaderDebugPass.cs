using System;
using System.Text;
using NativeRender;
using Nri;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Latest pick feedback (DebugFeedbackStruct, PathTracerDebug.hlsli) read back from
    /// u_FeedbackBuffer. Mirrors C++ Sample::m_feedbackData; displayed by the feature editor
    /// like the original ImGui "debugPrint %d: ..." block.
    /// </summary>
    public struct RtxptDebugFeedback
    {
        public Vector4[] DebugPrintSlots; // float4 debugPrint[16] — Print(i, value) from shaders
        public int       LineVertexCount;
        public int       PickedMaterialID;
        public bool      Valid;
    }

    /// <summary>
    /// Unity port of RTXPT's ShaderDebug (Misc/ShaderDebug.cpp + Libraries/ShaderDebug/ShaderDebug.hlsl):
    ///
    ///   BeginPass (start of frame)  — ShaderDebug::BeginFrame: rewrites the 96-byte buffer header
    ///                                 (zeroed print/triangle counters, VertexCountPerInstance = 3,
    ///                                 current worldToClip) via cmd.SetBufferData.
    ///   DrawPass  (before the blit) — ShaderDebug::EndFrameAndOutput:
    ///       1. DebugTriangle/DebugLine geometry from the raw buffer. The native raster path has no
    ///          ExecuteIndirect, so a fixed instance budget is drawn and the VS culls against the
    ///          GPU-written instance counter (see ShaderDebugTriangles/Lines.rastershader).
    ///       2. Picked-pixel debug lines (u_DebugLinesBuffer, ENABLE_DEBUG_LINES_VIZ) — the
    ///          Sample.cpp:2177 "Debug Lines" draw, vertex-pulled with GPU-side count culling.
    ///       3. Debug-viz overlay — alpha-blends the ShaderDebugViz texture over the final image
    ///          (skipped while a debug view is shown fullscreen by the output blit).
    ///       4. DebugPrint readback → Unity console (OutputLastBufferPrints port), and pick
    ///          feedback readback → <see cref="LastFeedback"/>. AsyncGPUReadback replaces the
    ///          original's 3-deep CPU ring (same few-frames latency).
    /// </summary>
    public static class NativeRtxptShaderDebug
    {
        /// <summary>Latest pick feedback for editor display (any camera).</summary>
        public static RtxptDebugFeedback LastFeedback;
    }

    /// <summary>Start-of-frame header reset (ShaderDebug::BeginFrame, ShaderDebug.cpp:174).</summary>
    public class NativeRtxptShaderDebugBeginPass : ScriptableRenderPass
    {
        private NativeRtxptPassContext _ctx;
        private readonly uint[] _header = new uint[NativeRtxptBufferResources.ShaderDebugHeaderBytes / 4];

        public void Setup(NativeRtxptPassContext ctx) => _ctx = ctx;

        private class PassData
        {
            internal GraphicsBuffer Buffer;
            internal uint[]         Header;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            var buffer = _ctx?.Buffers?.ShaderDebugBuffer;
            if (buffer == null) return;

            // ShaderDebugHeader: counters + DrawIndirect args {3, 0, 0, 0} + worldToClip.
            Array.Clear(_header, 0, _header.Length);
            _header[4] = 3; // VertexCountPerInstance

            // Unity's column-major Matrix4x4 memory (m[0..15] = column by column) read as a
            // row-major HLSL float4x4 used with mul(rowVector, M) applies exactly M_unity * v —
            // so the raw indexer order is the correct byte layout for the header.
            Matrix4x4 m = _ctx.FrameState.worldToClip;
            for (int i = 0; i < 16; i++)
                _header[8 + i] = BitsOf(m[i]);

            using var builder = renderGraph.AddUnsafePass<PassData>("ShaderDebugBegin", out var pd);
            pd.Buffer = buffer;
            pd.Header = _header;
            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) =>
            {
                var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
                cmd.SetBufferData(data.Buffer, data.Header);
            });
        }

        private static unsafe uint BitsOf(float f) => *(uint*)&f;
    }

    /// <summary>End-of-frame debug geometry draw + viz overlay + readbacks (EndFrameAndOutput).</summary>
    public class NativeRtxptShaderDebugDrawPass : ScriptableRenderPass, IDisposable
    {
        private const uint kRGBA16F = (uint)DXGI_FORMAT.DXGI_FORMAT_R16G16B16A16_FLOAT;

        // ExecuteIndirect replacement: instances drawn each frame; the VS culls against the GPU
        // counter, so this only caps how many debug primitives can be displayed per frame.
        private const int kGeometryInstanceBudget = 65536;

        private const uint kTopologyLineList = 2; // D3D_PRIMITIVE_TOPOLOGY_LINELIST

        private readonly NativeRasterPipeline      _trianglesRaster;
        private readonly NativeRasterDescriptorSet _trianglesDs;
        private readonly NativeRasterPipeline      _linesRaster;
        private readonly NativeRasterDescriptorSet _linesDs;
        private readonly NativeRasterPipeline      _feedbackLinesRaster;
        private readonly NativeRasterDescriptorSet _feedbackLinesDs;
        private readonly NativeRasterPipeline      _blendVizRaster;
        private readonly NativeRasterDescriptorSet _blendVizDs;

        private readonly uint[] _colorFmt = { kRGBA16F };

        private NativeRtxptPassContext _ctx;
        private IntPtr                 _targetPtr;
        private int2                   _targetRes;

        private static int s_pendingPrintReadbacks;
        private static int s_pendingFeedbackReadbacks;

        public NativeRtxptShaderDebugDrawPass(NativeRasterShader trianglesShader,
                                              NativeRasterShader linesShader,
                                              NativeRasterShader feedbackLinesShader,
                                              NativeRasterShader blendVizShader)
        {
            var alphaTris = NativeRenderPlugin.RasterPipelineStateDesc.FullscreenOpaque(
                kRGBA16F, NativeRenderPlugin.RasterPipelineStateDesc.TopologyTriangleList);
            alphaTris.blendMode = NativeRenderPlugin.RasterPipelineStateDesc.BlendModeAlpha;

            var alphaLines = alphaTris;
            alphaLines.primitiveTopology = kTopologyLineList;

            var alphaStrip = NativeRenderPlugin.RasterPipelineStateDesc.FullscreenOpaque(kRGBA16F);
            alphaStrip.blendMode = NativeRenderPlugin.RasterPipelineStateDesc.BlendModeAlpha;

            _trianglesRaster     = new NativeRasterPipeline(trianglesShader, alphaTris);
            _trianglesDs         = new NativeRasterDescriptorSet(_trianglesRaster);
            _linesRaster         = new NativeRasterPipeline(linesShader, alphaLines);
            _linesDs             = new NativeRasterDescriptorSet(_linesRaster);
            _feedbackLinesRaster = new NativeRasterPipeline(feedbackLinesShader, alphaLines);
            _feedbackLinesDs     = new NativeRasterDescriptorSet(_feedbackLinesRaster);
            _blendVizRaster      = new NativeRasterPipeline(blendVizShader, alphaStrip);
            _blendVizDs          = new NativeRasterDescriptorSet(_blendVizRaster);
        }

        public void Dispose()
        {
            _trianglesDs?.Dispose();
            _trianglesRaster?.Dispose();
            _linesDs?.Dispose();
            _linesRaster?.Dispose();
            _feedbackLinesDs?.Dispose();
            _feedbackLinesRaster?.Dispose();
            _blendVizDs?.Dispose();
            _blendVizRaster?.Dispose();
        }

        /// <summary>Target = the LDR/display image the output blit will show (C++ LdrColor).</summary>
        public void Setup(NativeRtxptPassContext ctx, NriTextureResource target, int2 targetRes)
        {
            _ctx       = ctx;
            _targetPtr = target.NativePtr;
            _targetRes = targetRes;
        }

        private class PassData
        {
            internal NativeRasterPipeline      TrianglesRaster, LinesRaster, FeedbackLinesRaster, BlendVizRaster;
            internal NativeRasterDescriptorSet TrianglesDs, LinesDs, FeedbackLinesDs, BlendVizDs;
            internal NativeRtxptPassContext    Ctx;
            internal IntPtr                    TargetPtr;
            internal int2                      TargetRes;
            internal uint[]                    ColorFmt;
            internal IntPtr[]                  ColorRes;
            internal bool                      DrawFeedbackLines;
            internal bool                      DrawVizOverlay;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            var s = _ctx?.Setting;
            if (s == null || !s.enableShaderDebug || _ctx.Buffers?.ShaderDebugBuffer == null)
                return;

            using var builder = renderGraph.AddUnsafePass<PassData>("ShaderDebug", out var pd);
            pd.TrianglesRaster     = _trianglesRaster;
            pd.TrianglesDs         = _trianglesDs;
            pd.LinesRaster         = _linesRaster;
            pd.LinesDs             = _linesDs;
            pd.FeedbackLinesRaster = _feedbackLinesRaster;
            pd.FeedbackLinesDs     = _feedbackLinesDs;
            pd.BlendVizRaster      = _blendVizRaster;
            pd.BlendVizDs          = _blendVizDs;
            pd.Ctx                 = _ctx;
            pd.TargetPtr           = _targetPtr;
            pd.TargetRes           = _targetRes;
            pd.ColorFmt            = _colorFmt;
            pd.ColorRes            = new IntPtr[1];
            pd.DrawFeedbackLines   = s.showDebugLines;
            // While a debug view is selected the output blit shows the viz texture fullscreen;
            // blending it over the LDR image as well would double-display it.
            pd.DrawVizOverlay      = s.debugViewType == RtxptDebugViewType.Disabled
                                     && s.showMode != NativeRtxptShowMode.NEELightColor;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var ctx = data.Ctx;
            var buf = ctx.Buffers;
            var res = ctx.Textures;

            cmd.BeginSample("ShaderDebug");

            var viewport = new Rect(0, 0, data.TargetRes.x, data.TargetRes.y);
            data.ColorRes[0] = data.TargetPtr;

            IntPtr debugBufPtr = buf.ShaderDebugBufferPtr;
            IntPtr depthPtr    = res.Depth.NativePtr;
            IntPtr vizPtr      = res.ShaderDebugViz.NativePtr;

            // 1. DebugTriangle geometry (ShaderDebug.cpp:372 DrawCurrentBufferGeometry, triangles).
            {
                var ds = data.TrianglesDs;
                ds.SetBuffer("t_ShaderDebugBuffer", debugBufPtr);
                ds.SetTexture("t_Depth", depthPtr);
                var draw = new RasterDrawDesc
                {
                    numRenderTargets = 1, colorResources = data.ColorRes, colorFormats = data.ColorFmt,
                    depthResource = IntPtr.Zero, viewport = viewport,
                    vertexCount = 3, instanceCount = kGeometryInstanceBudget,
                };
                data.TrianglesRaster.Draw(cmd, ds, in draw);
            }

            // 2. DebugLine geometry (same buffer, line pipeline).
            {
                var ds = data.LinesDs;
                ds.SetBuffer("t_ShaderDebugBuffer", debugBufPtr);
                ds.SetTexture("t_Depth", depthPtr);
                ds.SetTexture("t_DebugVizOutput", vizPtr);
                var draw = new RasterDrawDesc
                {
                    numRenderTargets = 1, colorResources = data.ColorRes, colorFormats = data.ColorFmt,
                    depthResource = IntPtr.Zero, viewport = viewport,
                    vertexCount = 3, instanceCount = kGeometryInstanceBudget,
                };
                data.LinesRaster.Draw(cmd, ds, in draw);
            }

            // 3. Picked-pixel debug lines (Sample.cpp:2177; needs ENABLE_DEBUG_LINES_VIZ=1 shaders).
            if (data.DrawFeedbackLines && buf.DebugLinesBuffer != null)
            {
                var ds = data.FeedbackLinesDs;
                ds.SetStructuredBuffer("t_DebugLines", buf.DebugLinesBufferPtr,
                                       NativeRtxptBufferResources.MaxDebugLines,
                                       NativeRtxptBufferResources.DebugLineStructSize);
                ds.SetBuffer("t_ShaderDebugBuffer", debugBufPtr);
                ds.SetBuffer("t_Feedback", buf.FeedbackBufferPtr);
                ds.SetTexture("t_Depth", depthPtr);
                var draw = new RasterDrawDesc
                {
                    numRenderTargets = 1, colorResources = data.ColorRes, colorFormats = data.ColorFmt,
                    depthResource = IntPtr.Zero, viewport = viewport,
                    vertexCount = NativeRtxptBufferResources.MaxDebugLines, instanceCount = 1,
                };
                data.FeedbackLinesRaster.Draw(cmd, ds, in draw);
            }

            // 4. Debug-viz overlay (ShaderDebug.cpp:204 — alpha blend of the viz texture).
            if (data.DrawVizOverlay)
            {
                var ds = data.BlendVizDs;
                ds.SetTexture("t_DebugVizOutput", vizPtr);
                var draw = new RasterDrawDesc
                {
                    numRenderTargets = 1, colorResources = data.ColorRes, colorFormats = data.ColorFmt,
                    depthResource = IntPtr.Zero, viewport = viewport,
                    vertexCount = 4, instanceCount = 1,
                };
                data.BlendVizRaster.Draw(cmd, ds, in draw);
            }

            // 5. DebugPrint readback → Unity console (OutputLastBufferPrints port).
            if (s_pendingPrintReadbacks < 3)
            {
                s_pendingPrintReadbacks++;
                cmd.RequestAsyncReadback(buf.ShaderDebugBuffer,
                    NativeRtxptBufferResources.ShaderDebugNoTrianglesBytes, 0,
                    req =>
                    {
                        s_pendingPrintReadbacks--;
                        if (!req.hasError)
                            ParseAndLogPrints(req.GetData<byte>());
                    });
            }

            // 6. Pick feedback readback (Sample.cpp:2220 → m_feedbackData).
            bool pick = ctx.Setting.continuousDebugFeedback;
            if (pick && s_pendingFeedbackReadbacks < 3)
            {
                s_pendingFeedbackReadbacks++;
                cmd.RequestAsyncReadback(buf.FeedbackBuffer,
                    NativeRtxptBufferResources.FeedbackStructSize, 0,
                    req =>
                    {
                        s_pendingFeedbackReadbacks--;
                        if (!req.hasError)
                            ParseFeedback(req.GetData<byte>());
                    });
            }

            cmd.EndSample("ShaderDebug");
        }

        // ── DebugFeedbackStruct parsing (PathTracerDebug.hlsli:170) ───────────
        private static unsafe void ParseFeedback(NativeArray<byte> bytes)
        {
            if (bytes.Length < NativeRtxptBufferResources.FeedbackStructSize) return;
            byte* p = (byte*)bytes.GetUnsafeReadOnlyPtr();

            var fb = new RtxptDebugFeedback
            {
                DebugPrintSlots = new Vector4[16],
                Valid           = true,
            };
            float* f = (float*)p;
            for (int i = 0; i < 16; i++)
                fb.DebugPrintSlots[i] = new Vector4(f[i*4 + 0], f[i*4 + 1], f[i*4 + 2], f[i*4 + 3]);
            int* ints = (int*)(p + 16 * 16);
            fb.LineVertexCount  = ints[0];
            fb.PickedMaterialID = ints[1];
            NativeRtxptShaderDebug.LastFeedback = fb;
        }

        // ── DebugPrint parsing (ShaderDebug.cpp:288 OutputLastBufferPrints) ───
        private const int kHeaderBytes  = NativeRtxptBufferResources.ShaderDebugHeaderBytes;
        private const int kPrintBytes   = NativeRtxptBufferResources.ShaderDebugPrintBytes;
        private const int kMaxPrintArgs = 8; // SHADER_PRINTF_MAX_DEBUG_PRINT_ARGS

        private static unsafe void ParseAndLogPrints(NativeArray<byte> bytes)
        {
            if (bytes.Length < kHeaderBytes) return;
            byte* raw = (byte*)bytes.GetUnsafeReadOnlyPtr();

            int printByteCount = *(int*)raw;
            if (printByteCount <= 0) return;

            bool hadOverflow = printByteCount > kPrintBytes;
            if (hadOverflow) printByteCount = kPrintBytes;

            byte* cur = raw + kHeaderBytes;
            byte* end = cur + Math.Min(printByteCount, bytes.Length - kHeaderBytes);

            var sb = new StringBuilder();
            while (cur + 12 <= end)
            {
                int numBytes   = *(int*)(cur + 0);
                int stringSize = *(int*)(cur + 4);
                int numArgs    = *(int*)(cur + 8);
                cur += 12;
                if (numArgs > kMaxPrintArgs || stringSize > (end - cur) || numBytes < 0)
                    break; // out of bounds — bug somewhere

                byte* itemStart = cur;

                string text = stringSize == 0 ? "" : ReadString(cur, stringSize);
                cur += stringSize;

                sb.Clear();
                string unformatted = null;
                for (int i = 0; i < numArgs; i++)
                {
                    string arg = ProcessArg(ref cur, end);
                    if (arg == null)
                        break;
                    string placeholder = "{" + i + "}";
                    int idx = text.IndexOf(placeholder, StringComparison.Ordinal);
                    if (idx < 0)
                        unformatted = (unformatted ?? " [unformatted args] ") + i + ": " + arg + " ";
                    else
                        text = text.Substring(0, idx) + arg + text.Substring(idx + placeholder.Length);
                }
                if (unformatted != null)
                    text += unformatted;

                Debug.Log("Shader: " + text);

                cur = itemStart + numBytes;
            }

            if (hadOverflow)
                Debug.LogWarning("ShaderDebug: insufficient space in SHADER_DEBUG_PRINT_BUFFER_IN_BYTES to store all DebugPrint-s");
        }

        private static unsafe string ReadString(byte* data, int size)
        {
            // StringSize includes the appended null terminator — trim it and any embedded nulls.
            int len = size;
            while (len > 0 && data[len - 1] == 0) len--;
            return len == 0 ? "" : Encoding.ASCII.GetString(data, len);
        }

        // ShaderDebug.cpp:244 ProcessArg — one type-code byte, then 4 bytes per element.
        private static unsafe string ProcessArg(ref byte* cur, byte* end)
        {
            if (cur + 1 > end) return null;
            int code = *cur;
            cur += 1;

            // ShaderDebugArgCode: 1-4 = uint1-4, 5-8 = int1-4, 9-12 = float1-4.
            if (code < 1 || code > 12) return null;
            int elementCount = (code - 1) % 4 + 1;
            int typeGroup    = (code - 1) / 4; // 0=uint, 1=int, 2=float

            if (cur + elementCount * 4 > end) return null;

            if (elementCount == 1)
                return ReadValue(ref cur, typeGroup);
 
            var sb = new StringBuilder("(");
            for (int i = 0; i < elementCount; i++)
            {
                if (i != 0) sb.Append(", ");
                sb.Append(ReadValue(ref cur, typeGroup));
            }
            return sb.Append(')').ToString();
        }

        private static unsafe string ReadValue(ref byte* cur, int typeGroup)
        {
            string v = typeGroup switch
            {
                0 => (*(uint*)cur).ToString(),
                1 => (*(int*)cur).ToString(),
                _ => (*(float*)cur).ToString("0.######"),
            };
            cur += 4;
            return v;
        }
    }
}
