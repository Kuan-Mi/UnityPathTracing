using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace NativeRender
{
    /// <summary>
    /// Per-draw render-target / viewport / draw-arg description for
    /// <see cref="NativeRasterPipeline.Draw"/>. Color arrays are caller-owned and may be
    /// cached/reused between frames. <c>numRenderTargets</c> entries of
    /// <c>colorResources</c> / <c>colorFormats</c> are read.
    /// </summary>
    public struct RasterDrawDesc
    {
        public uint    numRenderTargets;
        public IntPtr[] colorResources;   // ID3D12Resource* per RT (length >= numRenderTargets)
        public uint[]   colorFormats;     // DXGI_FORMAT per RT (must match the pipeline state)
        public IntPtr  depthResource;     // ID3D12Resource* or IntPtr.Zero
        public uint    depthFormat;       // DXGI_FORMAT for the depth view
        public bool    clearColor;
        public bool    clearDepth;
        public Color   clearColorValue;
        public float   clearDepthValue;
        public Rect    viewport;          // in pixels (x, y, width, height)
        public uint    vertexCount;       // e.g. 3 for a fullscreen triangle
        public uint    instanceCount;     // 0 => 1
    }

    /// <summary>
    /// Manages the D3D12 graphics pipeline (PSO + root signature + slot layout) created from a
    /// <see cref="NativeRasterShader"/> asset and a <see cref="NativeRenderPlugin.RasterPipelineStateDesc"/>.
    ///
    /// Resource bindings live in <see cref="NativeRasterDescriptorSet"/>; render targets are
    /// supplied per draw via <see cref="RasterDrawDesc"/>.
    /// Lifetime: must be explicitly disposed via <see cref="Dispose"/>.
    /// </summary>
    public sealed class NativeRasterPipeline : IDisposable
    {
        private ulong _handle;
        private NativeRasterShader _shader;
        private NativeRenderPlugin.RasterPipelineStateDesc _state;
        private RootConstantsHint[] _rootConstantsHints;
        private string[]            _rootSRVHints;

        private Dictionary<string, uint> _nameToSlot;
        private uint _slotCount;

        public bool IsValid => _handle != 0;

        internal IReadOnlyDictionary<string, uint> NameToSlot => _nameToSlot;
        internal uint SlotCount => _slotCount;
        internal ulong Handle => _handle;

        internal event Action<NativeRasterPipeline> OnRebuilt;

        public NativeRasterPipeline(NativeRasterShader shader,
                                    NativeRenderPlugin.RasterPipelineStateDesc state)
            : this(shader, state,
                   shader != null ? shader.RootConstantsHints : null,
                   shader != null ? shader.RootSRVHints        : null) { }

        public NativeRasterPipeline(NativeRasterShader shader,
                                    NativeRenderPlugin.RasterPipelineStateDesc state,
                                    RootConstantsHint[] rootConstantsHints,
                                    string[] rootSRVHints)
        {
            if (shader == null) throw new ArgumentNullException(nameof(shader));
            _shader             = shader;
            _state              = state;
            _rootConstantsHints = rootConstantsHints;
            _rootSRVHints       = rootSRVHints;
            BuildNativeHandle();
            BuildSlotLayout();
            NativeRasterShader.OnRecompiled += OnShaderRecompiled;
        }

        private void BuildNativeHandle()
        {
            byte[] vs = _shader.GetOrCompileVsDxil();
            byte[] ps = _shader.GetOrCompilePsDxil();
            if (vs == null || vs.Length == 0 || ps == null || ps.Length == 0)
                throw new InvalidOperationException(
                    $"[NativeRasterPipeline] Shader compilation failed for: {_shader.GetHlslPath()}");

            string hintsJson = BuildHintsJson(_rootConstantsHints, _rootSRVHints);
            if (hintsJson != null)
                _handle = NativeRenderPlugin.NR_CreateRasterShaderEx(
                    vs, (uint)vs.Length, ps, (uint)ps.Length, ref _state, _shader.name, hintsJson);
            else
                _handle = NativeRenderPlugin.NR_CreateRasterShader(
                    vs, (uint)vs.Length, ps, (uint)ps.Length, ref _state, _shader.name);

            if (_handle == 0)
                throw new InvalidOperationException(
                    $"[NativeRasterPipeline] NR_CreateRasterShader returned 0 for: {_shader.name}");
        }

        private static string BuildHintsJson(RootConstantsHint[] rcHints, string[] srvHints)
        {
            bool hasRC  = rcHints  != null && rcHints.Length  > 0;
            bool hasSRV = srvHints != null && srvHints.Length > 0;
            if (!hasRC && !hasSRV) return null;

            var sb = new System.Text.StringBuilder();
            sb.Append('{');
            if (hasRC)
            {
                sb.Append("\"rootConstants\":[");
                for (int i = 0; i < rcHints.Length; i++)
                {
                    if (i > 0) sb.Append(',');
                    sb.Append("{\"name\":\"").Append(rcHints[i].Name).Append("\",\"count\":").Append(rcHints[i].Count).Append('}');
                }
                sb.Append(']');
            }
            if (hasSRV)
            {
                if (hasRC) sb.Append(',');
                sb.Append("\"rootSRV\":[");
                for (int i = 0; i < srvHints.Length; i++)
                {
                    if (i > 0) sb.Append(',');
                    sb.Append('"').Append(srvHints[i]).Append('"');
                }
                sb.Append(']');
            }
            sb.Append('}');
            return sb.ToString();
        }

        private void BuildSlotLayout()
        {
            _slotCount  = NativeRenderPlugin.NR_RAS_GetBindingCount(_handle);
            _nameToSlot = new Dictionary<string, uint>((int)_slotCount);

            string json = _shader.ReflectionJson ?? "";
            if (_slotCount > 0 && json.Length > 0)
            {
                int bindingsIdx = json.IndexOf("\"bindings\"", StringComparison.Ordinal);
                int arrayStart  = bindingsIdx >= 0 ? json.IndexOf('[', bindingsIdx) : -1;
                if (arrayStart >= 0)
                {
                    int pos = arrayStart + 1;
                    while (pos < json.Length)
                    {
                        int objStart = json.IndexOf('{', pos);
                        if (objStart < 0) break;
                        int objEnd = json.IndexOf('}', objStart);
                        if (objEnd < 0) break;
                        string obj  = json.Substring(objStart + 1, objEnd - objStart - 1);
                        string name = ExtractJsonString(obj, "name");
                        if (!string.IsNullOrEmpty(name) && !_nameToSlot.ContainsKey(name))
                        {
                            uint idx = NativeRenderPlugin.NR_RAS_GetSlotIndex(_handle, name);
                            if (idx != uint.MaxValue)
                                _nameToSlot[name] = idx;
                        }
                        pos = objEnd + 1;
                    }
                }
            }
        }

        private void OnShaderRecompiled(NativeRasterShader shader)
        {
            if (shader != _shader) return;
            if (_handle != 0)
            {
                using var flushCmd = new CommandBuffer { name = "NR_GpuFlush" };
                flushCmd.IssuePluginEvent(NativeRenderPlugin.NR_GetGpuFlushEventFunc(), 0);
                Graphics.ExecuteCommandBuffer(flushCmd);
                NativeRenderPlugin.NR_WaitForGpuFlush();

                NativeRenderPlugin.NR_DestroyRasterShader(_handle);
                _handle = 0;
            }
            try
            {
                BuildNativeHandle();
                BuildSlotLayout();
                Debug.Log($"[NativeRasterPipeline] Rebuilt pipeline for: {shader.name}");
                OnRebuilt?.Invoke(this);
            }
            catch (Exception e) { Debug.LogError(e.Message); }
        }

        public void Dispose()
        {
            NativeRasterShader.OnRecompiled -= OnShaderRecompiled;
            if (_handle != 0)
            {
                GL.Flush();
                NativeRenderPlugin.NR_DestroyRasterShader(_handle);
                _handle = 0;
            }
        }

        // -------------------------------------------------------------------
        // Draw
        // -------------------------------------------------------------------

        /// <summary>
        /// Snapshots bindings from <paramref name="descriptorSet"/> and enqueues a DrawInstanced
        /// into the CommandBuffer using the render targets / viewport in <paramref name="draw"/>.
        /// </summary>
        public void Draw(CommandBuffer cmd, NativeRasterDescriptorSet descriptorSet, in RasterDrawDesc draw)
        {
            if (!IsValid || descriptorSet == null) return;

            IntPtr headerPtr = descriptorSet.SnapshotAndBuildHeader(in draw);
            if (headerPtr == IntPtr.Zero) return;

            cmd.IssuePluginEventAndData(NativeRenderPlugin.NR_RAS_GetRenderEventFunc(), 1, headerPtr);
        }

        private static string ExtractJsonString(string obj, string key)
        {
            string search = "\"" + key + "\"";
            int ki = obj.IndexOf(search, StringComparison.Ordinal);
            if (ki < 0) return null;
            int colon = obj.IndexOf(':', ki + search.Length);
            if (colon < 0) return null;
            int q1 = obj.IndexOf('"', colon + 1);
            if (q1 < 0) return null;
            int q2 = obj.IndexOf('"', q1 + 1);
            if (q2 < 0) return null;
            return obj.Substring(q1 + 1, q2 - q1 - 1);
        }
    }
}
